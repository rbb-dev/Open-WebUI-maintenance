"""
title: Open WebUI Maintenance Pipe
description: CLI-style maintenance pipe for Open WebUI deployments that audits uploads, storage, and user usage with guided remediation helpers
id: maintenance
version: 0.1.0
author: rbb-dev
author_url: https://github.com/rbb-dev
git_url: https://github.com/rbb-dev/Open-WebUI-maintenance
required_open_webui_version: 0.6.28
license: MIT
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import shlex
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Awaitable, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple
from uuid import UUID

from fastapi import Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

try:
    from open_webui.config import UPLOAD_DIR
except ModuleNotFoundError:  # pragma: no cover - local dev helper
    potential_backend = Path(__file__).resolve().parents[2] / "open-webui" / "backend"
    if potential_backend.exists():
        sys.path.append(str(potential_backend))
        from open_webui.config import UPLOAD_DIR
    else:
        raise
from open_webui.internal.db import get_db
from open_webui.models.chats import Chat
from open_webui.models.files import File, Files
try:  # pragma: no cover - guard for environments without knowledge module
    from open_webui.models.knowledge import KnowledgeFile
except Exception:  # pragma: no cover - fallback when table is absent
    KnowledgeFile = None  # type: ignore
from open_webui.models.users import Users

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


@dataclass
class FileSummary:
    """Lightweight projection of rows from the ``file`` table."""

    id: str
    user_id: Optional[str]
    filename: Optional[str]
    size: Optional[int]
    created_at: Optional[int]
    updated_at: Optional[int]
    meta: Dict[str, Any]
    path: Optional[str]


@dataclass
class DiskObject:
    file_id: Optional[str]
    path: str
    name: str
    size: int
    modified_at: int


@dataclass
class UserUsageStats:
    chat_count: int = 0
    chat_bytes: int = 0
    file_count: int = 0
    file_bytes: int = 0


class StorageInventory:
    """Abstract source of disk objects (local FS, S3, etc.)."""

    def iter_objects(self) -> Iterator[DiskObject]:  # pragma: no cover - interface
        raise NotImplementedError

    def delete(self, disk_object: DiskObject) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class LocalStorageInventory(StorageInventory):
    def __init__(self, root: Optional[str] = None):
        self.root = Path(root or UPLOAD_DIR).expanduser()

    def iter_objects(self) -> Iterator[DiskObject]:
        if not self.root.exists():
            return iter(())
        for entry in self.root.rglob("*"):
            if not entry.is_file():
                continue
            stat = entry.stat()
            yield DiskObject(
                file_id=self._derive_file_id(entry.name),
                path=str(entry),
                name=entry.name,
                size=stat.st_size,
                modified_at=int(stat.st_mtime),
            )

    @staticmethod
    def _derive_file_id(filename: str) -> Optional[str]:
        prefix, _, _ = filename.partition("_")
        try:
            UUID(prefix)
            return prefix
        except Exception:
            return None

    def delete(self, disk_object: DiskObject) -> None:
        try:
            path = Path(disk_object.path)
            if path.exists():
                path.unlink()
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Failed to delete {disk_object.path}: {exc}") from exc


class StorageAuditService:
    def __init__(self, inventory: StorageInventory):
        self.inventory = inventory

    def scan_db_orphans(
        self,
        files: Dict[str, FileSummary],
        *,
        limit: Optional[int],
        status_callback: Optional[Callable[[str, str], None]] = None,
    ) -> Dict[str, Any]:
        disk_index: Set[str] = set()
        missing: List[Dict[str, Any]] = []
        extras: List[Dict[str, Any]] = []
        extra_objects: List[DiskObject] = []
        missing_total = 0
        extras_total = 0
        disk_total = 0

        lim = None if (limit is None or limit == 0) else limit

        for disk_obj in self.inventory.iter_objects():
            disk_total += 1
            if disk_total % 200 == 0 and status_callback:
                status_callback("storage", f"Reviewed {disk_total} files on disk")
            file_id = disk_obj.file_id
            if file_id and file_id in files:
                disk_index.add(file_id)
            else:
                extras_total += 1
                if lim is None or len(extras) < lim:
                    extras.append(
                        {
                            "file_id": file_id,
                            "path": disk_obj.path,
                            "name": disk_obj.name,
                            "size": disk_obj.size,
                            "modified_at": disk_obj.modified_at,
                        }
                    )
                    extra_objects.append(disk_obj)

        for file_id, summary in files.items():
            if file_id not in disk_index:
                missing_total += 1
                if lim is None or len(missing) < lim:
                    missing.append(
                        {
                            "file_id": file_id,
                            "user_id": summary.user_id,
                            "filename": summary.filename,
                            "path": summary.path,
                            "updated_at": summary.updated_at,
                        }
                    )

        return {
            "disk_total": disk_total,
            "disk_without_db_total": extras_total,
            "db_missing_disk_total": missing_total,
            "disk_without_db": extras,
            "disk_without_db_objects": extra_objects,
            "missing_from_disk": missing,
            "has_more_disk": bool(lim and extras_total > lim),
            "has_more_missing": bool(lim and missing_total > lim),
        }


class UploadAuditService:
    """Performs cross-table scans to enumerate orphaned uploads."""

    def __init__(self, chunk_size: int = 400):
        self.chunk_size = max(50, chunk_size)

    def scan_db_orphans(
        self,
        *,
        user_filter: Optional[str],
        file_ids: Optional[Sequence[str]],
        limit: Optional[int],
        status_callback: Optional[Callable[[str, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, Any]:
        """Return file records that no longer have chat/knowledge references."""

        def emit(stage: str, message: str) -> None:
            if status_callback:
                status_callback(stage, message)

        with get_db() as db:
            files = self._load_files(
                db,
                user_filter=user_filter,
                file_ids=file_ids,
                cancel_event=cancel_event,
            )
            emit("files", f"Loaded {len(files)} uploaded files from the database")
            if cancel_event and cancel_event.is_set():
                return self._empty_summary(files)

            chat_refs = self._collect_chat_references(
                db,
                emit,
                cancel_event=cancel_event,
            )
            knowledge_refs = self._collect_knowledge_references(db, emit)
            referenced = chat_refs | knowledge_refs

            orphans: List[Dict[str, Any]] = []
            for record in files.values():
                if cancel_event and cancel_event.is_set():
                    break
                if record.id not in referenced:
                    orphans.append(self._summarize_file(record))

            orphans.sort(key=lambda row: row.get("created_at") or 0)
            has_more = False
            if limit and limit > 0 and len(orphans) > limit:
                has_more = True
                orphans = orphans[:limit]

            return {
                "files_total": len(files),
                "referenced_in_chats": len(chat_refs),
                "referenced_in_knowledge": len(knowledge_refs),
                "orphan_total": len([record for record in files.values() if record.id not in referenced]),
                "orphans": orphans,
                "has_more": has_more,
            }

    def scan_storage_mismatches(
        self,
        *,
        user_filter: Optional[str],
        file_ids: Optional[Sequence[str]],
        inventory: StorageInventory,
        limit: Optional[int],
        status_callback: Optional[Callable[[str, str], None]] = None,
    ) -> Dict[str, Any]:
        with get_db() as db:
            files = self._load_files(
                db,
                user_filter=user_filter,
                file_ids=file_ids,
                cancel_event=None,
            )
        if status_callback:
            status_callback("storage", f"Prepared {len(files)} database records for comparison")
        auditor = StorageAuditService(inventory)
        return auditor.scan_db_orphans(files, limit=limit, status_callback=status_callback)

    def _empty_summary(self, files: Dict[str, FileSummary]) -> Dict[str, Any]:
        return {
            "files_total": len(files),
            "referenced_in_chats": 0,
            "referenced_in_knowledge": 0,
            "orphan_total": 0,
            "orphans": [],
            "has_more": False,
        }

    def _load_files(
        self,
        session,
        *,
        user_filter: Optional[str],
        file_ids: Optional[Sequence[str]],
        cancel_event: Optional[threading.Event],
    ) -> Dict[str, FileSummary]:
        query = session.query(File)
        if user_filter:
            query = query.filter(File.user_id == user_filter)
        if file_ids:
            ids = [fid for fid in file_ids if fid]
            if ids:
                query = query.filter(File.id.in_(ids))
            else:
                return {}

        files: Dict[str, FileSummary] = {}
        for row in query.yield_per(self.chunk_size):
            if cancel_event and cancel_event.is_set():
                break
            record = FileSummary(
                id=row.id,
                user_id=getattr(row, "user_id", None),
                filename=getattr(row, "filename", None),
                size=self._extract_size(getattr(row, "meta", None)),
                created_at=getattr(row, "created_at", None),
                updated_at=getattr(row, "updated_at", None),
                meta=self._normalize_meta(getattr(row, "meta", None)),
                path=getattr(row, "path", None),
            )
            files[record.id] = record
        return files

    def _collect_chat_references(
        self,
        session,
        emit: Callable[[str, str], None],
        *,
        cancel_event: Optional[threading.Event],
    ) -> Set[str]:
        referenced: Set[str] = set()
        processed = 0
        query = session.query(Chat.chat).yield_per(self.chunk_size)
        for row in query:
            if cancel_event and cancel_event.is_set():
                break
            processed += 1
            payload = getattr(row, "chat", {}) or {}
            referenced.update(self._extract_file_ids_from_chat(payload))
            if processed % (self.chunk_size * 2) == 0:
                emit("chats", f"Scanned {processed} chats for attachments")
        emit("chats", f"Scanned {processed} chats for attachments")
        return referenced

    def _collect_knowledge_references(
        self,
        session,
        emit: Callable[[str, str], None],
    ) -> Set[str]:
        if KnowledgeFile is None:  # pragma: no cover - depends on deployment
            emit("knowledge", "Knowledge table unavailable; skipping")
            return set()
        referenced: Set[str] = set()
        query = session.query(KnowledgeFile.file_id).yield_per(self.chunk_size)
        count = 0
        for row in query:
            file_id = getattr(row, "file_id", None)
            if file_id:
                referenced.add(str(file_id))
                count += 1
        emit("knowledge", f"Collected {count} knowledge file references")
        return referenced

    def _extract_size(self, meta: Optional[dict]) -> Optional[int]:
        data = meta or {}
        size = data.get("size") if isinstance(data, dict) else None
        if size is None and isinstance(data, dict):
            inner = data.get("data")
            if isinstance(inner, dict):
                size = inner.get("size")
        try:
            return int(size) if size is not None else None
        except (TypeError, ValueError):
            return None

    def _normalize_meta(self, meta: Optional[Any]) -> Dict[str, Any]:
        if isinstance(meta, dict):
            return meta
        if isinstance(meta, str):
            try:
                return json.loads(meta)
            except json.JSONDecodeError:
                return {}
        return {}

    def _extract_file_ids_from_chat(self, payload: dict) -> Set[str]:
        history = payload.get("history", {}) if isinstance(payload, dict) else {}
        messages = history.get("messages", {}) if isinstance(history, dict) else {}
        file_ids: Set[str] = set()
        if not isinstance(messages, dict):
            return file_ids
        for message in messages.values():
            if not isinstance(message, dict):
                continue
            files = message.get("files")
            if not isinstance(files, list):
                continue
            for entry in files:
                file_id = self._coerce_file_id(entry)
                if file_id:
                    file_ids.add(file_id)
        return file_ids

    def _coerce_file_id(self, entry: Any) -> str:
        if isinstance(entry, str):
            return entry.strip()
        if not isinstance(entry, dict):
            return ""
        for key in ("id", "file_id", "fileId"):
            value = entry.get(key)
            if value:
                return str(value)
        nested = entry.get("file")
        if isinstance(nested, dict):
            for key in ("id", "file_id", "fileId"):
                value = nested.get(key)
                if value:
                    return str(value)
        return ""

    def _summarize_file(self, record: FileSummary) -> Dict[str, Any]:
        content_type = record.meta.get("content_type") if isinstance(record.meta, dict) else None
        return {
            "file_id": record.id,
            "user_id": record.user_id,
            "filename": record.filename or "(unnamed)",
            "size": record.size,
            "content_type": content_type,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "path": record.path,
        }


class Pipe:
    PIPE_ID = "maintenance"
    PIPE_NAME = "Open WebUI: Maintenance"

    class Valves(BaseModel):
        ENABLE_LOGGING: bool = Field(
            default=False,
            description="Emit INFO logs for the maintenance pipe (disabled by default).",
        )
        SCAN_DEFAULT_LIMIT: int = Field(
            default=25,
            ge=0,
            le=5000,
            description="Default number of orphaned files to display (0 = unlimited).",
        )
        SCAN_MAX_LIMIT: int = Field(
            default=500,
            ge=25,
            le=2000,
            description="Hard cap for scan results.",
        )
        STORAGE_SCAN_DEFAULT_LIMIT: int = Field(
            default=25,
            ge=0,
            le=5000,
            description="Default number of storage mismatches to display (0 = unlimited).",
        )
        STORAGE_SCAN_MAX_LIMIT: int = Field(
            default=500,
            ge=25,
            le=2000,
            description="Hard cap for storage scan output and cleanup batch sizes.",
        )
        DB_CHUNK_SIZE: int = Field(
            default=400,
            ge=50,
            le=2000,
            description="Rows fetched per database batch when scanning.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.service = UploadAuditService(chunk_size=self.valves.DB_CHUNK_SIZE)
        self._apply_logging_valve()

    def _apply_logging_valve(self) -> None:
        level = logging.DEBUG if self.valves.ENABLE_LOGGING else logging.INFO
        logger.setLevel(level)
        logger.propagate = True

    async def pipes(self) -> List[dict]:
        return [{"id": self.PIPE_ID, "name": self.PIPE_NAME}]

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __request__: Request,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> StreamingResponse | PlainTextResponse:
        command_text = self._extract_prompt_text(body)
        if not command_text:
            return await self._respond(True, self._help_markdown(), body)

        command, options = self._parse_command(command_text)
        raw_user_filter = options.get("user_id")
        user_filter: Optional[str]
        user_lookup_error: Optional[str] = None
        if raw_user_filter in {"me", "self"}:
            user_filter = __user__.get("id")
        else:
            user_filter, user_lookup_error = self._resolve_user_value(raw_user_filter)
        if user_lookup_error:
            return await self._respond(False, user_lookup_error, body)
        file_ids = options.get("ids") or []
        limit_option = options.get("limit")

        try:
            loop = asyncio.get_running_loop()
            if command in {"", "help"}:
                return await self._respond(True, self._help_markdown(), body)

            if command == "db-scan":
                limit = self._clamp_limit(
                    limit_option,
                    default=self.valves.SCAN_DEFAULT_LIMIT,
                    ceiling=self.valves.SCAN_MAX_LIMIT,
                    allow_zero=True,
                )
                status_callback = self._threadsafe_status_callback(loop, __event_emitter__)
                await self._emit_status(__event_emitter__, "Reviewing uploaded files...")
                result = await asyncio.to_thread(
                    self.service.scan_db_orphans,
                    user_filter=user_filter,
                    file_ids=file_ids,
                    limit=(None if limit == 0 else limit),
                    status_callback=status_callback,
                )
                storage_inventory = LocalStorageInventory()
                storage_scan = await asyncio.to_thread(
                    self.service.scan_storage_mismatches,
                    user_filter=user_filter,
                    file_ids=file_ids,
                    inventory=storage_inventory,
                    limit=(None if limit == 0 else limit),
                    status_callback=status_callback,
                )
                await self._emit_status(__event_emitter__, "Database scan complete", done=True)
                label_ids: List[str] = []
                label_ids.extend([row.get("user_id") for row in result["orphans"] if row.get("user_id")])
                label_ids.extend(
                    [row.get("user_id") for row in storage_scan.get("missing_from_disk", []) if row.get("user_id")]
                )
                user_labels = await self._resolve_user_labels(label_ids)
                scope = self._describe_scope(user_filter, file_ids)
                message = self._build_db_scan_report(
                    result,
                    missing_rows=storage_scan.get("missing_from_disk", []),
                    missing_total=storage_scan.get("db_missing_disk_total", 0),
                    missing_truncated=storage_scan.get("has_more_missing", False),
                    user_labels=user_labels,
                    scope=scope,
                    limit=limit,
                )
                return await self._respond(True, message, body)

            if command == "storage-scan":
                limit = self._clamp_limit(
                    limit_option,
                    default=self.valves.STORAGE_SCAN_DEFAULT_LIMIT,
                    ceiling=self.valves.STORAGE_SCAN_MAX_LIMIT,
                    allow_zero=True,
                )
                inventory = LocalStorageInventory()
                status_callback = self._threadsafe_status_callback(loop, __event_emitter__)
                await self._emit_status(__event_emitter__, "Reviewing files on disk...")
                result = await asyncio.to_thread(
                    self.service.scan_storage_mismatches,
                    user_filter=user_filter,
                    file_ids=file_ids,
                    inventory=inventory,
                    limit=(None if limit == 0 else limit),
                    status_callback=status_callback,
                )
                await self._emit_status(__event_emitter__, "Storage review complete", done=True)
                scope = self._describe_scope(user_filter, file_ids)
                message = self._build_storage_scan_report(result, scope=scope, limit=limit, user_labels={})
                return await self._respond(True, message, body)

            if command == "user-report":
                user_records = await asyncio.to_thread(self._load_user_records, user_filter)
                user_records = self._sort_user_records(user_records)
                if not user_records:
                    scope = self._describe_scope(user_filter, [])
                    message = f"No user data found ({scope})."
                    return await self._respond(True, message, body)
                status_callback = (
                    self._threadsafe_status_callback(loop, __event_emitter__) if __event_emitter__ else None
                )
                detail_user_id = user_filter if user_filter else None
                detail_label = None
                if detail_user_id:
                    # map to label using loaded records
                    for record in user_records:
                        rid = self._get_user_attribute(record, "id")
                        if rid == detail_user_id:
                            detail_label = self._user_label(record)
                            break
                if body.get("stream"):
                    return await self._stream_user_report(
                        user_records=user_records,
                        status_callback=status_callback,
                        emitter=__event_emitter__,
                        detail_user_id=detail_user_id,
                        detail_label=detail_label,
                    )
                rows = await asyncio.to_thread(
                    self._collect_user_report_rows,
                    user_records,
                    status_callback=status_callback,
                    row_callback=None,
                )
                detail_tables = ""
                if detail_user_id:
                    detail_tables = await asyncio.to_thread(
                        self._build_user_detail_tables,
                        detail_user_id,
                        detail_label or detail_user_id,
                    )
                if __event_emitter__:
                    await self._emit_user_report_table(__event_emitter__, rows, detail_tables=detail_tables)
                message = self._render_user_report(rows)
                if detail_tables:
                    message = f"{message}\n\n{detail_tables}"
                return await self._respond(True, message, body)

            if command == "db-clean":
                if not options.get("confirm"):
                    reminder = "Add `confirm` to remove database records whose files are missing."
                    return await self._respond(False, reminder, body)
                limit = self._clamp_limit(
                    limit_option,
                    default=self.valves.STORAGE_SCAN_DEFAULT_LIMIT,
                    ceiling=self.valves.STORAGE_SCAN_MAX_LIMIT,
                    allow_zero=True,
                )
                inventory = LocalStorageInventory()
                status_callback = self._threadsafe_status_callback(loop, __event_emitter__)
                scan_result = await asyncio.to_thread(
                    self.service.scan_storage_mismatches,
                    user_filter=user_filter,
                    file_ids=file_ids,
                    inventory=inventory,
                    limit=(None if limit == 0 else limit),
                    status_callback=status_callback,
                )
                entries = scan_result.get("missing_from_disk", [])
                if not entries:
                    scope = self._describe_scope(user_filter, file_ids)
                    return await self._respond(True, f"Every database record still has a matching file ({scope}).", body)
                clean_result = await asyncio.to_thread(self._clean_db_entries, entries)
                scope = self._describe_scope(user_filter, file_ids)
                user_labels = await self._resolve_user_labels(
                    [
                        entry.get("user_id")
                        for entry in clean_result.get("deleted_entries", [])
                        if entry.get("user_id")
                    ]
                )
                message = self._build_clean_db_report(clean_result, scope, user_labels)
                return await self._respond(True, message, body)

            if command == "storage-clean":
                if not options.get("confirm"):
                    reminder = "Add `confirm` to delete files on disk that are no longer tracked in the database."
                    return await self._respond(False, reminder, body)
                limit = self._clamp_limit(
                    limit_option,
                    default=self.valves.STORAGE_SCAN_DEFAULT_LIMIT,
                    ceiling=self.valves.STORAGE_SCAN_MAX_LIMIT,
                    allow_zero=True,
                )
                inventory = LocalStorageInventory()
                status_callback = self._threadsafe_status_callback(loop, __event_emitter__)
                scan_result = await asyncio.to_thread(
                    self.service.scan_storage_mismatches,
                    user_filter=user_filter,
                    file_ids=file_ids,
                    inventory=inventory,
                    limit=(None if limit == 0 else limit),
                    status_callback=status_callback,
                )
                disk_objects = scan_result.get("disk_without_db_objects", [])
                if not disk_objects:
                    scope = self._describe_scope(user_filter, file_ids)
                    return await self._respond(True, f"Every file on disk already has a database record ({scope}).", body)
                clean_result = await asyncio.to_thread(
                    self._clean_storage_entries,
                    disk_objects,
                    inventory,
                )
                scope = self._describe_scope(user_filter, file_ids)
                message = self._build_clean_storage_report(clean_result, scope)
                return await self._respond(True, message, body)

            message = (
                f"Unknown command `{command}`. Available commands: help, db-scan, storage-scan, user-report, db-clean, storage-clean.\n\n"
                + self._help_markdown()
            )
            return await self._respond(False, message, body)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.exception("Maintenance pipe failed")
            await self._emit_status(__event_emitter__, "Maintenance command failed", done=True)
            return await self._respond(False, f"Unable to complete command: {exc}", body)

    async def _respond(self, ok: bool, message: str, body: dict):
        is_stream = bool(body.get("stream"))
        status_code = 200 if ok else 400
        if not is_stream:
            safe_message = self._sanitize_output_text(message)
            return PlainTextResponse(safe_message, status_code=status_code)

        async def stream():
            payload = self._format_data(
                is_stream=True,
                model=self.PIPE_NAME,
                content=message,
                finish_reason="stop" if ok else "error",
            )
            yield payload
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    def _threadsafe_status_callback(
        self,
        loop: asyncio.AbstractEventLoop,
        emitter: Optional[Callable[[dict], Awaitable[None]]],
    ) -> Callable[[str, str], None]:
        def _callback(stage: str, message: str) -> None:
            if not emitter:
                return
            future = asyncio.run_coroutine_threadsafe(
                self._emit_status(emitter, f"{stage.capitalize()}: {message}"),
                loop,
            )
            future.add_done_callback(self._log_future_exception)

        return _callback

    @staticmethod
    def _log_future_exception(future):  # pragma: no cover - best-effort logging
        try:
            future.result()
        except Exception as exc:
            logger.debug("Status emission failed: %s", exc)

    async def _emit_status(
        self,
        emitter: Optional[Callable[[dict], Awaitable[None]]],
        message: str,
        *,
        done: bool = False,
    ) -> None:
        if emitter:
            await emitter({"type": "status", "data": {"description": message, "done": done}})

    async def _emit_message_chunk(
        self,
        emitter: Optional[Callable[[dict], Awaitable[None]]],
        content: Optional[str],
    ) -> None:
        if not emitter or not content:
            return
        safe = self._sanitize_output_text(content)
        if not safe:
            return
        await emitter({"type": "message", "data": {"content": safe}})

    def _parse_command(self, text: str):
        try:
            tokens = shlex.split(text.strip())
        except ValueError:
            tokens = text.strip().split()
        if not tokens:
            return "", {}
        command = tokens[0].lower()
        options: Dict[str, Any] = {"limit": None, "ids": [], "user_id": None, "confirm": False, "path": None}
        pending_key: Optional[str] = None
        pending_value: List[str] = []

        def assign_option(key: str, value: str) -> None:
            key = key.strip().lower()
            value = value.strip()
            if not key or not value:
                return
            if key == "limit":
                try:
                    options["limit"] = int(value)
                except ValueError:
                    pass
            elif key in {"id", "file", "file_id"}:
                ids = [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]
                options["ids"].extend(ids)
            elif key in {"user", "user_id"}:
                options["user_id"] = value
            elif key in {"path", "prefix"}:
                options["path"] = value

        recognized_keys = {"limit", "id", "file", "file_id", "user", "user_id", "path", "prefix"}

        def flush_pending() -> None:
            nonlocal pending_key, pending_value
            if pending_key and pending_value:
                assign_option(pending_key, " ".join(pending_value))
            pending_key = None
            pending_value = []

        for token in tokens[1:]:
            stripped = token.strip()
            if not stripped:
                continue
            lower = stripped.lower()
            if lower == "confirm":
                flush_pending()
                options["confirm"] = True
                continue
            if stripped == "=":
                continue
            if "=" in stripped:
                flush_pending()
                key, value = stripped.split("=", 1)
                assign_option(key, value)
                continue
            if pending_key:
                if lower in recognized_keys:
                    flush_pending()
                    pending_key = lower
                else:
                    pending_value.append(stripped)
                continue
            if lower in recognized_keys:
                flush_pending()
                pending_key = lower
                continue
        flush_pending()
        return command, options

    def _clamp_limit(
        self,
        value: Optional[int],
        *,
        default: int,
        ceiling: int,
        allow_zero: bool,
    ) -> int:
        if value is None:
            return default
        if value == 0 and allow_zero:
            return 0
        if value <= 0:
            return default
        return min(value, ceiling)

    def _describe_scope(self, user_filter: Optional[str], file_ids: Sequence[str]) -> str:
        scope = []
        if user_filter:
            scope.append(f"user `{user_filter}`")
        if file_ids:
            scope.append(f"specific file IDs ({len(file_ids)})")
        return ", ".join(scope) if scope else "entire workspace"

    def _build_db_scan_report(
        self,
        result: Dict[str, Any],
        *,
        missing_rows: List[Dict[str, Any]],
        missing_total: int,
        missing_truncated: bool,
        user_labels: Dict[str, str],
        scope: str,
        limit: int,
    ) -> str:
        lines = ["### Database scan results", ""]
        lines.append(f"- Files in database: {result['files_total']}")
        lines.append(f"- Referenced in chats: {result['referenced_in_chats']}")
        lines.append(f"- Referenced in knowledge bases: {result['referenced_in_knowledge']}")
        lines.append(f"- Files with no remaining references: {result['orphan_total']}")
        lines.append(f"- Database records with missing files on disk: {missing_total}")
        lines.append(f"- Scope: {scope}")
        if limit == 0:
            lines.append("- Output limit: unlimited")
        else:
            lines.append(f"- Output limit: {limit or self.valves.SCAN_DEFAULT_LIMIT}")
        if result.get("has_more"):
            lines.append("- Some orphaned files were not shown because of the output limit.")
        if missing_truncated:
            lines.append("- Some missing files were not shown because of the output limit.")
        lines.append("")

        orphans = result.get("orphans", [])
        if orphans:
            lines.append("#### Files with no remaining references")
            lines.append("| File ID | Owner | Name | Size | Created (UTC) | Path |")
            lines.append("| --- | --- | --- | --- | --- | --- |")
            for row in orphans:
                owner = user_labels.get(row.get("user_id"), row.get("user_id") or "—")
                file_id = row.get("file_id") or "?"
                name = self._shorten(row.get("filename") or "(unnamed)")
                size = self._format_filesize(row.get("size"))
                created = self._format_timestamp(row.get("created_at"))
                path = self._shorten(row.get("path") or "local upload dir")
                lines.append(f"| `{file_id}` | {owner} | {name} | {size} | {created} | {path} |")
            lines.append("")
        else:
            lines.append("No orphaned files detected. Storage matches chats and knowledge bases.")
            lines.append("")

        if missing_rows:
            lines.append("#### Database records with missing files")
            lines.append("| File ID | Owner | Name | Last Known Path | Updated (UTC) |")
            lines.append("| --- | --- | --- | --- | --- |")
            for row in missing_rows:
                owner = user_labels.get(row.get("user_id"), row.get("user_id") or "—")
                updated = self._format_timestamp(row.get("updated_at"))
                name = self._shorten(row.get("filename") or "(unnamed)")
                path = self._shorten(row.get("path") or "—")
                lines.append(f"| `{row.get('file_id')}` | {owner} | {name} | {path} | {updated} |")
        else:
            lines.append("No database records are missing their underlying files.")

        lines.append("")
        lines.append("Use `db-scan limit=0` to list every result or add `user=<id>` to focus on a single owner.")
        return "\n".join(lines)

    def _build_storage_scan_report(
        self,
        result: Dict[str, Any],
        *,
        scope: str,
        limit: int,
        user_labels: Dict[str, str],
    ) -> str:
        lines = ["### Storage scan results", ""]
        lines.append(f"- Files found on disk: {result['disk_total']}")
        lines.append(f"- Files on disk with no database record: {result['disk_without_db_total']}")
        if result.get("has_more_disk"):
            lines.append("  - Some files without database records were not shown because of the limit.")
        lines.append(f"- Scope: {scope}")
        if limit == 0:
            lines.append("- Output limit: unlimited")
        else:
            lines.append(f"- Output limit: {limit or self.valves.STORAGE_SCAN_DEFAULT_LIMIT}")
        lines.append("")

        extra_files = result.get("disk_without_db", [])
        if extra_files:
            lines.append("#### Files on disk without database records")
            lines.append("| Derived ID | Name | Size | Modified (UTC) | Path |")
            lines.append("| --- | --- | --- | --- | --- |")
            for item in extra_files:
                size = self._format_filesize(item.get("size"))
                modified = self._format_timestamp(item.get("modified_at"))
                name = self._shorten(item.get("name") or "(unnamed)")
                path = self._shorten(item.get("path") or "—")
                derived = item.get("file_id") or "—"
                lines.append(f"| `{derived}` | {name} | {size} | {modified} | {path} |")
        else:
            lines.append("No files on disk are missing database records.")

        return "\n".join(lines)

    def _load_user_records(self, user_filter: Optional[str]) -> List[Any]:
        if user_filter:
            user = Users.get_user_by_id(user_filter)
            return [user] if user else []
        result = Users.get_users()
        return result.get("users", []) if isinstance(result, dict) else []

    def _estimate_chat_bytes(self, payload: Any) -> int:
        if payload is None:
            return 0
        try:
            serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            return len(serialized.encode("utf-8"))
        except Exception:
            return len(str(payload).encode("utf-8"))

    def _calculate_usage_for_user(self, session, user_id: str) -> UserUsageStats:
        stats = UserUsageStats()
        chat_query = (
            session.query(Chat.chat)
            .filter(Chat.user_id == user_id)
            .yield_per(self.service.chunk_size)
        )
        for row in chat_query:
            stats.chat_count += 1
            stats.chat_bytes += self._estimate_chat_bytes(getattr(row, "chat", None))

        file_query = (
            session.query(File.meta)
            .filter(File.user_id == user_id)
            .yield_per(self.service.chunk_size)
        )
        for row in file_query:
            stats.file_count += 1
            size = self.service._extract_size(getattr(row, "meta", None)) or 0
            stats.file_bytes += size
        return stats

    def _collect_user_report_rows(
        self,
        user_records: Sequence[Any],
        *,
        status_callback: Optional[Callable[[str, str], None]],
        row_callback: Optional[Callable[[Optional[Dict[str, Any]]], None]],
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        if not user_records:
            if row_callback:
                row_callback(None)
            return rows

        with get_db() as db:
            for user in user_records:
                uid = self._get_user_attribute(user, "id")
                if not uid:
                    continue
                label_plain = self._user_label(user)
                if status_callback:
                    safe_label = self._sanitize_output_text(label_plain)
                    status_callback("users", f"Processing user {safe_label}")
                stats = self._calculate_usage_for_user(db, uid)
                total_bytes = stats.chat_bytes + stats.file_bytes
                row = {
                    "id": uid,
                    "label": self._sanitize_output_text(label_plain),
                    "label_plain": label_plain,
                    "chat_count": stats.chat_count,
                    "chat_bytes": stats.chat_bytes,
                    "file_count": stats.file_count,
                    "file_bytes": stats.file_bytes,
                    "total_bytes": total_bytes,
                }
                rows.append(row)
                if row_callback:
                    row_callback(row)

        if row_callback:
            row_callback(None)
        return rows

    def _get_top_user_chats(
        self,
        session,
        user_id: str,
        *,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        import heapq

        top: List[Tuple[int, str]] = []
        query = (
            session.query(Chat.title, Chat.chat)
            .filter(Chat.user_id == user_id)
            .yield_per(self.service.chunk_size)
        )
        for row in query:
            size = self._estimate_chat_bytes(getattr(row, "chat", None))
            if size <= 0:
                continue
            title = getattr(row, "title", None) or "(untitled chat)"
            heapq.heappush(top, (size, title))
            if len(top) > limit:
                heapq.heappop(top)
        top.sort(reverse=True)
        return [{"title": title, "size": size} for size, title in top]

    def _get_top_user_files(
        self,
        session,
        user_id: str,
        *,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        import heapq

        top: List[Tuple[int, str]] = []
        query = (
            session.query(File.filename, File.meta)
            .filter(File.user_id == user_id)
            .yield_per(self.service.chunk_size)
        )
        for row in query:
            size = self.service._extract_size(getattr(row, "meta", None)) or 0
            if size <= 0:
                continue
            name = getattr(row, "filename", None) or "(unnamed file)"
            heapq.heappush(top, (size, name))
            if len(top) > limit:
                heapq.heappop(top)
        top.sort(reverse=True)
        return [{"name": name, "size": size} for size, name in top]

    def _build_user_detail_tables(
        self,
        user_id: str,
        label: str,
        *,
        limit: int = 10,
    ) -> str:
        with get_db() as db:
            chats = self._get_top_user_chats(db, user_id, limit=limit)
            files = self._get_top_user_files(db, user_id, limit=limit)
        lines: List[str] = []
        safe_label = self._sanitize_output_text(label)
        if chats:
            lines.append(f"#### Largest chats for {safe_label}")
            lines.append("| Title | Size |")
            lines.append("| --- | --- |")
            for item in chats:
                title = self._shorten(item["title"])
                size = self._format_filesize(item["size"])
                lines.append(f"| {title} | {size} |")
            lines.append("")
        if files:
            lines.append(f"#### Largest files for {safe_label}")
            lines.append("| File name | Size |")
            lines.append("| --- | --- |")
            for item in files:
                name = self._shorten(item["name"])
                size = self._format_filesize(item["size"])
                lines.append(f"| {name} | {size} |")
        return "\n".join(lines).strip()

    @staticmethod
    def _get_user_attribute(user: Any, attr: str) -> Optional[str]:
        if hasattr(user, attr):
            value = getattr(user, attr)
        elif isinstance(user, dict):
            value = user.get(attr)
        else:
            value = None
        return str(value) if value is not None else None

    def _resolve_user_value(self, value: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        if value is None:
            return None, None
        candidate = value.strip()
        if not candidate:
            return None, None
        try:
            direct_user = Users.get_user_by_id(candidate)
        except Exception:
            direct_user = None
        if direct_user:
            return self._get_user_attribute(direct_user, "id"), None
        try:
            payload = Users.get_users()
        except Exception:
            payload = []
        users: List[Any]
        if isinstance(payload, dict):
            users = payload.get("users", []) or []
        elif isinstance(payload, list):
            users = payload
        else:
            users = []
        candidate_cf = candidate.casefold()
        exact: List[Any] = []
        partial: List[Any] = []
        for user in users:
            label = self._user_label(user)
            if not label:
                continue
            label_cf = label.casefold()
            if label_cf == candidate_cf:
                exact.append(user)
            elif candidate_cf in label_cf:
                partial.append(user)
        matches = exact or partial
        if len(matches) == 1:
            resolved = matches[0]
            return self._get_user_attribute(resolved, "id"), None
        if len(matches) > 1:
            labels = ", ".join(sorted({self._user_label(match) for match in matches}))
            return None, f"Multiple users match '{value}': {labels}. Please specify a unique ID or email."
        return None, f"No user found matching '{value}'. Provide a user ID or exact name/email."

    def _user_label(self, user: Any) -> str:
        for attr in ("name", "username", "email"):
            value = self._get_user_attribute(user, attr)
            if value:
                return value
        return self._get_user_attribute(user, "id") or "Unknown user"

    def _sort_user_records(self, users: Sequence[Any]) -> List[Any]:
        if not users:
            return []
        return sorted(
            [user for user in users if user is not None],
            key=lambda user: self._user_label(user).casefold(),
        )

    def _prepare_user_report_rows(
        self,
        users: Sequence[Any],
        usage: Dict[str, UserUsageStats],
        *,
        limit: int,
    ) -> List[Dict[str, Any]]:
        if not users:
            return []

        rows: List[Dict[str, Any]] = []
        for user in users:
            uid = self._get_user_attribute(user, "id")
            if not uid:
                continue
            stats = usage.get(uid, UserUsageStats())
            total_bytes = stats.chat_bytes + stats.file_bytes
            label_plain = self._user_label(user)
            rows.append(
                {
                    "id": uid,
                    "label": self._sanitize_output_text(label_plain),
                    "label_plain": label_plain,
                    "chat_count": stats.chat_count,
                    "chat_bytes": stats.chat_bytes,
                    "file_count": stats.file_count,
                    "file_bytes": stats.file_bytes,
                    "total_bytes": total_bytes,
                }
            )

        rows.sort(key=lambda row: row["label_plain"].casefold())
        if limit > 0:
            rows = rows[:limit]
        return rows

    def _format_user_report_row(self, row: Dict[str, Any]) -> str:
        chats_display = f"{row['chat_count']} ({self._format_filesize(row['chat_bytes'])})"
        files_display = f"{row['file_count']} ({self._format_filesize(row['file_bytes'])})"
        total_display = self._format_filesize(row["total_bytes"])
        return f"| {row['label']} | {chats_display} | {files_display} | {total_display} |"

    def _user_report_header(self) -> str:
        return "\n".join(
            [
                "### User usage report",
                "",
                "| User | Chats | Files | Total storage |",
                "| --- | --- | --- | --- |",
            ]
        )

    def _user_report_summary(self, rows: Sequence[Dict[str, Any]]) -> str:
        count = len(rows)
        label = "user" if count == 1 else "users"
        return f"\n\nReport complete – displayed {count} {label}."

    def _render_user_report(self, rows: Sequence[Dict[str, Any]]) -> str:
        if not rows:
            return "No user data available."
        lines = [self._user_report_header()]
        for row in rows:
            lines.append(self._format_user_report_row(row))
        lines.append(self._user_report_summary(rows))
        return "\n".join(lines)

    async def _emit_user_report_table(
        self,
        emitter: Optional[Callable[[dict], Awaitable[None]]],
        rows: Sequence[Dict[str, Any]],
        *,
        detail_tables: str = "",
    ) -> None:
        if not emitter:
            return
        header = self._user_report_header()
        await self._emit_message_chunk(emitter, header)
        if not rows:
            await self._emit_message_chunk(emitter, "\nNo user data available.")
        else:
            for row in rows:
                await self._emit_message_chunk(emitter, "\n" + self._format_user_report_row(row))
            await self._emit_message_chunk(emitter, self._user_report_summary(rows))
        if detail_tables:
            await self._emit_message_chunk(emitter, "\n\n" + detail_tables)

    async def _stream_user_report(
        self,
        *,
        user_records: Sequence[Any],
        status_callback: Optional[Callable[[str, str], None]],
        emitter: Optional[Callable[[dict], Awaitable[None]]],
        detail_user_id: Optional[str],
        detail_label: Optional[str],
    ) -> StreamingResponse:
        header = self._user_report_header()
        loop = asyncio.get_running_loop()
        queue: "asyncio.Queue[Optional[Dict[str, Any]]]" = asyncio.Queue()

        def _row_callback(row: Optional[Dict[str, Any]]) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, row)

        async def worker():
            return await asyncio.to_thread(
                self._collect_user_report_rows,
                user_records,
                status_callback=status_callback,
                row_callback=_row_callback,
            )

        worker_task = asyncio.create_task(worker())

        async def stream():
            await self._emit_message_chunk(emitter, header)
            yield self._format_data(
                is_stream=True,
                model=self.PIPE_NAME,
                content=header,
                finish_reason=None,
            )
            while True:
                row = await queue.get()
                if row is None:
                    break
                line = "\n" + self._format_user_report_row(row)
                await self._emit_message_chunk(emitter, line)
                yield self._format_data(
                    is_stream=True,
                    model=self.PIPE_NAME,
                    content=line,
                    finish_reason=None,
                )
            rows = await worker_task
            if not rows:
                final = "\nNo user data available."
                await self._emit_message_chunk(emitter, final)
                yield self._format_data(
                    is_stream=True,
                    model=self.PIPE_NAME,
                    content=final,
                    finish_reason="stop",
                )
                yield "data: [DONE]\n\n"
                return
            summary = self._user_report_summary(rows)
            await self._emit_message_chunk(emitter, summary)
            detail_text = ""
            if detail_user_id:
                detail_text = await asyncio.to_thread(
                    self._build_user_detail_tables,
                    detail_user_id,
                    detail_label or detail_user_id,
                )
            finish_reason = "stop" if not detail_text else None
            yield self._format_data(
                is_stream=True,
                model=self.PIPE_NAME,
                content=summary,
                finish_reason=finish_reason,
            )
            if detail_text:
                chunk = "\n\n" + detail_text
                await self._emit_message_chunk(emitter, chunk)
                yield self._format_data(
                    is_stream=True,
                    model=self.PIPE_NAME,
                    content=chunk,
                    finish_reason="stop",
                )
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    def _build_user_report(
        self,
        users: Sequence[Any],
        usage: Dict[str, UserUsageStats],
        *,
        limit: int,
    ) -> str:
        rows = self._prepare_user_report_rows(users, usage, limit=limit)
        return self._render_user_report(rows)

    def _clean_db_entries(self, entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        deleted: List[str] = []
        deleted_entries: List[Dict[str, Any]] = []
        failed: List[Dict[str, str]] = []
        entries_by_id = {entry.get("file_id"): entry for entry in entries if entry.get("file_id")}
        for entry in entries:
            file_id = entry.get("file_id")
            if not file_id:
                continue
            error: Optional[str] = None
            try:
                result = Files.delete_file_by_id(file_id)
            except Exception as exc:
                result = False
                error = str(exc)
            if result:
                deleted.append(file_id)
                if file_id in entries_by_id:
                    deleted_entries.append(entries_by_id[file_id])
            else:
                failed.append({"file_id": file_id, "error": error or "Delete failed"})
        return {"deleted": deleted, "deleted_entries": deleted_entries, "failed": failed}

    def _clean_storage_entries(
        self,
        disk_objects: List[DiskObject],
        inventory: StorageInventory,
    ) -> Dict[str, Any]:
        deleted: List[str] = []
        deleted_objects: List[Dict[str, Any]] = []
        failed: List[Dict[str, str]] = []
        for obj in disk_objects:
            try:
                inventory.delete(obj)
                deleted.append(obj.path)
                deleted_objects.append(
                    {
                        "file_id": obj.file_id,
                        "name": obj.name,
                        "size": obj.size,
                        "modified_at": obj.modified_at,
                        "path": obj.path,
                    }
                )
            except Exception as exc:
                failed.append({"path": obj.path, "error": str(exc)})
        return {"deleted": deleted, "deleted_objects": deleted_objects, "failed": failed}

    def _build_clean_db_report(
        self,
        result: Dict[str, Any],
        scope: str,
        user_labels: Dict[str, str],
    ) -> str:
        lines = ["### Database cleanup results", ""]
        lines.append(f"- Scope: {scope}")
        lines.append(f"- Database records removed: {len(result['deleted'])}")
        lines.append(f"- Failures: {len(result['failed'])}")
        if result["deleted"]:
            lines.append("- File IDs removed: " + ", ".join(f"`{fid}`" for fid in result["deleted"]))
        if result["failed"]:
            lines.append("\n#### Failures")
            lines.append("| File ID | Error |")
            lines.append("| --- | --- |")
            for row in result["failed"]:
                lines.append(f"| `{row['file_id']}` | {row['error']} |")
        deleted_entries = result.get("deleted_entries", [])
        if deleted_entries:
            lines.append("\n#### Deleted records")
            lines.append("| File ID | Owner | Name | Last Known Path | Updated (UTC) |")
            lines.append("| --- | --- | --- | --- | --- |")
            for entry in deleted_entries:
                owner = user_labels.get(entry.get("user_id"), entry.get("user_id") or "—")
                updated = self._format_timestamp(entry.get("updated_at"))
                name = self._shorten(entry.get("filename") or "(unnamed)")
                path = self._shorten(entry.get("path") or "—")
                lines.append(f"| `{entry.get('file_id')}` | {owner} | {name} | {path} | {updated} |")
        return "\n".join(lines)

    def _build_clean_storage_report(self, result: Dict[str, Any], scope: str) -> str:
        lines = ["### Storage cleanup results", ""]
        lines.append(f"- Scope: {scope}")
        lines.append(f"- Files deleted: {len(result['deleted'])}")
        lines.append(f"- Failures: {len(result['failed'])}")
        if result["failed"]:
            lines.append("\n#### Failures")
            lines.append("| Path | Error |")
            lines.append("| --- | --- |")
            for row in result["failed"]:
                lines.append(f"| {self._shorten(row['path'])} | {row['error']} |")
        deleted_objects = result.get("deleted_objects", [])
        if deleted_objects:
            lines.append("\n#### Deleted files on disk")
            lines.append("| Derived ID | Name | Size | Modified (UTC) | Path |")
            lines.append("| --- | --- | --- | --- | --- |")
            for item in deleted_objects:
                size = self._format_filesize(item.get("size"))
                modified = self._format_timestamp(item.get("modified_at"))
                name = self._shorten(item.get("name") or "(unnamed)")
                path = self._shorten(item.get("path") or "—")
                derived = item.get("file_id") or "—"
                lines.append(f"| `{derived}` | {name} | {size} | {modified} | {path} |")
        return "\n".join(lines)

    def _format_filesize(self, value: Optional[int]) -> str:
        if value is None or value < 0:
            return "—"
        if value == 0:
            return "0 B"
        units = ["B", "KB", "MB", "GB", "TB"]
        idx = min(int(math.log(value, 1024)), len(units) - 1)
        scaled = value / (1024 ** idx)
        return f"{scaled:.2f} {units[idx]}"

    def _format_timestamp(self, value: Optional[int]) -> str:
        if not value:
            return "—"
        try:
            dt = datetime.fromtimestamp(int(value), tz=timezone.utc)
            return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        except Exception:
            return str(value)

    def _shorten(self, text: str, limit: int = 60) -> str:
        safe = self._sanitize_output_text(text)
        safe = safe.replace("|", "\\|")
        return safe

    def _sanitize_output_text(self, text: Optional[str]) -> str:
        if text is None:
            return ""
        value = str(text)
        if not value:
            return ""
        builder: List[str] = []
        i = 0
        length = len(value)
        while i < length:
            ch = value[i]
            code = ord(ch)
            if ch == "\x00":
                i += 1
                continue
            if 0xD800 <= code <= 0xDBFF:
                if i + 1 < length:
                    next_code = ord(value[i + 1])
                    if 0xDC00 <= next_code <= 0xDFFF:
                        builder.append(ch)
                        builder.append(value[i + 1])
                        i += 2
                        continue
                builder.append("\ufffd")
                i += 1
                continue
            if 0xDC00 <= code <= 0xDFFF:
                builder.append("\ufffd")
                i += 1
                continue
            builder.append(ch)
            i += 1
        return "".join(builder)

    async def _resolve_user_labels(self, user_ids: Iterable[str]) -> Dict[str, str]:
        unique_ids = sorted({uid for uid in user_ids if uid})
        if not unique_ids:
            return {}

        def _load(ids: Sequence[str]):
            mapping: Dict[str, str] = {}
            for uid in ids:
                try:
                    user = Users.get_user_by_id(uid)
                except Exception:
                    user = None
                if user:
                    label = getattr(user, "name", None) or getattr(user, "username", None) or getattr(user, "email", None)
                    mapping[uid] = label or uid
                else:
                    mapping[uid] = uid
            return mapping

        return await asyncio.to_thread(_load, unique_ids)

    def _extract_prompt_text(self, body: dict) -> str:
        messages = body.get("messages") or []
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            text = self._collapse_content(message.get("content"))
            if text:
                return text.strip()
        for fallback_key in ("prompt", "input", "text"):
            if fallback_key in body and body[fallback_key]:
                return str(body[fallback_key]).strip()
        return ""

    def _collapse_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            return " ".join(parts)
        if isinstance(content, dict):
            if content.get("type") == "text":
                return content.get("text", "")
            if "content" in content:
                return str(content["content"])
        return ""

    def _help_markdown(self) -> str:
        return """# Open WebUI Maintenance Pipe

Use this pipe as a chat-style runbook for keeping Open WebUI deployments tidy. Each section below explains what the maintenance task does, how it evaluates data, and when administrators typically run it.

## Database maintenance
These commands operate only on the `file` table and its metadata.

- `db-scan [limit=<n>] [user=<id>|user=me] [id=<file-id>]`
  - **Purpose:** Identify uploads that are no longer referenced by chats or knowledge bases, and highlight database rows whose underlying files have vanished from storage.
  - **How it works:** Reads the `file` table in batches, extracts file IDs from `history.messages.*.files[]` and the knowledge tables, and optionally scopes the work via `user=`/`id=` filters. Matching storage lookups confirm whether each database row still has a binary on disk. Set `limit=0` to return everything; otherwise the limit caps both orphan listings and missing-file rows.
  - **When to run:** After content migrations, following bulk deletions, or on a regular cadence (weekly/monthly) to prevent unreferenced data from accumulating.

- `db-clean confirm [limit=<n>] [...]`
  - **Purpose:** Remove the database rows that `db-scan` proved were already missing their binaries.
  - **Safety:** Requires the literal word `confirm`, respects `limit`, deletes in small batches by default, and prints a table of every row removed (owner, filename, path, last update) for compliance records.
  - **Best practice:** Start with conservative limits (for example `limit=10`) in production and widen only after verifying results.

## Storage maintenance
These commands walk the upload directory and never touch database rows directly.

- `storage-scan [limit=<n>] [user=<id>|user=me] [id=<file-id>]`
  - **Purpose:** Locate files that live on disk but no longer have a database record—common after manual filesystem work or interrupted uploads.
  - **How it works:** Recursively traverses `UPLOAD_DIR`, derives file IDs from Open WebUI’s `UUID_prefix` naming convention, and compares each file to the `file` table. Anything that cannot be matched is reported along with size, timestamp, and path. The `limit` parameter controls how many such files are listed; `limit=0` lifts the cap.
  - **When to run:** During capacity reviews, when monitoring shows unexplained disk growth, or immediately after restoring/uploading data outside the normal pipeline.

- `storage-clean confirm [limit=<n>] [...]`
  - **Purpose:** Delete the disk files that `storage-scan` identified as untracked, freeing space without touching legitimate uploads.
  - **Safety:** Requires `confirm`, honors `limit`, and outputs a table of every deleted file (derived ID, name, size, path, last modified) so that enterprise admins can document the remediation.
  - **Best practice:** Combine with `storage-scan` output and process in small increments, especially on shared storage.

## User usage reporting
These commands summarize end-user footprints so you can plan capacity or chargeback.

- `user-report [user=<id>|user=me]`
  - **Purpose:** Produce a per-user table that lists chat counts and sizes, file counts and sizes, plus the combined storage footprint.
  - **How it works:** Resolves each user in scope, aggregates their chats (measuring JSON size in bytes) and file metadata, then sorts alphabetically so the full roster streams consistently.
  - **When to run:** Before large cleanup projects, when preparing usage reports for stakeholders, or to identify the accounts consuming the most storage.

## General guidance
1. Commands never delete anything unless you intentionally run `db-clean` or `storage-clean` with `confirm`.
2. Ownership labels use `file.user_id` and are resolved to user-friendly names where possible.
3. Paths shown in storage scans reference `UPLOAD_DIR` by default; future releases will add additional inventory backends for S3 or other providers.
4. Combine `limit`, `user`, and `id` options to align maintenance actions with enterprise change-control procedures.
"""

    def _format_data(
        self,
        *,
        is_stream: bool,
        model: str,
        content: Optional[str],
        usage: Optional[dict] = None,
        finish_reason: Optional[str] = None,
    ) -> str:
        safe_content = (
            self._sanitize_output_text(content) if isinstance(content, str) else content
        )
        data: Dict[str, Any] = {
            "id": f"chat.{int(time.time()*1000)}",
            "object": "chat.completion.chunk" if is_stream else "chat.completion",
            "created": int(time.time()),
            "model": model,
        }
        if is_stream:
            delta: Dict[str, Any] = {}
            if safe_content is not None:
                delta = {"role": "assistant", "content": safe_content}
            data["choices"] = [
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                    "delta": delta,
                }
            ]
        else:
            data["choices"] = [
                {
                    "index": 0,
                    "finish_reason": finish_reason or "stop",
                    "message": {"role": "assistant", "content": safe_content or ""},
                }
            ]
        if usage:
            data["usage"] = usage
        return f"data: {json.dumps(data)}\n\n"
