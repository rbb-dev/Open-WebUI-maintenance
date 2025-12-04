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
import base64
import contextlib
import io
import json
import logging
import math
import mimetypes
import re
import shlex
import threading
import time
import uuid
from collections import Counter, defaultdict
from concurrent.futures import Future
from dataclasses import dataclass, field
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
from open_webui.models.files import File, Files, FileForm
try:  # pragma: no cover - guard for environments without knowledge module
    from open_webui.models.knowledge import KnowledgeFile
except Exception:  # pragma: no cover - fallback when table is absent
    KnowledgeFile = None  # type: ignore
from open_webui.models.users import Users
from open_webui.storage.provider import Storage
from sqlalchemy import func, or_
from sqlalchemy.orm.attributes import flag_modified

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
    """File discovered on disk during a storage inventory walk."""

    file_id: Optional[str]
    path: str
    name: str
    size: int
    modified_at: int


@dataclass
class UserUsageStats:
    """Aggregated chat/file counts and byte totals for a single user."""

    chat_count: int = 0
    chat_bytes: int = 0
    file_count: int = 0
    file_bytes: int = 0


@dataclass
class ChatSanitizeReport:
    """Per-chat summary describing what would change after sanitization."""

    changed: bool
    counts: Counter = field(default_factory=Counter)
    fields: List[str] = field(default_factory=list)
    sanitized_values: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InlineImageMatch:
    """Single inline-image match pulled from chat content."""

    alt_text: str
    mime_type: str
    base64_data: str
    approx_bytes: int


@dataclass
class InlineImageUserSummary:
    """Per-user byte summary returned by image scans."""

    user_id: Optional[str]
    chats_with_inline: int
    inline_images: int
    inline_bytes: int


@dataclass
class InlineImageDetachRecord:
    """Audit entry describing the images detached from one chat."""

    chat_id: str
    user_id: Optional[str]
    images_detached: int
    bytes_detached: int
    file_ids: List[str]


@dataclass
class InlineImageSkipRecord:
    """Per-chat summary for inline images we couldn't detach."""

    chat_id: str
    user_id: Optional[str]
    skipped_images: int
    skipped_bytes: int
    reasons: Dict[str, int]


class InlineImageService:
    """Find and normalize inline base64 images embedded inside chat payloads."""
    MARKDOWN_DATA_URI_PATTERN = re.compile(
        r"!\[(?P<alt>[^\]]*)\]\(data:(?P<mime>[^;]+);base64,(?P<data>[^)]+)\)", re.IGNORECASE
    )
    BARE_DATA_URI_PATTERN = re.compile(
        r"^data:(?P<mime>[^;,]+)(?P<params>(;[^;,]+)*)?;base64,(?P<data>[A-Za-z0-9+/=\s]+)$",
        re.IGNORECASE,
    )
    SUPPORTED_MIME_EXTENSIONS = {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/jpg": "jpg",
        "image/webp": "webp",
        "image/gif": "gif",
    }
    MAX_STRING_MAP_DEPTH = 1000

    def __init__(self, chunk_size: int = 200):
        """Initialize the service with safe database batch sizes."""
        self.chunk_size = max(50, chunk_size)

    def scan(
        self,
        *,
        user_filter: Optional[str],
        user_query: Optional[str],
        chat_ids: Optional[Sequence[str]],
        limit: Optional[int],
        status_callback: Optional[Callable[[str, int], None]],
        result_callback: Optional[Callable[[InlineImageUserSummary], None]],
        user_sorter: Optional[Callable[[str], str]],
        cancel_event: Optional[threading.Event],
    ) -> Dict[str, Any]:
        """Aggregate inline-image usage per user and optionally stream rows."""
        summaries: List[InlineImageUserSummary] = []
        total_users = 0
        total_images = 0
        total_bytes = 0
        total_chats = 0
        has_more = False

        with get_db() as db:
            user_batches = self._collect_user_batches(
                db,
                user_filter=user_filter,
                user_query=user_query,
                chat_ids=chat_ids,
            )
            if user_sorter:
                user_batches.sort(key=lambda batch: user_sorter(batch[0]))
            else:
                user_batches.sort(key=lambda batch: (batch[0] or ""))

            for user_id, chat_count in user_batches:
                if cancel_event and cancel_event.is_set():
                    has_more = True
                    break
                if status_callback:
                    with contextlib.suppress(Exception):
                        status_callback(user_id, chat_count)
                query = self._build_query(db, user_filter=user_id, chat_ids=chat_ids)
                user_stats = self._scan_user_chats(query, cancel_event=cancel_event)
                if user_stats["inline_images"] == 0:
                    continue
                summary = InlineImageUserSummary(
                    user_id=user_id,
                    chats_with_inline=user_stats["chats_with_inline"],
                    inline_images=user_stats["inline_images"],
                    inline_bytes=user_stats["inline_bytes"],
                )
                summaries.append(summary)
                total_users += 1
                total_images += user_stats["inline_images"]
                total_bytes += user_stats["inline_bytes"]
                total_chats += user_stats["chats_with_inline"]
                if result_callback:
                    with contextlib.suppress(Exception):
                        result_callback(summary)
                if limit and limit > 0 and len(summaries) >= limit:
                    has_more = True
                    break

        return {
            "summaries": summaries,
            "users_with_inline": total_users,
            "total_inline_images": total_images,
            "total_inline_bytes": total_bytes,
            "total_chats_with_inline": total_chats,
            "has_more": has_more,
        }

    def detach(
        self,
        *,
        max_chats: Optional[int],
        user_filter: Optional[str],
        user_query: Optional[str],
        chat_ids: Optional[Sequence[str]],
        status_callback: Optional[Callable[[str, int], None]],
        user_sorter: Optional[Callable[[str], str]],
        cancel_event: Optional[threading.Event],
    ) -> Dict[str, Any]:
        """Persist inline blobs as managed files and rewrite chats in place."""
        processed = 0
        total_images = 0
        total_bytes = 0
        has_more = False
        records: List[InlineImageDetachRecord] = []
        skipped_chats = 0
        skipped_images = 0
        skipped_bytes = 0
        skipped_reason_counter: Counter[str] = Counter()
        skipped_records: List[InlineImageSkipRecord] = []

        with get_db() as db:
            user_batches = self._collect_user_batches(
                db,
                user_filter=user_filter,
                user_query=user_query,
                chat_ids=chat_ids,
            )
            if user_sorter:
                user_batches.sort(key=lambda batch: user_sorter(batch[0]))
            else:
                user_batches.sort(key=lambda batch: (batch[0] or ""))

            for user_id, chat_count in user_batches:
                if cancel_event and cancel_event.is_set():
                    has_more = True
                    break
                if status_callback:
                    with contextlib.suppress(Exception):
                        status_callback(user_id, chat_count)
                query = self._build_query(db, user_filter=user_id, chat_ids=chat_ids)
                for chat in query.yield_per(self.chunk_size):
                    if cancel_event and cancel_event.is_set():
                        has_more = True
                        break
                    result = self._detach_from_chat(db, chat)
                    matches_detected = result.get("matches_detected", result["images_detached"])
                    if matches_detected == 0:
                        continue
                    if result["images_detached"] == 0:
                        skipped_chats += 1
                        skipped_images += result.get("skipped_images", 0)
                        skipped_bytes += result.get("skipped_bytes", 0)
                        skipped_reason_counter.update(result.get("skipped_reasons", {}))
                        skipped_records.append(
                            InlineImageSkipRecord(
                                chat_id=chat.id,
                                user_id=chat.user_id,
                                skipped_images=result.get("skipped_images", 0),
                                skipped_bytes=result.get("skipped_bytes", 0),
                                reasons=dict(result.get("skipped_reasons", {})),
                            )
                        )
                        continue
                    processed += 1
                    total_images += result["images_detached"]
                    total_bytes += result["bytes_detached"]
                    records.append(
                        InlineImageDetachRecord(
                            chat_id=chat.id,
                            user_id=chat.user_id,
                            images_detached=result["images_detached"],
                            bytes_detached=result["bytes_detached"],
                            file_ids=result["file_ids"],
                        )
                    )
                    if max_chats and max_chats > 0 and processed >= max_chats:
                        has_more = True
                        break
                db.commit()
                if has_more:
                    break

        return {
            "processed_chats": processed,
            "images_detached": total_images,
            "bytes_detached": total_bytes,
            "records": records,
            "skipped_chats": skipped_chats,
            "skipped_images": skipped_images,
            "skipped_bytes": skipped_bytes,
            "skipped_reason_counts": dict(skipped_reason_counter),
            "skipped_records": skipped_records,
            "has_more": has_more,
        }

    def _scan_user_chats(self, query, cancel_event: Optional[threading.Event]) -> Dict[str, int]:
        """Walk chats for a single user and return aggregate inline-image counts."""
        chats_with_inline = 0
        inline_images = 0
        inline_bytes = 0
        for chat in query.yield_per(self.chunk_size):
            if cancel_event and cancel_event.is_set():
                break
            matches = self._extract_inline_images(chat.chat)
            if not matches:
                continue
            chats_with_inline += 1
            inline_images += len(matches)
            inline_bytes += sum(match.approx_bytes for match in matches)
        return {
            "chats_with_inline": chats_with_inline,
            "inline_images": inline_images,
            "inline_bytes": inline_bytes,
        }

    def _detach_from_chat(self, session, chat: Chat) -> Dict[str, Any]:
        """Handle inline image replacement for a single chat row."""
        payload = self._ensure_payload(chat.chat)

        def handler(text: str):
            return self._replace_inline_images(text, chat.user_id)

        try:
            new_payload, matches_info = self._map_strings(payload, handler)
        except ValueError:
            return {
                "images_detached": 0,
                "bytes_detached": 0,
                "file_ids": [],
                "matches_detected": 0,
                "skipped_images": 0,
                "skipped_bytes": 0,
                "skipped_reasons": Counter(),
            }
        if not matches_info:
            return {
                "images_detached": 0,
                "bytes_detached": 0,
                "file_ids": [],
                "matches_detected": 0,
                "skipped_images": 0,
                "skipped_bytes": 0,
                "skipped_reasons": Counter(),
            }

        successes = [entry for entry in matches_info if entry.get("status") == "detached"]
        skipped = [entry for entry in matches_info if entry.get("status") == "skipped"]
        skipped_counter = Counter(entry.get("reason", "unknown") for entry in skipped)
        skipped_bytes = sum(int(entry.get("bytes", 0)) for entry in skipped)
        if not successes:
            return {
                "images_detached": 0,
                "bytes_detached": 0,
                "file_ids": [],
                "matches_detected": len(skipped),
                "skipped_images": len(skipped),
                "skipped_bytes": skipped_bytes,
                "skipped_reasons": skipped_counter,
            }

        chat.chat = new_payload
        chat.updated_at = max(int(time.time()), chat.updated_at or 0)
        flag_modified(chat, "chat")
        session.flush()

        file_ids = [entry["file_id"] for entry in successes]
        total_bytes = sum(int(entry.get("bytes", 0)) for entry in successes)
        return {
            "images_detached": len(successes),
            "bytes_detached": total_bytes,
            "file_ids": file_ids,
            "matches_detected": len(successes) + len(skipped),
            "skipped_images": len(skipped),
            "skipped_bytes": skipped_bytes,
            "skipped_reasons": skipped_counter,
        }

    def _replace_inline_images(
        self,
        text: str,
        user_id: Optional[str],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Replace inline Markdown blobs in a string and return the replacements."""
        trimmed = text.strip()
        bare_data = self._parse_data_uri(trimmed)
        if bare_data:
            file_id, error_reason = self._persist_inline_image(user_id, bare_data["mime"], bare_data["data"])
            approx_bytes = self._estimate_base64_size(bare_data["data"])
            if not file_id:
                return text, [
                    {
                        "status": "skipped",
                        "reason": error_reason or "persist_failed",
                        "bytes": approx_bytes,
                        "mime": bare_data["mime"],
                    }
                ]
            replacement_url = f"/api/v1/files/{file_id}/content"
            if trimmed == text:
                new_text = replacement_url
            else:
                new_text = text.replace(trimmed, replacement_url, 1)
            return new_text, [
                {"status": "detached", "file_id": file_id, "bytes": approx_bytes}
            ]

        matches = list(self.MARKDOWN_DATA_URI_PATTERN.finditer(text))
        if not matches:
            return text, []
        rebuilt: List[str] = []
        last = 0
        entries: List[Dict[str, Any]] = []
        for match in matches:
            rebuilt.append(text[last : match.start()])
            alt_text = match.group("alt") or ""
            mime_type = (match.group("mime") or "").lower()
            base64_data = match.group("data") or ""
            file_id, error_reason = self._persist_inline_image(user_id, mime_type, base64_data)
            approx_bytes = self._estimate_base64_size(base64_data)
            if not file_id:
                entries.append(
                    {
                        "status": "skipped",
                        "reason": error_reason or "persist_failed",
                        "bytes": approx_bytes,
                        "mime": mime_type,
                    }
                )
                rebuilt.append(match.group(0))
                last = match.end()
                continue
            replacement = f"![{alt_text}](/api/v1/files/{file_id}/content)"
            rebuilt.append(replacement)
            entries.append({"status": "detached", "file_id": file_id, "bytes": approx_bytes})
            last = match.end()
        rebuilt.append(text[last:])
        return "".join(rebuilt), entries

    def _persist_inline_image(
        self,
        user_id: Optional[str],
        mime_type: str,
        base64_data: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Decode, validate, and store a single base64 payload via Storage."""
        mime = mime_type.lower() if mime_type else "image/png"
        extension = self.SUPPORTED_MIME_EXTENSIONS.get(mime)
        if not extension and mime.startswith("image/"):
            guessed = mimetypes.guess_extension(mime)
            if guessed:
                extension = guessed.lstrip(".")
        if not extension:
            return None, "unsupported_mime"
        try:
            cleaned = re.sub(r"\s+", "", base64_data or "")
            binary = base64.b64decode(cleaned, validate=True)
        except Exception:
            return None, "decode_error"
        if not binary:
            return None, "empty_payload"
        file_uuid = str(uuid.uuid4())
        filename = f"inline-image-{file_uuid}.{extension}"
        try:
            _, file_path = Storage.upload_file(
                io.BytesIO(binary),
                filename,
                {
                    "OpenWebUI-User-Id": (user_id or "system"),
                    "OpenWebUI-File-Id": file_uuid,
                },
            )
        except Exception:
            return None, "storage_error"
        meta = {
            "name": filename,
            "content_type": mime,
            "size": len(binary),
            "source": "maintenance:image-detach",
        }
        form = FileForm(
            id=file_uuid,
            filename=filename,
            path=file_path,
            data={},
            meta=meta,
        )
        inserted = Files.insert_new_file(user_id or "system", form)
        if inserted:
            return inserted.id, None
        return None, "db_error"

    def _collect_user_batches(
        self,
        session,
        *,
        user_filter: Optional[str],
        user_query: Optional[str],
        chat_ids: Optional[Sequence[str]],
    ) -> List[Tuple[str, int]]:
        """Return (user_id, chat_count) pairs honoring current filters."""
        query = session.query(Chat.user_id, func.count(Chat.id))
        if user_filter:
            query = query.filter(Chat.user_id == user_filter)
        if user_query:
            user_ids = self._lookup_user_ids_by_query(session, user_query)
            if not user_ids:
                return []
            query = query.filter(Chat.user_id.in_(user_ids))
        if chat_ids:
            ids = [cid for cid in chat_ids if cid]
            if ids:
                query = query.filter(Chat.id.in_(ids))
            else:
                return []
        query = query.group_by(Chat.user_id)
        return [(row[0], int(row[1])) for row in query]

    def _lookup_user_ids_by_query(self, session, query: str, limit: int = 50) -> List[str]:
        """Fuzzy match users and return a bounded list of IDs."""
        from open_webui.models.users import User

        text = (query or "").strip()
        if not text:
            return []
        escaped = text.replace("%", "\\%").replace("_", "\\_")
        like_pattern = f"%{escaped}%"
        user_rows = (
            session.query(User.id)
            .filter(
                or_(
                    User.name.ilike(like_pattern),
                    User.username.ilike(like_pattern),
                    User.email.ilike(like_pattern),
                )
            )
            .limit(limit)
            .all()
        )
        return [row[0] for row in user_rows]

    def _build_query(
        self,
        session,
        *,
        user_filter: Optional[str],
        chat_ids: Optional[Sequence[str]],
    ):
        """Create a chat query scoped to one user or explicit chat IDs."""
        query = session.query(Chat)
        if user_filter:
            query = query.filter(Chat.user_id == user_filter)
        if chat_ids:
            ids = [cid for cid in chat_ids if cid]
            if ids:
                query = query.filter(Chat.id.in_(ids))
            else:
                query = query.filter(False)
        return query.order_by(Chat.updated_at.desc())

    def _extract_inline_images(self, payload: Any) -> List[InlineImageMatch]:
        """Traverse a payload and collect every inline-image reference."""
        data = self._ensure_payload(payload)
        def handler(text: str):
            found = []
            trimmed = (text or "").strip()
            bare = self._parse_data_uri(trimmed)
            if bare:
                found.append(
                    InlineImageMatch(
                        alt_text="",
                        mime_type=bare["mime"],
                        base64_data=bare["data"],
                        approx_bytes=self._estimate_base64_size(bare["data"]),
                    )
                )
            for detected in self.MARKDOWN_DATA_URI_PATTERN.finditer(text):
                base64_data = detected.group("data") or ""
                mime_type = (detected.group("mime") or "").lower()
                approx_bytes = self._estimate_base64_size(base64_data)
                found.append(
                    InlineImageMatch(
                        alt_text=detected.group("alt") or "",
                        mime_type=mime_type,
                        base64_data=base64_data,
                        approx_bytes=approx_bytes,
                    )
                )
            return text, found

        try:
            _, found_matches = self._map_strings(data, handler)
        except ValueError:
            return []
        return [match for match in found_matches if isinstance(match, InlineImageMatch)]

    def _map_strings(
        self,
        node: Any,
        handler: Callable[[str], Tuple[str, List[Any]]],
        _depth: int = 0,
    ) -> Tuple[Any, List[Any]]:
        """Apply `handler` to every string in a nested structure, returning matches.

        Returns a NEW structure with transformed strings. Does not mutate the input.
        """
        if _depth > self.MAX_STRING_MAP_DEPTH:
            raise ValueError("Nested payload exceeded safe recursion depth")
        if isinstance(node, str):
            new_value, matches = handler(node)
            return new_value, matches
        if isinstance(node, list):
            new_list = []
            results = []
            for item in node:
                new_value, found = self._map_strings(item, handler, _depth + 1)
                new_list.append(new_value)
                results.extend(found)
            return new_list, results
        if isinstance(node, tuple):
            updated = []
            results = []
            for item in node:
                new_value, found = self._map_strings(item, handler, _depth + 1)
                updated.append(new_value)
                results.extend(found)
            return tuple(updated), results
        if isinstance(node, dict):
            new_dict = {}
            results = []
            for key, value in node.items():
                new_value, found = self._map_strings(value, handler, _depth + 1)
                new_dict[key] = new_value
                results.extend(found)
            return new_dict, results
        return node, []

    def _ensure_payload(self, payload: Any) -> Any:
        """Coerce serialized chat payloads into JSON-compatible Python objects."""
        if isinstance(payload, (dict, list, tuple)):
            return payload
        if isinstance(payload, str):
            trimmed = payload.strip()
            if trimmed.startswith("{") or trimmed.startswith("["):
                try:
                    return json.loads(payload)
                except Exception:
                    return payload
            return payload
        return payload

    @staticmethod
    def _estimate_base64_size(data: str) -> int:
        """Return a quick byte estimate for base64 data so reports stay lightweight."""
        if not data:
            return 0
        cleaned = re.sub(r"\s+", "", data)
        length = len(cleaned)
        if not length:
            return 0
        padding = cleaned.count("=")
        return max(0, (length * 3) // 4 - padding)

    def _parse_data_uri(self, text: str) -> Optional[Dict[str, str]]:
        """Extract MIME/data components from a bare `data:` URI string."""
        match = self.BARE_DATA_URI_PATTERN.match(text or "")
        if not match:
            return None
        mime = (match.group("mime") or "image/png").lower()
        data = match.group("data") or ""
        return {"mime": mime, "data": data}

class ChatRepairService:
    """Encapsulates scanning and repair logic so it can run in a thread."""

    DETAIL_SAMPLE_MAX = 20

    def __init__(self, chunk_size: int = 200):
        self.chunk_size = max(50, chunk_size)

    def scan(
        self,
        *,
        max_results: int,
        user_filter: Optional[str] = None,
        user_query: Optional[str] = None,
        chat_ids: Optional[Sequence[str]] = None,
        status_callback: Optional[Callable[[str, int], None]] = None,
        result_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        user_sorter: Optional[Callable[[str], str]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, Any]:
        """Return chats that would change if sanitized."""

        summary_counter: Counter = Counter()
        matches: List[Dict[str, Any]] = []
        examined = 0
        has_more = False

        with get_db() as db:
            user_batches = self._collect_user_batches(
                db,
                user_filter=user_filter,
                user_query=user_query,
                chat_ids=chat_ids,
            )
            if user_sorter:
                user_batches.sort(key=lambda batch: user_sorter(batch[0]))
            else:
                user_batches.sort(key=lambda batch: (batch[0] or ""))

            for user_id, chat_count in user_batches:
                if cancel_event and cancel_event.is_set():
                    has_more = True
                    break
                if status_callback:
                    with contextlib.suppress(Exception):
                        status_callback(user_id, chat_count)

                query = self._build_query(
                    db,
                    user_filter=user_id,
                    chat_ids=chat_ids,
                )
                iterator = query.yield_per(self.chunk_size)
                for chat in iterator:
                    if cancel_event and cancel_event.is_set():
                        has_more = True
                        break
                    examined += 1
                    report = self._analyse_chat(chat, mutate=False)
                    if not report.changed:
                        continue
                    summary_counter.update(report.counts)
                    row = {
                        "chat_id": chat.id,
                        "user_id": chat.user_id,
                        "title": chat.title or "(untitled)",
                        "updated_at": chat.updated_at,
                        "issue_counts": dict(report.counts),
                        "fields": list(report.fields),
                    }
                    matches.append(row)
                    if result_callback:
                        with contextlib.suppress(Exception):
                            result_callback(row)
                    if max_results and len(matches) >= max_results:
                        has_more = True
                        break
                if has_more:
                    break

        return {
            "examined": examined,
            "matches": len(matches),
            "results": matches,
            "counters": dict(summary_counter),
            "has_more": has_more,
        }

    def repair(
        self,
        *,
        max_repairs: Optional[int],
        user_filter: Optional[str] = None,
        user_query: Optional[str] = None,
        chat_ids: Optional[Sequence[str]] = None,
        status_callback: Optional[Callable[[str, int], None]] = None,
        user_sorter: Optional[Callable[[str], str]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, Any]:
        """Sanitize chats in-place and commit changes."""

        repaired = 0
        examined = 0
        has_more = False
        summary_counter: Counter = Counter()
        details: List[Dict[str, Any]] = []

        with get_db() as db:
            try:
                user_batches = self._collect_user_batches(
                    db,
                    user_filter=user_filter,
                    user_query=user_query,
                    chat_ids=chat_ids,
                )
                if user_sorter:
                    user_batches.sort(key=lambda batch: user_sorter(batch[0]))
                else:
                    user_batches.sort(key=lambda batch: (batch[0] or ""))

                for user_id, chat_count in user_batches:
                    if cancel_event and cancel_event.is_set():
                        has_more = True
                        break
                    if status_callback:
                        with contextlib.suppress(Exception):
                            status_callback(user_id, chat_count)

                    query = self._build_query(
                        db,
                        user_filter=user_id,
                        chat_ids=chat_ids,
                    )
                    iterator = query.yield_per(self.chunk_size)
                    for chat in iterator:
                        if cancel_event and cancel_event.is_set():
                            has_more = True
                            break
                        examined += 1
                        report = self._analyse_chat(chat, mutate=True)
                        if not report.changed:
                            continue
                        summary_counter.update(report.counts)
                        repaired += 1
                        if len(details) < self.DETAIL_SAMPLE_MAX:
                            details.append(
                                {
                                    "chat_id": chat.id,
                                    "user_id": chat.user_id,
                                    "title": chat.title or "(untitled)",
                                    "issue_counts": dict(report.counts),
                                    "fields": list(report.fields),
                                    "updated_at": chat.updated_at,
                                }
                            )
                        if max_repairs and repaired >= max_repairs:
                            has_more = True
                            break
                    if has_more:
                        break

                if repaired:
                    db.commit()
                else:
                    db.rollback()
            except Exception:
                db.rollback()
                raise

        return {
            "examined": examined,
            "repaired": repaired,
            "details": details,
            "counters": dict(summary_counter),
            "has_more": has_more,
        }

    def _build_query(
        self,
        session,
        *,
        user_filter: Optional[str],
        chat_ids: Optional[Sequence[str]],
    ):
        query = session.query(Chat)
        if user_filter:
            query = query.filter(Chat.user_id == user_filter)
        if chat_ids:
            ids = [cid for cid in chat_ids if cid]
            if ids:
                query = query.filter(Chat.id.in_(ids))
            else:
                query = query.filter(False)
        return query.order_by(Chat.updated_at.desc())

    def _collect_user_batches(
        self,
        session,
        *,
        user_filter: Optional[str],
        user_query: Optional[str],
        chat_ids: Optional[Sequence[str]],
    ) -> List[Tuple[str, int]]:
        query = session.query(Chat.user_id, func.count(Chat.id))
        if user_filter:
            query = query.filter(Chat.user_id == user_filter)
        if user_query:
            user_ids = self._lookup_user_ids_by_query(session, user_query)
            if not user_ids:
                return []
            query = query.filter(Chat.user_id.in_(user_ids))
        if chat_ids:
            ids = [cid for cid in chat_ids if cid]
            if ids:
                query = query.filter(Chat.id.in_(ids))
            else:
                return []
        query = query.group_by(Chat.user_id)
        return [(row[0], int(row[1])) for row in query]

    def _lookup_user_ids_by_query(self, session, query: str, limit: int = 50) -> List[str]:
        from open_webui.models.users import User

        text = (query or "").strip()
        if not text:
            return []
        escaped = text.replace("%", "\\%").replace("_", "\\_")
        like_pattern = f"%{escaped}%"
        user_rows = (
            session.query(User.id)
            .filter(
                or_(
                    User.name.ilike(like_pattern),
                    User.username.ilike(like_pattern),
                    User.email.ilike(like_pattern),
                )
            )
            .limit(limit)
            .all()
        )
        return [row[0] for row in user_rows]

    def _analyse_chat(self, chat: Chat, *, mutate: bool) -> ChatSanitizeReport:
        counts: Counter = Counter()
        fields: List[str] = []
        sanitized_values: Dict[str, Any] = {}

        title_value, title_changed, title_counts = self._sanitize_value(chat.title)
        counts.update(title_counts)
        if title_changed:
            sanitized_values["title"] = title_value
            fields.append("title")
            if mutate:
                chat.title = title_value

        chat_payload, chat_changed, chat_counts = self._sanitize_value(chat.chat)
        counts.update(chat_counts)
        if chat_changed:
            sanitized_values["chat"] = chat_payload
            fields.append("chat")
            if mutate:
                chat.chat = chat_payload

        meta_payload, meta_changed, meta_counts = self._sanitize_value(chat.meta)
        counts.update(meta_counts)
        if meta_changed:
            sanitized_values["meta"] = meta_payload
            fields.append("meta")
            if mutate:
                chat.meta = meta_payload

        if mutate and sanitized_values:
            chat.updated_at = max(int(time.time()), chat.updated_at or 0)

        return ChatSanitizeReport(
            changed=bool(sanitized_values),
            counts=counts,
            fields=fields,
            sanitized_values=sanitized_values,
        )

    def _sanitize_value(self, value: Any):
        """Sanitize any JSON-compatible value and report counts."""

        if value is None:
            return value, False, Counter()

        if isinstance(value, str):
            sanitized, counts, changed = self._sanitize_string(value)
            return sanitized if changed else value, changed, counts

        if isinstance(value, list):
            changed = False
            counts: Counter = Counter()
            new_items: List[Any] = []
            for item in value:
                sanitized_item, item_changed, item_counts = self._sanitize_value(item)
                counts.update(item_counts)
                changed = changed or item_changed
                new_items.append(sanitized_item)
            return (new_items if changed else value), changed, counts

        if isinstance(value, tuple):
            changed = False
            counts: Counter = Counter()
            new_items: List[Any] = []
            for item in value:
                sanitized_item, item_changed, item_counts = self._sanitize_value(item)
                counts.update(item_counts)
                changed = changed or item_changed
                new_items.append(sanitized_item)
            sanitized_value = tuple(new_items)
            return (sanitized_value if changed else value), changed, counts

        if isinstance(value, dict):
            changed = False
            counts: Counter = Counter()
            sanitized_dict: Dict[str, Any] = {}
            for key, item in value.items():
                sanitized_item, item_changed, item_counts = self._sanitize_value(item)
                counts.update(item_counts)
                if item_changed:
                    changed = True
                    sanitized_dict[key] = sanitized_item
                else:
                    sanitized_dict[key] = item
            return (sanitized_dict if changed else value), changed, counts

        return value, False, Counter()

    def _sanitize_string(self, value: str):
        if not value:
            return value, Counter(), False

        counts: Counter = Counter()
        builder: List[str] = []
        changed = False
        i = 0
        length = len(value)

        while i < length:
            ch = value[i]
            code = ord(ch)

            if ch == "\x00":
                counts["null_bytes"] += 1
                changed = True
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
                counts["lone_high"] += 1
                builder.append("\ufffd")
                changed = True
                i += 1
                continue

            if 0xDC00 <= code <= 0xDFFF:
                counts["lone_low"] += 1
                builder.append("\ufffd")
                changed = True
                i += 1
                continue

            builder.append(ch)
            i += 1

        sanitized = "".join(builder)
        if changed:
            counts["strings_touched"] += 1
            return sanitized, counts, True
        return value, counts, False


class StorageInventory:
    """Abstract source of disk objects (local FS, S3, etc.)."""

    def iter_objects(self) -> Iterator[DiskObject]:  # pragma: no cover - interface
        """Yield `DiskObject` entries for every file managed by this inventory."""
        raise NotImplementedError

    def delete(self, disk_object: DiskObject) -> None:  # pragma: no cover - interface
        """Remove a `DiskObject` from the backing store."""
        raise NotImplementedError


class LocalStorageInventory(StorageInventory):
    """Inventory implementation backed by the Open WebUI upload directory."""

    def __init__(self, root: Optional[str] = None):
        """Initialize the inventory with an optional override path."""
        self.root = Path(root or UPLOAD_DIR).expanduser()

    def iter_objects(self) -> Iterator[DiskObject]:
        """Yield every file below `UPLOAD_DIR`, tagging derived UUIDs when found."""
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
        """Return the first UUID embedded in the filename, if present."""

        if not filename:
            return None
        # Fast path: still support the original "<uuid>_something" layout.
        prefix, _, _ = filename.partition("_")
        try:
            UUID(prefix)
            return prefix
        except Exception:
            pass

        # Fallback: scan for any UUID substring (handles inline-image-<uuid>.png naming).
        match = re.search(
            r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
            filename,
        )
        if not match:
            return None
        candidate = match.group(0)
        try:
            UUID(candidate)
            return candidate
        except Exception:
            return None

    def delete(self, disk_object: DiskObject) -> None:
        """Remove the referenced file from disk."""
        try:
            path = Path(disk_object.path)
            if path.exists():
                path.unlink()
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Failed to delete {disk_object.path}: {exc}") from exc


class StorageAuditService:
    """Compare the database view of uploads with what actually sits on disk."""

    def __init__(self, inventory: StorageInventory):
        """Inject a storage inventory implementation (local disk, S3, etc.)."""
        self.inventory = inventory

    def scan_db_orphans(
        self,
        files: Dict[str, FileSummary],
        *,
        limit: Optional[int],
        status_callback: Optional[Callable[[str, str], None]] = None,
    ) -> Dict[str, Any]:
        """Return upload rows that no longer have binaries on disk, and vice versa."""
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
        """Store chunk size used across every SQLAlchemy iterator."""
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
        """Compare database rows against a storage inventory and report gaps."""
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
        """Return an empty result payload when work is cancelled early."""
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
        """Materialize a dictionary of FileSummary objects honoring filters."""
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
        """Stream chats to collect every file ID referenced inside payloads."""
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
        """Collect file IDs referenced by the optional knowledge base."""
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
        """Pull a best-effort file size from the serialized `meta` blob."""
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
        """Return a dict representation of `meta`, handling JSON strings gracefully."""
        if isinstance(meta, dict):
            return meta
        if isinstance(meta, str):
            try:
                return json.loads(meta)
            except json.JSONDecodeError:
                return {}
        return {}

    def _extract_file_ids_from_chat(self, payload: dict) -> Set[str]:
        """Collect file IDs from chat history payloads."""
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
        """Normalize potential file references to a plain ID string."""
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
        """Format a file summary for Markdown/JSON responses."""
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
    """Chat-facing command dispatcher that orchestrates every maintenance workflow."""
    PIPE_ID = "maintenance"
    PIPE_NAME = "Open WebUI: Maintenance"
    MAX_TOKEN_LENGTH = 256  # Cap individual CLI tokens to avoid runaway payloads.
    MAX_OPTION_VALUE_LENGTH = 2048  # Guard against unbounded option values.
    _QUOTE_TRANSLATION = str.maketrans(
        {
            "\u201c": '"',
            "\u201d": '"',
            "\u201e": '"',
            "\u201f": '"',
            "\u00ab": '"',
            "\u00bb": '"',
            "\u2033": '"',
            "\u2036": '"',
            "\u301d": '"',
            "\u301e": '"',
            "\u301f": '"',
            "\uff02": '"',
            "\u2018": "'",
            "\u2019": "'",
            "\u201a": "'",
            "\u201b": "'",
            "\u2032": "'",
            "\u2035": "'",
            "\u2039": "'",
            "\u203a": "'",
            "\uff07": "'",
        }
    )

    class Valves(BaseModel):
        ENABLE_LOGGING: bool = Field(
            default=False,
            description="Emit INFO logs for the maintenance pipe (disabled by default).",
        )
        SCAN_DEFAULT_LIMIT: int = Field(
            default=5,
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
            default=5,
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
        CHAT_SCAN_DEFAULT_LIMIT: int = Field(
            default=5,
            ge=0,
            le=5000,
            description="How many problematic chats to list per chat-scan command (0 = no limit).",
        )
        CHAT_SCAN_MAX_LIMIT: int = Field(
            default=200,
            ge=25,
            le=1000,
            description="Hard cap for chat scan output rows.",
        )
        CHAT_REPAIR_DEFAULT_LIMIT: int = Field(
            default=5,
            ge=0,
            le=200,
            description="How many chats to repair per run (0 = no cap).",
        )
        CHAT_REPAIR_MAX_LIMIT: int = Field(
            default=200,
            ge=10,
            le=1000,
            description="Safety ceiling for repairs per command.",
        )
        CHAT_DB_CHUNK_SIZE: int = Field(
            default=200,
            ge=50,
            le=1000,
            description="Rows fetched from the chat table per batch when scanning.",
        )
        IMAGE_SCAN_DEFAULT_LIMIT: int = Field(
            default=5,
            ge=0,
            le=5000,
            description="Default number of users shown in image-scan (0 = unlimited).",
        )
        IMAGE_SCAN_MAX_LIMIT: int = Field(
            default=200,
            ge=25,
            le=2000,
            description="Hard cap for image-scan rows.",
        )
        IMAGE_DETACH_DEFAULT_LIMIT: int = Field(
            default=5,
            ge=0,
            le=200,
            description="How many chats to process per image-detach command (0 = no cap).",
        )
        IMAGE_DETACH_MAX_LIMIT: int = Field(
            default=200,
            ge=10,
            le=1000,
            description="Safety ceiling for image-detach per command.",
        )

    def __init__(self):
        """Instantiate helper services and apply the initial logging valve."""
        self.valves = self.Valves()
        self.service = UploadAuditService(chunk_size=self.valves.DB_CHUNK_SIZE)
        self.chat_service = ChatRepairService(chunk_size=self.valves.CHAT_DB_CHUNK_SIZE)
        self.image_service = InlineImageService(chunk_size=self.valves.CHAT_DB_CHUNK_SIZE)
        self.upload_root = Path(UPLOAD_DIR).expanduser().resolve()
        self._apply_logging_valve()

    def _apply_logging_valve(self) -> None:
        """Set the module log level according to the ENABLE_LOGGING valve."""
        level = logging.DEBUG if self.valves.ENABLE_LOGGING else logging.INFO
        logger.setLevel(level)
        logger.propagate = True

    async def pipes(self) -> List[dict]:
        """Expose metadata for the Open WebUI functions registry."""
        return [{"id": self.PIPE_ID, "name": self.PIPE_NAME}]

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __request__: Request,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> StreamingResponse | PlainTextResponse:
        """Parse the incoming chat message, execute the command, and stream the result."""
        command_text = self._extract_prompt_text(body)
        if not command_text:
            return await self._respond(True, self._help_markdown(), body)

        try:
            command, options = self._parse_command(command_text)
        except ValueError as exc:
            return await self._respond(False, str(exc), body)
        raw_user_filter = options.get("user_id")
        user_filter: Optional[str]
        user_lookup_error: Optional[str] = None
        if raw_user_filter in {"me", "self"}:
            user_filter = __user__.get("id")
        else:
            user_filter, user_lookup_error = self._resolve_user_value(raw_user_filter)
        if user_lookup_error:
            return await self._respond(False, user_lookup_error, body)
        id_options = options.get("ids") or []
        file_ids = id_options
        limit_option = options.get("limit")
        user_query = options.get("user_query")

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

            if command == "chat-scan":
                target_ids = list(id_options)
                is_stream = bool(body.get("stream", True))
                status_cache: Dict[str, str] = {}
                user_sorter = self._make_user_sorter(status_cache)
                if is_stream:
                    return await self._stream_chat_scan(
                        loop=loop,
                        emitter=__event_emitter__,
                        limit_option=limit_option,
                        user_filter=user_filter,
                        user_query=user_query,
                        target_ids=target_ids,
                        body=body,
                        status_cache=status_cache,
                        user_sorter=user_sorter,
                    )
                status_callback = self._threadsafe_user_status_callback(
                    loop, __event_emitter__, status_cache, action="Scanning", purpose="for malformed Unicode"
                )
                limit = self._clamp_limit(
                    limit_option,
                    default=self.valves.CHAT_SCAN_DEFAULT_LIMIT,
                    ceiling=self.valves.CHAT_SCAN_MAX_LIMIT,
                    allow_zero=True,
                )
                await self._emit_status(
                    __event_emitter__,
                    "Scanning chats (no limit)" if limit == 0 else f"Scanning chats (limit={limit})...",
                )
                scan_result = await asyncio.to_thread(
                    self.chat_service.scan,
                    max_results=limit,
                    user_filter=user_filter,
                    user_query=user_query,
                    chat_ids=target_ids,
                    status_callback=status_callback,
                    user_sorter=user_sorter,
                )
                await self._emit_status(__event_emitter__, "Scan complete", done=True)
                user_labels = await self._resolve_user_labels([row["user_id"] for row in scan_result["results"]])
                scope = self._describe_scope(
                    user_filter,
                    target_ids,
                    user_query=user_query,
                    ids_label="chat IDs",
                )
                message = self._build_chat_scan_report(scan_result, user_labels=user_labels, scope=scope)
                return await self._respond(True, message, body)

            if command == "chat-repair":
                if not options.get("confirm"):
                    reminder = "Add `confirm` to run repairs (example: `chat-repair confirm limit=5`)."
                    return await self._respond(False, reminder, body)
                target_ids = list(id_options)
                if not target_ids:
                    target_ids = self._extract_chat_ids_from_history(body)
                repair_status_cache: Dict[str, str] = {}
                repair_user_sorter = self._make_user_sorter(repair_status_cache)
                status_callback = self._threadsafe_user_status_callback(
                    loop, __event_emitter__, repair_status_cache, action="Repairing", purpose="for malformed Unicode"
                )
                limit = self._clamp_limit(
                    limit_option,
                    default=self.valves.CHAT_REPAIR_DEFAULT_LIMIT,
                    ceiling=self.valves.CHAT_REPAIR_MAX_LIMIT,
                    allow_zero=True,
                )
                await self._emit_status(__event_emitter__, "Repair pass running...")
                repair_result = await asyncio.to_thread(
                    self.chat_service.repair,
                    max_repairs=(None if limit == 0 else limit),
                    user_filter=user_filter,
                    user_query=user_query,
                    chat_ids=target_ids,
                    status_callback=status_callback,
                    user_sorter=repair_user_sorter,
                )
                await self._emit_status(__event_emitter__, "Repair pass finished", done=True)
                user_labels = await self._resolve_user_labels([row["user_id"] for row in repair_result["details"]])
                scope = self._describe_scope(
                    user_filter,
                    target_ids,
                    user_query=user_query,
                    ids_label="chat IDs",
                )
                message = self._build_chat_repair_report(
                    repair_result,
                    user_labels=user_labels,
                    limit=limit,
                    scope=scope,
                )
                return await self._respond(True, message, body)

            if command == "image-scan":
                target_ids = list(id_options)
                is_stream = bool(body.get("stream", True))
                status_cache: Dict[str, str] = {}
                user_sorter = self._make_user_sorter(status_cache)
                if is_stream:
                    return await self._stream_image_scan(
                        loop=loop,
                        emitter=__event_emitter__,
                        limit_option=limit_option,
                        user_filter=user_filter,
                        user_query=user_query,
                        target_ids=target_ids,
                        status_cache=status_cache,
                        user_sorter=user_sorter,
                    )
                status_callback = self._threadsafe_user_status_callback(
                    loop, __event_emitter__, status_cache, action="Scanning", purpose="for inline images"
                )
                limit = self._clamp_limit(
                    limit_option,
                    default=self.valves.IMAGE_SCAN_DEFAULT_LIMIT,
                    ceiling=self.valves.IMAGE_SCAN_MAX_LIMIT,
                    allow_zero=True,
                )
                await self._emit_status(
                    __event_emitter__,
                    "Reviewing chats for inline images...",
                )
                scan_result = await asyncio.to_thread(
                    self.image_service.scan,
                    user_filter=user_filter,
                    user_query=user_query,
                    chat_ids=target_ids,
                    limit=(None if limit == 0 else limit),
                    status_callback=status_callback,
                    result_callback=None,
                    user_sorter=user_sorter,
                    cancel_event=None,
                )
                await self._emit_status(__event_emitter__, "Image scan complete", done=True)
                user_labels = await self._resolve_user_labels(
                    [summary.user_id for summary in scan_result["summaries"] if summary.user_id]
                )
                scope = self._describe_scope(
                    user_filter,
                    target_ids,
                    user_query=user_query,
                    ids_label="chat IDs",
                )
                message = self._build_image_scan_report(scan_result, user_labels=user_labels, scope=scope, limit=limit)
                return await self._respond(True, message, body)

            if command == "image-detach":
                if not options.get("confirm"):
                    reminder = "Add `confirm` to detach inline images (example: `image-detach confirm limit=5`)."
                    return await self._respond(False, reminder, body)
                target_ids = list(id_options)
                detach_status_cache: Dict[str, str] = {}
                detach_user_sorter = self._make_user_sorter(detach_status_cache)
                status_callback = self._threadsafe_user_status_callback(
                    loop, __event_emitter__, detach_status_cache, action="Detaching", purpose="inline images from"
                )
                limit = self._clamp_limit(
                    limit_option,
                    default=self.valves.IMAGE_DETACH_DEFAULT_LIMIT,
                    ceiling=self.valves.IMAGE_DETACH_MAX_LIMIT,
                    allow_zero=True,
                )
                await self._emit_status(__event_emitter__, "Detaching inline images...", done=False)
                detach_result = await asyncio.to_thread(
                    self.image_service.detach,
                    max_chats=(None if limit == 0 else limit),
                    user_filter=user_filter,
                    user_query=user_query,
                    chat_ids=target_ids,
                    status_callback=status_callback,
                    user_sorter=detach_user_sorter,
                    cancel_event=None,
                )
                await self._emit_status(__event_emitter__, "Image detach complete", done=True)
                user_labels = await self._resolve_user_labels(
                    [record.user_id for record in detach_result["records"] if record.user_id]
                )
                scope = self._describe_scope(
                    user_filter,
                    target_ids,
                    user_query=user_query,
                    ids_label="chat IDs",
                )
                message = self._build_image_detach_report(detach_result, user_labels=user_labels, limit=limit, scope=scope)
                return await self._respond(True, message, body)

            if command in {"db-clean", "db-clean-missing-files"}:
                if not options.get("confirm"):
                    reminder = "Add `confirm` to remove database records whose files are missing (example: `db-clean-missing-files confirm limit=5`)."
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

            if command == "db-clean-orphan-files":
                if not options.get("confirm"):
                    reminder = (
                        "Add `confirm` to remove database rows (and their binaries) for uploads with no references "
                        "(example: `db-clean-orphan-files confirm limit=5`)."
                    )
                    return await self._respond(False, reminder, body)
                limit = self._clamp_limit(
                    limit_option,
                    default=self.valves.SCAN_DEFAULT_LIMIT,
                    ceiling=self.valves.SCAN_MAX_LIMIT,
                    allow_zero=True,
                )
                status_callback = self._threadsafe_status_callback(loop, __event_emitter__)
                await self._emit_status(__event_emitter__, "Reviewing orphaned uploads...")
                scan_result = await asyncio.to_thread(
                    self.service.scan_db_orphans,
                    user_filter=user_filter,
                    file_ids=file_ids,
                    limit=(None if limit == 0 else limit),
                    status_callback=status_callback,
                )
                entries = scan_result.get("orphans", [])
                if not entries:
                    scope = self._describe_scope(user_filter, file_ids)
                    return await self._respond(
                        True,
                        f"No orphaned uploads remain for {scope}.",
                        body,
                    )
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
                    reminder = "Add `confirm` to delete files on disk that are no longer tracked in the database (example: `storage-clean confirm limit=5`)."
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
                "Unknown command `{}`. Available commands: help, db-scan, storage-scan, user-report, "
                "chat-scan, chat-repair, image-scan, image-detach, db-clean-missing-files, db-clean-orphan-files, "
                "storage-clean.\n\n"
            ).format(command) + self._help_markdown()
            return await self._respond(False, message, body)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.exception("Maintenance pipe failed")
            await self._emit_status(__event_emitter__, "Maintenance command failed", done=True)
            return await self._respond(False, f"Unable to complete command: {exc}", body)

    async def _respond(self, ok: bool, message: str, body: dict):
        """Return either a plain response or an SSE stream depending on the request."""
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
    def _log_future_exception(future: Future):  # pragma: no cover - best-effort logging
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

    def _threadsafe_user_status_callback(
        self,
        loop: asyncio.AbstractEventLoop,
        emitter: Optional[Callable[[dict], Awaitable[None]]],
        cache: Dict[str, str],
        *,
        action: str = "Scanning",
        purpose: str = "for malformed Unicode",
    ) -> Callable[[str, int], None]:
        seen_users: set[str] = set()

        def _callback(user_id: str, chat_count: int) -> None:
            user_id = user_id or "unknown"
            if user_id in seen_users:
                return
            seen_users.add(user_id)
            label = self._lookup_user_label_sync(user_id, cache)
            message = f"{action} {label}'s chats {purpose} (total chats: {chat_count})"
            logger.info(message)
            if not emitter:
                return
            future = asyncio.run_coroutine_threadsafe(self._emit_status(emitter, message), loop)
            future.add_done_callback(self._log_future_exception)

        return _callback

    def _threadsafe_result_callback(
        self,
        loop: asyncio.AbstractEventLoop,
        queue: "asyncio.Queue[Optional[Dict[str, Any]]]",
    ) -> Callable[[Dict[str, Any]], None]:
        def _callback(row: Dict[str, Any]) -> None:
            asyncio.run_coroutine_threadsafe(queue.put(row), loop)

        return _callback

    def _make_user_sorter(self, cache: Dict[str, str]) -> Callable[[str], str]:
        def _sorter(user_id: str) -> str:
            label = self._lookup_user_label_sync(user_id or "unknown", cache)
            return (label or user_id or "").lower()

        return _sorter

    async def _stream_chat_scan(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        emitter: Optional[Callable[[dict], Awaitable[None]]],
        limit_option: Optional[int],
        user_filter: Optional[str],
        user_query: Optional[str],
        target_ids: List[str],
        body: dict,
        status_cache: Dict[str, str],
        user_sorter: Callable[[str], str],
    ) -> StreamingResponse:
        limit = self._clamp_limit(
            limit_option,
            default=self.valves.CHAT_SCAN_DEFAULT_LIMIT,
            ceiling=self.valves.CHAT_SCAN_MAX_LIMIT,
            allow_zero=True,
        )
        queue: "asyncio.Queue[Optional[Dict[str, Any]]]" = asyncio.Queue()
        stream_user_cache: Dict[str, str] = {}
        status_callback = self._threadsafe_user_status_callback(
            loop, emitter, status_cache, action="Scanning", purpose="for malformed Unicode"
        )
        result_callback = self._threadsafe_result_callback(loop, queue)
        scope = self._describe_scope(
            user_filter,
            target_ids,
            user_query=user_query,
            ids_label="chat IDs",
        )
        thread_cancel = threading.Event()

        async def run_scan():
            await self._emit_status(
                emitter,
                "Scanning chats (no limit)" if limit == 0 else f"Scanning chats (limit={limit})...",
            )
            result = None

            def _scan_thread():
                return self.chat_service.scan(
                    max_results=limit,
                    user_filter=user_filter,
                    user_query=user_query,
                    chat_ids=target_ids,
                    status_callback=status_callback,
                    result_callback=result_callback,
                    user_sorter=user_sorter,
                    cancel_event=thread_cancel,
                )

            try:
                result = await asyncio.to_thread(_scan_thread)
                return result
            finally:
                await queue.put(None)
                await self._emit_status(emitter, "Scan complete", done=True)

        scan_task = asyncio.create_task(run_scan())

        async def stream():
            intro = (
                "Scan in progress - the table below will populate with chats that need fixing.\n\n"
                "| User | Chat ID | Title | Issues |\n"
                "| --- | --- | --- | --- |"
            )
            yield self._format_data(is_stream=True, model=self.PIPE_NAME, content=intro, finish_reason=None)
            await self._emit_message_chunk(emitter, intro)
            cancelled = False
            try:
                while True:
                    row = await queue.get()
                    if row is None:
                        break
                    line = await self._format_chat_stream_row(row, stream_user_cache)
                    await self._emit_message_chunk(emitter, line)
                    yield self._format_data(
                        is_stream=True,
                        model=self.PIPE_NAME,
                        content=line,
                        finish_reason=None,
                    )

                scan_result = await scan_task
                summary = self._build_stream_chat_scan_summary(scan_result, scope)
                await self._emit_message_chunk(emitter, summary)
                yield self._format_data(
                    is_stream=True,
                    model=self.PIPE_NAME,
                    content=summary,
                    finish_reason="stop",
                )
                yield "data: [DONE]\n\n"
            except asyncio.CancelledError:
                cancelled = True
                raise
            finally:
                thread_cancel.set()
                if cancelled:
                    with contextlib.suppress(asyncio.CancelledError):
                        await scan_task

        return StreamingResponse(stream(), media_type="text/event-stream")

    async def _stream_image_scan(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        emitter: Optional[Callable[[dict], Awaitable[None]]],
        limit_option: Optional[int],
        user_filter: Optional[str],
        user_query: Optional[str],
        target_ids: List[str],
        status_cache: Dict[str, str],
        user_sorter: Callable[[str], str],
    ) -> StreamingResponse:
        limit = self._clamp_limit(
            limit_option,
            default=self.valves.IMAGE_SCAN_DEFAULT_LIMIT,
            ceiling=self.valves.IMAGE_SCAN_MAX_LIMIT,
            allow_zero=True,
        )
        queue: "asyncio.Queue[Optional[InlineImageUserSummary]]" = asyncio.Queue()
        stream_user_cache: Dict[str, str] = {}
        status_callback = self._threadsafe_user_status_callback(
            loop, emitter, status_cache, action="Scanning", purpose="for inline images"
        )
        thread_cancel = threading.Event()

        def _result_callback(summary: InlineImageUserSummary) -> None:
            asyncio.run_coroutine_threadsafe(queue.put(summary), loop)

        async def run_scan():
            await self._emit_status(
                emitter,
                "Reviewing chats for inline images...",
            )

            def _scan_thread():
                return self.image_service.scan(
                    user_filter=user_filter,
                    user_query=user_query,
                    chat_ids=target_ids,
                    limit=(None if limit == 0 else limit),
                    status_callback=status_callback,
                    result_callback=_result_callback,
                    user_sorter=user_sorter,
                    cancel_event=thread_cancel,
                )

            try:
                result = await asyncio.to_thread(_scan_thread)
                return result
            finally:
                await queue.put(None)
                await self._emit_status(emitter, "Image scan complete", done=True)

        scan_task = asyncio.create_task(run_scan())

        async def stream():
            intro = (
                "Inline image scan in progress – users will appear as soon as results are ready.\n\n"
                "| User | Chats | Inline images | Estimated reclaim |\n"
                "| --- | --- | --- | --- |"
            )
            yield self._format_data(is_stream=True, model=self.PIPE_NAME, content=intro, finish_reason=None)
            await self._emit_message_chunk(emitter, intro)
            cancelled = False
            try:
                while True:
                    summary = await queue.get()
                    if summary is None:
                        break
                    line = await self._format_image_stream_row(summary, stream_user_cache)
                    await self._emit_message_chunk(emitter, line)
                    yield self._format_data(
                        is_stream=True,
                        model=self.PIPE_NAME,
                        content=line,
                        finish_reason=None,
                    )
                scan_result = await scan_task
                scope = self._describe_scope(
                    user_filter,
                    target_ids,
                    user_query=user_query,
                    ids_label="chat IDs",
                )
                summary_text = self._build_stream_image_scan_summary(scan_result, scope)
                await self._emit_message_chunk(emitter, summary_text)
                yield self._format_data(
                    is_stream=True,
                    model=self.PIPE_NAME,
                    content=summary_text,
                    finish_reason="stop",
                )
                yield "data: [DONE]\n\n"
            except asyncio.CancelledError:
                cancelled = True
                raise
            finally:
                thread_cancel.set()
                if cancelled:
                    with contextlib.suppress(asyncio.CancelledError):
                        await scan_task

        return StreamingResponse(stream(), media_type="text/event-stream")

    def _lookup_user_label_sync(self, user_id: str, cache: Dict[str, str]) -> str:
        if user_id in cache:
            return cache[user_id]
        label = user_id
        if user_id and user_id != "unknown":
            try:
                user = Users.get_user_by_id(user_id)
                if user:
                    label = (
                        getattr(user, "name", None)
                        or getattr(user, "username", None)
                        or getattr(user, "email", None)
                        or user_id
                    )
            except Exception:
                logger.debug("Failed to resolve user %s", user_id, exc_info=True)
        cache[user_id] = label
        return label

    async def _get_user_label_async(self, user_id: str, cache: Dict[str, str]) -> str:
        if user_id in cache:
            return cache[user_id]
        label = await asyncio.to_thread(self._lookup_user_label_sync, user_id, cache)
        return label

    async def _format_chat_stream_row(self, row: Dict[str, Any], cache: Dict[str, str]) -> str:
        user_label = await self._get_user_label_async(row.get("user_id") or "unknown", cache)
        chat_id = row.get("chat_id") or "unknown"
        title = self._shorten(self._sanitize_output_text(row.get("title") or "(untitled)"))
        issues = self._describe_counts(row.get("issue_counts", {}))
        if not issues:
            issues = ", ".join(row.get("fields", [])) or "Needs sanitizing"
        return f"\n| {user_label} | `{chat_id}` | {title} | {issues} |"

    def _build_stream_chat_scan_summary(self, scan_result: Dict[str, Any], scope: str) -> str:
        lines = ["", "### Scan summary", ""]
        lines.append(f"- Rows inspected: {scan_result['examined']}")
        lines.append(f"- Chats needing repair: {scan_result['matches']}")
        lines.append(f"- Scope: {scope}")
        counts_summary = self._describe_counts(scan_result.get("counters", {}))
        if counts_summary:
            lines.append(f"- Character fixes applied if you run repair: {counts_summary}")
        if scan_result.get("has_more"):
            lines.append("- Limit reached. Re-run chat-scan to continue browsing results.")
        if not scan_result.get("results"):
            lines.append("- No malformed Unicode detected in this batch.")
        lines.append("")
        lines.append(
            "Run `chat-repair confirm` (use `limit=<n>` or `limit=0` for unlimited) to sanitize the listed chats."
        )
        return "\n".join(lines)

    async def _format_image_stream_row(
        self,
        summary: InlineImageUserSummary,
        cache: Dict[str, str],
    ) -> str:
        user_label = await self._get_user_label_async(summary.user_id or "unknown", cache)
        est = self._format_filesize(summary.inline_bytes)
        return (
            f"\n| {user_label} | {summary.chats_with_inline} | "
            f"{summary.inline_images} | {est} |"
        )

    def _build_stream_image_scan_summary(self, scan_result: Dict[str, Any], scope: str) -> str:
        lines = ["", "### Image scan summary", ""]
        lines.append(f"- Users with inline images: {scan_result['users_with_inline']}")
        lines.append(f"- Chats containing inline images: {scan_result['total_chats_with_inline']}")
        lines.append(f"- Inline images detected: {scan_result['total_inline_images']}")
        lines.append(
            f"- Estimated reclaim: {self._format_filesize(scan_result['total_inline_bytes'])}"
        )
        lines.append(f"- Scope: {scope}")
        if scan_result.get("has_more"):
            lines.append("- Limit reached before finishing the result set.")
        lines.append("")
        lines.append("Run `image-detach confirm` to offload embedded blobs into the file store.")
        return "\n".join(lines)

    def _parse_command(self, text: str):
        text = self._normalize_quotes(text)
        try:
            tokens = shlex.split(text.strip())
        except ValueError:
            tokens = text.strip().split()
        if not tokens:
            return "", {}
        for token in tokens:
            if len(token) > self.MAX_TOKEN_LENGTH:
                raise ValueError("Command tokens must be shorter than 256 characters.")
        command_token = tokens[0].strip()
        if not command_token:
            raise ValueError("Missing command. Try `help` to see the available actions.")
        if command_token.startswith("/"):
            command = command_token[1:].lower()
        else:
            command = command_token.lower()
        if not command:
            raise ValueError("Missing command name. Try `help` to see the available actions.")
        options: Dict[str, Any] = {
            "limit": None,
            "ids": [],
            "user_id": None,
            "user_query": None,
            "confirm": False,
            "path": None,
        }
        pending_key: Optional[str] = None
        pending_value: List[str] = []

        def enforce_value_length(value: str) -> None:
            if len(value) > self.MAX_OPTION_VALUE_LENGTH:
                raise ValueError("Option values must be shorter than 2048 characters.")

        def assign_option(key: str, value: str) -> None:
            key = key.strip().lower()
            value = value.strip()
            enforce_value_length(value)
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
                cleaned_user = self._strip_matching_quotes(value)
                if cleaned_user not in {"me", "self"} and not self._is_valid_uuid(cleaned_user):
                    raise ValueError("User filters must be UUIDs (or use user=me).")
                options["user_id"] = cleaned_user
            elif key in {"path", "prefix"}:
                sanitized = self._sanitize_path_input(self._strip_matching_quotes(value) or "")
                options["path"] = sanitized
            elif key in {"user_query", "query"}:
                options["user_query"] = self._strip_matching_quotes(value)

        recognized_keys = {
            "limit",
            "id",
            "file",
            "file_id",
            "user",
            "user_id",
            "user_query",
            "query",
            "path",
            "prefix",
        }

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
        if options["ids"]:
            invalid_ids = [fid for fid in options["ids"] if not self._is_valid_uuid(fid)]
            if invalid_ids:
                raise ValueError(f"Invalid ID value '{invalid_ids[0]}' (expected UUID).")
        return command, options

    def _normalize_quotes(self, text: Optional[str]) -> str:
        if not text:
            return ""
        return str(text).translate(self._QUOTE_TRANSLATION)

    def _strip_matching_quotes(self, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = self._normalize_quotes(value).strip()
        if len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in {'"', "'"}:
            return normalized[1:-1].strip()
        return normalized

    def _sanitize_path_input(self, value: str) -> str:
        cleaned = (value or "").strip()
        if not cleaned:
            raise ValueError("Path/prefix cannot be empty.")
        candidate = Path(cleaned).expanduser()
        if not candidate.is_absolute():
            candidate = (self.upload_root / candidate).resolve()
        else:
            candidate = candidate.resolve()

        # Check if the resolved candidate is within the upload directory
        try:
            # Use relative_to() for robust path containment check (more Pythonic than str.startswith)
            candidate.relative_to(self.upload_root)
        except ValueError:
            # relative_to() raises ValueError if candidate is not under upload_root
            raise ValueError("Path/prefix must stay within the upload directory.")

        return str(candidate)

    @staticmethod
    def _is_valid_uuid(value: str) -> bool:
        try:
            UUID(value)
            return True
        except Exception:
            return False

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

    def _describe_scope(
        self,
        user_filter: Optional[str],
        ids: Sequence[str],
        *,
        user_query: Optional[str] = None,
        ids_label: str = "file IDs",
    ) -> str:
        scope = []
        if user_filter:
            scope.append(f"user `{user_filter}`")
        elif user_query:
            scope.append(f'users matching "{user_query}"')
        if ids:
            scope.append(f"specific {ids_label} ({len(ids)})")
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
        lines.append(f"- Orphaned files (no remaining references): {result['orphan_total']}")
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
            lines.append("#### Orphaned files (no remaining references)")
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

    def _build_chat_scan_report(
        self,
        scan_result: Dict[str, Any],
        *,
        user_labels: Dict[str, str],
        scope: str,
    ) -> str:
        lines = ["### Chat scan summary", ""]
        lines.append(f"- Rows inspected: {scan_result['examined']}")
        lines.append(f"- Problematic chats listed: {scan_result['matches']}")
        lines.append(f"- Scope: {scope}")
        counts_summary = self._describe_counts(scan_result.get("counters", {}))
        if counts_summary:
            lines.append(f"- Potential fixes: {counts_summary}")
        if scan_result.get("has_more"):
            lines.append("- Limit reached before finishing the data set.")
        lines.append("")
        results = scan_result.get("results", [])
        if not results:
            lines.append("No malformed Unicode was detected in the scanned chats.")
            lines.append("Add `limit=50` if you only want to spot-check a subset next time.")
            return "\n".join(lines)

        lines.append("| Chat ID | User | Title | Updated (UTC) | Issues | Fields |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for row in results:
            chat_id = row["chat_id"]
            user_label = user_labels.get(row["user_id"], row["user_id"])
            title = self._shorten(self._sanitize_output_text(row.get("title") or "(untitled)"))
            updated = self._format_timestamp(row.get("updated_at"))
            issues = self._describe_counts(row.get("issue_counts", {}))
            fields = ", ".join(row.get("fields", [])) or "title"
            lines.append(f"| `{chat_id}` | {user_label} | {title} | {updated} | {issues} | {fields} |")

        lines.append("")
        lines.append(
            "Next step: run `chat-repair confirm limit=5` (or `limit=0` for no cap) to clean the listed chats."
        )
        return "\n".join(lines)

    def _build_chat_repair_report(
        self,
        repair_result: Dict[str, Any],
        *,
        user_labels: Dict[str, str],
        limit: int,
        scope: str,
    ) -> str:
        lines = ["### Chat repair summary", ""]
        lines.append(f"- Rows inspected: {repair_result['examined']}")
        lines.append(f"- Chats repaired: {repair_result['repaired']}")
        lines.append(f"- Scope: {scope}")
        counts_summary = self._describe_counts(repair_result.get("counters", {}))
        if counts_summary:
            lines.append(f"- Characters replaced/removed: {counts_summary}")
        if limit == 0:
            lines.append("- Limit: unlimited (ran until scope finished)")
        else:
            lines.append(f"- Limit: {limit}")
        if repair_result.get("has_more"):
            lines.append("- Limit reached. Run the command again to continue.")
        lines.append("")
        details = repair_result.get("details", [])
        if not details:
            lines.append("No chats required changes. You're good to go!")
            return "\n".join(lines)

        lines.append("| Chat ID | User | Issues | Fields |")
        lines.append("| --- | --- | --- | --- |")
        for row in details:
            chat_id = row["chat_id"]
            user_label = user_labels.get(row["user_id"], row["user_id"])
            issues = self._describe_counts(row.get("issue_counts", {}))
            fields = ", ".join(row.get("fields", [])) or "title"
            lines.append(f"| `{chat_id}` | {user_label} | {issues} | {fields} |")

        if repair_result.get("has_more"):
            lines.append("")
            lines.append("Additional chats still need repairs. Re-run the command to keep going.")
        return "\n".join(lines)

    def _build_image_scan_report(
        self,
        result: Dict[str, Any],
        *,
        user_labels: Dict[str, str],
        scope: str,
        limit: int,
    ) -> str:
        lines = ["### Inline image scan results", ""]
        lines.append(f"- Users with inline images: {result['users_with_inline']}")
        lines.append(f"- Chats containing inline images: {result['total_chats_with_inline']}")
        lines.append(f"- Inline images detected: {result['total_inline_images']}")
        lines.append(
            f"- Estimated reclaim: {self._format_filesize(result['total_inline_bytes'])}"
        )
        lines.append(f"- Scope: {scope}")
        if limit == 0:
            lines.append("- Output limit: unlimited")
        else:
            lines.append(f"- Output limit: {limit or self.valves.IMAGE_SCAN_DEFAULT_LIMIT}")
        if result.get("has_more"):
            lines.append("- Some users were omitted because of the output limit.")
        lines.append("")

        summaries: List[InlineImageUserSummary] = result.get("summaries", [])
        if summaries:
            lines.append("| User | Chats with inline images | Inline images | Estimated reclaim |")
            lines.append("| --- | --- | --- | --- |")
            for summary in summaries:
                label = user_labels.get(summary.user_id, summary.user_id or "—")
                lines.append(
                    f"| {label} | {summary.chats_with_inline} | "
                    f"{summary.inline_images} | {self._format_filesize(summary.inline_bytes)} |"
                )
        else:
            lines.append("No inline base64 image blobs were detected in the selected scope.")

        lines.append("")
        lines.append("Next step: run `image-detach confirm` to move embedded blobs into the file store.")
        return "\n".join(lines)

    def _build_image_detach_report(
        self,
        result: Dict[str, Any],
        *,
        user_labels: Dict[str, str],
        limit: int,
        scope: str,
    ) -> str:
        lines = ["### Image detach summary", ""]
        lines.append(f"- Chats processed: {result['processed_chats']}")
        lines.append(f"- Inline images detached: {result['images_detached']}")
        lines.append(f"- Bytes migrated to file store: {self._format_filesize(result['bytes_detached'])}")
        lines.append(f"- Scope: {scope}")
        skipped_chats = result.get("skipped_chats", 0)
        skipped_images = result.get("skipped_images", 0)
        if skipped_chats:
            lines.append(f"- Chats skipped (detachment failed): {skipped_chats}")
        if skipped_images:
            lines.append(f"- Inline images skipped: {skipped_images}")
            reason_summary = self._format_skip_reason_summary(result.get("skipped_reason_counts", {}))
            if reason_summary:
                lines.append(f"- Skip reasons: {reason_summary}")
        if limit == 0:
            lines.append("- Limit: unlimited (ran until scope finished)")
        else:
            lines.append(f"- Limit: {limit}")
        if result.get("has_more"):
            lines.append("- Limit reached. Re-run the command to process more chats.")
        lines.append("")
        records: List[InlineImageDetachRecord] = result.get("records", [])
        skipped_records: List[InlineImageSkipRecord] = result.get("skipped_records", [])
        if records:
            lines.append("| Chat ID | User | Images detached | Bytes moved |")
            lines.append("| --- | --- | --- | --- |")
            for record in records:
                label = user_labels.get(record.user_id, record.user_id or "—")
                lines.append(
                    f"| `{record.chat_id}` | {label} | {record.images_detached} | "
                    f"{self._format_filesize(record.bytes_detached)} |"
                )
        if not records and not skipped_records:
            lines.append("No chats required changes.")

        if skipped_records:
            lines.append("")
            lines.append("#### Chats with inline images that could not be detached")
            lines.append("| Chat ID | User | Inline images | Estimated size | Reasons |")
            lines.append("| --- | --- | --- | --- | --- |")
            for record in skipped_records:
                label = user_labels.get(record.user_id, record.user_id or "—")
                reason_text = self._format_skip_reason_summary(record.reasons)
                lines.append(
                    f"| `{record.chat_id}` | {label} | {record.skipped_images} | "
                    f"{self._format_filesize(record.skipped_bytes)} | {reason_text or '—'} |"
                )

        lines.append("")
        lines.append("All detached assets are now accessible via the standard `/api/v1/files/{id}/content` endpoint.")
        return "\n".join(lines)

    def _format_skip_reason_summary(self, reasons: Dict[str, int]) -> str:
        if not reasons:
            return ""
        labels = {
            "unsupported_mime": "unsupported MIME type",
            "decode_error": "invalid base64",
            "empty_payload": "empty payload",
            "storage_error": "storage backend error",
            "db_error": "database insert error",
            "persist_failed": "unknown error",
        }
        parts: List[str] = []
        for key, count in sorted(reasons.items(), key=lambda item: (-item[1], item[0])):
            friendly = labels.get(key, key.replace("_", " "))
            parts.append(f"{count} {friendly}")
        return ", ".join(parts)

    def _describe_counts(self, counts: Dict[str, int]) -> str:
        if not counts:
            return ""
        parts = []
        nulls = counts.get("null_bytes", 0)
        if nulls:
            parts.append(f"{nulls} null byte{'s' if nulls != 1 else ''}")
        high = counts.get("lone_high", 0)
        if high:
            parts.append(f"{high} lone high surrogate{'s' if high != 1 else ''}")
        low = counts.get("lone_low", 0)
        if low:
            parts.append(f"{low} lone low surrogate{'s' if low != 1 else ''}")
        touched = counts.get("strings_touched", 0)
        if touched and not parts:
            parts.append(f"{touched} string{'s' if touched != 1 else ''} sanitized")
        elif touched:
            parts.append(f"{touched} string{'s' if touched != 1 else ''}")
        return ", ".join(parts)

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
        limit: int = 5,
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
        limit: int = 5,
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
        limit: int = 5,
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
        """Stream user usage rows as they are computed in a background thread."""
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
        """Render a Markdown table summarizing per-user usage statistics."""
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

    def _extract_chat_ids_from_history(self, body: dict) -> List[str]:
        ids: List[str] = []
        messages = body.get("messages") or []
        for message in reversed(messages):
            if message.get("role") != "assistant":
                continue
            content = self._collapse_content(message.get("content"))
            if not content:
                continue
            ids.extend(self._extract_chat_ids_from_text(content))
            if ids:
                break
        deduped: List[str] = []
        seen: Set[str] = set()
        for cid in ids:
            if cid not in seen:
                seen.add(cid)
                deduped.append(cid)
        return deduped

    def _extract_chat_ids_from_text(self, text: str) -> List[str]:
        ids: List[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("|"):
                continue
            cells = [cell.strip() for cell in line.split("|") if cell.strip()]
            if len(cells) < 2:
                continue
            for cell in cells:
                if cell.startswith("`") and cell.endswith("`"):
                    chat_id = cell.strip("`").strip()
                    if chat_id:
                        ids.append(chat_id)
                        break
        return ids

    def _help_markdown(self) -> str:
        return """# Open WebUI Maintenance Pipe

Use this pipe as a chat-style runbook for keeping Open WebUI deployments tidy. Each section below explains what the maintenance task does, how it evaluates data, and when administrators typically run it.

## Database maintenance
These commands operate only on the `file` table and its metadata.

- `db-scan [limit=<n>] [user=<id>|user=me] [id=<file-id>]`
  - **Purpose:** Identify uploads that are no longer referenced by chats or knowledge bases, and highlight database rows whose underlying files have vanished from storage.
  - **How it works:** Reads the `file` table in batches, extracts file IDs from `history.messages.*.files[]` and the knowledge tables, and optionally scopes the work via `user=`/`id=` filters. Matching storage lookups confirm whether each database row still has a binary on disk. Set `limit=0` to return everything; otherwise the limit caps both orphan listings and missing-file rows.
  - **When to run:** After content migrations, following bulk deletions, or on a regular cadence (weekly/monthly) to prevent unreferenced data from accumulating.

- `db-clean-missing-files confirm [limit=<n>] [...]` *(alias: `db-clean`)*
  - **Purpose:** Remove the database rows that `db-scan` proved were already missing their binaries.
  - **Safety:** Requires the literal word `confirm`, respects `limit`, deletes in small batches by default, and prints a table of every row removed (owner, filename, path, last update) for compliance records.
      - **Best practice:** Start with conservative limits (for example `limit=5`) in production and widen only after verifying results—the intentionally low default helps limit any damage if a scope turns out to be broader than expected.
- `db-clean-orphan-files confirm [limit=<n>] [...]`
  - **Purpose:** Delete uploads that still exist on disk but have zero references in chats or knowledge bases (the “Orphaned files (no remaining references)” group from `db-scan`).
  - **Safety:** Removes both the database row and the binary via `Files.delete_file_by_id`, honors `limit`, and produces the same audit table as other clean commands.
  - **Best practice:** Run `db-scan` first to confirm scope, then clean in manageable batches while capturing the Markdown table for change-control notes.

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

## Chat maintenance
These commands audit stored chats for malformed Unicode (null bytes, orphaned surrogate halves) and can optionally repair them in-place.

- `chat-scan [limit=<n>] [user=<id>|user=me] [user_query="name"] [id=<chat-id>]`
  - **Purpose:** Stream a live table of chats whose titles, payloads, or metadata would change if sanitized using the backend logic.
  - **How it works:** Batches chats by owner, inspects each JSON payload, and emits rows (plus running totals) via the event emitter so you can watch progress.
  - **When to run:** Before PostgreSQL index rebuilds, after imports from untrusted systems, or whenever assistants start throwing “invalid byte sequence” errors.

- `chat-repair confirm [limit=<n>] [user=<id>|user=me] [user_query="name"] [id=<chat-id>]`
  - **Purpose:** Apply the same sanitization logic as `chat-scan` but persist the cleaned values back to the `chat` table.
      - **Safety:** Requires `confirm`, honors `limit` (default 5, `limit=0` for unlimited), and shows a sample of the chats it fixed so you can audit changes.
      - **Best practice:** Run `chat-scan` first, then feed the resulting chat IDs (or rely on automatic history scraping) into `chat-repair confirm limit=5`; the reduced default is intentional to limit damage if repairs go sideways.

## Image maintenance
Inline images embedded as `data:image/...;base64` strings bloat the database and bypass access controls. Use these commands to locate and normalize them.

- `image-scan [limit=<n>] [user=<id>|user=me] [user_query="name"]`
  - **Purpose:** Identify users and chats that still contain inline base64 blobs along with the estimated database storage reclaim if you detach them.
  - **How it works:** Traverses each chat’s history, finds Markdown `![...](data:...)` links, and aggregates the byte estimates per user. Results stream live in table form and respect `limit`/`user` filters.
  - **When to run:** After migrations, when DB size spikes unexpectedly, or before enabling stricter file-governance controls.

- `image-detach confirm [limit=<n>] [user=<id>|user=me] [user_query="name"] [id=<chat-id>]`
  - **Purpose:** Convert those inline blobs into proper Open WebUI files by uploading them via the storage provider and rewriting the chat Markdown to reference `/api/v1/files/{id}/content`.
  - **Safety:** Requires `confirm`, honors `limit` (default 5 chats per run, `limit=0` for no cap), and records the chats/files it touched so you can track changes so any accidental scope issues stay small.
  - **Best practice:** Start with targeted scopes (e.g., `user=...` or `user_query="team"`) and keep copies of `image-scan` reports for change-control documentation.

## User usage reporting
These commands summarize end-user footprints so you can plan capacity or chargeback.

- `user-report [user=<id>|user=me]`
  - **Purpose:** Produce a per-user table that lists chat counts and sizes, file counts and sizes, plus the combined storage footprint.
  - **How it works:** Resolves each user in scope, aggregates their chats (measuring JSON size in bytes) and file metadata, then sorts alphabetically so the full roster streams consistently.
  - **When to run:** Before large cleanup projects, when preparing usage reports for stakeholders, or to identify the accounts consuming the most storage.

## General guidance
1. Commands never delete anything unless you intentionally run `db-clean-missing-files`, `db-clean-orphan-files`, `image-detach`, or `storage-clean` with `confirm`.
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
