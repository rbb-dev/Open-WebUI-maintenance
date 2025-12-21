from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

import open_webui_maintenance as maintenance
from open_webui_maintenance import (
    BaseChatService,
    ChatRepairService,
    DiskObject,
    FileSummary,
    InlineImageService,
    LocalStorageInventory,
    Pipe,
    StorageAuditService,
    UploadAuditService,
    sanitize_unicode,
)
from open_webui.models.users import User as UserModel


def _uuid(seed: int) -> str:
    return f"123e4567-e89b-12d3-a456-426614174{seed:03d}"


def _add_user(db_session, user_id: str, *, name: str, username: str | None = None, email: str | None = None):
    db_session.add(UserModel(id=user_id, name=name, username=username, email=email))
    db_session.commit()


def _add_chat(db_session, *, chat_id: str, user_id: str, title: str, payload):
    db_session.add(
        maintenance.Chat(
            id=chat_id,
            user_id=user_id,
            title=title,
            chat=payload,
            meta={},
            created_at=1,
            updated_at=1,
        )
    )
    db_session.commit()


def _make_inline_markdown(mime: str = "image/png", payload: bytes = b"img") -> str:
    b64 = base64.b64encode(payload).decode("utf-8")
    return f"![alt](data:{mime};base64,{b64})"


def _make_history_text(text: str) -> dict:
    return {"history": {"messages": {"1": {"files": [], "content": text}}}}


def test_sanitize_unicode_empty_and_surrogate_pair():
    assert sanitize_unicode("", track_counts=False) == ""
    empty_value, counts, changed = sanitize_unicode("", track_counts=True)
    assert empty_value == ""
    assert counts == {}
    assert changed is False

    surrogate_pair = "\ud83d\ude00"
    out, counts, changed = sanitize_unicode(surrogate_pair, track_counts=True)
    assert out == surrogate_pair
    assert counts == {}
    assert changed is False


def test_base_chat_service_user_lookup_and_build_query_filters(db_session):
    user_id = _uuid(1)
    _add_user(db_session, user_id, name="Alice", username="alice_user", email="alice@example.com")
    service = BaseChatService()

    with maintenance.get_db() as db:
        assert service._lookup_user_ids_by_query(db, "alice") == [user_id]
        assert service._lookup_user_ids_by_query(db, "") == []

    chat_id = _uuid(2)
    _add_chat(db_session, chat_id=chat_id, user_id=user_id, title="ok", payload=_make_history_text("hello"))

    with maintenance.get_db() as db:
        query = service._build_query(db, user_filter=None, chat_ids=[""])
        assert query.all() == []


def test_base_chat_service_collect_user_batches_filters(db_session):
    user_a = _uuid(3)
    user_b = _uuid(4)
    _add_user(db_session, user_a, name="Alice", username="alice", email="alice@example.com")
    _add_user(db_session, user_b, name="Bob", username="bobby", email="bob@example.com")

    _add_chat(db_session, chat_id=_uuid(5), user_id=user_a, title="ok", payload=_make_history_text("hello"))
    _add_chat(db_session, chat_id=_uuid(6), user_id=user_a, title="ok", payload=_make_history_text("hello"))
    _add_chat(db_session, chat_id=_uuid(7), user_id=user_b, title="ok", payload=_make_history_text("hello"))

    service = BaseChatService()
    with maintenance.get_db() as db:
        assert service._collect_user_batches(db, user_filter=user_a, user_query=None, chat_ids=None) == [(user_a, 2)]
        assert service._collect_user_batches(db, user_filter=None, user_query="ali", chat_ids=None) == [(user_a, 2)]
        assert service._collect_user_batches(db, user_filter=None, user_query=None, chat_ids=[_uuid(7)]) == [(user_b, 1)]
        assert service._collect_user_batches(db, user_filter=None, user_query=None, chat_ids=[""]) == []


def test_chat_repair_service_scan_cancel_and_limits(db_session):
    user_id = _uuid(10)
    _add_user(db_session, user_id, name="Alice")

    for idx in range(3):
        _add_chat(
            db_session,
            chat_id=_uuid(20 + idx),
            user_id=user_id,
            title="ok",
            payload=_make_history_text("bad\u0000text"),
        )

    service = ChatRepairService(chunk_size=50)
    result = service.scan(max_results=1, user_filter=None, user_query=None, chat_ids=None, user_sorter=None)
    assert result["matches"] == 1
    assert result["has_more"] is True

    cancel = threading.Event()
    cancel.set()
    cancelled = service.scan(max_results=10, user_filter=None, user_query=None, chat_ids=None, cancel_event=cancel)
    assert cancelled["has_more"] is True


def test_chat_repair_service_repair_rolls_back_when_no_changes(db_session, monkeypatch):
    user_id = _uuid(30)
    _add_user(db_session, user_id, name="Alice")
    _add_chat(
        db_session,
        chat_id=_uuid(31),
        user_id=user_id,
        title="clean",
        payload=_make_history_text("clean"),
    )

    service = ChatRepairService(chunk_size=50)
    result = service.repair(max_repairs=10, user_filter=None, user_query=None, chat_ids=None, user_sorter=None)
    assert result["repaired"] == 0

    def boom(*_, **__):
        raise RuntimeError("explode")

    monkeypatch.setattr(service, "_analyse_chat", boom)
    with pytest.raises(RuntimeError):
        service.repair(max_repairs=10, user_filter=None, user_query=None, chat_ids=None)


def test_inline_image_service_error_paths(monkeypatch):
    service = InlineImageService()

    file_id, reason = service._persist_inline_image("user-1", "text/plain", "abcd")
    assert file_id is None
    assert reason == "unsupported_mime"

    file_id, reason = service._persist_inline_image("user-1", "image/png", "not-base64")
    assert file_id is None
    assert reason == "decode_error"

    empty_b64 = base64.b64encode(b"").decode("utf-8")
    file_id, reason = service._persist_inline_image("user-1", "image/png", empty_b64)
    assert file_id is None
    assert reason == "empty_payload"

    def _boom_upload(*args, **kwargs):
        raise RuntimeError("storage down")

    original_upload = maintenance.Storage.upload_file
    monkeypatch.setattr(maintenance.Storage, "upload_file", _boom_upload)
    file_id, reason = service._persist_inline_image("user-1", "image/png", base64.b64encode(b"x").decode("utf-8"))
    assert file_id is None
    assert reason == "storage_error"

    monkeypatch.setattr(maintenance.Storage, "upload_file", original_upload)
    monkeypatch.setattr(maintenance.Files, "insert_new_file", lambda *_: None)
    file_id, reason = service._persist_inline_image("user-1", "image/png", base64.b64encode(b"x").decode("utf-8"))
    assert file_id is None
    assert reason == "db_error"


def test_inline_image_replace_bare_uri_branches(monkeypatch):
    service = InlineImageService()
    payload = base64.b64encode(b"hello").decode("utf-8")
    uri = f"data:image/png;base64,{payload}"

    monkeypatch.setattr(service, "_persist_inline_image", lambda *_: (None, "decode_error"))
    text, entries = service._replace_inline_images(uri, "user-1")
    assert text == uri
    assert entries[0]["status"] == "skipped"

    monkeypatch.setattr(service, "_persist_inline_image", lambda *_: (_uuid(1), None))
    wrapped = f"  {uri}  "
    text, entries = service._replace_inline_images(wrapped, "user-1")
    assert "/api/v1/files/" in text
    assert entries[0]["status"] == "detached"

    text, entries = service._replace_inline_images("hello world", "user-1")
    assert entries == []
    assert text == "hello world"


def test_inline_image_extract_recursion_guard(monkeypatch):
    service = InlineImageService()
    monkeypatch.setattr(service, "MAX_STRING_MAP_DEPTH", 1)
    payload = {"a": {"b": "c"}}
    assert service._extract_inline_images(payload) == []


def test_inline_image_scan_cancel_and_limits(db_session):
    user_id = _uuid(50)
    _add_user(db_session, user_id, name="Alice")
    _add_chat(db_session, chat_id=_uuid(51), user_id=user_id, title="noimg", payload=_make_history_text("hello"))
    _add_chat(db_session, chat_id=_uuid(52), user_id=user_id, title="img", payload=_make_history_text(_make_inline_markdown()))

    user_no_images = _uuid(53)
    _add_user(db_session, user_no_images, name="Bob")
    _add_chat(db_session, chat_id=_uuid(54), user_id=user_no_images, title="clean", payload=_make_history_text("hello"))

    service = InlineImageService(chunk_size=50)
    result = service.scan(
        user_filter=None,
        user_query=None,
        chat_ids=None,
        limit=1,
        status_callback=None,
        result_callback=None,
        user_sorter=None,
        cancel_event=None,
    )
    assert result["users_with_inline"] == 1
    assert result["has_more"] is True

    # Run without a tight limit so we also visit the no-image user and hit the skip branch.
    full = service.scan(
        user_filter=None,
        user_query=None,
        chat_ids=None,
        limit=10,
        status_callback=None,
        result_callback=None,
        user_sorter=None,
        cancel_event=None,
    )
    assert full["users_with_inline"] == 1

    cancel = threading.Event()
    cancel.set()
    cancelled = service.scan(
        user_filter=None,
        user_query=None,
        chat_ids=None,
        limit=10,
        status_callback=None,
        result_callback=None,
        user_sorter=None,
        cancel_event=cancel,
    )
    assert cancelled["has_more"] is True


def test_inline_image_detach_skips_and_limits(db_session, monkeypatch):
    user_id = _uuid(40)
    _add_user(db_session, user_id, name="Alice")

    markdown = _make_inline_markdown("image/png", b"img")
    _add_chat(db_session, chat_id=_uuid(41), user_id=user_id, title="img", payload=_make_history_text(markdown))

    # Force persist failures so detach reports skipped.
    original_insert = maintenance.Files.insert_new_file
    monkeypatch.setattr(maintenance.Files, "insert_new_file", lambda *_: None)
    service = InlineImageService(chunk_size=50)
    result = service.detach(max_chats=10, user_filter=None, user_query=None, chat_ids=None, status_callback=None, user_sorter=None, cancel_event=None)
    assert result["skipped_chats"] == 1
    assert result["skipped_images"] >= 1

    # Add another chat and verify max_chats triggers has_more.
    _add_chat(db_session, chat_id=_uuid(42), user_id=user_id, title="img2", payload=_make_history_text(markdown))
    monkeypatch.setattr(maintenance.Files, "insert_new_file", original_insert)
    result = service.detach(max_chats=1, user_filter=None, user_query=None, chat_ids=None, status_callback=None, user_sorter=None, cancel_event=None)
    assert result["has_more"] is True

    cancel = threading.Event()
    cancel.set()
    cancelled = service.detach(max_chats=10, user_filter=None, user_query=None, chat_ids=None, status_callback=None, user_sorter=None, cancel_event=cancel)
    assert cancelled["has_more"] is True


def test_storage_audit_service_has_more_and_status_callback():
    files = {
        "a": FileSummary(id="a", user_id=None, filename="a.bin", size=1, created_at=0, updated_at=0, meta={}, path="/tmp/a"),
        "b": FileSummary(id="b", user_id=None, filename="b.bin", size=1, created_at=0, updated_at=0, meta={}, path="/tmp/b"),
        "c": FileSummary(id="c", user_id=None, filename="c.bin", size=1, created_at=0, updated_at=0, meta={}, path="/tmp/c"),
    }

    class Inventory:
        def iter_objects(self):
            for idx in range(200):
                yield DiskObject(file_id=None, path=f"/tmp/extra-{idx}", name=f"extra-{idx}", size=1, modified_at=0)

    messages: list[str] = []

    def status(stage: str, message: str):
        messages.append(f"{stage}:{message}")

    auditor = StorageAuditService(Inventory())
    result = auditor.scan_db_orphans(files, limit=1, status_callback=status)
    assert result["has_more_disk"] is True
    assert result["has_more_missing"] is True
    assert any("Reviewed 200 files on disk" in msg for msg in messages)


def test_pipe_helpers_and_fallbacks(monkeypatch):
    pipe = Pipe()
    assert pipe._looks_like_remote_path("s3://bucket/key") is True
    assert pipe._looks_like_remote_path(str(Path(maintenance.UPLOAD_DIR) / "file.bin")) is False

    payload = pipe._format_data(is_stream=False, model="m", content="hi", usage={"tokens": 1}, finish_reason="stop")
    assert '"object": "chat.completion"' in payload
    assert '"usage"' in payload

    # Path sanitizer: reject escapes outside upload root.
    with pytest.raises(ValueError):
        pipe._sanitize_path_input("/etc/passwd")
    inside = pipe._sanitize_path_input("nested/file.txt")
    assert str(Path(inside)).startswith(str(Path(maintenance.UPLOAD_DIR)))

    # Vector client import failure path.
    monkeypatch.delitem(sys.modules, "open_webui.retrieval.vector.factory", raising=False)
    monkeypatch.delitem(sys.modules, "open_webui.retrieval.vector", raising=False)
    monkeypatch.delitem(sys.modules, "open_webui.retrieval", raising=False)
    assert pipe._get_vector_db_client() is None


@pytest.mark.asyncio
async def test_resolve_user_labels_fallback_loop(monkeypatch):
    pipe = Pipe()

    # Force the async batch query to fail so _resolve_user_labels uses the fallback loop.
    @contextmanager
    def broken_db():
        class Broken:
            def query(self, *_a, **_k):
                raise RuntimeError("nope")

        yield Broken()

    monkeypatch.setattr(maintenance, "get_db", broken_db)
    monkeypatch.setattr(maintenance.Users, "get_user_by_id", lambda uid: SimpleNamespace(id=uid, name=f"User {uid}"))

    labels = await pipe._resolve_user_labels([_uuid(1), _uuid(2)])
    assert labels[_uuid(1)].startswith("User ")


def test_local_storage_inventory_missing_root_and_empty_name(tmp_path: Path):
    inventory = LocalStorageInventory(root=str(tmp_path / "missing-root"))
    assert list(inventory.iter_objects()) == []
    assert LocalStorageInventory._derive_file_id("") is None


def test_upload_audit_service_cancel_and_limits(db_session):
    user_id = _uuid(60)
    _add_user(db_session, user_id, name="Alice")
    upload = UploadAuditService(chunk_size=50)

    file_a = maintenance.File(
        id=_uuid(61),
        user_id=user_id,
        filename="a.bin",
        path=str(Path(maintenance.UPLOAD_DIR) / f"{_uuid(61)}_a.bin"),
        meta={"data": {"size": "12"}},
        data={},
        created_at=1,
        updated_at=1,
    )
    file_b = maintenance.File(
        id=_uuid(62),
        user_id=user_id,
        filename="b.bin",
        path=str(Path(maintenance.UPLOAD_DIR) / f"{_uuid(62)}_b.bin"),
        meta={"size": "not-an-int"},
        data={},
        created_at=2,
        updated_at=2,
    )
    file_c = maintenance.File(
        id=_uuid(65),
        user_id=user_id,
        filename="c.bin",
        path=str(Path(maintenance.UPLOAD_DIR) / f"{_uuid(65)}_c.bin"),
        meta={"size": 1},
        data={},
        created_at=3,
        updated_at=3,
    )
    db_session.add_all([file_a, file_b, file_c])
    db_session.commit()

    # Chat payload as JSON string exercises UploadAuditService._ensure_payload (good JSON + bad JSON).
    _add_chat(
        db_session,
        chat_id=_uuid(63),
        user_id=user_id,
        title="c",
        payload=json.dumps({"history": {"messages": {"1": {"files": [{"id": _uuid(61)}]}}}}),
    )
    _add_chat(db_session, chat_id=_uuid(64), user_id=user_id, title="d", payload="{not-json")

    cancel = threading.Event()
    cancel.set()
    cancelled = upload.scan_db_orphans(user_filter=None, file_ids=None, limit=10, cancel_event=cancel)
    assert cancelled["orphans"] == []

    result = upload.scan_db_orphans(user_filter=user_id, file_ids=None, limit=1)
    assert result["has_more"] is True

    assert upload._extract_size({"data": {"size": "12"}}) == 12
    assert upload._extract_size({"size": "not-an-int"}) is None
