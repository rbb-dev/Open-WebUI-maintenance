from __future__ import annotations

import asyncio
import base64
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest

import open_webui_maintenance as maintenance
from open_webui_maintenance import BaseChatService, ChatRepairService, InlineImageService, LocalStorageInventory, UploadAuditService
from open_webui.models.users import User as UserModel


def _uuid(seed: int) -> str:
    return f"123e4567-e89b-12d3-a456-426614174{seed:03d}"


def _add_user(db_session, user_id: str, *, name: str):
    db_session.add(UserModel(id=user_id, name=name, username=None, email=None))
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


def _history(content: str) -> dict:
    return {"history": {"messages": {"1": {"files": [], "content": content}}}}


def _inline_data_uri(payload: bytes = b"x") -> str:
    b64 = base64.b64encode(payload).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def test_base_chat_service_build_query_with_chat_ids(db_session):
    user_id = _uuid(1)
    _add_user(db_session, user_id, name="Alice")
    chat_id = _uuid(2)
    _add_chat(db_session, chat_id=chat_id, user_id=user_id, title="ok", payload=_history("hello"))

    svc = BaseChatService()
    with maintenance.get_db() as db:
        results = svc._build_query(db, user_filter=None, chat_ids=[chat_id]).all()
        assert len(results) == 1


def test_collect_user_batches_returns_empty_when_user_query_matches_none(db_session):
    user_id = _uuid(3)
    _add_user(db_session, user_id, name="Alice")
    _add_chat(db_session, chat_id=_uuid(4), user_id=user_id, title="ok", payload=_history("hello"))

    svc = BaseChatService()
    with maintenance.get_db() as db:
        assert svc._collect_user_batches(db, user_filter=None, user_query="does-not-exist", chat_ids=None) == []


def test_inline_image_detach_inner_cancel_and_no_matches(db_session, monkeypatch):
    user_id = _uuid(10)
    _add_user(db_session, user_id, name="Alice")

    # One chat with no matches ensures matches_detected==0 path.
    _add_chat(db_session, chat_id=_uuid(11), user_id=user_id, title="noimg", payload=_history("hello"))
    # One chat with a match so the loop would continue if not cancelled.
    _add_chat(db_session, chat_id=_uuid(12), user_id=user_id, title="img", payload=_history(f"![a]({_inline_data_uri(b'img')})"))

    cancel = threading.Event()
    svc = InlineImageService(chunk_size=50)

    original = svc._detach_from_chat

    def _wrap(session, chat):
        result = original(session, chat)
        cancel.set()
        return result

    monkeypatch.setattr(svc, "_detach_from_chat", _wrap)
    result = svc.detach(
        max_chats=10,
        user_filter=None,
        user_query=None,
        chat_ids=None,
        status_callback=None,
        user_sorter=None,
        cancel_event=cancel,
    )
    assert result["has_more"] is True


def test_inline_image_scan_user_chats_cancel_break(db_session, monkeypatch):
    user_id = _uuid(13)
    _add_user(db_session, user_id, name="Alice")
    _add_chat(db_session, chat_id=_uuid(14), user_id=user_id, title="img1", payload=_history(f"![a]({_inline_data_uri(b'1')})"))
    _add_chat(db_session, chat_id=_uuid(15), user_id=user_id, title="img2", payload=_history(f"![a]({_inline_data_uri(b'2')})"))

    cancel = threading.Event()
    svc = InlineImageService(chunk_size=50)
    original = svc._extract_inline_images
    calls = {"n": 0}

    def _wrap(payload):
        calls["n"] += 1
        if calls["n"] == 1:
            cancel.set()
        return original(payload)

    monkeypatch.setattr(svc, "_extract_inline_images", _wrap)
    with maintenance.get_db() as db:
        query = svc._build_query(db, user_filter=user_id, chat_ids=None)
        stats = svc._scan_user_chats(query, cancel_event=cancel)
    assert stats["chats_with_inline"] == 1


def test_inline_image_detach_map_strings_value_error(monkeypatch):
    svc = InlineImageService()
    monkeypatch.setattr(svc, "MAX_STRING_MAP_DEPTH", 0)
    chat = type("ChatObj", (), {"chat": {"a": {"b": "c"}}, "user_id": "u", "id": "c", "updated_at": 0})()
    result = svc._detach_from_chat(None, chat)
    assert result["matches_detected"] == 0


def test_inline_image_map_strings_unknown_type_returns_empty_matches():
    svc = InlineImageService()

    def handler(text: str):
        return text, ["x"]

    assert handler("hello") == ("hello", ["x"])
    updated, matches = svc._map_strings(123, handler)
    assert updated == 123
    assert matches == []


def test_inline_image_guess_extension_path(monkeypatch):
    svc = InlineImageService()
    # Force a non-standard image MIME so we hit mimetypes.guess_extension().
    payload = base64.b64encode(b"hello").decode("utf-8")
    file_id, reason = svc._persist_inline_image("user-1", "image/tiff", payload)
    # Some platforms may not know tiff; accept either success or explicit unsupported.
    assert reason in {None, "unsupported_mime"}
    if reason is None:
        assert file_id is not None


def test_inline_image_map_strings_list_and_tuple():
    svc = InlineImageService()

    def handler(text: str):
        return text.upper(), ["x"]

    updated, matches = svc._map_strings(["a", ("b",)], handler)
    assert updated[0] == "A"
    assert updated[1][0] == "B"
    assert matches == ["x", "x"]


def test_inline_image_ensure_payload_and_estimate_size():
    svc = InlineImageService()
    assert svc._ensure_payload("{not-json") == "{not-json"
    assert svc._ensure_payload("hello") == "hello"
    assert svc._estimate_base64_size("") == 0
    assert svc._estimate_base64_size("   ") == 0


def test_chat_repair_service_inner_cancel_and_max_repairs(db_session, monkeypatch):
    user_id = _uuid(20)
    _add_user(db_session, user_id, name="Alice")
    for idx in range(3):
        _add_chat(db_session, chat_id=_uuid(21 + idx), user_id=user_id, title="ok", payload=_history("bad\u0000text"))

    svc = ChatRepairService(chunk_size=50)
    cancel = threading.Event()

    original_analyse = svc._analyse_chat

    def _wrap(chat, *, mutate: bool):
        report = original_analyse(chat, mutate=mutate)
        cancel.set()
        return report

    monkeypatch.setattr(svc, "_analyse_chat", _wrap)
    scan = svc.scan(max_results=10, user_filter=None, user_query=None, chat_ids=None, user_sorter=None, cancel_event=cancel)
    assert scan["has_more"] is True

    # Max repairs limit triggers has_more when more chats remain.
    svc2 = ChatRepairService(chunk_size=50)
    rep = svc2.repair(max_repairs=1, user_filter=None, user_query=None, chat_ids=None, user_sorter=None)
    assert rep["repaired"] == 1
    assert rep["has_more"] is True


def test_chat_repair_scan_skips_unchanged_chats(db_session):
    user_id = _uuid(80)
    _add_user(db_session, user_id, name="Alice")
    _add_chat(db_session, chat_id=_uuid(81), user_id=user_id, title="clean", payload=_history("clean"))

    svc = ChatRepairService(chunk_size=50)
    result = svc.scan(max_results=10, user_filter=None, user_query=None, chat_ids=None, user_sorter=None)
    assert result["matches"] == 0


def test_chat_repair_repair_cancel_paths(db_session, monkeypatch):
    user_id = _uuid(82)
    _add_user(db_session, user_id, name="Alice")
    _add_chat(db_session, chat_id=_uuid(83), user_id=user_id, title="ok", payload=_history("bad\u0000text"))
    _add_chat(db_session, chat_id=_uuid(84), user_id=user_id, title="ok", payload=_history("bad\u0000text"))

    svc = ChatRepairService(chunk_size=50)
    cancel_outer = threading.Event()
    cancel_outer.set()
    outer = svc.repair(max_repairs=10, user_filter=None, user_query=None, chat_ids=None, cancel_event=cancel_outer, user_sorter=None)
    assert outer["has_more"] is True

    cancel_inner = threading.Event()
    original = svc._analyse_chat
    calls = {"n": 0}

    def _wrap(chat, *, mutate: bool):
        calls["n"] += 1
        if calls["n"] == 1:
            cancel_inner.set()
        return original(chat, mutate=mutate)

    monkeypatch.setattr(svc, "_analyse_chat", _wrap)
    inner = svc.repair(max_repairs=10, user_filter=None, user_query=None, chat_ids=None, cancel_event=cancel_inner, user_sorter=None)
    assert inner["has_more"] is True


def test_sanitize_value_none_and_unknown_type():
    svc = ChatRepairService()
    assert svc._sanitize_value(None) == (None, False, {})
    assert svc._sanitize_value(123) == (123, False, {})


def test_local_storage_inventory_skips_directories(tmp_path: Path):
    upload = tmp_path / "upload"
    (upload / "nested").mkdir(parents=True)
    (upload / f"{_uuid(30)}_file.bin").write_bytes(b"x")

    inventory = LocalStorageInventory(root=str(upload))
    objects = list(inventory.iter_objects())
    assert len(objects) == 1


def test_derive_file_id_uuid_exception_branch(monkeypatch):
    inventory = LocalStorageInventory(root=str(Path(maintenance.UPLOAD_DIR)))
    original_uuid = maintenance.UUID

    def boom(*args, **kwargs):
        raise ValueError("nope")

    monkeypatch.setattr(maintenance, "UUID", boom)
    assert inventory._derive_file_id(f"inline-image-{_uuid(31)}.png") is None
    monkeypatch.setattr(maintenance, "UUID", original_uuid)


def test_upload_audit_collect_chat_and_knowledge_references(db_session, monkeypatch):
    user_id = _uuid(40)
    _add_user(db_session, user_id, name="Alice")
    file_id = _uuid(41)

    upload = UploadAuditService(chunk_size=1)
    upload.chunk_size = 1
    _add_chat(db_session, chat_id=_uuid(42), user_id=user_id, title="c", payload={"history": {"messages": {"1": {"files": [{"id": file_id}]}}}})
    _add_chat(db_session, chat_id=_uuid(43), user_id=user_id, title="d", payload={"history": {"messages": {"1": {"files": [{"id": file_id}]}}}})

    emitted: list[str] = []

    def emit(stage: str, msg: str):
        emitted.append(f"{stage}:{msg}")

    with maintenance.get_db() as db:
        refs = upload._collect_chat_references(db, emit, cancel_event=None)
        assert file_id in refs
        scanned_twos = [line for line in emitted if "Scanned 2 chats" in line]
        # Once from the periodic progress update, and once from the final emit.
        assert len(scanned_twos) == 2

        # Knowledge references.
        db.add(maintenance.KnowledgeFile(id=_uuid(44), knowledge_id="k", file_id=file_id, user_id=user_id, created_at=1, updated_at=1))
        db.commit()
        knowledge = upload._collect_knowledge_references(db, emit)
        assert file_id in knowledge


def test_upload_audit_collect_chat_references_cancel_break(db_session, monkeypatch):
    user_id = _uuid(45)
    _add_user(db_session, user_id, name="Alice")
    upload = UploadAuditService(chunk_size=1)

    _add_chat(db_session, chat_id=_uuid(46), user_id=user_id, title="c", payload=_history("hello"))
    _add_chat(db_session, chat_id=_uuid(47), user_id=user_id, title="d", payload=_history("hello"))

    cancel = threading.Event()

    original = upload._ensure_payload

    def _wrap(payload):
        cancel.set()
        return original(payload)

    monkeypatch.setattr(upload, "_ensure_payload", _wrap)

    with maintenance.get_db() as db:
        refs = upload._collect_chat_references(db, lambda *_: None, cancel_event=cancel)
    assert refs == set()


def test_upload_audit_meta_and_file_id_edge_cases():
    upload = UploadAuditService(chunk_size=10)

    assert upload._normalize_meta({"a": 1}) == {"a": 1}
    assert upload._normalize_meta('{"a": 2}') == {"a": 2}
    assert upload._normalize_meta("{not-json") == {}
    assert upload._normalize_meta(None) == {}

    assert upload._extract_file_ids_from_chat({"history": {"messages": []}}) == set()
    assert upload._extract_file_ids_from_chat({"history": {"messages": {"1": "not-a-dict"}}}) == set()
    assert upload._extract_file_ids_from_chat({"history": {"messages": {"1": {"files": "nope"}}}}) == set()
    # Non-dict entries should be ignored safely.
    assert upload._extract_file_ids_from_chat({"history": {"messages": {"1": {"files": [123]}}}}) == set()
    # Nested dict with no usable keys triggers the empty-string return.
    assert upload._extract_file_ids_from_chat({"history": {"messages": {"1": {"files": [{"file": {}}]}}}}) == set()


def test_upload_audit_ensure_payload_and_scan_cancellations(db_session, monkeypatch):
    upload = UploadAuditService(chunk_size=50)
    assert upload._ensure_payload("hello") == "hello"
    assert upload._ensure_payload(123) == 123

    user_id = _uuid(90)
    _add_user(db_session, user_id, name="Alice")
    file1 = maintenance.File(id=_uuid(91), user_id=user_id, filename="a", path="/tmp/a", meta={}, data={}, created_at=1, updated_at=1)
    file2 = maintenance.File(id=_uuid(92), user_id=user_id, filename="b", path="/tmp/b", meta={}, data={}, created_at=2, updated_at=2)
    db_session.add_all([file1, file2])
    db_session.commit()

    cancel = threading.Event()

    original_summarize = upload._summarize_file

    def _wrap(record):
        cancel.set()
        return original_summarize(record)

    monkeypatch.setattr(upload, "_summarize_file", _wrap)
    result = upload.scan_db_orphans(user_filter=None, file_ids=None, limit=10, cancel_event=cancel)
    assert result["orphans"] != []

    empty = upload.scan_db_orphans(user_filter=None, file_ids=[""], limit=10, cancel_event=None)
    assert empty["files_total"] == 0

@pytest.mark.asyncio
async def test_stream_cancellation_triggers_cleanup(db_session):
    pipe = maintenance.Pipe()
    user = {"id": _uuid(50), "name": "Admin"}
    _add_user(db_session, user["id"], name="Alice")
    _add_chat(db_session, chat_id=_uuid(51), user_id=user["id"], title="ok", payload=_history("bad\u0000text"))

    resp = await pipe.pipe({"messages": [{"role": "user", "content": "chat-scan limit=5"}]}, user, None)
    agen = resp.generator
    await agen.__anext__()  # first chunk
    started = asyncio.Event()

    async def consume():
        async for _ in agen:
            started.set()
            await asyncio.sleep(0)

    task = asyncio.create_task(consume())
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_pipe_emit_message_chunk_and_status_dedup(db_session):
    pipe = maintenance.Pipe()
    events: list[dict] = []
    got_status = asyncio.Event()

    async def emit(evt: dict):
        events.append(evt)
        if evt.get("type") == "status":
            got_status.set()

    await pipe._emit_message_chunk(emit, "\x00")
    assert events == []

    loop = asyncio.get_running_loop()
    cache: dict[str, str] = {}
    cb = pipe._threadsafe_user_status_callback(loop, emit, cache, action="Scanning", purpose="for tests")
    cb(_uuid(60), 1)
    cb(_uuid(60), 1)
    await asyncio.wait_for(got_status.wait(), timeout=1)


@pytest.mark.asyncio
async def test_image_scan_stream_cancellation_triggers_cleanup(db_session):
    pipe = maintenance.Pipe()
    user = {"id": _uuid(70), "name": "Admin"}
    _add_user(db_session, user["id"], name="Alice")
    _add_chat(db_session, chat_id=_uuid(71), user_id=user["id"], title="img", payload=_history(f"![a]({_inline_data_uri(b'img')})"))

    resp = await pipe.pipe({"messages": [{"role": "user", "content": "image-scan limit=5"}]}, user, None)
    agen = resp.generator
    await agen.__anext__()
    started = asyncio.Event()

    async def consume():
        async for _ in agen:
            started.set()
            await asyncio.sleep(0)

    task = asyncio.create_task(consume())
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
