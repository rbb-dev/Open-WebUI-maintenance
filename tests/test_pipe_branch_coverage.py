from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import open_webui_maintenance as maintenance
from open_webui.models.chats import Chat as ChatModel
from open_webui.models.files import File as FileModel
from open_webui.models.users import User as UserModel


def _uuid(seed: int) -> str:
    return f"123e4567-e89b-12d3-a456-426614174{seed:03d}"


def _add_user(db, user_id: str, *, name: str | None = None, username: str | None = None, email: str | None = None):
    db.add(UserModel(id=user_id, name=name, username=username, email=email))
    db.commit()


def _add_chat(db, *, chat_id: str, user_id: str, title: str, payload, updated_at: int = 0):
    db.add(ChatModel(id=chat_id, user_id=user_id, title=title, chat=payload, updated_at=updated_at))
    db.commit()


def _add_file(db, *, file_id: str, user_id: str, filename: str, meta, path: str | None = None, updated_at: int = 0):
    db.add(FileModel(id=file_id, user_id=user_id, filename=filename, meta=meta, path=path, updated_at=updated_at))
    db.commit()


def test_inline_image_service_ensure_payload_non_string():
    svc = maintenance.InlineImageService()
    assert svc._ensure_payload(123) == 123


def test_upload_audit_load_files_filters_ids(db_session):
    svc = maintenance.UploadAuditService(chunk_size=5)
    target_id = _uuid(1)
    _add_file(db_session, file_id=target_id, user_id=_uuid(2), filename="a.txt", meta={"size": 1})
    files = svc._load_files(db_session, user_filter=None, file_ids=[target_id], cancel_event=None)
    assert list(files) == [target_id]


def test_pipe_parse_command_error_and_edge_branches():
    pipe = maintenance.Pipe()

    assert pipe._parse_command("   ") == ("", {})

    with pytest.raises(ValueError, match="shorter than 256"):
        pipe._parse_command("x" * (pipe.MAX_TOKEN_LENGTH + 1))

    with pytest.raises(ValueError, match="Missing command"):
        pipe._parse_command('"" limit=5')

    with pytest.raises(ValueError, match="Missing command name"):
        pipe._parse_command("/")

    command, options = pipe._parse_command('"unterminated')
    assert command == '"unterminated'
    assert options["ids"] == []

    with pytest.raises(ValueError, match="shorter than 2048"):
        long_value_tokens = ["x" * 200] * 11  # > 2048 chars once joined, but each token stays < 256.
        pipe._parse_command("db-scan user_query " + " ".join(long_value_tokens))

    command, options = pipe._parse_command("db-scan user=")
    assert command == "db-scan"
    assert options["user_id"] is None

    command, options = pipe._parse_command("db-scan limit=abc")
    assert command == "db-scan"
    assert options["limit"] is None

    command, options = pipe._parse_command("storage-clean path=hello.txt")
    assert command == "storage-clean"
    assert isinstance(options["path"], str)
    assert options["path"].startswith(str(pipe.upload_root))

    command, options = pipe._parse_command('db-scan ""')
    assert command == "db-scan"
    assert options["confirm"] is False

    assert pipe._normalize_quotes(None) == ""
    assert pipe._strip_matching_quotes(None) is None
    assert pipe._strip_matching_quotes('" hello "') == "hello"

    with pytest.raises(ValueError, match="cannot be empty"):
        pipe._sanitize_path_input("")
    assert pipe._looks_like_remote_path(None) is False


@pytest.mark.asyncio
async def test_pipe_stream_helpers_user_label_and_summaries(monkeypatch):
    pipe = maintenance.Pipe()

    def boom(_user_id: str):
        raise RuntimeError("db down")

    monkeypatch.setattr(maintenance.Users, "get_user_by_id", boom)
    assert pipe._lookup_user_label_sync("boom", {}) == "boom"

    assert await pipe._get_user_label_async("cached", {"cached": "Cached User"}) == "Cached User"

    cache = {"u": "U"}
    row = {"user_id": "u", "chat_id": "c", "title": "t", "issue_counts": {}, "fields": ["title", "chat"]}
    formatted = await pipe._format_chat_stream_row(row, cache)
    assert "title, chat" in formatted

    summary = pipe._build_stream_chat_scan_summary(
        {"examined": 1, "matches": 0, "has_more": True, "results": [], "counters": {}},
        "entire workspace",
    )
    assert "Limit reached" in summary
    assert "No malformed Unicode" in summary

    image_summary = pipe._build_stream_image_scan_summary(
        {
            "users_with_inline": 1,
            "total_chats_with_inline": 2,
            "total_inline_images": 3,
            "total_inline_bytes": 4,
            "has_more": True,
        },
        "entire workspace",
    )
    assert "Limit reached" in image_summary


def test_pipe_report_builders_cover_empty_and_limit_branches():
    pipe = maintenance.Pipe()

    db_report = pipe._build_db_scan_report(
        {
            "files_total": 1,
            "referenced_in_chats": 0,
            "referenced_in_knowledge": 0,
            "orphan_total": 0,
            "orphans": [],
            "has_more": True,
        },
        missing_rows=[],
        missing_total=1,
        missing_truncated=True,
        user_labels={},
        scope="entire workspace",
        limit=0,
    )
    assert "Output limit: unlimited" in db_report
    assert "Some orphaned files were not shown" in db_report
    assert "Some missing files were not shown" in db_report

    storage_report = pipe._build_storage_scan_report(
        {
            "disk_total": 0,
            "disk_without_db_total": 0,
            "disk_without_db": [],
            "has_more_disk": True,
        },
        scope="entire workspace",
        limit=0,
        user_labels={},
    )
    assert "Output limit: unlimited" in storage_report
    assert "Some files without database records were not shown" in storage_report
    assert "No files on disk are missing database records" in storage_report

    chat_scan_report = pipe._build_chat_scan_report(
        {"examined": 1, "matches": 0, "has_more": True, "results": [], "counters": {}},
        user_labels={},
        scope="entire workspace",
    )
    assert "Limit reached before finishing" in chat_scan_report
    assert "No malformed Unicode" in chat_scan_report

    chat_repair_empty = pipe._build_chat_repair_report(
        {"examined": 1, "repaired": 0, "details": [], "counters": {}, "has_more": True},
        user_labels={},
        limit=0,
        scope="entire workspace",
    )
    assert "Limit: unlimited" in chat_repair_empty
    assert "Limit reached" in chat_repair_empty
    assert "No chats required changes" in chat_repair_empty

    chat_repair_more = pipe._build_chat_repair_report(
        {
            "examined": 2,
            "repaired": 1,
            "details": [{"chat_id": "c1", "user_id": "u1", "issue_counts": {}, "fields": []}],
            "counters": {},
            "has_more": True,
        },
        user_labels={"u1": "User"},
        limit=5,
        scope="entire workspace",
    )
    assert "Additional chats still need repairs" in chat_repair_more

    image_scan_report = pipe._build_image_scan_report(
        {
            "users_with_inline": 0,
            "total_chats_with_inline": 0,
            "total_inline_images": 0,
            "total_inline_bytes": 0,
            "summaries": [],
            "has_more": True,
        },
        user_labels={},
        scope="entire workspace",
        limit=0,
    )
    assert "No inline base64 image blobs" in image_scan_report
    assert "Output limit: unlimited" in image_scan_report
    assert "Some users were omitted" in image_scan_report

    detach_report = pipe._build_image_detach_report(
        {
            "processed_chats": 0,
            "images_detached": 0,
            "bytes_detached": 0,
            "skipped_chats": 2,
            "skipped_images": 3,
            "skipped_reason_counts": {"decode_error": 2, "unknown_reason": 1},
            "records": [],
            "skipped_records": [],
            "has_more": True,
        },
        user_labels={},
        limit=0,
        scope="entire workspace",
    )
    assert "Chats skipped" in detach_report
    assert "Skip reasons" in detach_report
    assert "Limit: unlimited" in detach_report
    assert "Limit reached" in detach_report
    assert "No chats required changes" in detach_report

    detach_report_skipped = pipe._build_image_detach_report(
        {
            "processed_chats": 1,
            "images_detached": 0,
            "bytes_detached": 0,
            "records": [],
            "skipped_records": [
                maintenance.InlineImageSkipRecord(
                    chat_id="c1",
                    user_id="u1",
                    skipped_images=1,
                    skipped_bytes=10,
                    reasons={"storage_error": 1},
                )
            ],
        },
        user_labels={"u1": "User"},
        limit=5,
        scope="entire workspace",
    )
    assert "could not be detached" in detach_report_skipped

    assert pipe._format_skip_reason_summary({}) == ""
    assert "unsupported MIME type" in pipe._format_skip_reason_summary({"unsupported_mime": 2})

    assert pipe._describe_counts({}) == ""
    assert pipe._describe_counts({"strings_touched": 2}) == "2 strings sanitized"

    assert pipe._estimate_chat_bytes(None) == 0
    assert pipe._estimate_chat_bytes({1, 2, 3}) > 0


def test_pipe_user_report_helpers_cover_branches(db_session, monkeypatch):
    pipe = maintenance.Pipe()

    calls: list[object] = []

    def row_callback(row):
        calls.append(row)

    assert pipe._collect_user_report_rows([], status_callback=None, row_callback=row_callback) == []
    assert calls == [None]

    calls.clear()
    assert pipe._collect_user_report_rows([{}], status_callback=None, row_callback=row_callback) == []
    assert calls == [None]

    assert pipe._get_user_attribute({"id": "x"}, "id") == "x"
    assert pipe._get_user_attribute(object(), "id") is None

    assert pipe._resolve_user_value("   ") == (None, None)

    def raising_get_user(_candidate: str):
        raise RuntimeError("fail")

    monkeypatch.setattr(maintenance.Users, "get_user_by_id", raising_get_user)
    monkeypatch.setattr(maintenance.Users, "get_users", lambda: [])
    resolved, error = pipe._resolve_user_value("bob")
    assert resolved is None
    assert error

    def raising_get_users():
        raise RuntimeError("fail")

    monkeypatch.setattr(maintenance.Users, "get_user_by_id", lambda _candidate: None)
    monkeypatch.setattr(maintenance.Users, "get_users", raising_get_users)
    resolved, error = pipe._resolve_user_value("bob")
    assert resolved is None
    assert error

    monkeypatch.setattr(maintenance.Users, "get_user_by_id", lambda _candidate: SimpleNamespace(id="resolved"))
    resolved, error = pipe._resolve_user_value("any")
    assert resolved == "resolved"
    assert error is None

    monkeypatch.setattr(maintenance.Users, "get_user_by_id", lambda _candidate: None)
    monkeypatch.setattr(maintenance.Users, "get_users", lambda: "not-a-list")
    resolved, error = pipe._resolve_user_value("any")
    assert resolved is None
    assert error

    monkeypatch.setattr(
        maintenance.Users,
        "get_users",
        lambda: {"users": [SimpleNamespace(id="u1", name="Alice Doe"), SimpleNamespace(id="u2", name="Bob")]},
    )
    resolved, error = pipe._resolve_user_value("ali")
    assert resolved == "u1"
    assert error is None

    pipe2 = maintenance.Pipe()
    monkeypatch.setattr(pipe2, "_user_label", lambda _user: "")
    monkeypatch.setattr(maintenance.Users, "get_users", lambda: {"users": [SimpleNamespace(id="u1", name="Alice")]})
    resolved, error = pipe2._resolve_user_value("ali")
    assert resolved is None
    assert error

    assert pipe._user_label({"id": "fallback"}) == "fallback"

    assert pipe._prepare_user_report_rows([], {}, limit=5) == []
    assert pipe._prepare_user_report_rows([{}], {}, limit=5) == []

    _add_user(db_session, "u1", name="Alice")
    _add_user(db_session, "u2", name="Bob")
    usage = {
        "u1": maintenance.UserUsageStats(chat_count=1, chat_bytes=10, file_count=0, file_bytes=0),
        "u2": maintenance.UserUsageStats(chat_count=1, chat_bytes=5, file_count=0, file_bytes=0),
    }
    rows = pipe._prepare_user_report_rows([{"id": "u1", "name": "Alice"}, {"id": "u2", "name": "Bob"}], usage, limit=1)
    assert len(rows) == 1

    assert pipe._render_user_report([]) == "No user data available."


@pytest.mark.asyncio
async def test_pipe_emit_user_report_table_and_stream_empty(monkeypatch):
    pipe = maintenance.Pipe()

    await pipe._emit_user_report_table(None, [], detail_tables="")

    events: list[dict] = []

    async def emitter(evt: dict):
        events.append(evt)

    await pipe._emit_user_report_table(emitter, [], detail_tables="")
    assert any("No user data available" in evt.get("data", {}).get("content", "") for evt in events)

    resp = await pipe._stream_user_report(
        user_records=[],
        status_callback=None,
        emitter=emitter,
        detail_user_id=None,
        detail_label=None,
    )
    chunks = []
    async for chunk in resp.generator:
        chunks.append(chunk)
    assert any("No user data available" in chunk for chunk in chunks)


def test_pipe_user_top_lists_cover_heappop_and_skip(db_session):
    pipe = maintenance.Pipe()
    user_id = "u-top"
    _add_user(db_session, user_id, name="Top")

    _add_chat(db_session, chat_id="c0", user_id=user_id, title="zero", payload=None)
    _add_chat(db_session, chat_id="c1", user_id=user_id, title="a", payload={"k": "x" * 10})
    _add_chat(db_session, chat_id="c2", user_id=user_id, title="b", payload={"k": "x" * 20})
    _add_chat(db_session, chat_id="c3", user_id=user_id, title="c", payload={"k": "x" * 30})
    top_chats = pipe._get_top_user_chats(db_session, user_id, limit=2)
    assert len(top_chats) == 2

    _add_file(db_session, file_id="f0", user_id=user_id, filename="z", meta={"size": 0})
    _add_file(db_session, file_id="f1", user_id=user_id, filename="a", meta={"size": 10})
    _add_file(db_session, file_id="f2", user_id=user_id, filename="b", meta={"size": 20})
    _add_file(db_session, file_id="f3", user_id=user_id, filename="c", meta={"size": 30})
    top_files = pipe._get_top_user_files(db_session, user_id, limit=2)
    assert len(top_files) == 2


def test_pipe_cleanup_helpers_cover_failures_and_warnings(monkeypatch):
    pipe = maintenance.Pipe()

    def delete_file(path: str) -> None:
        if path == "boom":
            raise RuntimeError("nope")

    def delete_file_by_id(file_id: str) -> bool:
        if file_id == "e":
            raise RuntimeError("db fail")
        return file_id != "c"

    class VectorClient:
        def delete(self, *, collection_name: str):
            raise RuntimeError("vector fail")

    monkeypatch.setattr(maintenance.Storage, "delete_file", delete_file)
    monkeypatch.setattr(maintenance.Files, "delete_file_by_id", delete_file_by_id)
    monkeypatch.setattr(pipe, "_get_vector_db_client", lambda: VectorClient())

    entries = [
        {"file_id": None},
        {"file_id": "a"},
        {"file_id": "b", "path": "boom"},
        {"file_id": "c", "path": "ok"},
        {"file_id": "d", "path": "ok"},
        {"file_id": "e", "path": "ok"},
    ]
    result = pipe._clean_db_entries(entries, delete_storage=True)
    assert "a" in {row["file_id"] for row in result["failed"]}
    assert "b" in {row["file_id"] for row in result["failed"]}
    assert "c" in {row["file_id"] for row in result["failed"]}
    assert "d" in result["deleted"]
    assert any(row["file_id"] == "d" for row in result["warnings"])
    assert any(row["file_id"] == "e" for row in result["failed"])

    report = pipe._build_clean_db_report(
        {"deleted": ["d"], "failed": [{"file_id": "x", "error": "nope"}], "warnings": [{"file_id": "d", "warning": "warn"}]},
        "entire workspace",
        {},
    )
    assert "Warnings:" in report
    assert "#### Failures" in report
    assert "#### Warnings" in report

    disk = [maintenance.DiskObject(file_id="x", name="n", size=1, modified_at=0, path="p")]

    class BadInventory:
        def delete(self, _obj):
            raise RuntimeError("cant")

    storage_result = pipe._clean_storage_entries(disk, BadInventory())
    assert storage_result["failed"]

    storage_report = pipe._build_clean_storage_report(
        {"deleted": [], "failed": [{"path": "p", "error": "cant"}], "deleted_objects": []},
        "entire workspace",
    )
    assert "#### Failures" in storage_report

    assert pipe._format_filesize(None) == "—"
    assert pipe._format_filesize(-1) == "—"
    assert pipe._format_timestamp("bad") == "bad"
    assert pipe._sanitize_output_text(None) == ""
    assert pipe._sanitize_output_text("") == ""


def test_pipe_prompt_and_history_helpers_cover_branches():
    pipe = maintenance.Pipe()

    assert pipe._extract_prompt_text({"messages": [{"role": "assistant", "content": "x"}], "prompt": "hi"}) == "hi"

    assert pipe._collapse_content([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "a b"
    assert pipe._collapse_content({"type": "text", "text": "hello"}) == "hello"
    assert pipe._collapse_content({"content": "wrapped"}) == "wrapped"
    assert pipe._collapse_content({"type": "image", "url": "x"}) == ""

    chat_id = _uuid(9)
    body = {
        "messages": [
            {"role": "user", "content": "ignored"},
            {"role": "assistant", "content": f"| Chat ID | User |\n| --- | --- |\n| `{chat_id}` | u |"},
            {"role": "assistant", "content": f"| `{chat_id}` | u |"},
            {"role": "assistant", "content": []},
        ]
    }
    ids = pipe._extract_chat_ids_from_history(body)
    assert ids == [chat_id]

    assert pipe._extract_chat_ids_from_text("| onlyone |") == []
