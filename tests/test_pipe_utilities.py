from __future__ import annotations

import pytest

from types import SimpleNamespace

import open_webui_maintenance as cleanup
from open_webui_maintenance import LocalStorageInventory, Pipe, UserUsageStats


@pytest.fixture(scope="module")
def pipe():
    return Pipe()


def test_parse_command_extracts_options(pipe):
    ids = [
        "123e4567-e89b-12d3-a456-426614174000",
        "123e4567-e89b-12d3-a456-426614174111",
        "123e4567-e89b-12d3-a456-426614174222",
    ]
    command, options = pipe._parse_command(
        f"db-clean confirm limit = 25 id = {ids[0]},{ids[1]};{ids[2]} user = me"
    )
    assert command == "db-clean"
    assert options["confirm"] is True
    assert options["limit"] == 25
    assert options["ids"] == ids
    assert options["user_id"] == "me"


def test_parse_command_accepts_space_separated_values(pipe):
    command, options = pipe._parse_command("storage-clean confirm limit 5 user me")
    assert command == "storage-clean"
    assert options["limit"] == 5
    assert options["user_id"] == "me"
    assert options["confirm"] is True


def test_parse_command_handles_uuid_user(pipe):
    user_uuid = "123e4567-e89b-12d3-a456-426614174333"
    command, options = pipe._parse_command(f"user-report user \"{user_uuid}\" limit 5")
    assert command == "user-report"
    assert options["user_id"] == user_uuid
    assert options["limit"] == 5


def test_parse_command_reads_user_query(pipe):
    command, options = pipe._parse_command('chat-scan user_query "John Citizen" limit 5')
    assert command == "chat-scan"
    assert options["user_query"] == "John Citizen"
    assert options["limit"] == 5


def test_parse_command_handles_quoted_user(pipe):
    user_uuid = "123e4567-e89b-12d3-a456-426614174444"
    command, options = pipe._parse_command(f'/image-detach confirm user="{user_uuid}" limit=1')
    assert command == "image-detach"
    assert options["user_id"] == user_uuid
    assert options["limit"] == 1


def test_parse_command_handles_smart_quotes(pipe):
    user_uuid = "123e4567-e89b-12d3-a456-426614174555"
    smart_command = f"/image-detach confirm user=\u201c{user_uuid}\u201d"
    command, options = pipe._parse_command(smart_command)
    assert command == "image-detach"
    assert options["user_id"] == user_uuid


def test_parse_command_accepts_plain_commands(pipe):
    command, options = pipe._parse_command("db-clean limit=5")
    assert command == "db-clean"
    assert options["limit"] == 5


def test_parse_command_rejects_invalid_user_id(pipe):
    with pytest.raises(ValueError):
        pipe._parse_command("/db-scan user not-a-uuid")


def test_parse_command_rejects_invalid_id_list(pipe):
    with pytest.raises(ValueError):
        pipe._parse_command("/db-clean limit=1 id=not-a-uuid user=me")


def test_clamp_limit_bounds(pipe):
    assert pipe._clamp_limit(None, default=10, ceiling=50, allow_zero=True) == 10
    assert pipe._clamp_limit(0, default=10, ceiling=50, allow_zero=True) == 0
    assert pipe._clamp_limit(-5, default=10, ceiling=50, allow_zero=False) == 10
    assert pipe._clamp_limit(500, default=10, ceiling=50, allow_zero=False) == 50


def test_describe_scope(pipe):
    assert pipe._describe_scope("user-1", []) == "user `user-1`"
    assert pipe._describe_scope(None, ["a", "b"]) == "specific file IDs (2)"
    assert pipe._describe_scope(None, []) == "entire workspace"


def test_describe_scope_with_user_query(pipe):
    assert pipe._describe_scope(None, [], user_query="alice") == 'users matching "alice"'
    assert (
        pipe._describe_scope(None, ["chat-1"], user_query="bob", ids_label="chat IDs")
        == 'users matching "bob", specific chat IDs (1)'
    )


def test_build_db_scan_report_contains_table(pipe):
    scan_result = {
        "files_total": 3,
        "referenced_in_chats": 2,
        "referenced_in_knowledge": 0,
        "orphan_total": 1,
        "orphans": [
            {
                "file_id": "file-1",
                "user_id": "user-1",
                "filename": "report.pdf",
                "size": 2048,
                "created_at": 0,
                "path": "/tmp/report.pdf",
            }
        ],
        "has_more": False,
    }
    missing_rows = [
        {
            "file_id": "file-2",
            "user_id": "user-2",
            "filename": "slides.pdf",
            "path": "/tmp/slides.pdf",
            "updated_at": 0,
        }
    ]
    user_labels = {"user-1": "John Citizen"}
    report = pipe._build_db_scan_report(
        scan_result,
        missing_rows=missing_rows,
        missing_total=1,
        missing_truncated=False,
        user_labels=user_labels,
        scope="entire workspace",
        limit=25,
    )
    assert "Database scan results" in report
    assert "| `file-1` | John Citizen |" in report
    assert "| `file-2` | user-2 |" in report


def test_format_filesize(pipe):
    assert pipe._format_filesize(0) == "0 B"
    assert pipe._format_filesize(2048).endswith("KB")


def test_build_storage_scan_report(pipe):
    scan_result = {
        "disk_total": 3,
        "disk_without_db_total": 1,
        "db_missing_disk_total": 1,
        "disk_without_db": [
            {
                "file_id": "extra-1",
                "name": "stray.bin",
                "size": 1024,
                "modified_at": 0,
                "path": "/tmp/stray.bin",
            }
        ],
        "disk_without_db_objects": [],
        "missing_from_disk": [
            {
                "file_id": "file-1",
                "user_id": "user-1",
                "filename": "report.pdf",
                "path": "/tmp/report.pdf",
                "updated_at": 0,
            }
        ],
        "has_more_disk": False,
        "has_more_missing": False,
    }
    user_labels = {"user-1": "John Citizen"}
    report = pipe._build_storage_scan_report(scan_result, scope="entire workspace", limit=25, user_labels=user_labels)
    assert "Storage scan results" in report
    assert "Files on disk without database records" in report


def test_build_chat_scan_report(pipe):
    scan_result = {
        "examined": 10,
        "matches": 1,
        "results": [
            {
                "chat_id": "chat-1",
                "user_id": "user-1",
                "title": "bad chat",
                "updated_at": 0,
                "issue_counts": {"null_bytes": 1},
                "fields": ["chat"],
            }
        ],
        "counters": {"null_bytes": 1},
        "has_more": False,
    }
    report = pipe._build_chat_scan_report(
        scan_result,
        user_labels={"user-1": "John Citizen"},
        scope="entire workspace",
    )
    assert "Chat scan summary" in report
    assert "| `chat-1` | John Citizen |" in report


def test_build_chat_repair_report(pipe):
    repair_result = {
        "examined": 5,
        "repaired": 2,
        "details": [
            {
                "chat_id": "chat-1",
                "user_id": "user-1",
                "issue_counts": {"lone_low": 1},
                "fields": ["chat"],
            }
        ],
        "counters": {"lone_low": 1},
        "has_more": False,
    }
    report = pipe._build_chat_repair_report(
        repair_result,
        user_labels={"user-1": "John Citizen"},
        limit=10,
        scope="entire workspace",
    )
    assert "Chat repair summary" in report
    assert "Chats repaired: 2" in report
    assert "| `chat-1` | John Citizen |" in report


def test_build_user_report(pipe):
    users = [
        SimpleNamespace(id="user-1", name="John Citizen", email="john1@example.com"),
        SimpleNamespace(id="user-2", name="John Citizen", email="john2@example.com"),
    ]
    usage = {
        "user-1": UserUsageStats(chat_count=2, chat_bytes=2048, file_count=1, file_bytes=1024),
        "user-2": UserUsageStats(chat_count=1, chat_bytes=512, file_count=0, file_bytes=0),
    }
    report = pipe._build_user_report(users, usage, limit=0)
    assert "User usage report" in report
    assert "John Citizen" in report


def test_prepare_user_report_rows_sorted_by_name(pipe):
    users = [
        SimpleNamespace(id="user-1", name="John Citizen", email="john1@example.com"),
        SimpleNamespace(id="user-2", name="John Citizen", email="john2@example.com"),
    ]
    usage = {
        "user-1": UserUsageStats(chat_count=1, chat_bytes=100, file_count=1, file_bytes=200),
        "user-2": UserUsageStats(chat_count=1, chat_bytes=100, file_count=1, file_bytes=200),
    }
    rows = pipe._prepare_user_report_rows(users, usage, limit=0)
    assert [row["label"] for row in rows] == ["John Citizen", "John Citizen"]


def test_resolve_user_value_matches_name(pipe, monkeypatch):
    fake_users = [
        SimpleNamespace(id="user-1", name="John Citizen", email="john@example.com"),
    ]

    monkeypatch.setattr(cleanup.Users, "get_user_by_id", lambda value: None)
    monkeypatch.setattr(cleanup.Users, "get_users", lambda: {"users": fake_users}, raising=False)

    user_id, error = pipe._resolve_user_value("John Citizen")
    assert user_id == "user-1"
    assert error is None


def test_resolve_user_value_detects_ambiguity(pipe, monkeypatch):
    fake_users = [
        SimpleNamespace(id="user-1", name="John Citizen", email="alpha@example.com"),
        SimpleNamespace(id="user-2", name="John Citizen", email="beta@example.com"),
    ]

    monkeypatch.setattr(cleanup.Users, "get_user_by_id", lambda value: None)
    monkeypatch.setattr(cleanup.Users, "get_users", lambda: {"users": fake_users}, raising=False)

    user_id, error = pipe._resolve_user_value("John Citizen")
    assert user_id is None
    assert "Multiple users match" in error


def test_describe_counts(pipe):
    counts = {"null_bytes": 2, "lone_high": 1, "strings_touched": 3}
    summary = pipe._describe_counts(counts)
    assert "null bytes" in summary
    assert "lone high" in summary


def test_extract_chat_ids_from_text(pipe):
    text = """
| User | Chat ID | Title | Issues |
| --- | --- | --- | --- |
| John Citizen | `chat-1` | Bad chat | 1 null byte |
"""
    ids = pipe._extract_chat_ids_from_text(text)
    assert ids == ["chat-1"]


def test_derive_file_id_supports_underscored_and_inline_names():
    uuid_value = "123e4567-e89b-12d3-a456-426614174abc"
    assert (
        LocalStorageInventory._derive_file_id(f"{uuid_value}_sample.txt")
        == uuid_value
    )
    assert (
        LocalStorageInventory._derive_file_id(f"inline-image-{uuid_value}.png")
        == uuid_value
    )
    assert LocalStorageInventory._derive_file_id("notes.txt") is None
