from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

import open_webui_maintenance as maintenance
from open_webui_maintenance import Pipe
from open_webui.models.users import User as UserModel


def _uuid(seed: int) -> str:
    return f"123e4567-e89b-12d3-a456-426614174{seed:03d}"


def _make_body(text: str | None, *, stream: bool | None = None, assistant_text: str | None = None) -> dict:
    messages = []
    if assistant_text is not None:
        messages.append({"role": "assistant", "content": assistant_text})
    if text is not None:
        messages.append({"role": "user", "content": text})
    body: dict = {"messages": messages}
    if stream is not None:
        body["stream"] = stream
    return body


async def _collect_stream(response) -> list[str]:
    chunks: list[str] = []
    async for chunk in response.generator:
        chunks.append(chunk)
    return chunks


@pytest.fixture
def pipe() -> Pipe:
    return Pipe()


@pytest.fixture
def user():
    return {"id": _uuid(1), "name": "Admin"}


@pytest.fixture
def event_log():
    events: list[dict] = []

    async def _emit(evt: dict):
        events.append(evt)

    return events, _emit


def _write_upload(file_id: str, filename: str, payload: bytes = b"hello") -> str:
    upload_dir = Path(maintenance.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    disk_name = f"{file_id}_{filename}"
    path = upload_dir / disk_name
    path.write_bytes(payload)
    return str(path)


def _add_user(db_session, user_id: str, *, name: str = "John Citizen"):
    db_session.add(UserModel(id=user_id, name=name, username=None, email=None))
    db_session.commit()


def _add_file(
    db_session,
    *,
    file_id: str,
    user_id: str,
    filename: str,
    path: str,
    size: int = 5,
):
    now = 1
    db_session.add(
        maintenance.File(
            id=file_id,
            user_id=user_id,
            filename=filename,
            path=path,
            meta={"content_type": "application/octet-stream", "size": size},
            data={},
            created_at=now,
            updated_at=now,
        )
    )
    db_session.commit()


def _add_chat(db_session, *, chat_id: str, user_id: str, title: str, payload):
    now = 1
    db_session.add(
        maintenance.Chat(
            id=chat_id,
            user_id=user_id,
            title=title,
            chat=payload,
            meta={},
            created_at=now,
            updated_at=now,
        )
    )
    db_session.commit()


def _make_history_with_files(file_ids: list[str], *, text: str | None = None) -> dict:
    message = {"files": [{"id": fid} for fid in file_ids]}
    if text is not None:
        message["content"] = text
    return {"history": {"messages": {"1": message}}}


@pytest.mark.asyncio
async def test_help_when_empty_prompt(pipe, user):
    resp = await pipe.pipe(_make_body(None), user, None)
    assert resp.status_code == 200
    assert "db-scan" in resp.content


@pytest.mark.asyncio
async def test_unknown_command_returns_help(pipe, user):
    resp = await pipe.pipe(_make_body("nope"), user, None)
    assert resp.status_code == 400
    assert "Unknown command" in resp.content
    assert "db-scan" in resp.content


@pytest.mark.asyncio
async def test_db_scan_reports_orphans_and_missing(pipe, user, db_session, event_log):
    events, emitter = event_log
    _add_user(db_session, user["id"], name="Alice")

    file_kept = _uuid(10)
    file_orphan = _uuid(11)

    kept_path = _write_upload(file_kept, "kept.bin", b"hello")
    _add_file(db_session, file_id=file_kept, user_id=user["id"], filename="kept.bin", path=kept_path, size=5)
    _add_file(db_session, file_id=file_orphan, user_id=user["id"], filename="orphan.bin", path=str(Path(maintenance.UPLOAD_DIR) / f"{file_orphan}_orphan.bin"), size=5)

    chat_id = _uuid(20)
    _add_chat(db_session, chat_id=chat_id, user_id=user["id"], title="ok", payload=_make_history_with_files([file_kept]))

    resp = await pipe.pipe(_make_body("db-scan limit=5"), user, None, emitter)
    assert resp.status_code == 200
    assert "Database scan results" in resp.content
    assert file_orphan in resp.content
    assert "Database records with missing files" in resp.content
    assert any(evt.get("type") == "status" for evt in events)


@pytest.mark.asyncio
async def test_storage_scan_lists_untracked_disk_files(pipe, user):
    stray_id = _uuid(30)
    _write_upload(stray_id, "stray.bin", b"hello")

    resp = await pipe.pipe(_make_body("storage-scan limit=5"), user, None)
    assert resp.status_code == 200
    assert "Storage scan results" in resp.content
    assert "Files on disk without database records" in resp.content


@pytest.mark.asyncio
async def test_user_report_handles_empty_user_list(pipe, user):
    resp = await pipe.pipe(_make_body("user-report", stream=False), user, None)
    assert resp.status_code == 200
    assert "No user data found" in resp.content


@pytest.mark.asyncio
async def test_user_report_non_stream(pipe, user, db_session):
    _add_user(db_session, user["id"], name="Alice")
    resp = await pipe.pipe(_make_body("user-report user=me", stream=False), user, None)
    assert resp.status_code == 200
    assert "User usage report" in resp.content


@pytest.mark.asyncio
async def test_user_report_non_stream_emits_table(pipe, user, db_session, event_log):
    events, emitter = event_log
    _add_user(db_session, user["id"], name="Alice")
    _add_chat(db_session, chat_id=_uuid(43), user_id=user["id"], title="hello", payload=_make_history_with_files([]))

    resp = await pipe.pipe(_make_body("user-report user=me", stream=False), user, None, emitter)
    assert resp.status_code == 200
    assert "Largest chats" in resp.content
    assert any(evt.get("type") == "message" for evt in events)


@pytest.mark.asyncio
async def test_user_report_streams_rows_and_details(pipe, user, db_session, event_log):
    events, emitter = event_log
    _add_user(db_session, user["id"], name="Alice")

    chat_id = _uuid(40)
    _add_chat(db_session, chat_id=chat_id, user_id=user["id"], title="hello", payload=_make_history_with_files([]))

    file_id = _uuid(41)
    file_path = _write_upload(file_id, "note.txt", b"hello")
    _add_file(db_session, file_id=file_id, user_id=user["id"], filename="note.txt", path=file_path, size=5)

    resp = await pipe.pipe(_make_body("user-report user=me", stream=True), user, None, emitter)
    chunks = await _collect_stream(resp)
    joined = "".join(chunks)
    assert "User usage report" in joined
    assert "data: [DONE]" in joined
    assert any(evt.get("type") == "message" for evt in events)


@pytest.mark.asyncio
async def test_chat_scan_streams_rows(pipe, user, db_session):
    _add_user(db_session, user["id"], name="Alice")
    chat_id = _uuid(50)
    payload = _make_history_with_files([], text="bad\x00text")
    _add_chat(db_session, chat_id=chat_id, user_id=user["id"], title="bad\x00title", payload=payload)

    resp = await pipe.pipe(_make_body("chat-scan limit=5"), user, None)
    chunks = await _collect_stream(resp)
    joined = "".join(chunks)
    assert "Scan in progress" in joined
    assert "| User | Chat ID | Title | Issues |" in joined
    assert chat_id in joined
    assert "### Scan summary" in joined
    assert "data: [DONE]" in joined


@pytest.mark.asyncio
async def test_chat_scan_non_stream(pipe, user, db_session):
    _add_user(db_session, user["id"], name="Alice")
    chat_id = _uuid(52)
    payload = _make_history_with_files([], text="bad\x00text")
    _add_chat(db_session, chat_id=chat_id, user_id=user["id"], title="bad\x00title", payload=payload)

    resp = await pipe.pipe(_make_body("chat-scan limit=5", stream=False), user, None)
    assert resp.status_code == 200
    assert "Chat scan summary" in resp.content
    assert chat_id in resp.content


@pytest.mark.asyncio
async def test_chat_repair_requires_confirm(pipe, user):
    resp = await pipe.pipe(_make_body("chat-repair limit=5"), user, None)
    assert resp.status_code == 400
    assert "confirm" in resp.content


@pytest.mark.asyncio
async def test_chat_repair_mutates_chats(pipe, user, db_session):
    _add_user(db_session, user["id"], name="Alice")
    chat_id = _uuid(51)
    payload = _make_history_with_files([], text="bad\x00text")
    _add_chat(db_session, chat_id=chat_id, user_id=user["id"], title="bad\x00title", payload=payload)

    resp = await pipe.pipe(_make_body("chat-repair confirm limit=5"), user, None)
    assert resp.status_code == 200
    assert "Chat repair summary" in resp.content

    repaired = db_session.get(maintenance.Chat, chat_id)
    assert repaired is not None
    assert "\x00" not in json.dumps(repaired.chat)
    assert "\x00" not in (repaired.title or "")


@pytest.mark.asyncio
async def test_image_scan_and_detach(pipe, user, db_session):
    _add_user(db_session, user["id"], name="Alice")
    payload = base64.b64encode(b"image-bytes").decode("utf-8")
    markdown = f"![alt](data:image/png;base64,{payload})"
    chat_id = _uuid(60)
    chat_payload = _make_history_with_files([], text=markdown)
    _add_chat(db_session, chat_id=chat_id, user_id=user["id"], title="has image", payload=chat_payload)

    scan_resp = await pipe.pipe(_make_body("image-scan limit=5"), user, None)
    scan_chunks = await _collect_stream(scan_resp)
    assert "Inline image scan" in "".join(scan_chunks)

    non_stream = await pipe.pipe(_make_body("image-scan limit=5", stream=False), user, None)
    assert non_stream.status_code == 200
    assert "Inline image scan results" in non_stream.content

    deny = await pipe.pipe(_make_body("image-detach limit=5"), user, None)
    assert deny.status_code == 400

    detach = await pipe.pipe(_make_body("image-detach confirm limit=5"), user, None)
    assert detach.status_code == 200
    assert "Inline images detached" in detach.content

    updated = db_session.get(maintenance.Chat, chat_id)
    assert updated is not None
    serialized = json.dumps(updated.chat)
    assert "/api/v1/files/" in serialized
    assert "data:image/png;base64" not in serialized


@pytest.mark.asyncio
async def test_db_clean_missing_files_skips_remote_paths(pipe, user, db_session):
    _add_user(db_session, user["id"], name="Alice")
    remote_id = _uuid(70)
    _add_file(
        db_session,
        file_id=remote_id,
        user_id=user["id"],
        filename="remote.bin",
        path="s3://bucket/key",
        size=5,
    )

    resp = await pipe.pipe(_make_body("db-clean confirm limit=5"), user, None)
    assert resp.status_code == 400
    assert "remote storage paths" in resp.content
    assert db_session.get(maintenance.File, remote_id) is not None


@pytest.mark.asyncio
async def test_db_clean_requires_confirm(pipe, user):
    resp = await pipe.pipe(_make_body("db-clean limit=5"), user, None)
    assert resp.status_code == 400
    assert "confirm" in resp.content


@pytest.mark.asyncio
async def test_db_clean_when_no_missing_files(pipe, user, db_session):
    _add_user(db_session, user["id"], name="Alice")
    file_id = _uuid(72)
    file_path = _write_upload(file_id, "present.bin", b"hello")
    _add_file(db_session, file_id=file_id, user_id=user["id"], filename="present.bin", path=file_path, size=5)

    resp = await pipe.pipe(_make_body("db-clean confirm limit=5"), user, None)
    assert resp.status_code == 200
    assert "Every database record still has a matching file" in resp.content


@pytest.mark.asyncio
async def test_db_clean_missing_files_deletes_local_rows(pipe, user, db_session):
    _add_user(db_session, user["id"], name="Alice")
    missing_id = _uuid(71)
    _add_file(
        db_session,
        file_id=missing_id,
        user_id=user["id"],
        filename="missing.bin",
        path=str(Path(maintenance.UPLOAD_DIR) / f"{missing_id}_missing.bin"),
        size=5,
    )

    resp = await pipe.pipe(_make_body("db-clean confirm limit=5"), user, None)
    assert resp.status_code == 200
    assert db_session.get(maintenance.File, missing_id) is None


@pytest.mark.asyncio
async def test_db_clean_mixed_remote_and_local_missing(pipe, user, db_session):
    _add_user(db_session, user["id"], name="Alice")
    remote_id = _uuid(73)
    local_id = _uuid(74)
    _add_file(db_session, file_id=remote_id, user_id=user["id"], filename="remote.bin", path="s3://bucket/key", size=5)
    _add_file(db_session, file_id=local_id, user_id=user["id"], filename="missing.bin", path=str(Path(maintenance.UPLOAD_DIR) / f"{local_id}_missing.bin"), size=5)

    resp = await pipe.pipe(_make_body("db-clean confirm limit=5"), user, None)
    assert resp.status_code == 200
    assert "Note: skipped" in resp.content
    assert db_session.get(maintenance.File, remote_id) is not None
    assert db_session.get(maintenance.File, local_id) is None


@pytest.mark.asyncio
async def test_db_clean_orphan_files_deletes_storage_and_vectors(pipe, user, db_session):
    _add_user(db_session, user["id"], name="Alice")
    orphan_id = _uuid(80)
    orphan_path = _write_upload(orphan_id, "orphan.bin", b"hello")
    _add_file(db_session, file_id=orphan_id, user_id=user["id"], filename="orphan.bin", path=orphan_path, size=5)

    resp = await pipe.pipe(_make_body("db-clean-orphan-files confirm limit=5"), user, None)
    assert resp.status_code == 200
    assert db_session.get(maintenance.File, orphan_id) is None
    assert not Path(orphan_path).exists()

    vector = pipe._get_vector_db_client()
    assert vector is not None
    assert f"file-{orphan_id}" in getattr(vector, "deleted", [])


@pytest.mark.asyncio
async def test_db_clean_orphan_requires_confirm(pipe, user):
    resp = await pipe.pipe(_make_body("db-clean-orphan-files limit=5"), user, None)
    assert resp.status_code == 400
    assert "confirm" in resp.content


@pytest.mark.asyncio
async def test_db_clean_orphan_when_no_orphans(pipe, user, db_session):
    _add_user(db_session, user["id"], name="Alice")
    file_id = _uuid(81)
    file_path = _write_upload(file_id, "kept.bin", b"hello")
    _add_file(db_session, file_id=file_id, user_id=user["id"], filename="kept.bin", path=file_path, size=5)
    _add_chat(db_session, chat_id=_uuid(82), user_id=user["id"], title="ok", payload=_make_history_with_files([file_id]))

    resp = await pipe.pipe(_make_body("db-clean-orphan-files confirm limit=5"), user, None)
    assert resp.status_code == 200
    assert "No orphaned uploads remain" in resp.content


@pytest.mark.asyncio
async def test_storage_clean_removes_untracked_disk_files(pipe, user):
    stray_id = _uuid(90)
    stray_path = _write_upload(stray_id, "stray.bin", b"hello")
    assert Path(stray_path).exists()

    resp = await pipe.pipe(_make_body("storage-clean confirm limit=5"), user, None)
    assert resp.status_code == 200
    assert not Path(stray_path).exists()


@pytest.mark.asyncio
async def test_storage_clean_requires_confirm(pipe, user):
    resp = await pipe.pipe(_make_body("storage-clean limit=5"), user, None)
    assert resp.status_code == 400
    assert "confirm" in resp.content


@pytest.mark.asyncio
async def test_storage_clean_when_no_untracked_files(pipe, user, db_session):
    _add_user(db_session, user["id"], name="Alice")
    file_id = _uuid(91)
    file_path = _write_upload(file_id, "kept.bin", b"hello")
    _add_file(db_session, file_id=file_id, user_id=user["id"], filename="kept.bin", path=file_path, size=5)

    resp = await pipe.pipe(_make_body("storage-clean confirm limit=5"), user, None)
    assert resp.status_code == 200
    assert "Every file on disk already has a database record" in resp.content


@pytest.mark.asyncio
async def test_db_scan_stream_response_uses_sse(pipe, user, db_session):
    _add_user(db_session, user["id"], name="Alice")
    file_id = _uuid(95)
    file_path = _write_upload(file_id, "kept.bin", b"hello")
    _add_file(db_session, file_id=file_id, user_id=user["id"], filename="kept.bin", path=file_path, size=5)
    _add_chat(db_session, chat_id=_uuid(96), user_id=user["id"], title="ok", payload=_make_history_with_files([file_id]))

    resp = await pipe.pipe(_make_body("db-scan limit=5", stream=True), user, None)
    chunks = await _collect_stream(resp)
    joined = "".join(chunks)
    assert "data:" in joined
    assert "data: [DONE]" in joined
