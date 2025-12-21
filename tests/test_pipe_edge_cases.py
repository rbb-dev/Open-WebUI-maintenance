from __future__ import annotations

import pytest

import open_webui_maintenance as maintenance
from open_webui_maintenance import Pipe
from open_webui.models.users import User as UserModel


def _uuid(seed: int) -> str:
    return f"123e4567-e89b-12d3-a456-426614174{seed:03d}"


@pytest.mark.asyncio
async def test_pipes_metadata():
    pipe = Pipe()
    rows = await pipe.pipes()
    assert rows == [{"id": pipe.PIPE_ID, "name": pipe.PIPE_NAME}]


def test_sync_services_from_valves_handles_bad_values():
    pipe = Pipe()
    pipe.valves.DB_CHUNK_SIZE = "bad"  # type: ignore[assignment]
    pipe.valves.CHAT_DB_CHUNK_SIZE = "bad"  # type: ignore[assignment]
    pipe._sync_services_from_valves()


@pytest.mark.asyncio
async def test_pipe_returns_parse_errors_as_400():
    pipe = Pipe()
    user = {"id": _uuid(1)}
    resp = await pipe.pipe({"messages": [{"role": "user", "content": "/db-scan user not-a-uuid"}]}, user, None)
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_pipe_returns_user_lookup_error_when_user_missing(db_session):
    pipe = Pipe()
    user = {"id": _uuid(1)}
    missing_user = _uuid(999)
    resp = await pipe.pipe({"messages": [{"role": "user", "content": f"db-scan user={missing_user}"}]}, user, None)
    assert resp.status_code == 400
    assert "No user found matching" in resp.content


@pytest.mark.asyncio
async def test_pipe_help_command():
    pipe = Pipe()
    user = {"id": _uuid(1)}
    resp = await pipe.pipe({"messages": [{"role": "user", "content": "help"}]}, user, None)
    assert resp.status_code == 200
    assert "db-scan" in resp.content


@pytest.mark.asyncio
async def test_resolve_user_labels_batch_missing_user_maps_to_id(db_session):
    pipe = Pipe()
    present = _uuid(10)
    db_session.add(UserModel(id=present, name="Alice", username=None, email=None))
    db_session.commit()

    labels = await pipe._resolve_user_labels([present, _uuid(11)])
    assert labels[present] == "Alice"
    assert labels[_uuid(11)] == _uuid(11)


@pytest.mark.asyncio
async def test_resolve_user_labels_fallback_handles_user_lookup_exceptions(monkeypatch):
    pipe = Pipe()
    u1 = _uuid(20)
    u2 = _uuid(21)

    def broken_db():
        class Broken:
            def query(self, *_a, **_k):
                raise RuntimeError("nope")

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

        return Broken()

    monkeypatch.setattr(maintenance, "get_db", lambda: broken_db())

    def get_user(uid: str):
        if uid == u1:
            raise RuntimeError("boom")
        return None

    monkeypatch.setattr(maintenance.Users, "get_user_by_id", get_user)
    labels = await pipe._resolve_user_labels([u1, u2])
    assert labels[u1] == u1
    assert labels[u2] == u2

