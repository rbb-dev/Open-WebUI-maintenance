"""Pytest configuration that stubs the Open WebUI modules expected by the pipe.

The real function runs inside Open WebUI, but unit tests only need to import the
module and exercise its pure-Python helpers. To keep the tests lightweight we
inject minimal stand-ins for ``open_webui.internal.db`` / ``open_webui.models``
modules before ``open_webui_maintenance`` is imported.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from contextlib import contextmanager
from pathlib import Path


class _QueryPlaceholder:
    """Minimal placeholder returned by DummySession.query."""

    def filter(self, *_, **__):  # pragma: no cover - only used when misconfigured
        return self

    def group_by(self, *_, **__):  # pragma: no cover
        return self

    def yield_per(self, *_):  # pragma: no cover
        return []

    def order_by(self, *_):  # pragma: no cover
        return self

    def limit(self, *_):  # pragma: no cover
        return self

    def all(self):  # pragma: no cover
        return []


class DummySession:
    def query(self, *_, **__):
        return _QueryPlaceholder()


@contextmanager
def dummy_db():
    yield DummySession()


# Build module hierarchy: open_webui, open_webui.internal, etc.
open_webui_module = types.ModuleType("open_webui")
internal_module = types.ModuleType("open_webui.internal")
internal_db_module = types.ModuleType("open_webui.internal.db")
internal_db_module.get_db = dummy_db
models_module = types.ModuleType("open_webui.models")
chats_module = types.ModuleType("open_webui.models.chats")
users_module = types.ModuleType("open_webui.models.users")
files_module = types.ModuleType("open_webui.models.files")
knowledge_module = types.ModuleType("open_webui.models.knowledge")
config_module = types.ModuleType("open_webui.config")
config_module.UPLOAD_DIR = "/tmp"
starlette_module = types.ModuleType("starlette")


# Minimal FastAPI/Starlette stubs so the module imports cleanly during tests.
fastapi_module = types.ModuleType("fastapi")


class Request:  # pragma: no cover - placeholder type
    ...


fastapi_module.Request = Request


class PlainTextResponse:  # pragma: no cover - used as a simple data holder in tests
    def __init__(self, content, status_code=200):
        self.content = content
        self.status_code = status_code


fastapi_responses = types.ModuleType("fastapi.responses")
fastapi_responses.PlainTextResponse = PlainTextResponse
fastapi_module.responses = fastapi_responses


class StreamingResponse:  # pragma: no cover - placeholder for type hints
    def __init__(self, generator, media_type=None):
        self.generator = generator
        self.media_type = media_type


starlette_responses = types.ModuleType("starlette.responses")
starlette_responses.StreamingResponse = StreamingResponse
starlette_module.responses = starlette_responses


class DummyUser:
    def __init__(self, user_id: str):
        self.id = user_id
        self.name = f"User {user_id}"
        self.username = f"user_{user_id}"
        self.email = f"{user_id}@example.com"


class DummyUsers:
    @staticmethod
    def get_user_by_id(user_id: str):
        return DummyUser(user_id)


class DummyChat:
    id: str
    user_id: str
    title: str
    chat: str
    meta: str
    updated_at: int


users_module.Users = DummyUsers
users_module.User = DummyUser
chats_module.Chat = DummyChat
files_module.File = type("File", (), {})
files_module.Files = type(
    "Files",
    (),
    {
        "delete_file_by_id": staticmethod(lambda file_id: bool(file_id)),
    },
)
knowledge_module.KnowledgeFile = type("KnowledgeFile", (), {})

sys.modules.setdefault("open_webui", open_webui_module)
sys.modules.setdefault("open_webui.internal", internal_module)
sys.modules.setdefault("open_webui.internal.db", internal_db_module)
sys.modules.setdefault("open_webui.models", models_module)
sys.modules.setdefault("open_webui.models.chats", chats_module)
sys.modules.setdefault("open_webui.models.users", users_module)
sys.modules.setdefault("open_webui.models.files", files_module)
sys.modules.setdefault("open_webui.models.knowledge", knowledge_module)
sys.modules.setdefault("open_webui.config", config_module)
sys.modules.setdefault("fastapi", fastapi_module)
sys.modules.setdefault("fastapi.responses", fastapi_responses)
sys.modules.setdefault("starlette", starlette_module)
sys.modules.setdefault("starlette.responses", starlette_responses)


# Load the actual pipe module (filename contains dashes, so we create a clean alias).
ROOT = Path(__file__).resolve().parents[1]
module_path = ROOT / "open-webui-maintenance.py"
spec = importlib.util.spec_from_file_location("open_webui_maintenance", module_path)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader, "Unable to create module spec for open-webui-maintenance"
sys.modules.setdefault("open_webui_maintenance", module)
spec.loader.exec_module(module)
