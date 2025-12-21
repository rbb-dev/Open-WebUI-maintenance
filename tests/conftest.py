"""Pytest configuration that stubs the Open WebUI modules expected by the pipe.

The real function runs inside Open WebUI, but unit tests only need to import the
module and exercise its logic. We provide a lightweight in-memory SQLAlchemy
database plus minimal stand-ins for Open WebUI modules/classes.
"""

from __future__ import annotations

import importlib.util
import sys
import time
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator, Optional

import pytest
from sqlalchemy import BigInteger, Column, JSON, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import StaticPool


ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = str((ROOT / ".tmp_uploads").resolve())
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)


Base = declarative_base()


class User(Base):
    __tablename__ = "user"

    id = Column(Text, primary_key=True)
    name = Column(Text, nullable=True)
    username = Column(Text, nullable=True)
    email = Column(Text, nullable=True)


class Chat(Base):
    __tablename__ = "chat"

    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=True)
    title = Column(Text, nullable=True)
    chat = Column(JSON, nullable=True)
    meta = Column(JSON, nullable=True)
    created_at = Column(BigInteger, nullable=True)
    updated_at = Column(BigInteger, nullable=True)


class File(Base):
    __tablename__ = "file"

    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=True)
    filename = Column(Text, nullable=True)
    path = Column(Text, nullable=True)
    data = Column(JSON, nullable=True)
    meta = Column(JSON, nullable=True)
    created_at = Column(BigInteger, nullable=True)
    updated_at = Column(BigInteger, nullable=True)


class KnowledgeFile(Base):
    __tablename__ = "knowledge_file"

    id = Column(String, primary_key=True)
    knowledge_id = Column(Text, nullable=True)
    file_id = Column(Text, nullable=False)
    user_id = Column(Text, nullable=True)
    created_at = Column(BigInteger, nullable=True)
    updated_at = Column(BigInteger, nullable=True)


engine = create_engine(
    "sqlite+pysqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
    future=True,
)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)
Base.metadata.create_all(bind=engine)


@contextmanager
def get_db() -> Iterator:
    session = SessionLocal()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


class Users:
    @staticmethod
    def get_user_by_id(user_id: str) -> Optional[User]:
        with get_db() as db:
            return db.get(User, user_id)

    @staticmethod
    def get_users():
        with get_db() as db:
            rows = db.query(User).all()
        return {"users": rows}


class FileForm:
    def __init__(
        self,
        *,
        id: str,
        filename: str,
        path: str,
        data: Optional[dict] = None,
        meta: Optional[dict] = None,
    ):
        self.id = id
        self.filename = filename
        self.path = path
        self.data = data or {}
        self.meta = meta or {}


class Files:
    @staticmethod
    def insert_new_file(user_id: str, form: FileForm):
        now = int(time.time())
        with get_db() as db:
            row = File(
                id=form.id,
                user_id=user_id,
                filename=form.filename,
                path=form.path,
                data=form.data,
                meta=form.meta,
                created_at=now,
                updated_at=now,
            )
            db.add(row)
            db.commit()
        return SimpleNamespace(id=form.id, path=form.path, user_id=user_id)

    @staticmethod
    def get_file_by_id(file_id: str):
        with get_db() as db:
            row = db.get(File, file_id)
            if not row:
                return None
            return SimpleNamespace(id=row.id, path=row.path, user_id=row.user_id)

    @staticmethod
    def delete_file_by_id(file_id: str) -> bool:
        with get_db() as db:
            deleted = db.query(File).filter(File.id == file_id).delete()
            db.commit()
            return bool(deleted)


class _Storage:
    @staticmethod
    def upload_file(fileobj, filename: str, _tags: dict):
        payload = fileobj.read()
        path = str(Path(UPLOAD_DIR) / filename)
        Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(payload)
        return payload, path

    @staticmethod
    def delete_file(file_path: str) -> None:
        if not file_path:
            return
        path = Path(file_path)
        if path.exists():
            path.unlink()


class _VectorClient:
    def __init__(self):
        self.deleted = []

    def delete(self, *, collection_name: str):
        self.deleted.append(collection_name)


VECTOR_DB_CLIENT = _VectorClient()


@pytest.fixture(autouse=True)
def _clean_state(tmp_path: Path):
    # Reset database tables.
    with get_db() as db:
        db.query(KnowledgeFile).delete()
        db.query(File).delete()
        db.query(Chat).delete()
        db.query(User).delete()
        db.commit()

    # Reset upload directory.
    upload_root = Path(UPLOAD_DIR)
    upload_root.mkdir(parents=True, exist_ok=True)
    for entry in upload_root.glob("**/*"):
        if entry.is_file():
            entry.unlink()

    VECTOR_DB_CLIENT.deleted.clear()
    yield


@pytest.fixture
def db_session():
    with get_db() as db:
        yield db


# --- Stub module hierarchy: open_webui.* ---
open_webui_module = types.ModuleType("open_webui")
internal_module = types.ModuleType("open_webui.internal")
internal_db_module = types.ModuleType("open_webui.internal.db")
internal_db_module.get_db = get_db

models_module = types.ModuleType("open_webui.models")
chats_module = types.ModuleType("open_webui.models.chats")
users_module = types.ModuleType("open_webui.models.users")
files_module = types.ModuleType("open_webui.models.files")
knowledge_module = types.ModuleType("open_webui.models.knowledge")

config_module = types.ModuleType("open_webui.config")
config_module.UPLOAD_DIR = UPLOAD_DIR

storage_provider_module = types.ModuleType("open_webui.storage.provider")
storage_provider_module.Storage = _Storage

retrieval_module = types.ModuleType("open_webui.retrieval")
vector_module = types.ModuleType("open_webui.retrieval.vector")
vector_factory_module = types.ModuleType("open_webui.retrieval.vector.factory")
vector_factory_module.VECTOR_DB_CLIENT = VECTOR_DB_CLIENT

chats_module.Chat = Chat
users_module.User = User
users_module.Users = Users
files_module.File = File
files_module.Files = Files
files_module.FileForm = FileForm
knowledge_module.KnowledgeFile = KnowledgeFile

sys.modules.setdefault("open_webui", open_webui_module)
sys.modules.setdefault("open_webui.internal", internal_module)
sys.modules.setdefault("open_webui.internal.db", internal_db_module)
sys.modules.setdefault("open_webui.models", models_module)
sys.modules.setdefault("open_webui.models.chats", chats_module)
sys.modules.setdefault("open_webui.models.users", users_module)
sys.modules.setdefault("open_webui.models.files", files_module)
sys.modules.setdefault("open_webui.models.knowledge", knowledge_module)
sys.modules.setdefault("open_webui.config", config_module)
sys.modules.setdefault("open_webui.storage", types.ModuleType("open_webui.storage"))
sys.modules.setdefault("open_webui.storage.provider", storage_provider_module)
sys.modules.setdefault("open_webui.retrieval", retrieval_module)
sys.modules.setdefault("open_webui.retrieval.vector", vector_module)
sys.modules.setdefault("open_webui.retrieval.vector.factory", vector_factory_module)


# Minimal FastAPI/Starlette stubs so the module imports cleanly during tests.
fastapi_module = types.ModuleType("fastapi")


class Request:  # pragma: no cover - placeholder type
    ...


fastapi_module.Request = Request


class PlainTextResponse:
    def __init__(self, content, status_code=200):
        self.content = content
        self.status_code = status_code


fastapi_responses = types.ModuleType("fastapi.responses")
fastapi_responses.PlainTextResponse = PlainTextResponse
fastapi_module.responses = fastapi_responses

starlette_module = types.ModuleType("starlette")


class StreamingResponse:
    def __init__(self, generator, media_type=None):
        self.generator = generator
        self.media_type = media_type


starlette_responses = types.ModuleType("starlette.responses")
starlette_responses.StreamingResponse = StreamingResponse
starlette_module.responses = starlette_responses

sys.modules.setdefault("fastapi", fastapi_module)
sys.modules.setdefault("fastapi.responses", fastapi_responses)
sys.modules.setdefault("starlette", starlette_module)
sys.modules.setdefault("starlette.responses", starlette_responses)


# Load the actual pipe module (filename contains dashes, so we create a clean alias).
module_path = ROOT / "open-webui-maintenance.py"
spec = importlib.util.spec_from_file_location("open_webui_maintenance", module_path)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader, "Unable to create module spec for open-webui-maintenance"
sys.modules.setdefault("open_webui_maintenance", module)
spec.loader.exec_module(module)
