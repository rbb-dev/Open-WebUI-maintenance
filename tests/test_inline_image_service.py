from __future__ import annotations

import base64

from open_webui_maintenance import InlineImageService


def _make_data_uri(payload: bytes, mime: str = "image/png") -> str:
    b64 = base64.b64encode(payload).decode("utf-8")
    return f"![test]({f'data:{mime};base64,'}{b64})"


def test_estimate_base64_size_matches_payload_length():
    service = InlineImageService()
    data = base64.b64encode(b"hello world").decode("utf-8")
    estimated = service._estimate_base64_size(data)
    assert estimated >= len(b"hello world")


def test_extract_inline_images_detects_data_uri():
    service = InlineImageService()
    payload = _make_data_uri(b"abc123", "image/jpeg")
    matches = service._extract_inline_images(payload)
    assert len(matches) == 1
    assert matches[0].mime_type == "image/jpeg"
    assert matches[0].approx_bytes >= 6


def test_replace_inline_images_uses_persisted_file(monkeypatch):
    service = InlineImageService()
    monkeypatch.setattr(service, "_persist_inline_image", lambda user_id, mime, data: "file-123")
    text = f"example {_make_data_uri(b'payload', 'image/png')}"
    new_text, entries = service._replace_inline_images(text, "user-1")
    assert "file-123" in new_text
    assert len(entries) == 1
    assert entries[0]["file_id"] == "file-123"


def test_bare_data_uri_detection(monkeypatch):
    service = InlineImageService()
    base = base64.b64encode(b"inline-bytes").decode("utf-8")
    uri = f"data:image/png;base64,{base}"
    matches = service._extract_inline_images(uri)
    assert len(matches) == 1
    monkeypatch.setattr(service, "_persist_inline_image", lambda user_id, mime, data: "file-789")
    new_text, entries = service._replace_inline_images(uri, "user-9")
    assert new_text.endswith("/api/v1/files/file-789/content")
    assert entries[0]["file_id"] == "file-789"
