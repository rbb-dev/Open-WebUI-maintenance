from __future__ import annotations

from collections import Counter

from open_webui_maintenance import ChatRepairService


class DummyChat:
    def __init__(self, title, chat, meta, updated_at=0):
        self.id = "chat-1"
        self.user_id = "user-1"
        self.title = title
        self.chat = chat
        self.meta = meta
        self.updated_at = updated_at


def test_sanitize_string_replaces_null_and_lone_surrogates():
    service = ChatRepairService()
    dirty = "ok\x00bad\ud800tail\udc96"

    sanitized, counts, changed = service._sanitize_string(dirty)

    assert changed is True
    assert sanitized == "okbad\ufffdtail\ufffd"
    assert counts == Counter(
        {
            "null_bytes": 1,
            "lone_high": 1,
            "lone_low": 1,
            "strings_touched": 1,
        }
    )


def test_sanitize_value_traverses_nested_structures():
    service = ChatRepairService()
    nested = {
        "list": ["good", "bad\udc96"],
        "tuple": ("ok", "broken\ud800"),
        "dict": {"inner": "still\x00bad"},
    }

    sanitized, changed, counts = service._sanitize_value(nested)

    assert changed is True
    assert sanitized["list"][1].endswith("\ufffd")
    assert sanitized["tuple"][1].endswith("\ufffd")
    assert "\x00" not in sanitized["dict"]["inner"]
    assert counts["strings_touched"] == 3


def test_analyse_chat_mutates_fields_and_tracks_counts(monkeypatch):
    service = ChatRepairService()
    chat = DummyChat(
        title="bad\ud800title",
        chat={"messages": ["ok", "bad\x00msg"]},
        meta="meta\udc00",
        updated_at=10,
    )

    report = service._analyse_chat(chat, mutate=True)

    assert report.changed is True
    assert "\ufffd" in chat.title
    assert "\x00" not in chat.chat["messages"][1]
    assert chat.meta.endswith("\ufffd")
    assert chat.updated_at >= 10
    assert report.counts["strings_touched"] == 3
