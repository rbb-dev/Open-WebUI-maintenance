from __future__ import annotations

from open_webui_maintenance import UploadAuditService, FileSummary


def test_extract_file_ids_from_chat_handles_various_shapes():
    service = UploadAuditService()
    payload = {
        "history": {
            "messages": {
                "1": {"files": [{"id": "file-a"}, {"file": {"file_id": "file-b"}}]},
                "2": {"files": ["file-c", {"fileId": "file-d"}]},
                "3": {"files": None},
            }
        }
    }

    file_ids = service._extract_file_ids_from_chat(payload)

    assert file_ids == {"file-a", "file-b", "file-c", "file-d"}


def test_summarize_file_generates_metadata_snapshot():
    service = UploadAuditService()
    record = FileSummary(
        id="file-123",
        user_id="user-1",
        filename="report.pdf",
        size=2048,
        created_at=100,
        updated_at=200,
        meta={"content_type": "application/pdf"},
        path="/tmp/report.pdf",
    )

    summary = service._summarize_file(record)

    assert summary["file_id"] == "file-123"
    assert summary["user_id"] == "user-1"
    assert summary["size"] == 2048
    assert summary["content_type"] == "application/pdf"
    assert summary["path"] == "/tmp/report.pdf"
