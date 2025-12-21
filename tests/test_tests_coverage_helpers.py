from __future__ import annotations

import pytest

from open_webui.models.files import Files as FilesAPI
from open_webui.storage.provider import Storage as StorageAPI


def test_conftest_files_get_file_by_id_missing_returns_none():
    assert FilesAPI.get_file_by_id("does-not-exist") is None


def test_conftest_storage_delete_file_ignores_empty_path():
    StorageAPI.delete_file("")


def test_pipe_command_integration_helper_make_body_covers_assistant_branch():
    from tests import test_pipe_commands_integration as integration

    body = integration._make_body("help", assistant_text="previous output")
    assert body["messages"][0]["role"] == "assistant"


def test_user_label_batch_legacy_original_load_exception_branch():
    from tests.test_user_label_batch_legacy import original_load

    class UsersBoom:
        def get_user_by_id(self, _uid: str):
            raise RuntimeError("boom")

    mapping = original_load(["u1"], UsersBoom())
    assert mapping == {"u1": "u1"}

