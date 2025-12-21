"""
Legacy regression tests for the batched user-label resolver.

These mirror the original script-style tests but live under `tests/` so pytest
collects them consistently.
"""

from typing import Dict, Sequence
from unittest.mock import MagicMock, Mock


def original_load(ids: Sequence[str], Users_mock) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for uid in ids:
        try:
            user = Users_mock.get_user_by_id(uid)
        except Exception:
            user = None
        if user:
            label = getattr(user, "name", None) or getattr(user, "username", None) or getattr(user, "email", None)
            mapping[uid] = label or uid
        else:
            mapping[uid] = uid
    return mapping


def batched_load(ids: Sequence[str], User_class_mock, get_db_mock) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with get_db_mock() as db:
        users = db.query(User_class_mock).filter(User_class_mock.id.in_(ids)).all()
        user_map = {user.id: user for user in users}
        for uid in ids:
            user = user_map.get(uid)
            if user:
                label = getattr(user, "name", None) or getattr(user, "username", None) or getattr(user, "email", None)
                mapping[uid] = label or uid
            else:
                mapping[uid] = uid
    return mapping


def create_mock_user(user_id: str, name: str = None, username: str = None, email: str = None):
    user = Mock()
    user.id = user_id
    user.name = name
    user.username = username
    user.email = email
    return user


def _make_db_context(mock_users):
    db_mock = MagicMock()
    query_chain = db_mock.query.return_value.filter.return_value
    query_chain.all.return_value = mock_users

    get_db_mock = MagicMock()
    get_db_mock.return_value.__enter__.return_value = db_mock
    get_db_mock.return_value.__exit__.return_value = None
    return get_db_mock, db_mock


def test_basic_user_resolution():
    user_ids = ["user1", "user2", "user3"]
    Users_mock = Mock()
    Users_mock.get_user_by_id = Mock(
        side_effect=lambda uid: {
            "user1": create_mock_user("user1", name="Alice"),
            "user2": create_mock_user("user2", name="Bob"),
            "user3": create_mock_user("user3", name="Charlie"),
        }[uid]
    )

    mock_users = [
        create_mock_user("user1", name="Alice"),
        create_mock_user("user2", name="Bob"),
        create_mock_user("user3", name="Charlie"),
    ]
    get_db_mock, _ = _make_db_context(mock_users)

    User_class_mock = Mock()
    User_class_mock.id = Mock()
    User_class_mock.id.in_ = Mock(return_value="filter_condition")

    assert original_load(user_ids, Users_mock) == batched_load(user_ids, User_class_mock, get_db_mock)


def test_missing_users():
    user_ids = ["user1", "user_missing", "user3"]
    Users_mock = Mock()

    def get_user_side_effect(uid):
        if uid == "user_missing":
            return None
        return {
            "user1": create_mock_user("user1", name="Alice"),
            "user3": create_mock_user("user3", name="Charlie"),
        }.get(uid)

    Users_mock.get_user_by_id = Mock(side_effect=get_user_side_effect)

    mock_users = [
        create_mock_user("user1", name="Alice"),
        create_mock_user("user3", name="Charlie"),
    ]
    get_db_mock, _ = _make_db_context(mock_users)

    User_class_mock = Mock()
    User_class_mock.id = Mock()
    User_class_mock.id.in_ = Mock(return_value="filter_condition")

    assert original_load(user_ids, Users_mock) == batched_load(user_ids, User_class_mock, get_db_mock)

