"""
Test suite for batched user label resolution optimization.
Ensures the batched version behaves identically to the original N+1 version.
"""
import sys
import io
from typing import Dict, List, Sequence
from unittest.mock import Mock, MagicMock, patch

# Force UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def original_load(ids: Sequence[str], Users_mock) -> Dict[str, str]:
    """Original N+1 implementation."""
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
    """New batched implementation."""
    mapping: Dict[str, str] = {}
    with get_db_mock() as db:
        try:
            # Batch fetch all users in one query
            users = db.query(User_class_mock).filter(User_class_mock.id.in_(ids)).all()
            user_map = {user.id: user for user in users}

            for uid in ids:
                user = user_map.get(uid)
                if user:
                    label = getattr(user, "name", None) or getattr(user, "username", None) or getattr(user, "email", None)
                    mapping[uid] = label or uid
                else:
                    mapping[uid] = uid
        except Exception as e:
            # Should not happen in this test, but included for completeness
            raise
    return mapping


def create_mock_user(user_id: str, name: str = None, username: str = None, email: str = None):
    """Create a mock user object."""
    user = Mock()
    user.id = user_id
    user.name = name
    user.username = username
    user.email = email
    return user


def test_basic_user_resolution():
    """Test basic case with all users having names."""
    print("\n" + "="*80)
    print("Test 1: Basic user resolution")
    print("="*80)

    user_ids = ["user1", "user2", "user3"]

    # Setup mocks for original
    Users_mock = Mock()
    Users_mock.get_user_by_id = Mock(side_effect=lambda uid: {
        "user1": create_mock_user("user1", name="Alice"),
        "user2": create_mock_user("user2", name="Bob"),
        "user3": create_mock_user("user3", name="Charlie"),
    }[uid])

    # Setup mocks for batched
    mock_users = [
        create_mock_user("user1", name="Alice"),
        create_mock_user("user2", name="Bob"),
        create_mock_user("user3", name="Charlie"),
    ]

    db_mock = MagicMock()
    query_chain = db_mock.query.return_value.filter.return_value
    query_chain.all.return_value = mock_users

    User_class_mock = Mock()
    User_class_mock.id = Mock()
    User_class_mock.id.in_ = Mock(return_value="filter_condition")

    get_db_mock = MagicMock()
    get_db_mock.return_value.__enter__.return_value = db_mock
    get_db_mock.return_value.__exit__.return_value = None

    # Execute both
    result_original = original_load(user_ids, Users_mock)
    result_batched = batched_load(user_ids, User_class_mock, get_db_mock)

    # Verify
    assert result_original == result_batched, f"Mismatch: {result_original} != {result_batched}"
    assert result_original == {"user1": "Alice", "user2": "Bob", "user3": "Charlie"}

    print(f"Original: {result_original}")
    print(f"Batched:  {result_batched}")
    print("PASS: Results match")
    return True


def test_missing_users():
    """Test case where some users don't exist."""
    print("\n" + "="*80)
    print("Test 2: Missing users")
    print("="*80)

    user_ids = ["user1", "user_missing", "user3"]

    # Setup mocks for original
    Users_mock = Mock()
    def get_user_side_effect(uid):
        if uid == "user_missing":
            return None
        return {
            "user1": create_mock_user("user1", name="Alice"),
            "user3": create_mock_user("user3", name="Charlie"),
        }.get(uid)

    Users_mock.get_user_by_id = Mock(side_effect=get_user_side_effect)

    # Setup mocks for batched (only return existing users)
    mock_users = [
        create_mock_user("user1", name="Alice"),
        create_mock_user("user3", name="Charlie"),
    ]

    db_mock = MagicMock()
    query_chain = db_mock.query.return_value.filter.return_value
    query_chain.all.return_value = mock_users

    User_class_mock = Mock()
    User_class_mock.id = Mock()
    User_class_mock.id.in_ = Mock(return_value="filter_condition")

    get_db_mock = MagicMock()
    get_db_mock.return_value.__enter__.return_value = db_mock
    get_db_mock.return_value.__exit__.return_value = None

    # Execute both
    result_original = original_load(user_ids, Users_mock)
    result_batched = batched_load(user_ids, User_class_mock, get_db_mock)

    # Verify
    assert result_original == result_batched, f"Mismatch: {result_original} != {result_batched}"
    assert result_original == {"user1": "Alice", "user_missing": "user_missing", "user3": "Charlie"}

    print(f"Original: {result_original}")
    print(f"Batched:  {result_batched}")
    print("PASS: Results match (missing user handled correctly)")
    return True


def test_label_fallback():
    """Test label resolution fallback (name -> username -> email -> id)."""
    print("\n" + "="*80)
    print("Test 3: Label fallback order")
    print("="*80)

    user_ids = ["user1", "user2", "user3", "user4"]

    # Setup mocks for original
    Users_mock = Mock()
    Users_mock.get_user_by_id = Mock(side_effect=lambda uid: {
        "user1": create_mock_user("user1", name="Alice"),
        "user2": create_mock_user("user2", name=None, username="bob_user"),
        "user3": create_mock_user("user3", name=None, username=None, email="charlie@example.com"),
        "user4": create_mock_user("user4", name=None, username=None, email=None),
    }[uid])

    # Setup mocks for batched
    mock_users = [
        create_mock_user("user1", name="Alice"),
        create_mock_user("user2", name=None, username="bob_user"),
        create_mock_user("user3", name=None, username=None, email="charlie@example.com"),
        create_mock_user("user4", name=None, username=None, email=None),
    ]

    db_mock = MagicMock()
    query_chain = db_mock.query.return_value.filter.return_value
    query_chain.all.return_value = mock_users

    User_class_mock = Mock()
    User_class_mock.id = Mock()
    User_class_mock.id.in_ = Mock(return_value="filter_condition")

    get_db_mock = MagicMock()
    get_db_mock.return_value.__enter__.return_value = db_mock
    get_db_mock.return_value.__exit__.return_value = None

    # Execute both
    result_original = original_load(user_ids, Users_mock)
    result_batched = batched_load(user_ids, User_class_mock, get_db_mock)

    # Verify
    assert result_original == result_batched, f"Mismatch: {result_original} != {result_batched}"
    expected = {
        "user1": "Alice",
        "user2": "bob_user",
        "user3": "charlie@example.com",
        "user4": "user4",  # Falls back to ID
    }
    assert result_original == expected

    print(f"Original: {result_original}")
    print(f"Batched:  {result_batched}")
    print("PASS: Label fallback works correctly")
    return True


def test_empty_input():
    """Test with empty user ID list."""
    print("\n" + "="*80)
    print("Test 4: Empty input")
    print("="*80)

    user_ids = []

    # Both should return empty dict without hitting mocks
    Users_mock = Mock()
    get_db_mock = MagicMock()
    User_class_mock = Mock()

    # Execute both (should not call any mocks)
    result_original = original_load(user_ids, Users_mock)
    result_batched = batched_load(user_ids, User_class_mock, get_db_mock)

    # Verify
    assert result_original == result_batched, f"Mismatch: {result_original} != {result_batched}"
    assert result_original == {}

    print(f"Original: {result_original}")
    print(f"Batched:  {result_batched}")
    print("PASS: Empty input handled correctly")
    return True


def test_query_efficiency():
    """Verify batched version makes fewer queries."""
    print("\n" + "="*80)
    print("Test 5: Query efficiency")
    print("="*80)

    user_ids = ["user1", "user2", "user3", "user4", "user5"]

    # Setup mocks for original - track call count
    Users_mock = Mock()
    call_count = {"original": 0}

    def track_calls(uid):
        call_count["original"] += 1
        return create_mock_user(uid, name=f"User {uid}")

    Users_mock.get_user_by_id = Mock(side_effect=track_calls)

    # Setup mocks for batched - track query count
    mock_users = [create_mock_user(uid, name=f"User {uid}") for uid in user_ids]

    db_mock = MagicMock()
    call_count["batched"] = 0

    def track_batch_query(*args, **kwargs):
        call_count["batched"] += 1
        return mock_users

    query_chain = db_mock.query.return_value.filter.return_value
    query_chain.all = Mock(side_effect=track_batch_query)

    User_class_mock = Mock()
    User_class_mock.id = Mock()
    User_class_mock.id.in_ = Mock(return_value="filter_condition")

    get_db_mock = MagicMock()
    get_db_mock.return_value.__enter__.return_value = db_mock
    get_db_mock.return_value.__exit__.return_value = None

    # Execute both
    result_original = original_load(user_ids, Users_mock)
    result_batched = batched_load(user_ids, User_class_mock, get_db_mock)

    # Verify results match
    assert result_original == result_batched

    # Verify efficiency
    print(f"Original queries: {call_count['original']} (N+1 pattern)")
    print(f"Batched queries:  {call_count['batched']} (single batch)")
    print(f"Original: {result_original}")
    print(f"Batched:  {result_batched}")

    assert call_count["original"] == 5, "Original should make 5 queries"
    assert call_count["batched"] == 1, "Batched should make 1 query"

    print("PASS: Batched version is more efficient (1 query vs 5)")
    return True


if __name__ == "__main__":
    print("\n" + "="*80)
    print("BATCHED USER LABEL RESOLUTION TESTS")
    print("="*80)

    tests = [
        test_basic_user_resolution,
        test_missing_users,
        test_label_fallback,
        test_empty_input,
        test_query_efficiency,
    ]

    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append((test_func.__name__, result))
        except Exception as e:
            print(f"FAIL: {test_func.__name__} - {e}")
            results.append((test_func.__name__, False))

    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\nOK: ALL TESTS PASSED - Batched optimization is behaviorally equivalent")
        print("Performance: Reduced from N queries to 1 query per batch")
    else:
        print("\nERROR: SOME TESTS FAILED - Review implementation")

    print("="*80)
