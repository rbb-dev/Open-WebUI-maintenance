"""
Legacy regression tests for string sanitization refactoring.

These validate the original behavior we wanted to preserve while refactoring.
"""

from collections import Counter
from typing import List


TEST_CASES = [
    # (input, expected_output, expected_changed, expected_counts_keys)
    ("", "", False, []),
    ("hello world", "hello world", False, []),
    ("hello\x00world", "helloworld", True, ["null_bytes", "strings_touched"]),
    ("multiple\x00null\x00bytes", "multiplenullbytes", True, ["null_bytes", "strings_touched"]),
    # High surrogate alone (0xD800-0xDBFF)
    ("hi\ud800there", "hi\ufffdthere", True, ["lone_high", "strings_touched"]),
    ("start\udbff", "start\ufffd", True, ["lone_high", "strings_touched"]),
    # Low surrogate alone (0xDC00-0xDFFF)
    ("hi\udc00there", "hi\ufffdthere", True, ["lone_low", "strings_touched"]),
    ("end\udfff", "end\ufffd", True, ["lone_low", "strings_touched"]),
    # Valid surrogate pair (should NOT be replaced)
    ("emoji\ud83d\ude00end", "emoji\ud83d\ude00end", False, []),
    # Mixed issues (note: \ud800\udc00 is a VALID surrogate pair)
    ("\x00\ud800\udc00normal", "\ud800\udc00normal", True, ["null_bytes", "strings_touched"]),
    ("test\x00null\ud800lone", "testnull\ufffdlone", True, ["null_bytes", "lone_high", "strings_touched"]),
    # Edge case: None (handled differently by each method)
    (None, None, False, []),
]


def _sanitize_string_original(value: str):
    """Original implementation from ChatRepairService (pre-refactor)."""
    if not value:
        return value, Counter(), False

    counts: Counter = Counter()
    builder: List[str] = []
    changed = False
    i = 0
    length = len(value)

    while i < length:
        ch = value[i]
        code = ord(ch)

        if ch == "\x00":
            counts["null_bytes"] += 1
            changed = True
            i += 1
            continue

        if 0xD800 <= code <= 0xDBFF:
            if i + 1 < length:
                next_code = ord(value[i + 1])
                if 0xDC00 <= next_code <= 0xDFFF:
                    builder.append(ch)
                    builder.append(value[i + 1])
                    i += 2
                    continue
            counts["lone_high"] += 1
            builder.append("\ufffd")
            changed = True
            i += 1
            continue

        if 0xDC00 <= code <= 0xDFFF:
            counts["lone_low"] += 1
            builder.append("\ufffd")
            changed = True
            i += 1
            continue

        builder.append(ch)
        i += 1

    sanitized = "".join(builder)
    if changed:
        counts["strings_touched"] += 1
        return sanitized, counts, True
    return value, counts, False


def _sanitize_output_text_original(text):
    """Original implementation from Pipe._sanitize_output_text (pre-refactor)."""
    if text is None:
        return ""
    value = str(text)
    if not value:
        return ""
    builder: List[str] = []
    i = 0
    length = len(value)
    while i < length:
        ch = value[i]
        code = ord(ch)
        if ch == "\x00":
            i += 1
            continue
        if 0xD800 <= code <= 0xDBFF:
            if i + 1 < length:
                next_code = ord(value[i + 1])
                if 0xDC00 <= next_code <= 0xDFFF:
                    builder.append(ch)
                    builder.append(value[i + 1])
                    i += 2
                    continue
            builder.append("\ufffd")
            i += 1
            continue
        if 0xDC00 <= code <= 0xDFFF:
            builder.append("\ufffd")
            i += 1
            continue
        builder.append(ch)
        i += 1
    return "".join(builder)


def test_sanitize_string_original_cases():
    for input_val, expected_output, expected_changed, expected_count_keys in TEST_CASES:
        if input_val is None:
            continue
        output, counts, changed = _sanitize_string_original(input_val)
        assert output == expected_output
        assert changed == expected_changed
        assert set(counts.keys()) == set(expected_count_keys)


def test_sanitize_output_text_original_cases():
    for input_val, expected_output, _, _ in TEST_CASES:
        if input_val is None:
            expected_output = ""
        output = _sanitize_output_text_original(input_val)
        assert output == expected_output
