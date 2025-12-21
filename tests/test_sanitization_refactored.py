"""
Regression tests for the refactored `sanitize_unicode` helper.
"""

from collections import Counter
from typing import List, Tuple, Union


def sanitize_unicode(
    value: str,
    *,
    track_counts: bool = False,
    replace_char: str = "\ufffd",
) -> Union[str, Tuple[str, Counter, bool]]:
    """Remove null bytes and lone surrogates from Unicode strings."""
    if not value:
        return (value, Counter(), False) if track_counts else value

    counts: Counter = Counter()
    builder: List[str] = []
    changed = False
    i = 0
    length = len(value)

    while i < length:
        ch = value[i]
        code = ord(ch)

        if ch == "\x00":
            if track_counts:
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
            if track_counts:
                counts["lone_high"] += 1
            builder.append(replace_char)
            changed = True
            i += 1
            continue

        if 0xDC00 <= code <= 0xDFFF:
            if track_counts:
                counts["lone_low"] += 1
            builder.append(replace_char)
            changed = True
            i += 1
            continue

        builder.append(ch)
        i += 1

    sanitized = "".join(builder)

    if track_counts:
        if changed:
            counts["strings_touched"] += 1
            return sanitized, counts, True
        return value, counts, False

    return sanitized


TEST_CASES = [
    ("", "", False, []),
    ("hello world", "hello world", False, []),
    ("hello\x00world", "helloworld", True, ["null_bytes", "strings_touched"]),
    ("multiple\x00null\x00bytes", "multiplenullbytes", True, ["null_bytes", "strings_touched"]),
    ("hi\ud800there", "hi\ufffdthere", True, ["lone_high", "strings_touched"]),
    ("start\udbff", "start\ufffd", True, ["lone_high", "strings_touched"]),
    ("hi\udc00there", "hi\ufffdthere", True, ["lone_low", "strings_touched"]),
    ("end\udfff", "end\ufffd", True, ["lone_low", "strings_touched"]),
    ("emoji\ud83d\ude00end", "emoji\ud83d\ude00end", False, []),
    ("\x00\ud800\udc00normal", "\ud800\udc00normal", True, ["null_bytes", "strings_touched"]),
    ("test\x00null\ud800lone", "testnull\ufffdlone", True, ["null_bytes", "lone_high", "strings_touched"]),
    (None, None, False, []),
]


def test_sanitize_unicode_with_tracking():
    for input_val, expected_output, expected_changed, expected_count_keys in TEST_CASES:
        if input_val is None:
            continue
        output, counts, changed = sanitize_unicode(input_val, track_counts=True)
        assert output == expected_output
        assert changed == expected_changed
        assert set(counts.keys()) == set(expected_count_keys)


def test_sanitize_unicode_without_tracking():
    for input_val, expected_output, _, _ in TEST_CASES:
        if input_val is None:
            expected_output = ""
        output = sanitize_unicode(input_val or "", track_counts=False)
        assert output == expected_output
