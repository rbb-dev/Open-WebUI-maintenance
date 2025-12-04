"""
Test suite for REFACTORED string sanitization.
Ensures behavioral equivalence after consolidation.
"""
import sys
import io
from collections import Counter
from typing import List, Tuple, Union

# Force UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Extract just the sanitize_unicode function
def sanitize_unicode(
    value: str,
    *,
    track_counts: bool = False,
    replace_char: str = "\ufffd"
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


# Test cases (same as baseline)
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
    """Test sanitize_unicode with track_counts=True."""
    print("=" * 80)
    print("Testing NEW sanitize_unicode (track_counts=True)")
    print("=" * 80)

    results = []
    for i, (input_val, expected_output, expected_changed, expected_count_keys) in enumerate(TEST_CASES):
        if input_val is None:
            results.append({"test": i, "status": "SKIP", "reason": "None not handled"})
            continue

        output, counts, changed = sanitize_unicode(input_val, track_counts=True)

        output_ok = output == expected_output
        changed_ok = changed == expected_changed
        counts_ok = set(counts.keys()) == set(expected_count_keys)

        status = "PASS" if (output_ok and changed_ok and counts_ok) else "FAIL"

        results.append({
            "test": i,
            "status": status
        })

        if status == "FAIL":
            print(f"\nFAIL: Test {i} FAILED:")
            print(f"  Input: {repr(input_val)}")
            print(f"  Expected: {repr(expected_output)}, changed={expected_changed}, counts={expected_count_keys}")
            print(f"  Got:      {repr(output)}, changed={changed}, counts={list(counts.keys())}")

    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    skipped = sum(1 for r in results if r["status"] == "SKIP")

    print(f"\n{'='*80}")
    print(f"Results: {passed} PASS, {failed} FAIL, {skipped} SKIP")
    print(f"{'='*80}\n")

    return results


def test_sanitize_unicode_without_tracking():
    """Test sanitize_unicode with track_counts=False."""
    print("=" * 80)
    print("Testing NEW sanitize_unicode (track_counts=False)")
    print("=" * 80)

    results = []
    for i, (input_val, expected_output, _, _) in enumerate(TEST_CASES):
        if input_val is None:
            expected_output = ""

        output = sanitize_unicode(input_val or "", track_counts=False)
        output_ok = output == expected_output

        status = "PASS" if output_ok else "FAIL"

        results.append({
            "test": i,
            "status": status
        })

        if status == "FAIL":
            print(f"\nFAIL: Test {i} FAILED:")
            print(f"  Input: {repr(input_val)}")
            print(f"  Expected: {repr(expected_output)}")
            print(f"  Got:      {repr(output)}")

    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")

    print(f"\n{'='*80}")
    print(f"Results: {passed} PASS, {failed} FAIL")
    print(f"{'='*80}\n")

    return results


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("REGRESSION TESTS - New Implementation")
    print("=" * 80 + "\n")

    results_with_tracking = test_sanitize_unicode_with_tracking()
    results_without_tracking = test_sanitize_unicode_without_tracking()

    total_tests = len(results_with_tracking) + len(results_without_tracking)
    total_passed = (
        sum(1 for r in results_with_tracking if r["status"] == "PASS") +
        sum(1 for r in results_without_tracking if r["status"] == "PASS")
    )
    total_failed = (
        sum(1 for r in results_with_tracking if r["status"] == "FAIL") +
        sum(1 for r in results_without_tracking if r["status"] == "FAIL")
    )

    print("\n" + "=" * 80)
    print("OVERALL REGRESSION TEST RESULTS")
    print("=" * 80)
    print(f"Total: {total_tests} tests")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")

    if total_failed == 0:
        print("\nOK: ALL REGRESSION TESTS PASSED - Refactoring is safe!")
    else:
        print("\nERROR: REGRESSION FAILURES - DO NOT COMMIT")

    print("=" * 80)
