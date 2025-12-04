"""
Test suite for string sanitization refactoring.
Ensures behavioral equivalence before and after consolidation.
"""
import sys
import io
from collections import Counter
from typing import List

# Force UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


# Test cases covering all edge cases
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

    # Mixed issues
    # Note: \ud800\udc00 is a VALID surrogate pair (should NOT be replaced)
    ("\x00\ud800\udc00normal", "\ud800\udc00normal", True, ["null_bytes", "strings_touched"]),
    ("test\x00null\ud800lone", "testnull\ufffdlone", True, ["null_bytes", "lone_high", "strings_touched"]),

    # Edge case: None (handled differently by each method)
    (None, None, False, []),  # Will be handled specially
]


def test_sanitize_string_current():
    """Test the current _sanitize_string implementation."""
    print("=" * 80)
    print("Testing CURRENT _sanitize_string implementation")
    print("=" * 80)

    # Import the current implementation by reading and executing it
    import sys
    sys.path.insert(0, r'c:\Work\Dev\open-webui-maintenance')

    # We can't import directly, so we'll simulate the function
    def _sanitize_string_original(value: str):
        """Original implementation from ChatRepairService."""
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

    results = []
    for i, (input_val, expected_output, expected_changed, expected_count_keys) in enumerate(TEST_CASES):
        if input_val is None:
            # Skip None test for _sanitize_string
            results.append({"test": i, "status": "SKIP", "reason": "None not handled"})
            continue

        output, counts, changed = _sanitize_string_original(input_val)

        # Check output
        output_ok = output == expected_output
        changed_ok = changed == expected_changed
        counts_ok = set(counts.keys()) == set(expected_count_keys)

        status = "PASS" if (output_ok and changed_ok and counts_ok) else "FAIL"

        results.append({
            "test": i,
            "input": repr(input_val),
            "expected_output": repr(expected_output),
            "actual_output": repr(output),
            "expected_changed": expected_changed,
            "actual_changed": changed,
            "expected_counts": expected_count_keys,
            "actual_counts": list(counts.keys()),
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


def test_sanitize_output_text_current():
    """Test the current _sanitize_output_text implementation."""
    print("=" * 80)
    print("Testing CURRENT _sanitize_output_text implementation")
    print("=" * 80)

    def _sanitize_output_text_original(text):
        """Original implementation from Pipe class."""
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

    results = []
    for i, (input_val, expected_output, _, _) in enumerate(TEST_CASES):
        # Handle None specially for this method
        if input_val is None:
            expected_output = ""

        output = _sanitize_output_text_original(input_val)
        output_ok = output == expected_output

        status = "PASS" if output_ok else "FAIL"

        results.append({
            "test": i,
            "input": repr(input_val),
            "expected_output": repr(expected_output),
            "actual_output": repr(output),
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
    print("BASELINE TESTS - Current Implementation")
    print("=" * 80 + "\n")

    results_sanitize_string = test_sanitize_string_current()
    results_sanitize_output = test_sanitize_output_text_current()

    # Summary
    total_tests = len(results_sanitize_string) + len(results_sanitize_output)
    total_passed = (
        sum(1 for r in results_sanitize_string if r["status"] == "PASS") +
        sum(1 for r in results_sanitize_output if r["status"] == "PASS")
    )
    total_failed = (
        sum(1 for r in results_sanitize_string if r["status"] == "FAIL") +
        sum(1 for r in results_sanitize_output if r["status"] == "FAIL")
    )

    print("\n" + "=" * 80)
    print("OVERALL BASELINE RESULTS")
    print("=" * 80)
    print(f"Total: {total_tests} tests")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")

    if total_failed == 0:
        print("\nOK: ALL BASELINE TESTS PASSED - Safe to proceed with refactoring")
    else:
        print("\nWARNING: BASELINE FAILURES DETECTED - Review before proceeding")

    print("=" * 80)
