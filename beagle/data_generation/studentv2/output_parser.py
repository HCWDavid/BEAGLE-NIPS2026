"""
Output Parser Module for BEAGLE Simulation

This module provides deterministic parsing of pytest/python execution output
to create a "Truth Anchor" for the LLM. It distinguishes between:

1. CRASHES (Syntax/Runtime errors) - Code didn't run at all
2. FAILURES (Test failures including AttributeError) - Tests ran but failed
3. SUCCESS - All tests passed

The parser extracts a SINGLE focused error to prevent cognitive overload
and creates clear semantic signals for the LLM.

CRITICAL: The display_log must show ONLY ONE error to prevent the LLM
from picking the wrong error and causing desync issues.
"""

import re
from dataclasses import dataclass


@dataclass
class ParsedOutput:
    """Structured representation of execution output for LLM consumption."""
    status_header: str      # e.g., "SUCCESS", "CRASH", "TEST FAILED"
    summary_line: str       # e.g., "21 passed, 3 failed"
    primary_error: str      # The SINGLE error detail to focus on
    display_log: str        # The sanitized log (SINGLE focused error only!)
    n_passed: int = 0
    n_failed: int = 0
    error_type: str = ""    # "crash", "failure", "success", "draft"


def parse_execution_output(raw_output: str) -> ParsedOutput:
    """
    Parses pytest/python output into a structured semantic signal for the LLM.
    Distinguishes between CRASHES (Syntax/Runtime) and FAILURES (Logic/Missing methods).
    
    CRITICAL: Always returns ONLY ONE focused error to prevent desync.
    
    Args:
        raw_output: Raw pytest or python execution output
        
    Returns:
        ParsedOutput with clear status header, summary, and SINGLE focused error
    """
    if not raw_output or raw_output == "(Code drafted but not executed)":
        return ParsedOutput(
            status_header="STATUS: DRAFTING MODE",
            summary_line="Code drafted. Ready to execute.",
            primary_error="",
            display_log="(No execution output yet)",
            error_type="draft"
        )

    # Check if this is pytest output (has "passed" or "failed" counts)
    # If so, use test failure logic even for AttributeError/TypeError
    is_pytest_output = bool(re.search(r"(\d+) (passed|failed)", raw_output))
    
    # 1. Check for CRITICAL CRASHES (Code didn't run at all)
    # These are TRUE crashes - not test failures with errors
    # Only apply if NOT a pytest test result
    if not is_pytest_output:
        crash_patterns = [
            (r"IndentationError: ([^\n]+)", "CRASH: INDENTATION ERROR"),
            (r"SyntaxError: ([^\n]+)", "CRASH: SYNTAX ERROR"),
            (r"NameError: ([^\n]+)", "CRASH: NAME ERROR (Typo or missing var?)"),
            (r"TypeError: ([^\n]+)", "CRASH: TYPE ERROR"),
            (r"AttributeError: ([^\n]+)", "CRASH: ATTRIBUTE ERROR (Wrong method/var name?)"),
            (r"ZeroDivisionError: ([^\n]+)", "CRASH: ZERO DIVISION ERROR"),
            (r"ImportError: ([^\n]+)", "CRASH: IMPORT ERROR"),
            (r"ModuleNotFoundError: ([^\n]+)", "CRASH: MODULE NOT FOUND"),
        ]
        
        for pattern, label in crash_patterns:
            match = re.search(pattern, raw_output)
            if match:
                detail = match.group(0).strip()
                
                return ParsedOutput(
                    status_header=label,
                    summary_line=f"The code crashed: {detail}",
                    primary_error=detail,
                    display_log=(
                        f"===============================================================\n"
                        f"{label}\n"
                        f"===============================================================\n\n"
                        f"Error: {detail}\n\n"
                        f"(Fix this error first. Tests could not run.)"
                    ),
                    error_type="crash"
                )

    # 2. Check for TEST RESULTS (Pytest)
    passed_match = re.search(r"(\d+) passed", raw_output)
    failed_match = re.search(r"(\d+) failed", raw_output)
    
    n_passed = int(passed_match.group(1)) if passed_match else 0
    n_failed = int(failed_match.group(1)) if failed_match else 0
    total = n_passed + n_failed

    # SUCCESS CASE - All tests passed
    if n_failed == 0 and n_passed > 0:
        return ParsedOutput(
            status_header="*** SUCCESS: ALL TESTS PASSED ***",
            summary_line=f"All {total} tests PASSED. The code is correct!",
            primary_error="None",
            display_log=(
                f"===============================================================\n"
                f"*** SUCCESS: ALL {n_passed}/{total} TESTS PASSED ***\n"
                f"===============================================================\n"
                f"The code is correct. You solved the problem!"
            ),
            n_passed=n_passed,
            n_failed=0,
            error_type="success"
        )
    
    # FAILURE CASE (Test failures) - Some tests failed
    if n_failed > 0:
        # Extract the FIRST error only - this is CRITICAL to prevent desync
        first_error = _extract_first_error(raw_output)
        
        # V34.4: Extract SPECIFIC error type from error message
        # This is critical for the truth anchor to show correct type
        extracted_error_type = "failure"  # default
        error_type_patterns = [
            (r'TypeError', 'typeerror'),
            (r'AttributeError', 'attributeerror'),
            (r'NameError', 'nameerror'),
            (r'SyntaxError', 'syntaxerror'),
            (r'IndentationError', 'indentationerror'),
            (r'ImportError', 'importerror'),
            (r'ValueError', 'valueerror'),
            (r'KeyError', 'keyerror'),
            (r'ZeroDivisionError', 'zerodivisionerror'),
            (r'AssertionError', 'assertionerror'),
        ]
        for pattern, error_type_name in error_type_patterns:
            if re.search(pattern, first_error, re.IGNORECASE):
                extracted_error_type = error_type_name
                break
        
        return ParsedOutput(
            status_header=f"*** TEST FAILURE: {n_failed} TESTS FAILED ***",
            summary_line=f"Results: {n_passed} passed, {n_failed} failed.",
            primary_error=first_error,
            display_log=(
                f"===============================================================\n"
                f"TEST SUMMARY: {n_passed} passed, {n_failed} failed.\n"
                f"===============================================================\n\n"
                f"FOCUS ON THIS ERROR (fix this one first):\n"
                f"--------------------------------------------------\n"
                f"{first_error}\n"
                f"--------------------------------------------------\n"
                f"(Other errors are hidden. Fix this error first, then run again.)"
            ),
            n_passed=n_passed,
            n_failed=n_failed,
            error_type=extracted_error_type  # V34.4: Use specific error type
        )

    # Fallback - Generic Output (no clear test results)
    return ParsedOutput(
        status_header="STATUS: EXECUTION COMPLETE",
        summary_line="Check output below.",
        primary_error="",
        display_log=raw_output[:1000] if len(raw_output) > 1000 else raw_output,
        error_type="unknown"
    )


def _extract_first_error(raw_output: str) -> str:
    """
    Extract the FIRST error from pytest output.
    
    CRITICAL: Must return ONLY ONE error to prevent desync where LLM
    picks the wrong error from a list.
    
    Priority:
    1. First IndentationError/SyntaxError (blocking)
    2. First NameError/TypeError/AttributeError (runtime)
    3. First AssertionError (logic)
    """
    # Priority 1: Syntax errors (first match only)
    syntax_patterns = [
        r"(IndentationError: [^\n]+)",
        r"(SyntaxError: [^\n]+)",
    ]
    for pattern in syntax_patterns:
        match = re.search(pattern, raw_output)
        if match:
            return match.group(1)
    
    # Priority 2: Runtime errors (first match only)
    runtime_patterns = [
        r"(NameError: [^\n]+)",
        r"(TypeError: [^\n]+)",
        r"(AttributeError: [^\n]+)",
        r"(KeyError: [^\n]+)",
        r"(ValueError: [^\n]+)",
        r"(ZeroDivisionError: [^\n]+)",
    ]
    for pattern in runtime_patterns:
        match = re.search(pattern, raw_output)
        if match:
            # Get the test name if available
            error_text = match.group(1)
            # Look for the FAILED line just before this error
            failed_match = re.search(r"FAILED ([^\s]+)", raw_output[:match.start()])
            if failed_match:
                test_name = failed_match.group(1).split("::")[-1]  # Get just test function name
                return f"{test_name}: {error_text}"
            return error_text
    
    # Priority 3: Assertion errors (first match only)
    assert_match = re.search(r"(AssertionError: [^\n]+)", raw_output)
    if assert_match:
        return assert_match.group(1)
    
    # Priority 4: Any FAILED line
    failed_match = re.search(r"FAILED ([^\n]+)", raw_output)
    if failed_match:
        return f"Test failed: {failed_match.group(1)}"
    
    return "(Could not extract specific error - check output)"


# Focused error display applies uniformly to ALL students (both low and high performers)
# to prevent cognitive overload and avoid adding confounding variables between levels.
