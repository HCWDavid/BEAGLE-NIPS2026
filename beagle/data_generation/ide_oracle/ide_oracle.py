"""
IDE Oracle - Simulates an IDE testing environment

Instead of using JSON test cases, this oracle:
1. Saves student code to beagle/data_generation/tmp/solution.py
2. Runs actual pytest tests from the problem directory
3. Parses pytest output to determine pass/fail and error messages

This is more realistic than the old JSON-based approach.
"""

import os
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import json
import re

# Hard ceiling on pytest invocations. Reference solutions across all problems
# run in ~0.33s; 5s gives 15x headroom while catching infinite loops in student
# code (e.g., `while True:` in update()) before they freeze the batch. Verified
# against the paper's 150 particle_simulator runs — none of them ever exceeded
# the previous 30s timeout, so tightening to 5s does not alter their behaviour.
_PYTEST_TIMEOUT_SECONDS = 5

if TYPE_CHECKING:
    from beagle.data_generation.classroom.problem_definition import ProblemDefinition


@dataclass
class TestResult:
    """Result from running pytest tests."""
    passed: bool
    total_tests: int
    passed_tests: int
    failed_tests: int
    test_details: List[Dict]  # Details for each test
    stdout: str  # Full pytest output
    stderr: str  # Error output
    execution_time: float
    captured_output: str = ""  # Student's print() statements


class IDEOracle:
    """
    Simulates an IDE environment that runs pytest tests.
    
    Usage:
        oracle = IDEOracle()
        result = oracle.test_code(
            code="def calculate_range(v, a): ...",
            problem_id="projectile_motion"
        )
    """

    def __init__(
        self,
        tmp_dir: Optional[Path] = None,
        problems_dir: Optional[Path] = None,
        save_history: bool = False
    ):
        """
        Initialize IDE Oracle.
        
        Args:
            tmp_dir: Directory to save student code (default: beagle/data_generation/tmp)
            problems_dir: Directory containing problems (default: beagle/data_generation/problems)
            save_history: If True, saves each attempt to tmp/history/ with timestamp
        """
        if tmp_dir is None:
            # Default: beagle/data_generation/tmp
            self.tmp_dir = Path(__file__).parent.parent / "tmp"
        else:
            self.tmp_dir = Path(tmp_dir)

        if problems_dir is None:
            # Default: beagle/data_generation/problems
            self.problems_dir = Path(__file__).parent.parent / "problems"
        else:
            self.problems_dir = Path(problems_dir)

        self.save_history = save_history
        self.attempt_count = 0

        # Ensure tmp directory exists
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

        if self.save_history:
            (self.tmp_dir / "history").mkdir(exist_ok=True)

    @staticmethod
    def load_problem(
        problem_name: str,
        problems_dir: Optional[Path] = None
    ) -> "ProblemDefinition":
        """
        Load and validate a problem definition from its directory.
        
        Args:
            problem_name: Name of the problem (e.g., "projectile_motion")
            problems_dir: Directory containing problems (default: beagle/data_generation/problems)
            
        Returns:
            ProblemDefinition loaded from description.json
            
        Raises:
            FileNotFoundError: If problem directory or files don't exist
            ValueError: If problem structure is invalid or KC IDs are malformed
        """
        # Import here to avoid circular dependency
        from beagle.data_generation.classroom.problem_definition import ProblemDefinition

        if problems_dir is None:
            problems_dir = Path(__file__).parent.parent / "problems"
        else:
            problems_dir = Path(problems_dir)

        problem_dir = problems_dir / problem_name
        problem_file = problem_dir / "description.json"

        if not problem_dir.exists():
            raise FileNotFoundError(
                f"Problem directory not found: {problem_dir}"
            )

        if not problem_file.exists():
            raise FileNotFoundError(f"Problem file not found: {problem_file}")

        # Validate problem directory structure
        required_files = {
            "description.json", "reference_solution.py",
            f"test_{problem_name}.py"
        }
        # Optional files that are allowed but not required
        optional_files = {
            "student_progression_examples.txt",  # In-context examples for granularity modes
        }
        allowed_files = required_files | optional_files

        actual_files = {f.name
                        for f in problem_dir.iterdir()
                        if f.is_file()}

        # Check for missing required files
        missing_files = required_files - actual_files
        if missing_files:
            raise ValueError(
                f"Problem directory '{problem_name}' is missing required files:\n"
                f"  Missing: {sorted(missing_files)}\n"
                f"  Found: {sorted(actual_files)}\n"
                f"  Required: {sorted(required_files)}"
            )

        # Check for extra files (excluding optional ones)
        extra_files = actual_files - allowed_files
        if extra_files:
            raise ValueError(
                f"Problem directory '{problem_name}' contains extra files:\n"
                f"  Extra: {sorted(extra_files)}\n"
                f"  Allowed: {sorted(allowed_files)}\n"
                f"  Please remove extra files or add them to .gitignore"
            )

        # Load JSON
        with open(problem_file, 'r') as f:
            data = json.load(f)

        # Validate required_kcs
        if "required_kcs" not in data:
            raise ValueError(
                f"Problem JSON missing 'required_kcs' field: {problem_file}"
            )

        required_kcs = data["required_kcs"]
        if not required_kcs or not isinstance(required_kcs, list):
            raise ValueError(
                f"'required_kcs' must be a non-empty list: {problem_file}"
            )

        # Validate KC ID format (KC_C1_..., KC_P5_..., KC_M1_...)
        # C = Coding, P = Physics, M = Math
        valid_kc_pattern = re.compile(r'^KC_[CPM]\d+_[A-Z_]+$')
        invalid_kcs = [
            kc for kc in required_kcs if not valid_kc_pattern.match(kc)
        ]

        if invalid_kcs:
            raise ValueError(
                f"Invalid KC IDs in {problem_file}:\n"
                f"  Invalid: {invalid_kcs}\n"
                f"  Expected format: KC_C#_..., KC_P#_..., or KC_M#_...\n"
                f"  Examples: KC_C2_MATH_LIBRARY, KC_P5_UNIT_RADIANS, KC_M1_DERIVATIVE_CONCEPT\n"
                f"  Note: Old format like 'KC_IMPORT_MATH' is not supported."
            )

        # Create ProblemDefinition
        return ProblemDefinition(
            problem_id=data["problem_id"],
            title=data["title"],
            description=data["description"],
            function_name=data["function_name"],
            required_kcs=required_kcs,
            starting_code=data.get("starting_code", "")
        )

    def test_code(self, code: str, problem_id: str) -> TestResult:
        """
        Test student code against problem's pytest tests.
        
        Args:
            code: Python code written by student
            problem_id: Problem identifier (e.g., "projectile_motion")
            
        Returns:
            TestResult with pass/fail and detailed feedback
        """
        self.attempt_count += 1

        # Save code to tmp/solution.py (overwrite)
        solution_path = self.tmp_dir / "solution.py"
        solution_path.write_text(code)

        # Optionally save to history
        if self.save_history:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            history_path = self.tmp_dir / "history" / f"attempt_{self.attempt_count:03d}_{timestamp}.py"
            history_path.write_text(code)

        # Find problem directory
        problem_dir = self.problems_dir / problem_id
        if not problem_dir.exists():
            return TestResult(
                passed=False,
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                test_details=[],
                stdout="",
                stderr=f"Problem directory not found: {problem_dir}",
                execution_time=0.0
            )

        # Find test file
        test_files = list(problem_dir.glob("test_*.py"))
        if not test_files:
            return TestResult(
                passed=False,
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                test_details=[],
                stdout="",
                stderr=f"No test files found in {problem_dir}",
                execution_time=0.0
            )

        test_file = test_files[0]

        # Run pytest with JSON report
        return self._run_pytest(test_file, problem_dir)

    def _run_pytest(self, test_file: Path, problem_dir: Path) -> TestResult:
        """
        Run pytest and parse results.
        
        Args:
            test_file: Path to test file
            problem_dir: Problem directory (for working directory)
            
        Returns:
            TestResult with parsed pytest output
        """
        import time

        start_time = time.time()

        # Run pytest with verbose output and capture details
        # -s flag: don't capture print() output so we can see it
        # --rootdir: set to problems/ so conftest.py is found
        # start_new_session=True puts pytest in its own process group so we can
        # kill the whole group on timeout — important because student code with
        # `while True:` would otherwise survive subprocess.run's SIGKILL of just
        # the immediate child.
        problems_dir = problem_dir.parent  # Go up to problems/ directory
        cmd = [
            sys.executable, "-m", "pytest",
            str(test_file), "-v", "--tb=short", "-vv", "-s",
            f"--rootdir={problems_dir}"
        ]
        proc = subprocess.Popen(
            cmd,
            cwd=str(problem_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=_PYTEST_TIMEOUT_SECONDS)
            return_code = proc.returncode
            timed_out = False
        except subprocess.TimeoutExpired:
            # Kill the entire process group, not just pytest, so any subprocesses
            # the student's code spawned die too.
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                stdout, stderr = "", ""
            return_code = -1
            timed_out = True

        execution_time = time.time() - start_time

        # On timeout, surface a single synthetic failure that downstream parsing
        # and the BKT update can read, instead of an empty result that looks like
        # a passing run.
        if timed_out:
            timeout_msg = (
                f"Test execution timed out after {_PYTEST_TIMEOUT_SECONDS}s — "
                "likely an infinite loop in the student's code."
            )
            stderr = (stderr or "") + "\n" + timeout_msg
            test_details = [{
                "name": "<timeout>",
                "passed": False,
                "error": timeout_msg,
                "error_type": "TimeoutError",
            }]
            captured_output = ""
        else:
            test_details = self._parse_pytest_output(stdout, stderr)
            captured_output = self._extract_captured_output(stdout)

        total_tests = len(test_details)
        passed_tests = sum(1 for t in test_details if t["passed"])
        failed_tests = total_tests - passed_tests

        return TestResult(
            passed=(return_code == 0),
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            test_details=test_details,
            stdout=stdout,
            stderr=stderr,
            execution_time=execution_time,
            captured_output=captured_output
        )

    def _parse_pytest_output(self, stdout: str, stderr: str) -> List[Dict]:
        """
        Parse pytest output to extract test details.
        
        Returns:
            List of test details with name, passed, and error info
        """
        test_details = []

        # Only parse the main test output, not the short summary
        # Split at "short test summary" to avoid duplicates
        main_output = stdout.split("short test summary")[0]

        # Pattern: test_file.py::test_name ... PASSED/FAILED
        # With -s flag, there might be output between test name and result
        pattern = r"(test_\w+\.py)::(test_\w+).*?(PASSED|FAILED)"

        for match in re.finditer(pattern, main_output, re.DOTALL):
            test_file, test_name, status = match.groups()

            test_details.append(
                {
                    "name":
                    test_name,
                    "file":
                    test_file,
                    "passed": (status == "PASSED"),
                    "error":
                    self._extract_error_for_test(stdout, test_name)
                    if status == "FAILED" else None
                }
            )

        return test_details

    def _extract_error_for_test(self, output: str,
                                test_name: str) -> Optional[str]:
        """Extract error message for a specific failed test."""
        # Look for the FAILURES section
        failures_match = re.search(
            r"=+ FAILURES =+(.+?)(?:=+ short test summary|$)", output,
            re.DOTALL
        )
        if not failures_match:
            return None

        failures_section = failures_match.group(1)

        # Find this specific test's failure
        test_pattern = rf"_{test_name}_(.+?)(?=_{2,}|$)"
        test_match = re.search(test_pattern, failures_section, re.DOTALL)

        if test_match:
            error_text = test_match.group(1).strip()
            # Extract just the assertion error or exception
            lines = error_text.split("\n")
            # Get last few meaningful lines
            relevant_lines = [
                l for l in lines if l.strip() and not l.startswith("test_")
            ]
            return "\n".join(relevant_lines[-3:]
                             ) if relevant_lines else error_text[:200]

        return None

    def _extract_captured_output(self, stdout: str) -> str:
        """
        Extract print() statements from student code.
        
        With -s flag, student print() output appears between test name and PASSED/FAILED.
        Example:
            test_file.py::test_name ['some', 'output']
            more output
            PASSED
        """
        captured_lines = []

        # Find all test executions and capture output between test name and result
        # Pattern: after "test_X.py::test_name" until "PASSED" or "FAILED"
        pattern = r"test_\w+\.py::test_\w+\s+(.*?)(?:PASSED|FAILED)"

        for match in re.finditer(pattern, stdout, re.DOTALL):
            output = match.group(1).strip()
            if output:
                # Filter out pytest's own output lines
                lines = output.split('\n')
                student_lines = [
                    line for line in lines if line.strip() and not any(
                        marker in line for marker in [
                            '====', '----', 'platform', 'rootdir', 'plugins',
                            'cachedir', 'collected'
                        ]
                    )
                ]
                if student_lines:
                    captured_lines.extend(student_lines)

        result_str = '\n'.join(captured_lines).strip()
        
        # KEY SAFETY: Truncate output to avoid massive token payloads (e.g., infinite loops)
        if len(result_str) > 10000:
            result_str = result_str[:10000] + "\n... [TRUNCATED DUE TO EXCESSIVE OUTPUT] ..."
            
        return result_str
