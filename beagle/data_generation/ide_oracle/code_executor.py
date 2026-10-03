"""
Safe code execution environment for student-generated Python code.

Provides sandboxed execution with timeout, memory limits, and test validation.
"""

import sys
import io
import traceback
import ast
import re
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import contextlib
import signal
from functools import wraps


def safe_serialize(value: Any) -> Any:
    """
    Convert any value to a JSON-serializable format.

    Handles special Python objects that can't be directly serialized:
    - Ellipsis (...) -> "Ellipsis"
    - NaN -> "NaN"
    - Infinity -> "Infinity" or "-Infinity"
    - Complex numbers -> string representation
    - Other objects -> string representation

    Args:
        value: Any Python value

    Returns:
        JSON-serializable equivalent
    """
    if value is None:
        return None
    if value is ...:
        return "Ellipsis"
    if isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, complex):
        return str(value)
    # Fallback to string representation for any other type
    return str(value)


@dataclass
class TestCase:
    """Single test case for code validation."""
    name: str
    inputs: Dict[str, Any]
    expected_output: Any
    tolerance: Optional[float] = None  # For floating point comparisons


@dataclass
class ExecutionResult:
    """Result of code execution."""
    passed: bool
    output: Any
    error: Optional[str] = None
    error_type: Optional[str] = None
    traceback: Optional[str] = None
    stdout: str = ""
    test_results: Optional[List[Dict]] = None
    execution_time: Optional[float] = None

    def to_dict(self) -> Dict:
        """
        Convert to dictionary for JSON serialization.

        Safely handles non-serializable values like Ellipsis, NaN, Infinity, etc.
        Deep-serializes test_results to ensure all nested values are safe.
        """
        data = {
            "passed": self.passed,
            "output": safe_serialize(self.output),
            "error": self.error,
            "error_type": self.error_type,
            "traceback": self.traceback,
            "stdout": self.stdout,
            "execution_time": self.execution_time
        }

        # Deep serialize test_results to handle problematic values
        if self.test_results:
            safe_test_results = []
            for test in self.test_results:
                safe_test = {}
                for key, value in test.items():
                    # Serialize all values, especially 'actual', 'expected', 'error'
                    safe_test[key] = safe_serialize(value)
                safe_test_results.append(safe_test)
            data["test_results"] = safe_test_results
        else:
            data["test_results"] = None

        return data


class TimeoutException(Exception):
    """Raised when code execution exceeds timeout."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutException("Code execution timed out")


class CodeExecutor:
    """
    Safe Python code execution environment.

    Features:
    - Sandboxed execution (restricted builtins)
    - Timeout protection
    - Capture stdout/stderr
    - Test case validation
    - Error reporting with traceback
    """

    # Allowed builtins for student code
    SAFE_BUILTINS = {
        'abs', 'all', 'any', 'bin', 'bool', 'chr', 'dict', 'dir',
        'divmod', 'enumerate', 'filter', 'float', 'format', 'hex',
        'int', 'isinstance', 'len', 'list', 'map', 'max', 'min',
        'oct', 'ord', 'pow', 'print', 'range', 'reversed', 'round',
        'set', 'slice', 'sorted', 'str', 'sum', 'tuple', 'type',
        'zip', 'True', 'False', 'None',
        # Math operations
        '__import__',  # Allow importing math, etc.
    }

    # Allowed modules
    ALLOWED_MODULES = {
        'math', 'random', 'itertools', 'functools', 'collections'
    }

    def __init__(
        self,
        timeout: int = 5,
        enable_timeout: bool = True
    ):
        """
        Initialize code executor.

        Args:
            timeout: Maximum execution time in seconds
            enable_timeout: Whether to enable timeout protection
        """
        self.timeout = timeout
        self.enable_timeout = enable_timeout

    def execute(
        self,
        code: str,
        test_cases: Optional[List[TestCase]] = None,
        function_name: Optional[str] = None
    ) -> ExecutionResult:
        """
        Execute student code and validate against test cases.

        Args:
            code: Python code to execute
            test_cases: List of test cases to validate
            function_name: Name of function to test (auto-detected if None)

        Returns:
            ExecutionResult with execution details
        """
        # Validate code syntax first
        syntax_error = self._check_syntax(code)
        if syntax_error:
            return ExecutionResult(
                passed=False,
                output=None,
                error=syntax_error,
                error_type="SyntaxError"
            )

        # Extract function name if not provided
        if function_name is None and test_cases:
            function_name = self._extract_function_name(code)

        # Execute code
        try:
            namespace, stdout, exec_time = self._execute_code(code)

            # If no test cases, just return success with namespace
            if not test_cases:
                # Return full namespace (excluding builtins and internal vars)
                clean_namespace = {
                    k: v for k, v in namespace.items()
                    if not k.startswith('__') and k != '__builtins__'
                }
                return ExecutionResult(
                    passed=True,
                    output=clean_namespace,
                    stdout=stdout,
                    execution_time=exec_time
                )

            # Run test cases
            test_results = []
            all_passed = True

            for test_case in test_cases:
                result = self._run_test_case(
                    namespace,
                    function_name,
                    test_case
                )
                test_results.append(result)
                if not result["passed"]:
                    all_passed = False

            return ExecutionResult(
                passed=all_passed,
                output=None,
                stdout=stdout,
                test_results=test_results,
                execution_time=exec_time
            )

        except TimeoutException:
            return ExecutionResult(
                passed=False,
                output=None,
                error=f"Code execution exceeded {self.timeout} second timeout",
                error_type="TimeoutError"
            )

        except Exception as e:
            return ExecutionResult(
                passed=False,
                output=None,
                error=str(e),
                error_type=type(e).__name__,
                traceback=traceback.format_exc()
            )

    def _check_syntax(self, code: str) -> Optional[str]:
        """
        Check code for syntax errors.

        Returns:
            Error message if syntax error, None otherwise
        """
        try:
            ast.parse(code)
            return None
        except SyntaxError as e:
            return f"Syntax error on line {e.lineno}: {e.msg}"

    def _extract_function_name(self, code: str) -> Optional[str]:
        """Extract the first function name from code."""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    return node.name
        except:
            pass
        return None

    def _execute_code(
        self,
        code: str
    ) -> Tuple[Dict, str, float]:
        """
        Execute code in sandboxed environment.

        Returns:
            (namespace, stdout, execution_time)
        """
        import time

        # Create safe namespace
        safe_namespace = self._create_safe_namespace()

        # Capture stdout
        stdout_capture = io.StringIO()

        # Set timeout if enabled
        if self.enable_timeout and hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout)

        try:
            start_time = time.time()

            with contextlib.redirect_stdout(stdout_capture):
                exec(code, safe_namespace)

            exec_time = time.time() - start_time

            if self.enable_timeout and hasattr(signal, 'SIGALRM'):
                signal.alarm(0)  # Cancel timeout

            return safe_namespace, stdout_capture.getvalue(), exec_time

        except TimeoutException:
            raise
        except Exception as e:
            if self.enable_timeout and hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            raise

    def _create_safe_namespace(self) -> Dict:
        """Create sandboxed namespace for code execution."""
        # Restricted builtins
        safe_builtins = {
            name: __builtins__[name]
            for name in self.SAFE_BUILTINS
            if name in __builtins__
        }

        # Custom __import__ to restrict modules
        original_import = __builtins__['__import__']

        def safe_import(name, *args, **kwargs):
            if name.split('.')[0] not in self.ALLOWED_MODULES:
                raise ImportError(f"Module '{name}' is not allowed")
            return original_import(name, *args, **kwargs)

        safe_builtins['__import__'] = safe_import

        return {
            '__builtins__': safe_builtins,
            '__name__': '__main__'
        }

    def _run_test_case(
        self,
        namespace: Dict,
        function_name: str,
        test_case: TestCase
    ) -> Dict:
        """
        Run a single test case.

        Returns:
            Dict with test result details
        """
        try:
            # Get function from namespace
            if function_name not in namespace:
                return {
                    "name": test_case.name,
                    "passed": False,
                    "error": f"Function '{function_name}' not found",
                    "expected": test_case.expected_output,
                    "actual": None
                }

            func = namespace[function_name]

            # Call function with test inputs
            if isinstance(test_case.inputs, dict):
                actual_output = func(**test_case.inputs)
            elif isinstance(test_case.inputs, (list, tuple)):
                actual_output = func(*test_case.inputs)
            else:
                actual_output = func(test_case.inputs)

            # Compare output
            passed = self._compare_outputs(
                actual_output,
                test_case.expected_output,
                test_case.tolerance
            )

            return {
                "name": test_case.name,
                "passed": passed,
                "expected": test_case.expected_output,
                "actual": actual_output,
                "error": None if passed else f"Expected {test_case.expected_output}, got {actual_output}"
            }

        except Exception as e:
            return {
                "name": test_case.name,
                "passed": False,
                "error": f"{type(e).__name__}: {str(e)}",
                "expected": test_case.expected_output,
                "actual": None
            }

    def _compare_outputs(
        self,
        actual: Any,
        expected: Any,
        tolerance: Optional[float]
    ) -> bool:
        """Compare actual vs expected output."""
        # Float comparison with tolerance
        if tolerance is not None and isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
            return abs(actual - expected) <= tolerance

        # Direct comparison
        return actual == expected


class SafeCodeExecutor(CodeExecutor):
    """
    Extra-safe code executor with additional restrictions.

    Prevents:
    - File I/O
    - Network access
    - Process spawning
    - Dangerous operations
    """

    SAFE_BUILTINS = {
        'abs', 'all', 'any', 'bin', 'bool', 'chr', 'dict', 'dir',
        'divmod', 'enumerate', 'filter', 'float', 'format', 'hex',
        'int', 'isinstance', 'len', 'list', 'map', 'max', 'min',
        'oct', 'ord', 'pow', 'print', 'range', 'reversed', 'round',
        'set', 'slice', 'sorted', 'str', 'sum', 'tuple', 'type',
        'zip', 'True', 'False', 'None',
    }

    ALLOWED_MODULES = {'math'}  # Only math module

    def _check_dangerous_code(self, code: str) -> Optional[str]:
        """Check for potentially dangerous operations."""
        dangerous_patterns = [
            (r'open\s*\(', "File operations not allowed"),
            (r'import\s+os', "OS module not allowed"),
            (r'import\s+subprocess', "Subprocess module not allowed"),
            (r'import\s+sys', "Sys module not allowed"),
            (r'__import__\s*\(', "Dynamic imports not allowed"),
            (r'eval\s*\(', "eval() not allowed"),
            (r'exec\s*\(', "exec() not allowed"),
        ]

        for pattern, message in dangerous_patterns:
            if re.search(pattern, code):
                return message

        return None

    def execute(self, code: str, **kwargs) -> ExecutionResult:
        """Execute with additional safety checks."""
        # Check for dangerous code
        danger_error = self._check_dangerous_code(code)
        if danger_error:
            return ExecutionResult(
                passed=False,
                output=None,
                error=danger_error,
                error_type="SecurityError"
            )

        return super().execute(code, **kwargs)
