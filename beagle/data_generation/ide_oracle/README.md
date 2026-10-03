# IDE Oracle - Code Testing Environment

The IDE Oracle simulates a realistic IDE environment where:
1. Student code is saved to `beagle/data_generation/tmp/solution.py`
2. Actual pytest tests are run from the problem directory
3. Test results are parsed and returned with detailed feedback

## Structure

```
beagle/data_generation/
├── ide_oracle/
│   ├── __init__.py
│   ├── ide_oracle.py          # Main pytest runner
│   └── code_executor.py       # Legacy low-level executor (kept for compatibility)
├── problems/
│   ├── projectile_motion/
│   │   ├── description.json   # Problem metadata and required KCs
│   │   ├── test_projectile_motion.py  # Pytest unit tests
│   │   ├── solution.py                # Teacher's reference solution
│   │   └── README.md
│   └── particle_simulator/
│       └── ...
└── tmp/
    └── solution.py            # Student code saved here for testing
```

## Usage

### Testing Student Code

```python
from beagle.data_generation.ide_oracle import IDEOracle

oracle = IDEOracle()

student_code = """
import math

def calculate_range(initial_velocity, angle):
    theta_rad = math.radians(angle)
    # ... rest of solution
    return range_distance
"""

result = oracle.test_code(student_code, "projectile_motion")

print(f"Passed: {result.passed}")
print(f"Tests: {result.passed_tests}/{result.total_tests}")

for test in result.test_details:
    print(f"  {test['name']}: {'PASS' if test['passed'] else 'FAIL'}")
    if test.get('error'):
        print(f"    Error: {test['error']}")
```

### Test Result Structure

```python
@dataclass
class TestResult:
    passed: bool              # All tests passed?
    total_tests: int          # Number of tests
    passed_tests: int         # Number passing
    failed_tests: int         # Number failing
    test_details: List[Dict]  # Per-test details
    stdout: str               # Full pytest output
    stderr: str               # Error output
    execution_time: float     # Execution time in seconds
```

## Creating New Problems

1. Create problem directory: `beagle/data_generation/problems/my_problem/`

2. Add `description.json`:
```json
{
  "problem_id": "my_problem_01",
  "title": "Problem Title",
  "description": "Problem description...",
  "function_name": "my_function",
  "required_kcs": ["KC_C1", "KC_C2", ...]
}
```

3. Add `test_my_problem.py`:
```python
import pytest

def test_case_1():
    from solution import my_function  # Imports from tmp/solution.py
    result = my_function(input1, input2)
    assert result == expected
```

4. Add `solution.py` with teacher's correct implementation

## Design Benefits

✅ **Realistic**: Uses actual pytest, not JSON test cases
✅ **Flexible**: Supports any pytest features (fixtures, parametrize, etc.)
✅ **Debuggable**: Students see real pytest output
✅ **Extensible**: Easy to add new test cases
✅ **Professional**: Mirrors real-world development workflow
