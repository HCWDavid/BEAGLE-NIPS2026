"""
Pytest configuration for all problem tests.

This file is automatically loaded by pytest and sets up the import path
so that test files can import from solution.py in the tmp directory
where student code is saved during testing.

IMPORTANT: Teacher reference solutions should be named 'reference_solution.py',
NOT 'solution.py', to avoid conflicts with student code imports.
"""

import sys
from pathlib import Path

# Add tmp directory to path so all tests can import the student's solution.py
# tmp is at beagle/data_generation/tmp (one level up from problems/)
tmp_dir = Path(__file__).parent.parent / "tmp"
tmp_dir = tmp_dir.resolve()  # Convert to absolute path
if str(tmp_dir) not in sys.path:
    sys.path.insert(0, str(tmp_dir))
