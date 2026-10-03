"""
IDE Oracle - Code Testing and Execution Environment

Simulates an IDE environment that:
1. Saves student code to tmp directory
2. Runs pytest tests against the code
3. Reports test results and errors

Components:
- code_executor.py: Low-level code execution (legacy, for simple tests)
- ide_oracle.py: High-level pytest runner and result parser
"""

from beagle.data_generation.ide_oracle.ide_oracle import IDEOracle, TestResult

__all__ = ['IDEOracle', 'TestResult']
