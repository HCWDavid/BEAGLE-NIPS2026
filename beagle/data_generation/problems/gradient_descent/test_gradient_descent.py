"""
Unit tests for Gradient Descent Optimizer.

Tests the f, f_grad, and gradient_descent functions.
These tests are ordered from simple to complex to allow incremental progress.
Based on the Bielefeld Python Programming Study structure.
"""

import pytest
import numpy as np


# =============================================================================
# LEVEL 1: Function f(x) Definition (4 tests)
# =============================================================================

def test_01_f_exists():
    """Test that f function is defined."""
    from solution import f
    assert f is not None, "f function should be defined"
    assert callable(f), "f should be callable"


def test_02_f_returns_number():
    """Test that f returns a numeric value."""
    from solution import f
    result = f(0)
    assert isinstance(result, (int, float, np.number)), "f should return a number"


def test_03_f_at_zero():
    """Test f(0) = 0."""
    from solution import f
    result = f(0)
    assert abs(result - 0) < 0.01, f"f(0) should be 0, got {result}"


def test_04_f_at_one():
    """Test f(1) = 1 - 1 + 0.25 = 0.25."""
    from solution import f
    result = f(1)
    expected = 1**4 - 1**2 + 0.25 * 1  # = 0.25
    assert abs(result - expected) < 0.01, f"f(1) should be {expected}, got {result}"


# =============================================================================
# LEVEL 2: Function f(x) Correctness (4 tests)
# =============================================================================

def test_05_f_at_negative():
    """Test f(-1) = 1 - 1 - 0.25 = -0.25."""
    from solution import f
    result = f(-1)
    expected = (-1)**4 - (-1)**2 + 0.25 * (-1)  # = -0.25
    assert abs(result - expected) < 0.01, f"f(-1) should be {expected}, got {result}"


def test_06_f_at_half():
    """Test f(0.5)."""
    from solution import f
    result = f(0.5)
    expected = 0.5**4 - 0.5**2 + 0.25 * 0.5  # = 0.0625 - 0.25 + 0.125 = -0.0625
    assert abs(result - expected) < 0.01, f"f(0.5) should be {expected}, got {result}"


def test_07_f_uses_power_correctly():
    """Test that f uses x^4 correctly (not x^2 or other)."""
    from solution import f
    # At x=2: 2^4 - 2^2 + 0.5 = 16 - 4 + 0.5 = 12.5
    result = f(2)
    expected = 2**4 - 2**2 + 0.25 * 2  # = 12.5
    assert abs(result - expected) < 0.1, f"f(2) should be {expected}, got {result}"


def test_08_f_handles_array():
    """Test that f can handle numpy arrays (optional but useful)."""
    from solution import f
    try:
        x = np.array([0, 1, -1])
        result = f(x)
        assert len(result) == 3, "f should handle array input"
    except Exception:
        # It's okay if array input is not supported
        pass


# =============================================================================
# LEVEL 3: Gradient Function Definition (4 tests)
# =============================================================================

def test_09_f_grad_exists():
    """Test that f_grad function is defined."""
    from solution import f_grad
    assert f_grad is not None, "f_grad function should be defined"
    assert callable(f_grad), "f_grad should be callable"


def test_10_f_grad_returns_number():
    """Test that f_grad returns a numeric value."""
    from solution import f_grad
    result = f_grad(0)
    assert isinstance(result, (int, float, np.number)), "f_grad should return a number"


def test_11_f_grad_at_zero():
    """Test gradient at x=0. Analytical: f'(x) = 4x^3 - 2x + 0.25, f'(0) = 0.25."""
    from solution import f_grad
    result = f_grad(0)
    expected = 0.25  # 4*0 - 0 + 0.25
    assert abs(result - expected) < 0.01, f"f_grad(0) should be ~{expected}, got {result}"


def test_12_f_grad_at_one():
    """Test gradient at x=1. Analytical: f'(1) = 4 - 2 + 0.25 = 2.25."""
    from solution import f_grad
    result = f_grad(1)
    expected = 2.25  # 4*1 - 2*1 + 0.25
    assert abs(result - expected) < 0.05, f"f_grad(1) should be ~{expected}, got {result}"


# =============================================================================
# LEVEL 4: Gradient Function Correctness (4 tests)
# =============================================================================

def test_13_f_grad_negative():
    """Test gradient at negative x. f'(-1) = -4 + 2 + 0.25 = -1.75."""
    from solution import f_grad
    result = f_grad(-1)
    expected = -1.75
    assert abs(result - expected) < 0.05, f"f_grad(-1) should be ~{expected}, got {result}"


def test_14_f_grad_sign_positive():
    """Test that gradient is positive where function is increasing."""
    from solution import f_grad
    # At x=1, function is increasing, so gradient should be positive
    result = f_grad(1)
    assert result > 0, f"f_grad(1) should be positive, got {result}"


def test_15_f_grad_sign_negative():
    """Test that gradient is negative where function is decreasing."""
    from solution import f_grad
    # At x=-1, function is decreasing (towards minimum), gradient should be negative
    result = f_grad(-1)
    assert result < 0, f"f_grad(-1) should be negative, got {result}"


def test_16_f_grad_uses_numerical_diff():
    """Test that gradient uses reasonable numerical differentiation."""
    from solution import f_grad
    # Test at a few points - should be reasonably accurate
    test_points = [0, 0.5, 1, -0.5]
    for x in test_points:
        result = f_grad(x)
        analytical = 4 * x**3 - 2 * x + 0.25  # True derivative
        assert abs(result - analytical) < 0.1, f"f_grad({x})={result} not close to analytical {analytical}"


# =============================================================================
# LEVEL 5: Gradient Descent Function Definition (4 tests)
# =============================================================================

def test_17_gradient_descent_exists():
    """Test that gradient_descent function is defined."""
    from solution import gradient_descent
    assert gradient_descent is not None, "gradient_descent should be defined"
    assert callable(gradient_descent), "gradient_descent should be callable"


def test_18_gradient_descent_returns_number():
    """Test that gradient_descent returns a numeric value."""
    from solution import gradient_descent, f_grad
    result = gradient_descent(f_grad, 0, 0.01)
    assert isinstance(result, (int, float, np.number)), "gradient_descent should return a number"


def test_19_gradient_descent_accepts_params():
    """Test that gradient_descent accepts f_grad, x0, and eta parameters."""
    from solution import gradient_descent, f_grad
    # Should not raise an error with these parameters
    result = gradient_descent(f_grad, 0.5, 0.01)
    assert result is not None, "gradient_descent should return a result"


def test_20_gradient_descent_changes_x():
    """Test that gradient_descent doesn't just return the initial value."""
    from solution import gradient_descent, f_grad
    x0 = 1.0
    result = gradient_descent(f_grad, x0, 0.01)
    # Starting at x=1 (not a minimum), result should be different
    assert abs(result - x0) > 0.01, "gradient_descent should move from initial point"


# =============================================================================
# LEVEL 6: Gradient Descent Behavior (4 tests)
# =============================================================================

def test_21_gradient_descent_moves_downhill():
    """Test that gradient descent moves toward lower f values."""
    from solution import gradient_descent, f_grad, f
    x0 = 1.0
    result = gradient_descent(f_grad, x0, 0.01)
    # f at result should be less than f at x0
    assert f(result) < f(x0), "gradient_descent should find lower f value"


def test_22_gradient_descent_finds_minimum_region():
    """Test that gradient descent gets close to a minimum."""
    from solution import gradient_descent, f_grad
    x0 = 0.5
    result = gradient_descent(f_grad, x0, 0.01)
    # The global minimum is around x ≈ -0.66
    # Result should be in the vicinity of a minimum
    assert -1.5 < result < 1.5, f"Result {result} seems too far from expected minimum region"


def test_23_gradient_descent_converges():
    """Test that gradient is small at the result (convergence)."""
    from solution import gradient_descent, f_grad
    x0 = 0.5
    result = gradient_descent(f_grad, x0, 0.01)
    final_grad = f_grad(result)
    assert abs(final_grad) < 0.1, f"Gradient at result should be small, got {final_grad}"


def test_24_gradient_descent_eta_effect():
    """Test that different eta values still converge (robustness)."""
    from solution import gradient_descent, f_grad
    result_small_eta = gradient_descent(f_grad, 0.5, 0.001)
    result_large_eta = gradient_descent(f_grad, 0.5, 0.1)
    # Both should converge to similar region (maybe different local minima)
    assert -1.5 < result_small_eta < 1.5, "Should converge with small eta"
    assert -1.5 < result_large_eta < 1.5, "Should converge with large eta"


# =============================================================================
# LEVEL 7: Gradient Descent Accuracy (4 tests)
# =============================================================================

def test_25_gradient_descent_from_positive():
    """Test gradient descent starting from positive x."""
    from solution import gradient_descent, f_grad, f
    result = gradient_descent(f_grad, 1.0, 0.01)
    # Should find a minimum
    assert f(result) < f(1.0), "Should find lower point than starting"


def test_26_gradient_descent_from_negative():
    """Test gradient descent starting from negative x."""
    from solution import gradient_descent, f_grad, f
    result = gradient_descent(f_grad, -1.0, 0.01)
    # Should find a minimum
    assert f(result) < f(-1.0), "Should find lower point than starting"


def test_27_gradient_descent_finds_global_min():
    """Test that gradient descent can find the global minimum."""
    from solution import gradient_descent, f_grad, f
    # Try from x0 = -0.5, which should go to global min around -0.66
    result = gradient_descent(f_grad, -0.5, 0.01)
    # Global minimum is around x ≈ -0.66, f(x) ≈ -0.33
    assert abs(result - (-0.66)) < 0.2, f"Expected global min around -0.66, got {result}"


def test_28_gradient_descent_minimum_value():
    """Test that the function value at result is close to global minimum."""
    from solution import gradient_descent, f_grad, f
    result = gradient_descent(f_grad, -0.5, 0.01)
    f_at_result = f(result)
    # Global minimum f value is approximately -0.33
    assert f_at_result < -0.2, f"Expected f(result) < -0.2, got {f_at_result}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
