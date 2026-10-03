"""
Unit tests for Inclined Plane Slider.

Tests the Slider class that simulates an object sliding down a ramp.
These tests are ordered from simple to complex to allow incremental progress.
"""

import pytest
import math


# =============================================================================
# LEVEL 1: Class Definition & Basic Structure (4 tests)
# =============================================================================

def test_01_class_exists():
    """Test that Slider class is defined."""
    from solution import Slider
    assert Slider is not None, "Slider class should be defined"


def test_02_can_instantiate():
    """Test that Slider can be instantiated with basic parameters."""
    from solution import Slider
    s = Slider(30, 10)
    assert s is not None, "Should be able to create a Slider instance"


def test_03_has_get_position_method():
    """Test that Slider has get_position method."""
    from solution import Slider
    s = Slider(30, 10)
    assert hasattr(s, 'get_position'), "Slider should have get_position method"
    assert callable(s.get_position), "get_position should be callable"


def test_04_has_get_velocity_method():
    """Test that Slider has get_velocity method."""
    from solution import Slider
    s = Slider(30, 10)
    assert hasattr(s, 'get_velocity'), "Slider should have get_velocity method"
    assert callable(s.get_velocity), "get_velocity should be callable"


# =============================================================================
# LEVEL 2: Initial State (4 tests)
# =============================================================================

def test_05_starts_at_position_zero():
    """Test that slider starts at position 0 (top of ramp)."""
    from solution import Slider
    s = Slider(45, 20)
    pos = s.get_position()
    assert abs(pos) < 0.01, f"Initial position should be 0, got {pos}"


def test_06_starts_at_rest():
    """Test that slider starts with zero velocity."""
    from solution import Slider
    s = Slider(45, 20)
    vel = s.get_velocity()
    assert abs(vel) < 0.01, f"Initial velocity should be 0, got {vel}"


def test_07_has_get_time_method():
    """Test that Slider has get_time method."""
    from solution import Slider
    s = Slider(30, 10)
    assert hasattr(s, 'get_time'), "Slider should have get_time method"
    time = s.get_time()
    assert abs(time) < 0.01, f"Initial time should be 0, got {time}"


def test_08_has_get_acceleration_method():
    """Test that Slider has get_acceleration method."""
    from solution import Slider
    s = Slider(30, 10)
    assert hasattr(s, 'get_acceleration'), "Slider should have get_acceleration method"
    assert callable(s.get_acceleration), "get_acceleration should be callable"


# =============================================================================
# LEVEL 3: Update Method & Basic Motion (3 tests)
# =============================================================================

def test_09_has_update_method():
    """Test that Slider has update method."""
    from solution import Slider
    s = Slider(45, 10)
    assert hasattr(s, 'update'), "Slider should have update method"
    assert callable(s.update), "update should be callable"


def test_10_update_increases_position():
    """Test that update causes slider to move down ramp."""
    from solution import Slider
    s = Slider(45, 10, friction=0)
    initial_pos = s.get_position()
    s.update(0.1)
    new_pos = s.get_position()
    assert new_pos > initial_pos, "Slider should move down ramp (position increases)"


def test_11_update_increases_velocity():
    """Test that velocity increases during slide."""
    from solution import Slider
    s = Slider(45, 10, friction=0)
    initial_vel = s.get_velocity()
    s.update(0.1)
    new_vel = s.get_velocity()
    assert new_vel > initial_vel, "Velocity should increase while sliding"


# =============================================================================
# LEVEL 4: Gravity & Angle Physics (4 tests)
# =============================================================================

def test_12_steeper_angle_faster_acceleration():
    """Test that steeper angles cause greater acceleration."""
    from solution import Slider
    s_gentle = Slider(20, 10, friction=0)
    s_steep = Slider(60, 10, friction=0)
    
    a_gentle = s_gentle.get_acceleration()
    a_steep = s_steep.get_acceleration()
    
    assert a_steep > a_gentle, "Steeper angle should have greater acceleration"


def test_13_frictionless_45_degree():
    """Test acceleration at 45 degrees without friction."""
    from solution import Slider
    # At 45°, a = g * sin(45) ≈ 9.8 * 0.707 ≈ 6.93
    s = Slider(45, 10, friction=0)
    a = s.get_acceleration()
    expected = 9.8 * math.sin(math.radians(45))
    assert abs(a - expected) < 0.5, f"Expected a≈{expected:.2f}, got {a}"


def test_14_frictionless_30_degree():
    """Test acceleration at 30 degrees without friction."""
    from solution import Slider
    # At 30°, a = g * sin(30) = 9.8 * 0.5 = 4.9
    s = Slider(30, 10, friction=0)
    a = s.get_acceleration()
    expected = 9.8 * 0.5
    assert abs(a - expected) < 0.3, f"Expected a≈{expected:.2f}, got {a}"


def test_15_flat_surface_no_acceleration():
    """Test that a flat surface (0 degrees) has no acceleration."""
    from solution import Slider
    s = Slider(0, 10, friction=0)
    a = s.get_acceleration()
    assert abs(a) < 0.01, f"Flat surface should have zero acceleration, got {a}"


# =============================================================================
# LEVEL 5: Friction Physics (4 tests)
# =============================================================================

def test_16_friction_reduces_acceleration():
    """Test that friction reduces acceleration."""
    from solution import Slider
    s_smooth = Slider(45, 10, friction=0)
    s_rough = Slider(45, 10, friction=0.3)
    
    a_smooth = s_smooth.get_acceleration()
    a_rough = s_rough.get_acceleration()
    
    assert a_rough < a_smooth, "Friction should reduce acceleration"


def test_17_friction_acceleration_formula():
    """Test that acceleration follows: a = g(sin θ - μ cos θ)."""
    from solution import Slider
    # At 45°, friction=0.2: a = 9.8 * (sin(45) - 0.2*cos(45))
    s = Slider(45, 10, friction=0.2)
    a = s.get_acceleration()
    sin_45 = math.sin(math.radians(45))
    cos_45 = math.cos(math.radians(45))
    expected = 9.8 * (sin_45 - 0.2 * cos_45)
    assert abs(a - expected) < 0.3, f"Expected a≈{expected:.2f}, got {a}"


def test_18_high_friction_prevents_sliding():
    """Test that high friction on gentle slope prevents sliding."""
    from solution import Slider
    # At 20° with friction=0.5, slider shouldn't move
    # tan(20°) ≈ 0.36 < 0.5, so friction > gravity component
    s = Slider(20, 10, friction=0.5)
    
    for _ in range(10):
        s.update(0.1)
    
    pos = s.get_position()
    assert pos < 0.1, f"High friction should prevent sliding, got position {pos}"


def test_19_critical_angle():
    """Test behavior near critical friction angle."""
    from solution import Slider
    # At 30°, tan(30°) ≈ 0.577
    # With friction = 0.5 < 0.577, should still slide
    s = Slider(30, 10, friction=0.5)
    a = s.get_acceleration()
    assert a > 0, "Should still slide when friction < tan(angle)"


# =============================================================================
# LEVEL 6: Bottom Detection (3 tests)
# =============================================================================

def test_20_has_has_reached_bottom_method():
    """Test that Slider has has_reached_bottom method."""
    from solution import Slider
    s = Slider(45, 10)
    assert hasattr(s, 'has_reached_bottom'), "Slider should have has_reached_bottom method"


def test_21_not_at_bottom_initially():
    """Test that slider has not reached bottom initially."""
    from solution import Slider
    s = Slider(45, 10)
    assert not s.has_reached_bottom(), "Should not be at bottom initially"


def test_22_detects_reaching_bottom():
    """Test that reaching bottom is detected."""
    from solution import Slider
    s = Slider(45, 5, friction=0)  # Short ramp
    
    for _ in range(100):
        s.update(0.1)
        if s.has_reached_bottom():
            break
    
    assert s.has_reached_bottom(), "Should detect reaching bottom"
    assert s.get_position() >= 5.0, f"Position should be at least ramp length"


# =============================================================================
# LEVEL 7: Full Simulation Accuracy (2 tests)
# =============================================================================

def test_23_kinematics_accuracy():
    """Test position/velocity after known time with known acceleration."""
    from solution import Slider
    # Frictionless 30° ramp: a = g*sin(30) = 4.9 m/s²
    # After t=2s from rest: v = a*t = 9.8, x = 0.5*a*t² = 9.8
    s = Slider(30, 50, friction=0)
    
    for _ in range(200):
        s.update(0.01)
    
    # After 2 seconds
    vel = s.get_velocity()
    pos = s.get_position()
    
    assert abs(vel - 9.8) < 0.5, f"Expected v≈9.8 m/s, got {vel}"
    assert abs(pos - 9.8) < 0.5, f"Expected x≈9.8 m, got {pos}"


def test_24_complete_slide_scenario():
    """Test a complete slide down the ramp."""
    from solution import Slider
    # 45° ramp, 10m long, friction=0.1
    s = Slider(45, 10, friction=0.1)
    
    max_iterations = 5000  # Safety: prevent infinite loop if get_time() is broken
    iterations = 0
    while not s.has_reached_bottom():
        s.update(0.01)
        iterations += 1
        if iterations >= max_iterations:
            break  # Hard limit to prevent infinite loop
        if hasattr(s, 'get_time') and callable(s.get_time):
            if s.get_time() > 10:
                break  # Safety timeout if get_time works
    
    assert s.has_reached_bottom(), "Should reach bottom"
    
    # Calculate expected time: x = 0.5*a*t² → t = sqrt(2x/a)
    sin_45 = math.sin(math.radians(45))
    cos_45 = math.cos(math.radians(45))
    a = 9.8 * (sin_45 - 0.1 * cos_45)
    expected_time = math.sqrt(2 * 10 / a)
    
    actual_time = s.get_time()
    assert abs(actual_time - expected_time) < 0.3, f"Expected time≈{expected_time:.2f}s, got {actual_time}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
