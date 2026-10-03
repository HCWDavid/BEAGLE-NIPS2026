"""
Unit tests for Bouncing Ball Simulator.

Tests the BouncingBall class that simulates a ball bouncing under gravity.
These tests are ordered from simple to complex to allow incremental progress.
"""

import pytest


# =============================================================================
# LEVEL 1: Class Definition & Basic Structure (4 tests)
# =============================================================================

def test_01_class_exists():
    """Test that BouncingBall class is defined."""
    from solution import BouncingBall
    assert BouncingBall is not None, "BouncingBall class should be defined"


def test_02_can_instantiate():
    """Test that BouncingBall can be instantiated with basic parameters."""
    from solution import BouncingBall
    b = BouncingBall(10.0, 0)
    assert b is not None, "Should be able to create a BouncingBall instance"


def test_03_has_get_height_method():
    """Test that BouncingBall has get_height method."""
    from solution import BouncingBall
    b = BouncingBall(10.0, 0)
    assert hasattr(b, 'get_height'), "BouncingBall should have get_height method"
    assert callable(b.get_height), "get_height should be callable"


def test_04_has_get_velocity_method():
    """Test that BouncingBall has get_velocity method."""
    from solution import BouncingBall
    b = BouncingBall(10.0, 0)
    assert hasattr(b, 'get_velocity'), "BouncingBall should have get_velocity method"
    assert callable(b.get_velocity), "get_velocity should be callable"


# =============================================================================
# LEVEL 2: Initial State (4 tests)
# =============================================================================

def test_05_initial_height():
    """Test initial height is set correctly."""
    from solution import BouncingBall
    b = BouncingBall(15.0, 0)
    height = b.get_height()
    assert abs(height - 15.0) < 0.01, f"Initial height should be 15.0, got {height}"


def test_06_initial_velocity_zero():
    """Test initial velocity when dropped from rest."""
    from solution import BouncingBall
    b = BouncingBall(10.0, 0)
    vel = b.get_velocity()
    assert abs(vel) < 0.01, f"Initial velocity should be 0, got {vel}"


def test_07_initial_velocity_upward():
    """Test initial upward velocity."""
    from solution import BouncingBall
    b = BouncingBall(5.0, 10.0)
    vel = b.get_velocity()
    assert abs(vel - 10.0) < 0.01, f"Initial velocity should be 10.0, got {vel}"


def test_08_initial_velocity_downward():
    """Test initial downward velocity."""
    from solution import BouncingBall
    b = BouncingBall(20.0, -5.0)
    vel = b.get_velocity()
    assert abs(vel - (-5.0)) < 0.01, f"Initial velocity should be -5.0, got {vel}"


# =============================================================================
# LEVEL 3: Update Method & Gravity (3 tests)
# =============================================================================

def test_09_has_update_method():
    """Test that BouncingBall has update method."""
    from solution import BouncingBall
    b = BouncingBall(10.0, 0)
    assert hasattr(b, 'update'), "BouncingBall should have update method"
    assert callable(b.update), "update should be callable"


def test_10_falls_under_gravity():
    """Test that ball falls when dropped."""
    from solution import BouncingBall
    b = BouncingBall(10.0, 0)
    initial_height = b.get_height()
    b.update(0.1)
    new_height = b.get_height()
    assert new_height < initial_height, "Ball should fall under gravity"


def test_11_velocity_increases_downward():
    """Test that velocity becomes more negative (faster fall) due to gravity."""
    from solution import BouncingBall
    b = BouncingBall(10.0, 0)
    initial_vel = b.get_velocity()
    b.update(0.1)
    new_vel = b.get_velocity()
    assert new_vel < initial_vel, "Velocity should decrease (become more negative) under gravity"


# =============================================================================
# LEVEL 4: Bouncing Physics (4 tests)
# =============================================================================

def test_12_has_bounce_count_method():
    """Test that BouncingBall has get_bounce_count method."""
    from solution import BouncingBall
    b = BouncingBall(10.0, 0)
    assert hasattr(b, 'get_bounce_count'), "BouncingBall should have get_bounce_count method"


def test_13_initial_bounce_count_zero():
    """Test that initial bounce count is zero."""
    from solution import BouncingBall
    b = BouncingBall(10.0, 0)
    count = b.get_bounce_count()
    assert count == 0, f"Initial bounce count should be 0, got {count}"


def test_14_bounce_increments_count():
    """Test that bouncing increments the bounce counter."""
    from solution import BouncingBall
    b = BouncingBall(1.0, 0)  # Low height for quick bounce
    
    # Simulate until bounce occurs
    for _ in range(100):
        b.update(0.05)
        if b.get_bounce_count() > 0:
            break
    
    assert b.get_bounce_count() >= 1, "Bounce count should increase after hitting floor"


def test_15_velocity_reverses_on_bounce():
    """Test that velocity reverses direction after bounce."""
    from solution import BouncingBall
    b = BouncingBall(0.5, 0)  # Low height
    
    # Fall until just before bounce
    for _ in range(20):
        b.update(0.02)
    
    # Velocity should be negative (falling) before bounce
    # After bounce, velocity should be positive (rising)
    vel_before_bounce = b.get_velocity()
    
    # Continue until bounce
    for _ in range(50):
        if b.get_bounce_count() > 0:
            break
        b.update(0.02)
    
    vel_after_bounce = b.get_velocity()
    
    # After bounce, velocity should be positive or ball at rest
    assert vel_after_bounce >= 0, "Velocity should be upward after bounce"


# =============================================================================
# LEVEL 5: Restitution & Energy Loss (3 tests)
# =============================================================================

def test_16_restitution_reduces_velocity():
    """Test that restitution coefficient reduces velocity on bounce."""
    from solution import BouncingBall
    b = BouncingBall(2.0, 0, restitution=0.5)
    
    # Record velocity just before first bounce
    pre_bounce_vel = 0
    for _ in range(100):
        if b.get_height() <= 0.1 and b.get_velocity() < 0:
            pre_bounce_vel = abs(b.get_velocity())
        b.update(0.02)
        if b.get_bounce_count() == 1:
            break
    
    # Velocity after bounce should be less than before
    post_bounce_vel = abs(b.get_velocity())
    
    assert post_bounce_vel < pre_bounce_vel, "Velocity magnitude should decrease after bounce"


def test_17_low_restitution_loses_more_energy():
    """Test that lower restitution loses more energy per bounce."""
    from solution import BouncingBall
    b_high = BouncingBall(5.0, 0, restitution=0.9)
    b_low = BouncingBall(5.0, 0, restitution=0.5)
    
    # Simulate both until after first bounce
    for _ in range(200):
        b_high.update(0.02)
        b_low.update(0.02)
    
    # After several bounces, high restitution should maintain more height/velocity
    assert b_high.get_height() >= b_low.get_height() or abs(b_high.get_velocity()) >= abs(b_low.get_velocity()), \
        "Higher restitution should maintain more energy"


def test_18_multiple_bounces_decrease_height():
    """Test that maximum height decreases with each bounce."""
    from solution import BouncingBall
    b = BouncingBall(5.0, 0, restitution=0.7)
    
    max_heights = [5.0]
    current_max = 0
    last_bounce_count = 0
    
    for _ in range(500):
        b.update(0.02)
        if b.get_height() > current_max:
            current_max = b.get_height()
        if b.get_bounce_count() > last_bounce_count:
            max_heights.append(current_max)
            current_max = 0
            last_bounce_count = b.get_bounce_count()
        if b.get_bounce_count() >= 3:
            break
    
    # Each successive max height should be smaller
    for i in range(1, min(len(max_heights), 3)):
        assert max_heights[i] < max_heights[i-1], f"Max heights should decrease: {max_heights}"


# =============================================================================
# LEVEL 6: Energy Methods (4 tests)
# =============================================================================

def test_19_has_energy_methods():
    """Test that BouncingBall has energy methods."""
    from solution import BouncingBall
    b = BouncingBall(10.0, 0)
    assert hasattr(b, 'get_kinetic_energy'), "Should have get_kinetic_energy"
    assert hasattr(b, 'get_potential_energy'), "Should have get_potential_energy"


def test_20_potential_energy_at_height():
    """Test potential energy calculation: PE = m * g * h."""
    from solution import BouncingBall
    # h = 10, g = 9.8, m = 1, PE = 98
    b = BouncingBall(10.0, 0)
    pe = b.get_potential_energy()
    assert abs(pe - 98.0) < 1.0, f"Expected PE≈98 J, got {pe}"


def test_21_kinetic_energy_calculation():
    """Test kinetic energy calculation: KE = 0.5 * m * v²."""
    from solution import BouncingBall
    # v = 10, m = 1, KE = 50
    b = BouncingBall(0.1, 10.0)
    ke = b.get_kinetic_energy()
    assert abs(ke - 50.0) < 1.0, f"Expected KE≈50 J, got {ke}"


def test_22_energy_converts_during_fall():
    """Test that PE converts to KE during fall (energy conservation)."""
    from solution import BouncingBall
    b = BouncingBall(10.0, 0)
    initial_pe = b.get_potential_energy()
    
    # Fall for a bit
    for _ in range(30):
        b.update(0.02)
    
    # Before hitting ground, KE should have increased from PE
    ke = b.get_kinetic_energy()
    assert ke > 1.0, "KE should increase as ball falls"


# =============================================================================
# LEVEL 7: Rest Detection & Full Simulation (2 tests)
# =============================================================================

def test_23_has_is_at_rest_method():
    """Test that BouncingBall has is_at_rest method."""
    from solution import BouncingBall
    b = BouncingBall(10.0, 0)
    assert hasattr(b, 'is_at_rest'), "Should have is_at_rest method"
    result = b.is_at_rest()
    assert isinstance(result, bool), "is_at_rest should return a boolean"


def test_24_eventually_comes_to_rest():
    """Test that ball eventually comes to rest."""
    from solution import BouncingBall
    b = BouncingBall(2.0, 0, restitution=0.6)
    
    # Simulate for a long time
    for _ in range(2000):
        b.update(0.02)
        if b.is_at_rest():
            break
    
    assert b.is_at_rest(), "Ball should eventually come to rest"
    assert b.get_bounce_count() > 2, "Ball should have bounced several times before resting"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
