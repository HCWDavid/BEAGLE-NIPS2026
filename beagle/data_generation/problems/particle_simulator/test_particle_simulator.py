"""
Unit tests for 2D Particle Physics Simulator.

Tests the Particle class that simulates particle motion under gravity and drag.
These tests are ordered from simple to complex to allow incremental progress.
"""

import pytest


# =============================================================================
# LEVEL 1: Class Definition & Basic Structure (4 tests)
# =============================================================================

def test_01_class_exists():
    """Test that Particle class is defined."""
    from solution import Particle
    assert Particle is not None, "Particle class should be defined"


def test_02_can_instantiate():
    """Test that Particle can be instantiated with basic parameters."""
    from solution import Particle
    p = Particle(0, 0, 0, 0, 1.0)
    assert p is not None, "Should be able to create a Particle instance"


def test_03_has_get_position_method():
    """Test that Particle has get_position method."""
    from solution import Particle
    p = Particle(0, 0, 0, 0, 1.0)
    assert hasattr(p, 'get_position'), "Particle should have get_position method"
    assert callable(p.get_position), "get_position should be callable"


def test_04_has_get_velocity_method():
    """Test that Particle has get_velocity method."""
    from solution import Particle
    p = Particle(0, 0, 0, 0, 1.0)
    assert hasattr(p, 'get_velocity'), "Particle should have get_velocity method"
    assert callable(p.get_velocity), "get_velocity should be callable"


# =============================================================================
# LEVEL 2: Initial State (4 tests)
# =============================================================================

def test_05_initial_position_origin():
    """Test initial position at origin."""
    from solution import Particle
    p = Particle(0, 0, 5, 10, 1.0)
    pos = p.get_position()
    assert isinstance(pos, tuple), "get_position should return a tuple"
    assert len(pos) == 2, "Position should have 2 components (x, y)"
    assert abs(pos[0] - 0) < 0.01, f"Initial x should be 0, got {pos[0]}"
    assert abs(pos[1] - 0) < 0.01, f"Initial y should be 0, got {pos[1]}"


def test_06_initial_position_nonzero():
    """Test initial position at non-zero location."""
    from solution import Particle
    p = Particle(10, 20, 0, 0, 1.0)
    pos = p.get_position()
    assert abs(pos[0] - 10) < 0.01, f"Initial x should be 10, got {pos[0]}"
    assert abs(pos[1] - 20) < 0.01, f"Initial y should be 20, got {pos[1]}"


def test_07_initial_velocity_zero():
    """Test initial velocity of zero."""
    from solution import Particle
    p = Particle(5, 5, 0, 0, 2.0)
    vel = p.get_velocity()
    assert isinstance(vel, tuple), "get_velocity should return a tuple"
    assert len(vel) == 2, "Velocity should have 2 components (vx, vy)"
    assert abs(vel[0] - 0) < 0.01, f"Initial vx should be 0, got {vel[0]}"
    assert abs(vel[1] - 0) < 0.01, f"Initial vy should be 0, got {vel[1]}"


def test_08_initial_velocity_nonzero():
    """Test initial velocity with non-zero values."""
    from solution import Particle
    p = Particle(0, 0, 15, -5, 1.5)
    vel = p.get_velocity()
    assert abs(vel[0] - 15) < 0.01, f"Initial vx should be 15, got {vel[0]}"
    assert abs(vel[1] - (-5)) < 0.01, f"Initial vy should be -5, got {vel[1]}"


# =============================================================================
# LEVEL 3: Update Method & Basic Motion (3 tests)
# =============================================================================

def test_09_has_update_method():
    """Test that Particle has update method."""
    from solution import Particle
    p = Particle(0, 0, 0, 0, 1.0)
    assert hasattr(p, 'update'), "Particle should have update method"
    assert callable(p.update), "update should be callable"


def test_10_update_changes_position():
    """Test that update() changes position when particle has velocity."""
    from solution import Particle
    p = Particle(0, 0, 10, 0, 1.0)
    initial_x = p.get_position()[0]
    p.update(0.1)
    new_x = p.get_position()[0]
    assert new_x != initial_x, "Position should change after update when particle has velocity"
    assert new_x > initial_x, "Particle moving right should increase x position"


def test_11_stationary_particle_falls():
    """Test that a stationary particle falls due to gravity."""
    from solution import Particle
    p = Particle(0, 100, 0, 0, 1.0)
    initial_y = p.get_position()[1]
    p.update(0.1)
    new_y = p.get_position()[1]
    assert new_y < initial_y, "Stationary particle should fall (y should decrease due to gravity)"


# =============================================================================
# LEVEL 4: Gravity Physics (3 tests)
# =============================================================================

def test_12_gravity_accelerates_downward():
    """Test that gravity causes downward acceleration (negative vy)."""
    from solution import Particle
    p = Particle(0, 100, 0, 0, 1.0)
    initial_vy = p.get_velocity()[1]
    p.update(0.1)
    new_vy = p.get_velocity()[1]
    assert new_vy < initial_vy, "Velocity should become more negative due to gravity"


def test_13_gravity_value_approximately_correct():
    """Test that gravity acceleration is approximately 9.8 m/s²."""
    from solution import Particle
    # Start at rest, after 0.1s, vy should be approximately -0.98 m/s
    # (ignoring drag for simplicity, but with drag it should be close)
    p = Particle(0, 100, 0, 0, 1.0)
    p.update(0.1)
    vy = p.get_velocity()[1]
    # With g=9.8 and dt=0.1, expected vy ≈ -0.98 (slightly less due to drag)
    assert vy < -0.5, f"After 0.1s, vy should be significantly negative, got {vy}"
    assert vy > -2.0, f"vy shouldn't be too extreme, got {vy}"


def test_14_heavier_particle_same_gravity():
    """Test that heavier particles fall at the same rate (gravity is mass-independent)."""
    from solution import Particle
    p_light = Particle(0, 100, 0, 0, 1.0)
    p_heavy = Particle(0, 100, 0, 0, 10.0)

    p_light.update(0.1)
    p_heavy.update(0.1)

    y_light = p_light.get_position()[1]
    y_heavy = p_heavy.get_position()[1]

    # Both should fall approximately the same distance (drag effect differs slightly)
    # With drag proportional to velocity (not mass-dependent), heavier particles
    # actually fall slightly faster due to less relative drag effect
    # But for initial steps, difference should be small
    assert abs(y_light - y_heavy) < 1.0, "Heavy and light particles should fall similarly initially"


# =============================================================================
# LEVEL 5: Drag/Air Resistance Physics (3 tests)
# =============================================================================

def test_15_drag_slows_horizontal_motion():
    """Test that air resistance slows down horizontal motion."""
    from solution import Particle
    p = Particle(0, 0, 50, 0, 1.0)
    initial_vx = p.get_velocity()[0]

    p.update(0.1)
    new_vx = p.get_velocity()[0]

    assert new_vx < initial_vx, "Horizontal velocity should decrease due to drag"
    assert new_vx > 0, "Velocity should remain positive (same direction)"


def test_16_drag_slows_over_time():
    """Test that velocity continues to decrease over multiple updates."""
    from solution import Particle
    p = Particle(0, 0, 100, 0, 1.0)

    velocities = [p.get_velocity()[0]]
    for _ in range(5):
        p.update(0.1)
        velocities.append(p.get_velocity()[0])

    # Each velocity should be less than the previous
    for i in range(1, len(velocities)):
        assert velocities[i] < velocities[i-1], f"Velocity should keep decreasing: {velocities}"


def test_17_drag_affects_both_directions():
    """Test that drag affects both vx and vy components."""
    from solution import Particle
    # Particle moving up and to the right
    p = Particle(0, 0, 20, 20, 1.0)
    initial_vx = p.get_velocity()[0]
    initial_vy = p.get_velocity()[1]

    p.update(0.1)
    new_vx = p.get_velocity()[0]
    new_vy = p.get_velocity()[1]

    # vx should decrease due to drag
    assert new_vx < initial_vx, "vx should decrease due to drag"
    # vy should decrease more (due to both drag and gravity)
    assert new_vy < initial_vy, "vy should decrease due to drag and gravity"


# =============================================================================
# LEVEL 6: Kinetic Energy (3 tests)
# =============================================================================

def test_18_has_kinetic_energy_method():
    """Test that Particle has get_kinetic_energy method."""
    from solution import Particle
    p = Particle(0, 0, 10, 0, 1.0)
    assert hasattr(p, 'get_kinetic_energy'), "Particle should have get_kinetic_energy method"
    assert callable(p.get_kinetic_energy), "get_kinetic_energy should be callable"


def test_19_kinetic_energy_stationary():
    """Test that stationary particle has zero kinetic energy."""
    from solution import Particle
    p = Particle(0, 0, 0, 0, 2.0)
    ke = p.get_kinetic_energy()
    assert abs(ke) < 0.01, f"Stationary particle should have KE=0, got {ke}"


def test_20_kinetic_energy_simple():
    """Test kinetic energy calculation: KE = 0.5 * m * v²."""
    from solution import Particle
    # v = 10 m/s (only horizontal), m = 2 kg
    # KE = 0.5 * 2 * 10² = 100 J
    p = Particle(0, 0, 10, 0, 2.0)
    ke = p.get_kinetic_energy()
    assert abs(ke - 100.0) < 0.01, f"Expected KE=100 J, got {ke}"


def test_21_kinetic_energy_both_components():
    """Test kinetic energy with both velocity components."""
    from solution import Particle
    # vx = 3, vy = 4, so v² = 9 + 16 = 25, m = 2
    # KE = 0.5 * 2 * 25 = 25 J
    p = Particle(0, 0, 3, 4, 2.0)
    ke = p.get_kinetic_energy()
    assert abs(ke - 25.0) < 0.01, f"Expected KE=25 J, got {ke}"


# =============================================================================
# LEVEL 7: Integration & Accuracy (3 tests)
# =============================================================================

def test_22_single_step_position_accuracy():
    """Test position after single update step with known parameters."""
    from solution import Particle
    # Start at (0, 10), v=(5, 0), mass=1
    # After dt=0.1:
    # - Drag on vx: F_drag_x = -0.1 * 5 = -0.5, ax = -0.5/1 = -0.5
    # - vx_new = 5 + (-0.5) * 0.1 = 4.95
    # - x_new = 0 + 4.95 * 0.1 = 0.495
    # - Gravity: ay = -9.8
    # - vy_new = 0 + (-9.8) * 0.1 = -0.98
    # - y_new = 10 + (-0.98) * 0.1 = 9.902
    p = Particle(0, 10, 5, 0, 1.0)
    p.update(0.1)
    pos = p.get_position()

    assert abs(pos[0] - 0.495) < 0.02, f"Expected x≈0.495, got {pos[0]}"
    assert abs(pos[1] - 9.902) < 0.02, f"Expected y≈9.902, got {pos[1]}"


def test_23_single_step_velocity_accuracy():
    """Test velocity after single update step with known parameters."""
    from solution import Particle
    p = Particle(0, 10, 5, 0, 1.0)
    p.update(0.1)
    vel = p.get_velocity()

    assert abs(vel[0] - 4.95) < 0.02, f"Expected vx≈4.95, got {vel[0]}"
    assert abs(vel[1] - (-0.98)) < 0.02, f"Expected vy≈-0.98, got {vel[1]}"


def test_24_multiple_steps_trajectory():
    """Test that particle follows expected trajectory over multiple steps."""
    from solution import Particle
    p = Particle(0, 0, 20, 20, 1.5)

    # Simulate for 2.5 seconds (50 steps of 0.05s)
    for _ in range(50):
        p.update(0.05)

    pos = p.get_position()
    ke = p.get_kinetic_energy()

    # After 2.5s with these parameters, x should be positive and reasonable
    assert pos[0] > 0 and pos[0] < 50, f"Final x position {pos[0]} out of expected range"
    # KE should have decreased from initial due to drag and gravity doing negative work
    assert abs(ke - 239.0) < 15, f"Final KE {ke} should be approximately 239 J"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
