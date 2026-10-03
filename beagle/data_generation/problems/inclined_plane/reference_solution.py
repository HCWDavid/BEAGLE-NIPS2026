"""
Reference solution for Inclined Plane Slider.
This is the correct implementation that students should work towards.
"""

import math


class Slider:
    """
    Simulates an object sliding down an inclined plane.
    
    Physics model:
    - Gravity component along ramp: g * sin(angle)
    - Friction opposing motion: friction * g * cos(angle)
    - Net acceleration: a = g * (sin(angle) - friction * cos(angle))
    - Numerical integration: Euler method with dt time steps
    """

    def __init__(self, angle, length, friction=0.1):
        """
        Initialize slider on inclined plane.
        
        Args:
            angle: angle of incline (degrees)
            length: length of ramp (meters)
            friction: coefficient of kinetic friction (0-1)
        """
        self.angle = angle
        self.length = length
        self.friction = friction
        self.g = 9.8  # gravity (m/s²)
        
        # Convert angle to radians
        self.angle_rad = math.radians(angle)
        
        # State variables
        self.position = 0.0  # distance along ramp
        self.velocity = 0.0  # velocity along ramp
        self.time = 0.0
        
        # Calculate acceleration
        # a = g * (sin(angle) - friction * cos(angle))
        sin_a = math.sin(self.angle_rad)
        cos_a = math.cos(self.angle_rad)
        self.acceleration = self.g * (sin_a - self.friction * cos_a)
        
        # If friction is too high, object won't slide
        if self.acceleration < 0:
            self.acceleration = 0

    def update(self, dt):
        """
        Update slider state after time step dt.
        
        Args:
            dt: time step in seconds
        """
        if self.has_reached_bottom():
            return
        
        # Only move if acceleration is positive (can overcome friction)
        if self.acceleration > 0:
            # Update velocity (v = v0 + a*dt)
            self.velocity += self.acceleration * dt
            
            # Update position (x = x0 + v*dt)
            self.position += self.velocity * dt
        
        # Update time
        self.time += dt
        
        # Clamp position to ramp length
        if self.position > self.length:
            self.position = self.length

    def get_position(self):
        """Return distance traveled along ramp (meters)."""
        return self.position

    def get_velocity(self):
        """Return current velocity along ramp (m/s)."""
        return self.velocity

    def get_acceleration(self):
        """Return current acceleration along ramp (m/s²)."""
        return self.acceleration

    def has_reached_bottom(self):
        """Return True if slider has reached bottom of ramp."""
        return self.position >= self.length

    def get_time(self):
        """Return total time elapsed (seconds)."""
        return self.time
