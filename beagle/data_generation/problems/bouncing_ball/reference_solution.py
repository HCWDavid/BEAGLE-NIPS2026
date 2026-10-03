"""
Reference solution for Bouncing Ball Simulator.
This is the correct implementation that students should work towards.
"""


class BouncingBall:
    """
    Simulates a ball falling under gravity and bouncing off the floor.
    
    Physics model:
    - Gravity: constant downward acceleration g = 9.8 m/s²
    - Bounce: velocity reverses and reduces by restitution coefficient
    - Numerical integration: Euler method with dt time steps
    """

    def __init__(self, height, velocity, restitution=0.8):
        """
        Initialize bouncing ball.
        
        Args:
            height: initial height above floor (meters)
            velocity: initial vertical velocity (m/s, positive = upward)
            restitution: coefficient of restitution (0-1, energy kept on bounce)
        """
        self.height = height
        self.velocity = velocity
        self.restitution = restitution
        self.g = 9.8  # gravity (m/s²)
        self.mass = 1.0  # assume unit mass for energy calculations
        self.bounce_count = 0
        self.at_rest = False

    def update(self, dt):
        """
        Update ball state after time step dt using Euler integration.
        
        Args:
            dt: time step in seconds
        """
        if self.at_rest:
            return

        # Apply gravity (downward acceleration)
        self.velocity -= self.g * dt

        # Update position
        self.height += self.velocity * dt

        # Check for floor collision
        if self.height <= 0:
            # Ball hit the floor
            self.height = 0  # Keep ball at floor level
            
            # Reverse and reduce velocity (bounce)
            self.velocity = -self.restitution * self.velocity
            self.bounce_count += 1

            # Check if ball has essentially stopped
            # If the velocity is too small to reach a meaningful height
            max_height = (self.velocity ** 2) / (2 * self.g) if self.velocity > 0 else 0
            if max_height < 0.01:
                self.at_rest = True
                self.velocity = 0

    def get_height(self):
        """Return current height above floor (meters)."""
        return self.height

    def get_velocity(self):
        """Return current vertical velocity (m/s)."""
        return self.velocity

    def get_bounce_count(self):
        """Return number of times ball has bounced."""
        return self.bounce_count

    def get_kinetic_energy(self):
        """
        Calculate kinetic energy: KE = 0.5 * m * v²
        """
        return 0.5 * self.mass * self.velocity**2

    def get_potential_energy(self):
        """
        Calculate gravitational potential energy: PE = m * g * h
        """
        return self.mass * self.g * self.height

    def is_at_rest(self):
        """Return True if ball has stopped bouncing."""
        return self.at_rest
