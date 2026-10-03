"""
Reference solution for 2D Particle Physics Simulator.
This is the correct implementation that students should work towards.
"""


class Particle:
    """
    Simulates a particle moving in 2D space under gravity and air resistance.
    
    Physics model:
    - Gravity: constant downward acceleration g = 9.8 m/s²
    - Air resistance (drag): F_drag = -k * v, where k = 0.1
    - Acceleration: a = F_total / mass
    - Numerical integration: Euler method with dt time steps
    """

    def __init__(self, x, y, vx, vy, mass):
        """
        Initialize particle with position, velocity, and mass.
        
        Args:
            x: initial x position (meters)
            y: initial y position (meters)
            vx: initial x velocity (m/s)
            vy: initial y velocity (m/s)
            mass: particle mass (kg)
        """
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.mass = mass
        self.g = 9.8  # gravity (m/s²)
        self.k = 0.1  # drag coefficient

    def update(self, dt):
        """
        Update particle state after time step dt using Euler integration.
        
        Args:
            dt: time step in seconds
        """
        # Calculate forces
        # Gravity force (downward)
        F_gravity_y = -self.mass * self.g

        # Drag force (opposes motion)
        F_drag_x = -self.k * self.vx
        F_drag_y = -self.k * self.vy

        # Total force
        F_total_x = F_drag_x
        F_total_y = F_gravity_y + F_drag_y

        # Acceleration (F = ma, so a = F/m)
        ax = F_total_x / self.mass
        ay = F_total_y / self.mass

        # Update velocity (v = v0 + a*dt)
        self.vx += ax * dt
        self.vy += ay * dt

        # Update position (x = x0 + v*dt)
        self.x += self.vx * dt
        self.y += self.vy * dt

    def get_position(self):
        """Return current position as (x, y) tuple."""
        return (self.x, self.y)

    def get_velocity(self):
        """Return current velocity as (vx, vy) tuple."""
        return (self.vx, self.vy)

    def get_kinetic_energy(self):
        """
        Calculate kinetic energy: KE = 0.5 * m * v²
        where v² = vx² + vy²
        """
        v_squared = self.vx**2 + self.vy**2
        return 0.5 * self.mass * v_squared
