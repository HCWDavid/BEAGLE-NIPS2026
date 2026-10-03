"""
Reference solution for Gradient Descent Optimizer.
This is the correct implementation that students should work towards.
Based on the Bielefeld Python Programming Study tasks.
"""

import numpy as np


def f(x):
    """
    The objective function to minimize: f(x) = x^4 - x^2 + 0.25*x

    This function has a global minimum around x ≈ -0.66

    Args:
        x: input value (scalar or numpy array)

    Returns:
        f(x) value
    """
    return x**4 - x**2 + 0.25 * x


def f_grad(x):
    """
    Compute the gradient (derivative) of f at point x using numerical differentiation.

    Uses central difference formula: (f(x+delta) - f(x-delta)) / (2*delta)

    Args:
        x: point at which to compute gradient

    Returns:
        Approximate gradient value
    """
    delta = 0.0001
    x1 = x - delta
    x2 = x + delta
    y1 = f(x1)
    y2 = f(x2)
    return (y2 - y1) / (2 * delta)


def gradient_descent(f_grad, x0, eta):
    """
    Perform gradient descent optimization to find the minimum.

    Updates x iteratively: x = x - eta * gradient
    Stops when |gradient| < 0.001 or after max_iterations

    Args:
        f_grad: function that computes gradient at a point
        x0: initial starting point
        eta: learning rate (step size)

    Returns:
        x value at the minimum
    """
    x = x0
    n_steps = 0
    max_iterations = 10000
    tolerance = 0.001

    grad = f_grad(x)

    while np.abs(grad) >= tolerance and n_steps < max_iterations:
        # Gradient descent update: move in opposite direction of gradient
        x = x - eta * grad

        # Compute new gradient
        grad = f_grad(x)

        # Increment step counter
        n_steps += 1

    return x
