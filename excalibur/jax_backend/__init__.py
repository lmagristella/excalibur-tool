"""JAX backend for high-performance relativistic ray tracing.

This package provides JAX-based implementations of:
- geodesic equations
- numerical integrators (RK4 / RK45)
- batch integration of many photons

It is designed as a drop-in high-performance backend for the existing
NumPy/Numba-based excalibur code, without changing the physical models.
"""
