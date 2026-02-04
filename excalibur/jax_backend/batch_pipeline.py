"""High-level JAX batch integration API.

This module exposes a simple function to integrate many photons using the
JAX backend, while relying on the existing cosmology and interpolator
objects from excalibur.
"""

import numpy as np
import jax.numpy as jnp

from .metrics import build_metric_batch_params
from .integrators import integrate_batch_rk4


def integrate_photons_jax(initial_states, cosmology, interpolator, n_steps, dt,
                           field_name="Phi", eta_mode="midpoint"):
    """Integrate a batch of photons with JAX RK4 backend.

    Parameters
    ----------
    initial_states : array, shape (N, 8)
    cosmology : object with a_of_eta and adot_of_eta
    interpolator : grid interpolator with value_gradient_and_time_derivative
    n_steps : int
    dt : float
        Conformal time step.

    Returns
    -------
    trajectory : np.ndarray, shape (n_steps+1, N, 8)
    """
    # Prepare metric parameters from existing Python code
    a, adot, phi_norms_np, grad_phi_np, phi_dot_norms_np = build_metric_batch_params(
        initial_states, cosmology, interpolator, field_name=field_name, eta_mode=eta_mode
    )

    # Move data to JAX
    phi_norms = jnp.array(phi_norms_np)
    grad_phi_batch = jnp.array(grad_phi_np)
    phi_dot_norms = jnp.array(phi_dot_norms_np)

    # Run JAX integrator
    traj_jax = integrate_batch_rk4(
        initial_states,
        n_steps,
        dt,
        a,
        adot,
        phi_norms,
        grad_phi_batch,
        phi_dot_norms,
    )

    # Convert back to NumPy for compatibility with existing pipeline
    return np.array(traj_jax)
