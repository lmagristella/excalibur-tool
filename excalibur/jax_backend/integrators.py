"""JAX-based numerical integrators for photon geodesics.

This module provides vectorized RK4 integration using the JAX geodesic kernels
from `geodesics.py`.

State format for photons (per row):
    state = [eta, x, y, z, u_eta, u_x, u_y, u_z]
"""

import jax
import jax.numpy as jnp
from jax import jit

from .geodesics import geodesic_acceleration_perturbed_flrw_batch

jax.config.update("jax_enable_x64", True)


@jit
def rk4_step_batch(states_batch, dt, a, adot, phi_norms, grad_phi_batch, phi_dot_norms):
    """Single RK4 step for a batch of photons.

    Parameters
    ----------
    states_batch : array, shape (N, 8)
        Photon states [eta, x, y, z, u_eta, u_x, u_y, u_z] for N photons.
    dt : float
        Integration step in conformal time (can be negative for backward tracing).
    a, adot : float
        Scale factor and its derivative (can be taken at representative eta).
    phi_norms : array, shape (N,)
        Phi/c^2 values for each photon (dimensionless).
    grad_phi_batch : array, shape (N, 3)
        Spatial gradients of Phi in SI units for each photon.
    phi_dot_norms : array, shape (N,)
        Phi_dot/c^2 values for each photon (dimensionless).

    Returns
    -------
    new_states : array, shape (N, 8)
        Updated states after one RK4 step.
    """
    # Split into position and velocity
    velocities = states_batch[:, 4:]

    # k1
    k1 = jnp.zeros_like(states_batch)
    k1 = k1.at[:, :4].set(velocities)
    k1_accel = geodesic_acceleration_perturbed_flrw_batch(
        velocities, a, adot, phi_norms, grad_phi_batch, phi_dot_norms
    )
    k1 = k1.at[:, 4:].set(k1_accel)

    # k2
    states2 = states_batch + 0.5 * dt * k1
    vel2 = states2[:, 4:]
    k2 = jnp.zeros_like(states_batch)
    k2 = k2.at[:, :4].set(vel2)
    k2_accel = geodesic_acceleration_perturbed_flrw_batch(
        vel2, a, adot, phi_norms, grad_phi_batch, phi_dot_norms
    )
    k2 = k2.at[:, 4:].set(k2_accel)

    # k3
    states3 = states_batch + 0.5 * dt * k2
    vel3 = states3[:, 4:]
    k3 = jnp.zeros_like(states_batch)
    k3 = k3.at[:, :4].set(vel3)
    k3_accel = geodesic_acceleration_perturbed_flrw_batch(
        vel3, a, adot, phi_norms, grad_phi_batch, phi_dot_norms
    )
    k3 = k3.at[:, 4:].set(k3_accel)

    # k4
    states4 = states_batch + dt * k3
    vel4 = states4[:, 4:]
    k4 = jnp.zeros_like(states_batch)
    k4 = k4.at[:, :4].set(vel4)
    k4_accel = geodesic_acceleration_perturbed_flrw_batch(
        vel4, a, adot, phi_norms, grad_phi_batch, phi_dot_norms
    )
    k4 = k4.at[:, 4:].set(k4_accel)

    # Combine
    new_states = states_batch + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return new_states


def integrate_batch_rk4(initial_states, n_steps, dt, a, adot, phi_norms, grad_phi_batch, phi_dot_norms):
    """Integrate a batch of photons using RK4 in Python loop over time.

    This keeps the per-step heavy math JIT-compiled, while the outer
    time loop stays in Python for simplicity.

    Parameters
    ----------
    initial_states : array, shape (N, 8)
    n_steps : int
    dt : float
    a, adot : float
    phi_norms : array, shape (N,)
    grad_phi_batch : array, shape (N, 3)
    phi_dot_norms : array, shape (N,)

    Returns
    -------
    trajectory : array, shape (n_steps+1, N, 8)
        Full trajectory for each photon.
    """
    states = jnp.array(initial_states)
    traj = [states]

    for _ in range(n_steps):
        states = rk4_step_batch(states, dt, a, adot, phi_norms, grad_phi_batch, phi_dot_norms)
        traj.append(states)

    return jnp.stack(traj, axis=0)
