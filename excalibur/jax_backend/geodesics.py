"""JAX implementations of geodesic equations for perturbed FLRW.

These functions mirror the physics of `perturbed_flrw_metric_fast.compute_geodesic_acceleration`,
but are written in pure JAX for JIT and vectorization.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap

from excalibur.core.constants import c

jax.config.update("jax_enable_x64", True)


@jit
def geodesic_acceleration_perturbed_flrw(u, a, adot, phi_norm, grad_phi_si, phi_dot_norm):
    """Geodesic acceleration for perturbed FLRW (Newtonian gauge, psi=phi).

    Parameters
    ----------
    u : array, shape (4,)
        4-velocity components [u0, u1, u2, u3].
    a : float
        Scale factor a(eta).
    adot : float
        Derivative da/deta.
    phi_norm : float
        Gravitational potential Phi/c^2 (dimensionless).
    grad_phi_si : array, shape (3,)
        Spatial gradient of Phi in SI units (m/s^2).
    phi_dot_norm : float
        Time derivative of Phi/c^2 (dimensionless).

    Returns
    -------
    du : array, shape (4,)
        Derivatives [du0, du1, du2, du3].
    """
    c_val = c
    c2 = c_val * c_val
    c4 = c2 * c2
    a2 = a * a
    adot_a = adot / a

    u0, u1, u2, u3 = u[0], u[1], u[2], u[3]
    u_spatial_sq = u1 * u1 + u2 * u2 + u3 * u3

    # Newtonian gauge: psi = phi
    psi = phi_norm

    # du0/dlambda (temporal acceleration)
    # Gamma_000_term = 0 for static field (psi_dot ~ 0)
    gamma_0ij = (a * adot / c2
                 + 2.0 * a * adot / c4 * (phi_norm + psi)
                 - a2 / c4 * phi_dot_norm)
    gamma_00i = 2.0 * (
        (grad_phi_si[0] / c2) * u0 * u1
        + (grad_phi_si[1] / c2) * u0 * u2
        + (grad_phi_si[2] / c2) * u0 * u3
    )
    du0 = -(gamma_0ij * u_spatial_sq + gamma_00i)

    # du1/dlambda (x)
    gamma_100 = (grad_phi_si[0] / c2) / a2 * u0 * u0
    gamma_10i = 2.0 * (adot_a - phi_dot_norm / c2) * u0 * u1
    gamma_1ii = (
        -(grad_phi_si[0] / c2) * u1 * u1
        + (grad_phi_si[0] / c2) * (u2 * u2 + u3 * u3)
        - (grad_phi_si[1] / c2) * u1 * u2
        - (grad_phi_si[2] / c2) * u1 * u3
    )
    du1 = -(gamma_100 + gamma_10i + gamma_1ii)

    # du2/dlambda (y) - symmetric
    gamma_200 = (grad_phi_si[1] / c2) / a2 * u0 * u0
    gamma_20i = 2.0 * (adot_a - phi_dot_norm / c2) * u0 * u2
    gamma_2ii = (
        -(grad_phi_si[1] / c2) * u2 * u2
        + (grad_phi_si[1] / c2) * (u1 * u1 + u3 * u3)
        - (grad_phi_si[0] / c2) * u2 * u1
        - (grad_phi_si[2] / c2) * u2 * u3
    )
    du2 = -(gamma_200 + gamma_20i + gamma_2ii)

    # du3/dlambda (z) - symmetric
    gamma_300 = (grad_phi_si[2] / c2) / a2 * u0 * u0
    gamma_30i = 2.0 * (adot_a - phi_dot_norm / c2) * u0 * u3
    gamma_3ii = (
        -(grad_phi_si[2] / c2) * u3 * u3
        + (grad_phi_si[2] / c2) * (u1 * u1 + u2 * u2)
        - (grad_phi_si[0] / c2) * u3 * u1
        - (grad_phi_si[1] / c2) * u3 * u2
    )
    du3 = -(gamma_300 + gamma_30i + gamma_3ii)

    return jnp.array([du0, du1, du2, du3])


# Vectorized version over a batch of photons
geodesic_acceleration_perturbed_flrw_batch = vmap(
    geodesic_acceleration_perturbed_flrw,
    in_axes=(0, None, None, 0, 0, 0),  # batch over u, phi, grad_phi, phi_dot
)
