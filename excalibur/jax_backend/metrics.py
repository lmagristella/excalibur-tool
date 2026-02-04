"""Wrappers to connect existing metric/interpolator code to the JAX backend.

These functions stay in NumPy space, preparing arrays of metric quantities
(phi, grad phi, phi_dot, a, adot) that are then consumed by JAX kernels.
"""

import numpy as np
from excalibur.core.constants import c


def build_metric_batch_params(photon_states, cosmology, interpolator, field_name="Phi", eta_mode="midpoint"):
    """Prepare metric parameters for a batch of photons.

    Parameters
    ----------
    photon_states : array, shape (N, 8)
        Initial photon states [eta, x, y, z, u_eta, u_x, u_y, u_z].
    cosmology : object
        Must provide a_of_eta(eta) and adot_of_eta(eta).
    interpolator : object
        Must provide value_gradient_and_time_derivative(pos, field_name, eta).
    field_name : str
        Name of the potential field to use (default "Phi").
    eta_mode : str
        How to choose eta for metric evaluation ("midpoint" or "initial").

    Returns
    -------
    a : float
    adot : float
    phi_norms : array, shape (N,)
    grad_phi_batch : array, shape (N, 3)
    phi_dot_norms : array, shape (N,)
    """
    photon_states = np.asarray(photon_states)
    etas = photon_states[:, 0]
    positions = photon_states[:, 1:4]

    if eta_mode == "midpoint":
        eta_ref = 0.5 * (etas.min() + etas.max())
    else:
        eta_ref = etas[0]

    a = cosmology.a_of_eta(eta_ref)
    adot = cosmology.adot_of_eta(eta_ref)

    phi_list = []
    grad_list = []
    phi_dot_list = []

    for pos in positions:
        phi_SI, grad_phi_SI, phi_dot_SI = interpolator.value_gradient_and_time_derivative(
            pos, field_name, eta_ref
        )
        phi_list.append(phi_SI / (c ** 2))  # dimensionless
        grad_list.append(grad_phi_SI)
        phi_dot_list.append(phi_dot_SI / (c ** 2))

    phi_norms = np.array(phi_list)
    grad_phi_batch = np.array(grad_list)
    phi_dot_norms = np.array(phi_dot_list)

    return a, adot, phi_norms, grad_phi_batch, phi_dot_norms
