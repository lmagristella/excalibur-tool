"""Helpers for weak-lensing normalization conventions.

The NFW benchmarking scripts in this repository compare two closely related
normalizations for the critical surface density:

- the conformal/comoving convention, consistent with the conformal Jacobi
  dictionary used by the cosmological optical solver;
- the physical weak-lensing convention, which differs by an extra
  ``1 / (1 + z_l)`` factor.
"""

import numpy as np

from excalibur.core.constants import G, c


DEFAULT_LENSING_REFERENCE_CONVENTION = "conformal_comoving"
PHYSICAL_LENSING_REFERENCE_CONVENTION = "physical"


def sigma_cr_conventions(d_l, d_s, d_ls, z_l):
    """Return ``(Sigma_cr_comoving, Sigma_cr_physical)``.

    The distances ``d_l``, ``d_s`` and ``d_ls`` are the comoving distances used
    throughout the NFW benchmarking scripts.
    """
    if d_l <= 0.0 or d_s <= 0.0 or d_ls <= 0.0:
        raise ValueError("Sigma_cr requires strictly positive lensing distances")

    sigma_cr_comoving = (c ** 2 / (4.0 * np.pi * G)) * d_s / (d_l * d_ls)
    sigma_cr_physical = sigma_cr_comoving / (1.0 + z_l)
    return sigma_cr_comoving, sigma_cr_physical


def lensing_convention_label(convention):
    """Human-readable label for a stored lensing convention key."""
    if convention == DEFAULT_LENSING_REFERENCE_CONVENTION:
        return "conformal/comoving"
    if convention == PHYSICAL_LENSING_REFERENCE_CONVENTION:
        return "physical"
    return str(convention)