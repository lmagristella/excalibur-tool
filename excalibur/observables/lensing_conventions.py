"""Helpers for weak-lensing normalization conventions.

Two conventions for the critical surface density are exposed:

- ``Sigma_cr_comoving``: the BS-2001 formula structure with comoving distances
  substituted directly. Preserved for backward compatibility with runs that
  used the now-removed (1+z_l)-biased simulation path; not standard.
- ``Sigma_cr_physical``: standard Bartelmann-Schneider 2001 critical surface
  density (angular-diameter distances), expressible from the comoving form as
  ``Sigma_cr_phys = Sigma_cr_comoving * (1 + z_l)``. This is the convention
  used in the weak-lensing literature.

Root-cause fix for the (1+z_l) bias
-----------------------------------
The simulation pipeline previously produced a kappa biased by a factor (1+z_l)
relative to the physical observable, because the NFW potential is defined in
physical coordinates while the simulator's box coordinates are comoving
(Fleury eq 4.69, *Light propagation in inhomogeneous and anisotropic
cosmologies*, 2015). This is now fixed at the source: pass ``bardeen_a_lens =
1/(1+z_l)`` to ``NumbaAMRBackend`` and the kernel rescales r_s and rho_s
internally to evaluate the correct Bardeen potential, with the photon's
physical impact parameter satisfying b_phys = a_l * b_co.

No post-processing factor is required when ``bardeen_a_lens`` is set.
"""

import numpy as np

from excalibur.core.constants import G, c


DEFAULT_LENSING_REFERENCE_CONVENTION = "conformal_comoving"
PHYSICAL_LENSING_REFERENCE_CONVENTION = "physical"


def sigma_cr_conventions(d_l, d_s, d_ls, z_l):
    """Return ``(Sigma_cr_comoving, Sigma_cr_physical)``.

    The distances ``d_l``, ``d_s`` and ``d_ls`` are the **comoving** distances
    used throughout the NFW benchmarking scripts.

    Two conventions are returned:

    - ``Sigma_cr_comoving`` is the BS-2001 formula structure with comoving
      distances substituted directly (non-standard but matches the simulated
      kappa from the conformal-metric pipeline). It carries an extra
      ``1 / (1 + z_l)`` factor relative to the standard physical convention.

    - ``Sigma_cr_physical`` is the standard Bartelmann-Schneider 2001 critical
      surface density, expressible from comoving distances as

          Sigma_cr_physical = (c^2 / 4 pi G) * D_A_s / (D_A_l * D_A_ls)
                            = Sigma_cr_comoving * (1 + z_l)

      This is the convention used in the weak-lensing literature for
      comparison with observations.
    """
    if d_l <= 0.0 or d_s <= 0.0 or d_ls <= 0.0:
        raise ValueError("Sigma_cr requires strictly positive lensing distances")

    sigma_cr_comoving = (c ** 2 / (4.0 * np.pi * G)) * d_s / (d_l * d_ls)
    sigma_cr_physical = sigma_cr_comoving * (1.0 + z_l)
    return sigma_cr_comoving, sigma_cr_physical


def lensing_convention_label(convention):
    """Human-readable label for a stored lensing convention key."""
    if convention == DEFAULT_LENSING_REFERENCE_CONVENTION:
        return "conformal/comoving"
    if convention == PHYSICAL_LENSING_REFERENCE_CONVENTION:
        return "physical"
    return str(convention)
