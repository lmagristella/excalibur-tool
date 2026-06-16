"""
Underdensity (cosmic void) lens model.

A void is a region whose matter density lies *below* the cosmic mean, so its
density **contrast** delta = (rho - rho_bar)/rho_bar is negative. Lensing
responds to the contrast (the perturbation away from the smooth background), so
an underdensity produces a *negative* convergence kappa < 0: it **defocuses**
light and aligns background galaxies **radially** (tangential shear of the
opposite sign to a cluster).

``VoidNFW`` models a compact underdensity with an NFW *shape* but a negative
amplitude. Because every NFW lensing quantity (potential, gradient, Hessian,
surface density) is **linear in rho_s**, flipping the sign of ``rho_s`` turns the
overdense halo into an underdense void with the correct signs everywhere:

    rho(r)    = -|rho_s| / [(r/r_s)(1 + r/r_s)^2]            < 0
    Phi(r)    = +4 pi G |rho_s| r_s^3 ln(1 + r/r_s) / r       (repulsive)
    Sigma(b)  = -2 |rho_s| r_s f(b/r_s)                       < 0
    kappa(b)  = Sigma(b) / Sigma_cr                           < 0

The model is parameterised by an *equivalent* ``M_200`` magnitude (the mass that
an overdense NFW of the same |rho_s|, r_s would carry) and a concentration, so a
void of "depth" M_200 = 2e15 Msun fills the same angular scale as the cluster
halos used elsewhere in the toolkit. It exposes the same interface as
``NFWHalo`` (center, r_s, rho_s, potential/gradient/Hessian) plus the
``q_intermediate``/``q_minor``/``rotation_matrix`` attributes so it is a drop-in
analytical source for the lensing runners and the (spherical) Numba NFW bypass.

References
----------
Amendola, Frieman & Waga 1999; Krause et al. 2013 -- void weak lensing.
Hamaus, Sutter & Wandelt 2014 -- universal void density profile (compensated).
"""

import numpy as np

from excalibur.core.constants import one_Mpc, one_Msun
from excalibur.objects.nfw_halo import NFWHalo


class VoidNFW(NFWHalo):
    r"""Compact underdensity with an NFW shape and negative amplitude.

    Parameters
    ----------
    M_200_equiv : float
        Magnitude of the equivalent NFW mass [kg]. Sets r_s and |rho_s| via the
        usual NFW relations; the stored ``rho_s`` is then negated.
    c_NFW : float
        Concentration of the equivalent NFW (controls how peaked the void is).
    center : array (3,)
        Void center [m].
    rho_cr : float, optional
        Critical density [kg/m^3]; defaults to the present-day value used by
        ``NFWHalo``.
    """

    def __init__(self, M_200_equiv, c_NFW, center, rho_cr=None):
        super().__init__(M_200_equiv, c_NFW, center, rho_cr=rho_cr)
        # Flip the amplitude: same NFW shape, underdense (delta < 0).
        self.rho_s = -self.rho_s
        self.Sigma_s = -self.Sigma_s
        # Keep a positive bookkeeping handle on the depth.
        self.M_200_equiv = M_200_equiv
        self.is_underdensity = True
        # Spherical: expose the triaxial-style metadata the runners read.
        self.q_intermediate = 1.0
        self.q_minor = 1.0
        self.rotation_matrix = np.eye(3, dtype=float)

    def __repr__(self):
        return (
            f"VoidNFW(|M_200|={self.M_200/one_Msun:.2e} Msun (underdense), "
            f"c={self.c_NFW}, "
            f"R_200={self.R_200/one_Mpc:.3f} Mpc, "
            f"r_s={self.r_s/one_Mpc*1000:.0f} kpc)"
        )
