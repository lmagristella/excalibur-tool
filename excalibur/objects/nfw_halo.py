"""
Navarro-Frenk-White (NFW) halo profile.

Provides the 3-D gravitational potential Phi(r) for grid-based ray-tracing
and the analytic projected quantities Sigma(b), kappa(b), gamma(b) for validation.

References
----------
Navarro, Frenk & White 1996 (ApJ 462, 563)
Bartelmann 1996 (A&A 313, 697) -- lensing formulae for NFW
Wright & Brainerd 2000 (ApJ 534, 34) -- kappa(x) and gamma(x) closed forms
"""

import numpy as np
from excalibur.core.constants import G, c, one_Mpc, one_Msun


class NFWHalo:
    r"""
    NFW halo with mass *M_200*, concentration *c_NFW*, at position *center*.

    The density profile is

    .. math::
        \rho(r) = \frac{\rho_s}{(r/r_s)(1 + r/r_s)^2}

    where :math:`r_s = R_{200}/c` and :math:`\rho_s` is fixed by
    :math:`M_{200} = 4\pi\rho_s r_s^3 [\ln(1+c) - c/(1+c)]`.

    Parameters
    ----------
    M_200 : float
        Mass inside R_200 [kg].
    c_NFW : float
        NFW concentration parameter.
    center : array (3,)
        Spatial position of the halo centre [m].
    rho_cr : float, optional
        Critical density of the universe [kg/m^3].  Defaults to the
        present-day value for H0 = 70 km/s/Mpc.
    """

    def __init__(self, M_200, c_NFW, center, rho_cr=None):
        self.M_200 = M_200
        self.c_NFW = c_NFW
        self.center = np.asarray(center, dtype=float)
        self.x0, self.y0, self.z0 = self.center

        if rho_cr is None:
            H0 = 70e3 / one_Mpc                       # s^-^1
            rho_cr = 3.0 * H0**2 / (8.0 * np.pi * G)  # kg/m^3
        self.rho_cr = rho_cr

        # R_200 from M_200 = (4pi/3) x 200 rho_cr x R_200^3
        self.R_200 = (3.0 * M_200 / (4.0 * np.pi * 200.0 * rho_cr))**(1.0 / 3.0)
        self.r_s = self.R_200 / c_NFW

        # Characteristic density
        fc = np.log(1.0 + c_NFW) - c_NFW / (1.0 + c_NFW)
        self.rho_s = M_200 / (4.0 * np.pi * self.r_s**3 * fc)

        # Characteristic surface density (for lensing)
        self.Sigma_s = self.rho_s * self.r_s   # kg/m^2

    # ------------------------------------------------------------------
    #  3-D profiles
    # ------------------------------------------------------------------
    def density(self, x, y, z):
        """NFW density rho(r) [kg/m^3]."""
        r = self._radius(x, y, z)
        s = r / self.r_s
        s = np.maximum(s, 1e-10)  # avoid /0
        return self.rho_s / (s * (1.0 + s)**2)

    def mass_enclosed(self, r):
        """Mass inside radius r:  M(<r) = 4pi rho_s r_s^3 [ln(1+r/r_s) - r/r_s/(1+r/r_s)]."""
        s = r / self.r_s
        return 4.0 * np.pi * self.rho_s * self.r_s**3 * (
            np.log(1.0 + s) - s / (1.0 + s)
        )

    def potential(self, x, y, z):
        r"""Gravitational potential Phi(r) for the NFW profile.

        .. math::
            \Phi(r) = -\frac{4\pi G \rho_s r_s^3}{r}\,\ln\!\left(1 + \frac{r}{r_s}\right)

        Accepts meshgrid-style (X, Y, Z) arrays or scalar coordinates.
        """
        x, y, z = np.broadcast_arrays(np.asarray(x, dtype=float),
                                       np.asarray(y, dtype=float),
                                       np.asarray(z, dtype=float))
        r = np.sqrt((x - self.x0)**2 + (y - self.y0)**2 + (z - self.z0)**2)
        r = np.maximum(r, 1e-6 * self.r_s)  # soften singularity at r=0
        s = r / self.r_s
        prefac = 4.0 * np.pi * G * self.rho_s * self.r_s**3
        return -prefac * np.log(1.0 + s) / r

    # ------------------------------------------------------------------
    #  Projected (lensing) quantities  --  Bartelmann 1996 / Wright & Brainerd 2000
    # ------------------------------------------------------------------
    @staticmethod
    def _f_nfw(x):
        r"""
        The function f(x) appearing in the NFW surface-density formula.

        .. math::
            f(x) = \begin{cases}
                \frac{1}{x^2 - 1}\left(1 - \frac{1}{\sqrt{1 - x^2}}
                    \mathrm{arccosh}\,\frac{1}{x}\right) & x < 1 \\[6pt]
                \frac{1}{3} & x = 1 \\[6pt]
                \frac{1}{x^2 - 1}\left(1 - \frac{1}{\sqrt{x^2 - 1}}
                    \arctan\frac{\sqrt{x^2-1}}{1}\right) & x > 1
            \end{cases}
        """
        x = np.asarray(x, dtype=float)
        f = np.empty_like(x)
        lo = x < 1.0 - 1e-6
        hi = x > 1.0 + 1e-6
        eq = ~lo & ~hi

        # x < 1
        if np.any(lo):
            xl = x[lo]
            sqrt_term = np.sqrt(1.0 - xl**2)
            f[lo] = (1.0 / (xl**2 - 1.0)) * (
                1.0 - np.arccosh(1.0 / xl) / sqrt_term
            )

        # x > 1
        if np.any(hi):
            xh = x[hi]
            sqrt_term = np.sqrt(xh**2 - 1.0)
            f[hi] = (1.0 / (xh**2 - 1.0)) * (
                1.0 - np.arctan(sqrt_term) / sqrt_term
            )

        # x ~ 1
        f[eq] = 1.0 / 3.0

        return f

    @staticmethod
    def _g_nfw(x):
        r"""
        The function g(x) appearing in the NFW mean-surface-density formula.

        .. math::
            g(x) = \ln\frac{x}{2} + \begin{cases}
                \frac{1}{\sqrt{1-x^2}}\,\mathrm{arccosh}\frac{1}{x} & x < 1 \\
                1 & x = 1 \\
                \frac{1}{\sqrt{x^2-1}}\,\arctan\frac{\sqrt{x^2-1}}{1} & x > 1
            \end{cases}
        """
        x = np.asarray(x, dtype=float)
        g = np.log(x / 2.0)
        lo = x < 1.0 - 1e-6
        hi = x > 1.0 + 1e-6
        eq = ~lo & ~hi

        if np.any(lo):
            xl = x[lo]
            g[lo] += np.arccosh(1.0 / xl) / np.sqrt(1.0 - xl**2)

        if np.any(hi):
            xh = x[hi]
            g[hi] += np.arctan(np.sqrt(xh**2 - 1.0)) / np.sqrt(xh**2 - 1.0)

        g[eq] += 1.0

        return g

    def surface_density(self, b):
        r"""Projected surface mass density Sigma(b) [kg/m^2].

        .. math::
            \Sigma(b) = 2\,\rho_s\,r_s\,f(b/r_s)

        Parameters
        ----------
        b : float or array
            Impact parameter [m].
        """
        x = np.asarray(b, dtype=float) / self.r_s
        x = np.maximum(x, 1e-6)
        return 2.0 * self.Sigma_s * self._f_nfw(x)

    def mean_surface_density(self, b):
        r"""
        Mean projected surface density inside radius b:

        .. math::
            \bar\Sigma(<b) = \frac{4\,\rho_s\,r_s}{(b/r_s)^2}\,g(b/r_s)
        """
        x = np.asarray(b, dtype=float) / self.r_s
        x = np.maximum(x, 1e-6)
        return (4.0 * self.Sigma_s / x**2) * self._g_nfw(x)

    def kappa_analytic(self, b, Sigma_cr):
        r"""Convergence kappa(b) = Sigma(b) / Sigma_cr."""
        return self.surface_density(b) / Sigma_cr

    def gamma_analytic(self, b, Sigma_cr):
        r"""
        Tangential shear for NFW:

        .. math::
            \gamma_t(b) = \frac{\bar\Sigma(<b) - \Sigma(b)}{\Sigma_{cr}}
        """
        return (self.mean_surface_density(b) - self.surface_density(b)) / Sigma_cr

    # ------------------------------------------------------------------
    #  Utilities
    # ------------------------------------------------------------------
    def _radius(self, x, y, z):
        x, y, z = np.broadcast_arrays(np.asarray(x, dtype=float),
                                       np.asarray(y, dtype=float),
                                       np.asarray(z, dtype=float))
        return np.sqrt((x - self.x0)**2 + (y - self.y0)**2 + (z - self.z0)**2)

    def __repr__(self):
        return (
            f"NFWHalo(M_200={self.M_200/one_Msun:.2e} Msun, "
            f"c={self.c_NFW}, "
            f"R_200={self.R_200/one_Mpc:.3f} Mpc, "
            f"r_s={self.r_s/one_Mpc*1000:.0f} kpc)"
        )
