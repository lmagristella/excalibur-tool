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


_GAUSS_LEGENDRE_CACHE = {}


def _gauss_legendre_to_infinity(order):
    cached = _GAUSS_LEGENDRE_CACHE.get(order)
    if cached is not None:
        return cached

    nodes, weights = np.polynomial.legendre.leggauss(order)
    u = 0.5 * (nodes + 1.0)
    tau = u / (1.0 - u)
    mapped_weights = 0.5 * weights / (1.0 - u) ** 2
    _GAUSS_LEGENDRE_CACHE[order] = (tau, mapped_weights)
    return tau, mapped_weights


def _rotation_matrix_xyz(orientation_euler_deg):
    angles = np.asarray(orientation_euler_deg, dtype=float)
    if angles.shape != (3,):
        raise ValueError("orientation_euler_deg must contain exactly three values")

    ax, ay, az = np.deg2rad(angles)

    rot_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(ax), -np.sin(ax)],
            [0.0, np.sin(ax), np.cos(ax)],
        ],
        dtype=float,
    )
    rot_y = np.array(
        [
            [np.cos(ay), 0.0, np.sin(ay)],
            [0.0, 1.0, 0.0],
            [-np.sin(ay), 0.0, np.cos(ay)],
        ],
        dtype=float,
    )
    rot_z = np.array(
        [
            [np.cos(az), -np.sin(az), 0.0],
            [np.sin(az), np.cos(az), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return rot_z @ rot_y @ rot_x


def _validate_rotation_matrix(rotation_matrix):
    rotation = np.asarray(rotation_matrix, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError("rotation_matrix must have shape (3, 3)")

    should_be_identity = rotation.T @ rotation
    if not np.allclose(should_be_identity, np.eye(3), atol=1e-12):
        raise ValueError("rotation_matrix must be orthonormal")

    if np.linalg.det(rotation) <= 0.0:
        raise ValueError("rotation_matrix must be right-handed")
    return rotation


def _validate_axis_ratios(axis_ratios):
    ratios = np.asarray(axis_ratios, dtype=float)
    if ratios.shape != (2,):
        raise ValueError("axis_ratios must contain exactly two values: (b_over_a, c_over_a)")

    q_intermediate, q_minor = ratios
    if not (0.0 < q_minor <= q_intermediate <= 1.0):
        raise ValueError("axis_ratios must satisfy 0 < c/a <= b/a <= 1")
    return q_intermediate, q_minor


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

    def potential_gradient(self, x, y, z):
        r"""Analytic gradient :math:`\nabla\Phi = (GM(<r)/r^2)\,\hat r`.

        Returns ``(gx, gy, gz)`` as plain Python floats for a single point,
        or ndarray components for broadcasted inputs.  Uses the shell
        theorem: for any spherical profile, the gradient depends only on
        the mass enclosed.
        """
        dx = np.asarray(x, dtype=float) - self.x0
        dy = np.asarray(y, dtype=float) - self.y0
        dz = np.asarray(z, dtype=float) - self.z0
        r2 = dx * dx + dy * dy + dz * dz
        r2 = np.maximum(r2, (1e-6 * self.r_s) ** 2)
        r = np.sqrt(r2)
        M = self.mass_enclosed(r)
        f_over_r = G * M / (r2 * r)     # f(r)/r = GM/r^3
        gx = f_over_r * dx
        gy = f_over_r * dy
        gz = f_over_r * dz
        if np.ndim(gx) == 0:
            return float(gx), float(gy), float(gz)
        return gx, gy, gz

    def potential_hessian(self, x, y, z):
        r"""Analytic Hessian :math:`H_{ij} = \partial_i\partial_j \Phi`.

        .. math::
            H_{ij} = \left[4\pi G \rho(r) - \frac{3 G M(<r)}{r^3}\right]
                     \frac{(x_i-x_{0i})(x_j-x_{0j})}{r^2}
                     + \delta_{ij}\,\frac{G M(<r)}{r^3}

        Scalar inputs return a (3, 3) ndarray.  The trace satisfies
        :math:`\nabla^2\Phi = 4\pi G \rho(r)` (Poisson).
        """
        dx = float(x) - self.x0
        dy = float(y) - self.y0
        dz = float(z) - self.z0
        r2 = dx * dx + dy * dy + dz * dz
        r_min2 = (1e-6 * self.r_s) ** 2
        if r2 < r_min2:
            r2 = r_min2
        r = np.sqrt(r2)
        s = r / self.r_s
        rho = self.rho_s / (s * (1.0 + s) ** 2)
        M = 4.0 * np.pi * self.rho_s * self.r_s ** 3 * (
            np.log(1.0 + s) - s / (1.0 + s)
        )
        GM_over_r3 = G * M / (r2 * r)
        radial_coef = 4.0 * np.pi * G * rho - 3.0 * GM_over_r3
        H = np.empty((3, 3))
        d = (dx, dy, dz)
        for i in range(3):
            for j in range(3):
                H[i, j] = radial_coef * d[i] * d[j] / r2
                if i == j:
                    H[i, j] += GM_over_r3
        return H

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
        scalar_input = x.ndim == 0
        x_work = np.atleast_1d(x)

        g = np.log(x_work / 2.0)
        lo = x_work < 1.0 - 1e-6
        hi = x_work > 1.0 + 1e-6
        eq = ~lo & ~hi

        if np.any(lo):
            xl = x_work[lo]
            g[lo] += np.arccosh(1.0 / xl) / np.sqrt(1.0 - xl**2)

        if np.any(hi):
            xh = x_work[hi]
            g[hi] += np.arctan(np.sqrt(xh**2 - 1.0)) / np.sqrt(xh**2 - 1.0)

        g[eq] += 1.0

        if scalar_input:
            return float(g[0])
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
        r"""Convergence kappa(b) = Sigma(b) / Sigma_cr.

        This helper is intentionally agnostic about the critical surface-density
        convention. The returned kappa matches whichever ``Sigma_cr`` the
        caller supplies:

        - using ``Sigma_cr = c^2/(4 pi G) * D_s/(D_l D_ls)/(1+z_l)`` yields the
          usual physical-screen weak-lensing normalization used by several NFW
          comparison scripts in this repository;
        - using the comoving ``Sigma_cr`` without the extra ``/(1+z_l)`` factor
          yields the conformal-screen reference appropriate for Fleury's
          conformal Jacobi dictionary.
        """
        return self.surface_density(b) / Sigma_cr

    def gamma_analytic(self, b, Sigma_cr):
        r"""
        Tangential shear for NFW:

        .. math::
            \gamma_t(b) = \frac{\bar\Sigma(<b) - \Sigma(b)}{\Sigma_{cr}}

        As for ``kappa_analytic``, the numerical value depends on the caller's
        choice of ``Sigma_cr`` convention.
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


class TriaxialNFWHalo:
    r"""
    Triaxial NFW halo with homoeoidal ellipsoidal symmetry.

    The density follows the triaxial NFW form discussed by Heyrovsky &
    Karamazov (2024),

    .. math::
        \rho(\hat a) = \frac{\rho_s}{(\hat a/r_s)(1 + \hat a/r_s)^2},

    with ellipsoidal semi-major-axis coordinate

    .. math::
        \hat a^2 = x_1^2 + \frac{x_2^2}{q_2^2} + \frac{x_3^2}{q_3^2},

    where ``q_2 = b/a`` and ``q_3 = c/a`` in the principal-axes frame.

    The three-dimensional Newtonian potential and its derivatives are evaluated
    from the standard one-dimensional homoeoidal integrals for densities
    stratified on similar ellipsoids. In the spherical limit ``q_2 = q_3 = 1``,
    the class reduces to ``NFWHalo``.

    Parameters
    ----------
    M_200 : float
        Mass inside the ellipsoid of semi-major axis ``a_200`` whose mean
        density is ``200 * rho_cr`` [kg].
    c_NFW : float
        NFW concentration defined with respect to ``a_200``.
    center : array-like, shape (3,)
        Halo center in world coordinates [m].
    axis_ratios : tuple(float, float), optional
        Principal-axis ratios ``(b/a, c/a)`` with ``0 < c/a <= b/a <= 1``.
    rotation_matrix : array-like, shape (3, 3), optional
        Right-handed orthonormal matrix whose columns are the principal axes in
        world coordinates.
    orientation_euler_deg : array-like, shape (3,), optional
        Convenience alternative to ``rotation_matrix`` using the same XYZ Euler
        convention as ``equivalent_mass_profiles``.
    rho_cr : float, optional
        Critical density [kg/m^3]. Defaults to the same present-day value used
        by ``NFWHalo``.
    quadrature_order : int, optional
        Gauss-Legendre order used for the semi-infinite homoeoidal integrals.
    """

    def __init__(
        self,
        M_200,
        c_NFW,
        center,
        axis_ratios=(1.0, 1.0),
        rotation_matrix=None,
        orientation_euler_deg=None,
        rho_cr=None,
        quadrature_order=96,
    ):
        if rotation_matrix is not None and orientation_euler_deg is not None:
            raise ValueError("Specify either rotation_matrix or orientation_euler_deg, not both")

        self.M_200 = float(M_200)
        self.c_NFW = float(c_NFW)
        self.center = np.asarray(center, dtype=float)
        self.x0, self.y0, self.z0 = self.center

        self.q_intermediate, self.q_minor = _validate_axis_ratios(axis_ratios)
        self.axis_ratios = (self.q_intermediate, self.q_minor)
        self.e1 = np.sqrt(1.0 - self.q_intermediate ** 2)
        self.e2 = np.sqrt(1.0 - self.q_minor ** 2)

        if rotation_matrix is None and orientation_euler_deg is None:
            self.rotation_matrix = np.eye(3, dtype=float)
        elif rotation_matrix is not None:
            self.rotation_matrix = _validate_rotation_matrix(rotation_matrix)
        else:
            self.rotation_matrix = _rotation_matrix_xyz(orientation_euler_deg)

        if rho_cr is None:
            H0 = 70e3 / one_Mpc
            rho_cr = 3.0 * H0**2 / (8.0 * np.pi * G)
        self.rho_cr = float(rho_cr)

        shape_factor = self.q_intermediate * self.q_minor
        self.a_200 = (
            3.0 * self.M_200 / (4.0 * np.pi * 200.0 * self.rho_cr * shape_factor)
        ) ** (1.0 / 3.0)
        self.R_200 = self.a_200
        self.r_s = self.a_200 / self.c_NFW

        fc = np.log(1.0 + self.c_NFW) - self.c_NFW / (1.0 + self.c_NFW)
        self.rho_s = self.M_200 / (4.0 * np.pi * shape_factor * self.r_s**3 * fc)

        self._axis_scales = np.array([1.0, self.q_intermediate, self.q_minor], dtype=float)
        self._axis_scales_sq = self._axis_scales**2
        self._axis_product = float(np.prod(self._axis_scales))
        self._tau_nodes, self._tau_weights = _gauss_legendre_to_infinity(int(quadrature_order))
        interval_nodes, interval_weights = np.polynomial.legendre.leggauss(max(256, 2 * int(quadrature_order)))
        interval_u = 0.5 * (interval_nodes + 1.0)
        interval_integrand = 1.0 / np.sqrt(
            (1.0 + (self.q_intermediate**2 - 1.0) * interval_u**2)
            * (1.0 + (self.q_minor**2 - 1.0) * interval_u**2)
        )
        self._inv_delta_integral = float(np.sum(interval_weights * interval_integrand))
        self._xi_center = self.rho_s * self.r_s**2
        self._m_soft = 1e-6 * self.r_s

    def density(self, x, y, z):
        """Triaxial NFW density evaluated at world coordinates [kg/m^3]."""
        ell_radius = self._ellipsoidal_radius(x, y, z)
        scaled_radius = np.maximum(ell_radius / self.r_s, 1e-10)
        return self.rho_s / (scaled_radius * (1.0 + scaled_radius) ** 2)

    def mass_enclosed(self, ellipsoidal_radius):
        r"""Mass enclosed inside ellipsoids with semi-major axis ``ellipsoidal_radius``."""
        s = np.asarray(ellipsoidal_radius, dtype=float) / self.r_s
        return 4.0 * np.pi * self._axis_product * self.rho_s * self.r_s**3 * (
            np.log(1.0 + s) - s / (1.0 + s)
        )

    def potential(self, x, y, z):
        r"""Triaxial NFW potential ``Phi`` evaluated by homoeoidal quadrature."""
        x_arr, y_arr, z_arr = np.broadcast_arrays(
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
            np.asarray(z, dtype=float),
        )

        values = np.empty_like(x_arr, dtype=float)
        flat_values = values.reshape(-1)
        dx = (x_arr - self.x0).reshape(-1)
        dy = (y_arr - self.y0).reshape(-1)
        dz = (z_arr - self.z0).reshape(-1)
        local_points = self.rotation_matrix.T @ np.vstack([dx, dy, dz])

        for index in range(flat_values.size):
            flat_values[index] = self._potential_local(local_points[:, index])

        if values.ndim == 0:
            return float(values)
        return values

    def potential_gradient(self, x, y, z):
        r"""Gradient of the triaxial NFW potential at one point."""
        point_local = self._point_to_local(x, y, z)
        local_gradient = self._gradient_local(point_local)
        world_gradient = self.rotation_matrix @ local_gradient
        return tuple(float(component) for component in world_gradient)

    def potential_hessian(self, x, y, z):
        r"""Hessian of the triaxial NFW potential at one point."""
        point_local = self._point_to_local(x, y, z)
        hessian_local = self._hessian_local(point_local)
        return self.rotation_matrix @ hessian_local @ self.rotation_matrix.T

    def _point_to_local(self, x, y, z):
        point = np.array([float(x) - self.x0, float(y) - self.y0, float(z) - self.z0], dtype=float)
        return self.rotation_matrix.T @ point

    def _ellipsoidal_radius(self, x, y, z):
        x_arr, y_arr, z_arr = np.broadcast_arrays(
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
            np.asarray(z, dtype=float),
        )
        dx = (x_arr - self.x0).reshape(-1)
        dy = (y_arr - self.y0).reshape(-1)
        dz = (z_arr - self.z0).reshape(-1)
        local_points = self.rotation_matrix.T @ np.vstack([dx, dy, dz])
        ell_radius = np.sqrt(
            local_points[0] ** 2
            + (local_points[1] / self.q_intermediate) ** 2
            + (local_points[2] / self.q_minor) ** 2
        )
        ell_radius = ell_radius.reshape(x_arr.shape)
        if ell_radius.ndim == 0:
            return float(ell_radius)
        return ell_radius

    def _common_integrand_terms(self, point_local, *, soften_density):
        tau = self._tau_nodes
        denom_terms = self._axis_scales_sq[:, None] + tau[None, :]
        delta = np.sqrt(np.prod(denom_terms, axis=0))
        ell_radius_sq = np.sum((point_local[:, None] ** 2) / denom_terms, axis=0)
        ell_radius = np.sqrt(ell_radius_sq)
        if soften_density:
            ell_radius_density = np.maximum(ell_radius, self._m_soft)
        else:
            ell_radius_density = ell_radius
        common_weights = self._tau_weights / delta
        return denom_terms, common_weights, ell_radius, ell_radius_density

    def _nfw_xi(self, ell_radius):
        return self.rho_s * self.r_s**3 / (ell_radius + self.r_s)

    def _nfw_density(self, ell_radius):
        scaled_radius = np.maximum(ell_radius / self.r_s, 1e-12)
        return self.rho_s / (scaled_radius * (1.0 + scaled_radius) ** 2)

    def _nfw_density_prime_over_radius(self, ell_radius):
        scaled_radius = np.maximum(ell_radius / self.r_s, 1e-12)
        return (
            -self.rho_s / self.r_s**2
            * (1.0 + 3.0 * scaled_radius)
            / (scaled_radius**3 * (1.0 + scaled_radius) ** 3)
        )

    def _potential_local(self, point_local):
        _, common_weights, ell_radius, _ = self._common_integrand_terms(
            point_local,
            soften_density=False,
        )
        xi = self._nfw_xi(ell_radius) - self._xi_center
        prefactor = -2.0 * np.pi * G * self._axis_product
        return prefactor * (self._xi_center * self._inv_delta_integral + np.sum(common_weights * xi))

    def _gradient_local(self, point_local):
        if np.allclose(point_local, 0.0, atol=0.0):
            return np.zeros(3, dtype=float)

        denom_terms, common_weights, _, ell_radius_density = self._common_integrand_terms(
            point_local,
            soften_density=True,
        )
        rho = self._nfw_density(ell_radius_density)
        prefactor = 2.0 * np.pi * G * self._axis_product
        gradient = np.empty(3, dtype=float)
        for axis_index in range(3):
            integral = np.sum(common_weights * rho / denom_terms[axis_index])
            gradient[axis_index] = prefactor * point_local[axis_index] * integral
        return gradient

    def _hessian_local(self, point_local):
        denom_terms, common_weights, _, ell_radius_density = self._common_integrand_terms(
            point_local,
            soften_density=True,
        )
        rho = self._nfw_density(ell_radius_density)
        rho_prime_over_radius = self._nfw_density_prime_over_radius(ell_radius_density)
        prefactor = 2.0 * np.pi * G * self._axis_product

        diagonal_integrals = np.empty(3, dtype=float)
        for axis_index in range(3):
            diagonal_integrals[axis_index] = np.sum(
                common_weights * rho / denom_terms[axis_index]
            )

        hessian = np.zeros((3, 3), dtype=float)
        for i in range(3):
            hessian[i, i] = prefactor * diagonal_integrals[i]
            for j in range(i, 3):
                mixed_integral = np.sum(
                    common_weights
                    * rho_prime_over_radius
                    / (denom_terms[i] * denom_terms[j])
                )
                hessian[i, j] += prefactor * point_local[i] * point_local[j] * mixed_integral
                if i != j:
                    hessian[j, i] = hessian[i, j]
        return hessian

    def __repr__(self):
        return (
            f"TriaxialNFWHalo(M_200={self.M_200/one_Msun:.2e} Msun, "
            f"c={self.c_NFW}, "
            f"b/a={self.q_intermediate:.3f}, "
            f"c/a={self.q_minor:.3f}, "
            f"a_200={self.a_200/one_Mpc:.3f} Mpc, "
            f"r_s={self.r_s/one_Mpc*1000:.0f} kpc)"
        )
