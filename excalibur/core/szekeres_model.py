"""
Quasi-spherical Szekeres (QSS, epsilon = +1) background model for EXCALIBUR.

This is the inhomogeneous analogue of :mod:`excalibur.core.cosmology`.  It stores
the Szekeres free functions of ``r``, solves the areal-radius dynamics
``Phi(t, r)`` (Friedmann--LTB equation with dust + Lambda), and exposes all the
quantities the metric / geodesic / curvature layers need:

    Phi, Phi_,t, Phi_,r, Phi_,tr, Phi_,rr        (areal radius and derivatives)
    E, E_,p, E_,q, E_,r, E_,rr, E_,rp, E_,rq     (Szekeres dipole structure)
    rho(t, r, p, q)                              (rest-mass density)

Reference
---------
M.-N. Celerier, *Precision cosmology with exact inhomogeneous solutions of
General Relativity: the Szekeres models*, Phys. Rev. D 110, 123526 (2024).
arXiv:2407.04452.

Conventions (cf. CODEBASE_OVERVIEW.md, sec. 6)
----------------------------------------------
* Comoving synchronous coordinates ``x^mu = (t, r, p, q)`` with ``t`` the
  **cosmic time** (index 0).  Signature ``(-, +, +, +)``; in the metric
  ``g_tt = -c^2``.
* **Unit-system agnostic.**  ``G`` and ``c`` are read from
  :mod:`excalibur.core.constants` and appear explicitly.  Setting ``c = G = 1``
  reduces every formula to the geometric form of the paper (used as a
  cross-check).  Run in ``"cosmo"`` units (Msun / Gyr / Gpc) for physical
  distances; ``M`` is a physical mass (Msun), ``rho`` a physical density.

Metric (built in :mod:`excalibur.metrics.szekeres_metric`)::

    ds^2 = -c^2 dt^2 + (Phi_,r - Phi E_,r/E)^2 / (epsilon - k) dr^2
                     + Phi^2 / E^2 (dp^2 + dq^2)

Field equations (units restored, the only place we restore G/c by hand)::

    (Phi_,t)^2     = 2 G M / Phi  -  k c^2  +  (Lambda c^2 / 3) Phi^2
    4 pi rho       = (M_,r - 3 M E_,r/E) / [ Phi^2 (Phi_,r - Phi E_,r/E) ]
    t - t_B(r)     = int_0^Phi dPhi~ / sqrt( 2 G M / Phi~ - k c^2 + (Lambda c^2/3) Phi~^2 )

The density equation is **G-free** by construction (it is dM = rho dV); see the
note in the brief.  Setting ``E_,r/E = 0`` (constant ``S, P, Q``) reduces the
model to LTB; choosing ``M ~ r^3``, ``k ~ r^2``, ``t_B = const`` further reduces
it to FLRW with ``Phi = a(t) r`` (see :func:`flrw_equivalent`).
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad, cumulative_simpson
from scipy.interpolate import RectBivariateSpline, PchipInterpolator

from excalibur.core.constants import G, c


# ----------------------------------------------------------------------
#  Small numerical-derivative helper for the (smooth) free functions of r
# ----------------------------------------------------------------------
def _fd1(f, r, h):
    """Central first derivative of a scalar callable ``f(r)``."""
    return (f(r + h) - f(r - h)) / (2.0 * h)


def _fd2(f, r, h):
    """Central second derivative of a scalar callable ``f(r)``."""
    return (f(r + h) - 2.0 * f(r) + f(r - h)) / (h * h)


class SzekeresModel:
    r"""Quasi-spherical Szekeres model (epsilon = +1) with dust + Lambda.

    A model is fully specified by **six free functions of** ``r`` plus ``epsilon``
    and ``Lambda``.  Changing the model is a one-liner -- just pass new callables::

        model = SzekeresModel(M=lambda r: ..., k=lambda r: ..., S=lambda r: ...,
                              P=lambda r: ..., Q=lambda r: ..., t_B=lambda r: ...,
                              Lambda=0.0, epsilon=1)

    Physical meaning of the free functions (Celerier 2024, Sec. II)
    --------------------------------------------------------------
    * ``M(r)``  -- in the quasi-spherical case (``epsilon=+1``) the **total active
      gravitational mass** inside the comoving shell ``r = const`` (the monopole).
    * ``k(r)``  -- the **local spatial curvature** of the shell.  Requires
      ``epsilon - k(r) > 0`` everywhere (so ``k < 1`` for QSS); its sign sets
      bound/marginal/unbound (recollapsing/critical/ever-expanding) shells.
    * ``S(r), P(r), Q(r)`` -- the **dipole** structure.  The whole "Szekeres"
      (non-LTB) character lives in ``E_,r/E``, built solely from ``S, P, Q``:
      on each ``{t, r}=const`` 2-sphere ``E_,r/E`` vanishes on an equator and is
      extremal (``+/-`` max) at two poles, i.e. a mass dipole superposed on the
      ``M`` monopole.  ``S`` sets the dipole strength, ``P, Q`` its orientation
      (the non-concentric shift of successive spheres).  Constant ``S, P, Q`` =>
      ``E_,r = 0`` => the LTB (spherically symmetric) sub-case.
    * ``t_B(r)`` -- the **bang time** (local age of the Universe at shell ``r``);
      a non-constant ``t_B`` is a non-simultaneous big bang (a decaying-mode
      inhomogeneity).  Constant ``t_B`` is the usual "growing-mode-only" choice.
    * ``epsilon`` -- foliation class: ``+1`` quasi-spherical (only fully supported
      case), ``0`` quasi-planar, ``-1`` quasi-hyperbolic.
    * ``Lambda`` -- cosmological constant.

    FLRW / LTB limits.  ``S,P,Q = const`` (dipole off) gives LTB.  Adding
    ``M(r) = M0 r^3``, ``k(r) = k0 r^2`` and ``t_B = const`` gives FLRW with
    ``Phi = a(t) r`` (see :func:`flrw_equivalent`).  One of the six functions can
    always be fixed by the gauge freedom to rescale ``r``.

    Parameters
    ----------
    M, k, S, P, Q, t_B : callable
        The six free functions of ``r`` (each ``float -> float``); units follow
        the active system (``M`` in Msun, ``t_B`` in Gyr for ``"cosmo"``).
    Lambda : float, optional
        Cosmological constant (units ``1/length^2`` in the active system).
    epsilon : int, optional
        Szekeres class.  ``+1`` = quasi-spherical (the only fully supported
        case in v1).  ``0`` / ``-1`` accepted but untested.
    r_min, r_max, n_r : float, float, int
        Radial tabulation range and resolution for the background spline.
    t_min, t_max, n_t : float, float, int
        Cosmic-time tabulation range/resolution (in the active time unit).
        Must stay safely above ``t_B(r)`` everywhere (areal radius derivatives
        diverge at the bang).
    free_derivs : dict, optional
        Optional analytic overrides for r-derivatives of the free functions,
        keyed by ``"dM", "dk", "dS", "dP", "dQ", "dt_B", "ddS", "ddP", "ddQ"``.
        Anything not supplied is finite-differenced (the free functions are
        smooth 1-D callables, so this is safe -- unlike differencing the
        tabulated ``Phi``).
    fd_rel : float
        Relative step used for finite-differencing the free functions.
    spline_k : int
        Spline order for the ``Phi(t, r)`` interpolant (5 recommended so the
        second r-derivative ``Phi_,rr`` is smooth).
    """

    def __init__(
        self,
        M,
        k,
        S,
        P,
        Q,
        t_B,
        Lambda=0.0,
        epsilon=1,
        *,
        r_min=1e-4,
        r_max=1.0,
        n_r=400,
        t_min=None,
        t_max=None,
        n_t=400,
        free_derivs=None,
        fd_rel=1e-6,
        spline_k=5,
    ):
        self.M = M
        self.k = k
        self.S = S
        self.P = P
        self.Q = Q
        self.t_B = t_B
        self.Lambda = float(Lambda)
        self.epsilon = int(epsilon)

        self._free_derivs = dict(free_derivs or {})
        self._fd_rel = float(fd_rel)

        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.n_r = int(n_r)

        # Default cosmic-time window: from just after the latest bang time up to
        # a few expansion times.  The caller should override for tailored runs.
        if t_min is None or t_max is None:
            r_probe = np.linspace(self.r_min, self.r_max, 32)
            tb_vals = np.array([self.t_B(rr) for rr in r_probe])
            tb_hi = float(np.max(tb_vals))
            # rough present-time scale from the flat-dust age ~ (2/3)/H if Λ=0;
            # we just need a safe, generous window for tabulation.
            span = max(self.r_max, 1.0) * 30.0
            t_min = tb_hi + 1e-3 * span if t_min is None else t_min
            t_max = tb_hi + span if t_max is None else t_max
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.n_t = int(n_t)

        self.spline_k = int(spline_k)

        self._build_background()

    # ------------------------------------------------------------------
    #  Free-function r-derivatives (analytic override or finite difference)
    # ------------------------------------------------------------------
    def _h(self, r):
        return self._fd_rel * max(abs(r), 1.0)

    def dM(self, r):
        f = self._free_derivs.get("dM")
        return f(r) if f else _fd1(self.M, r, self._h(r))

    def dk(self, r):
        f = self._free_derivs.get("dk")
        return f(r) if f else _fd1(self.k, r, self._h(r))

    def dS(self, r):
        f = self._free_derivs.get("dS")
        return f(r) if f else _fd1(self.S, r, self._h(r))

    def dP(self, r):
        f = self._free_derivs.get("dP")
        return f(r) if f else _fd1(self.P, r, self._h(r))

    def dQ(self, r):
        f = self._free_derivs.get("dQ")
        return f(r) if f else _fd1(self.Q, r, self._h(r))

    def dt_B(self, r):
        f = self._free_derivs.get("dt_B")
        return f(r) if f else _fd1(self.t_B, r, self._h(r))

    def ddS(self, r):
        f = self._free_derivs.get("ddS")
        return f(r) if f else _fd2(self.S, r, self._h(r))

    def ddP(self, r):
        f = self._free_derivs.get("ddP")
        return f(r) if f else _fd2(self.P, r, self._h(r))

    def ddQ(self, r):
        f = self._free_derivs.get("ddQ")
        return f(r) if f else _fd2(self.Q, r, self._h(r))

    # ==================================================================
    #  Szekeres dipole structure  E(r, p, q)  and derivatives (closed form)
    # ==================================================================
    #
    #  E = (p - P)^2 / (2 S) + (q - Q)^2 / (2 S) + epsilon S / 2
    #
    #  Derivatives (derived analytically; the brief's E_,r prefactor is
    #  S_,r / 2, NOT S_,r/(2 S) -- the brief flags this to verify):
    #
    #    E_,p  = (p - P) / S
    #    E_,q  = (q - Q) / S
    #    E_,r  = -(S_,r / 2) [ ((p-P)/S)^2 + ((q-Q)/S)^2 - epsilon ]
    #            - ( (p-P) P_,r + (q-Q) Q_,r ) / S
    #    E_,rp = -P_,r / S - (p-P) S_,r / S^2
    #    E_,rq = -Q_,r / S - (q-Q) S_,r / S^2
    #    E_,rr : differentiate E_,r once more (uses S_,rr, P_,rr, Q_,rr)
    #
    def E(self, r, p, q):
        S = self.S(r)
        dp = p - self.P(r)
        dq = q - self.Q(r)
        return (dp * dp + dq * dq) / (2.0 * S) + self.epsilon * S / 2.0

    def E_p(self, r, p, q):
        return (p - self.P(r)) / self.S(r)

    def E_q(self, r, p, q):
        return (q - self.Q(r)) / self.S(r)

    def E_r(self, r, p, q):
        S = self.S(r)
        Sr = self.dS(r)
        Pr = self.dP(r)
        Qr = self.dQ(r)
        dp = p - self.P(r)
        dq = q - self.Q(r)
        u = dp / S
        v = dq / S
        return -(Sr / 2.0) * (u * u + v * v - self.epsilon) - (dp * Pr + dq * Qr) / S

    def E_rp(self, r, p, q):
        S = self.S(r)
        Sr = self.dS(r)
        Pr = self.dP(r)
        dp = p - self.P(r)
        return -Pr / S - dp * Sr / (S * S)

    def E_rq(self, r, p, q):
        S = self.S(r)
        Sr = self.dS(r)
        Qr = self.dQ(r)
        dq = q - self.Q(r)
        return -Qr / S - dq * Sr / (S * S)

    def E_rr(self, r, p, q):
        S = self.S(r)
        Sr = self.dS(r)
        Srr = self.ddS(r)
        Pr = self.dP(r)
        Prr = self.ddP(r)
        Qr = self.dQ(r)
        Qrr = self.ddQ(r)
        dp = p - self.P(r)
        dq = q - self.Q(r)

        # B = ((p-P)/S)^2 + ((q-Q)/S)^2 - epsilon ;  E_,r = -(Sr/2) B - C
        B = dp * dp / (S * S) + dq * dq / (S * S) - self.epsilon
        # B_,r
        Br = (-2.0 * (dp * Pr + dq * Qr) / (S * S)
              - 2.0 * (dp * dp + dq * dq) * Sr / (S ** 3))
        # C = ((p-P) P_,r + (q-Q) Q_,r)/S ;  C_,r
        C_r = ((-Pr * Pr + dp * Prr - Qr * Qr + dq * Qrr) / S
               - (dp * Pr + dq * Qr) * Sr / (S * S))
        return -(Srr / 2.0) * B - (Sr / 2.0) * Br - C_r

    def E_ratio_r(self, r, p, q):
        """Convenience: the Szekeres dipole term ``E_,r / E``."""
        return self.E_r(r, p, q) / self.E(r, p, q)

    # ==================================================================
    #  Background dynamics  Phi(t, r)
    # ==================================================================
    def _W(self, Phi, r, Mr=None, kr=None):
        r"""Friedmann--LTB velocity ``W = sqrt(2 G M / Phi - k c^2 + (Lambda c^2/3) Phi^2)``.

        This is exactly ``Phi_,t`` for the expanding branch.
        """
        if Mr is None:
            Mr = self.M(r)
        if kr is None:
            kr = self.k(r)
        val = 2.0 * G * Mr / Phi - kr * c * c + (self.Lambda * c * c / 3.0) * Phi * Phi
        return np.sqrt(val)

    def _t_of_Phi(self, Phi, r, Mr=None, kr=None):
        r"""Cosmic time elapsed since the bang to reach areal radius ``Phi`` at ``r``.

        ``t - t_B(r) = int_0^Phi dPhi~ / W(Phi~)``.  The integrand ``1/W`` is
        finite on ``(0, Phi]`` and ``-> 0`` at the origin, so plain adaptive
        quadrature is well behaved.  Used only to seed the cumulative tables.
        """
        if Mr is None:
            Mr = self.M(r)
        if kr is None:
            kr = self.k(r)

        def integrand(Ph):
            return 1.0 / self._W(Ph, r, Mr, kr)

        val, _ = quad(integrand, 0.0, Phi, limit=200)
        return self.t_B(r) + val

    def _build_background(self):
        r"""Tabulate ``Phi(t, r)`` and the *analytic* ``Phi_,r(t, r)`` on a grid.

        For each ``r`` we sample the areal radius ``Phi`` from the bang upward and
        build, by cumulative quadrature,

            t(Phi)   = t_B(r) + int_0^Phi dPhi~ / W
            I_r(Phi) = -int_0^Phi (2 G M_,r/Phi~ - k_,r c^2) / (2 W^3) dPhi~

        Inverting ``t(Phi)`` gives ``Phi`` on the cosmic-time grid; the exact
        radial derivative follows analytically (avoids differencing a noisy
        table, cf. brief sec. 3.1):

            Phi_,r = -W(Phi) [ t_B,_r + I_r(Phi) ]

        Two splines are stored: ``Phi`` (for the areal radius) and the *smooth*
        analytic ``Phi_,r`` (whose own derivatives give ``Phi_,rr`` and
        ``Phi_,tr`` with only ``1/Dr`` noise amplification).  ``Phi_,t`` is taken
        directly from Friedmann, ``Phi_,t = W(Phi)`` -- exact.
        """
        self.r_grid = np.linspace(self.r_min, self.r_max, self.n_r)
        self.t_grid = np.linspace(self.t_min, self.t_max, self.n_t)

        Phi_table = np.empty((self.n_t, self.n_r), dtype=float)
        Phi_r_table = np.empty((self.n_t, self.n_r), dtype=float)

        # The areal-radius integrals have a sqrt(Phi) integrand near the bang
        # (W ~ Phi^{-1/2}), whose infinite slope at Phi=0 cripples uniform-grid
        # quadrature.  Substituting Phi = u^2 (dPhi = 2u du) regularises it: the
        # integrands become smooth in u, and Simpson + monotone-cubic (PCHIP)
        # inversion then reach ~1e-10 instead of ~1e-5 (cf. brief sec. 4.4).
        n_samp = 2000
        for j, r in enumerate(self.r_grid):
            Mr = self.M(r)
            kr = self.k(r)
            Mr_r = self.dM(r)
            kr_r = self.dk(r)
            tb_r = self.dt_B(r)

            # Upper Phi bound: grow until cosmic time exceeds t_max.
            Phi_hi = max(self.r_max, 1.0) * 1e-3
            guard = 0
            while self._t_of_Phi(Phi_hi, r, Mr, kr) < self.t_max and guard < 200:
                Phi_hi *= 1.5
                guard += 1

            # Sample u uniformly; Phi = u^2 clusters points near the bang.
            u = np.linspace(0.0, np.sqrt(Phi_hi), n_samp)
            Phi_s = u * u
            W_s = np.empty_like(u)
            W_s[0] = np.inf
            W_s[1:] = self._W(Phi_s[1:], r, Mr, kr)

            # t(Phi) = int dPhi/W = int (2u/W) du  -- smooth integrand in u.
            g_t = np.zeros_like(u)
            g_t[1:] = 2.0 * u[1:] / W_s[1:]
            t_s = self.t_B(r) + cumulative_simpson(g_t, x=u, initial=0.0)

            # I_r(Phi) = -int (2 G M_,r/Phi - k_,r c^2)/(2 W^3) dPhi, in u.
            g_ir = np.zeros_like(u)
            g_ir[1:] = -(2.0 * G * Mr_r / Phi_s[1:] - kr_r * c * c) \
                / (2.0 * W_s[1:] ** 3) * 2.0 * u[1:]
            I_r_s = cumulative_simpson(g_ir, x=u, initial=0.0)

            # Monotone-cubic inversions (t and Phi are both monotone in u).
            Phi_col = PchipInterpolator(t_s, Phi_s)(self.t_grid)
            Phi_table[:, j] = Phi_col

            # Analytic Phi_,r at the grid times.
            W_col = self._W(Phi_col, r, Mr, kr)
            I_r_col = PchipInterpolator(Phi_s, I_r_s)(Phi_col)
            Phi_r_table[:, j] = -W_col * (tb_r + I_r_col)

        self._Phi_table = Phi_table
        self._Phi_r_table = Phi_r_table

        self._phi_spline = RectBivariateSpline(
            self.t_grid, self.r_grid, Phi_table,
            kx=self.spline_k, ky=self.spline_k,
        )
        # Spline of the *analytic* Phi_,r: its derivatives give Phi_,rr, Phi_,tr.
        self._phi_r_spline = RectBivariateSpline(
            self.t_grid, self.r_grid, Phi_r_table,
            kx=self.spline_k, ky=self.spline_k,
        )

    # ------------------------------------------------------------------
    #  Areal radius and derivatives
    # ------------------------------------------------------------------
    def Phi(self, t, r):
        return float(self._phi_spline(t, r, dx=0, dy=0)[0, 0])

    def Phi_t(self, t, r):
        r"""``Phi_,t = W(Phi)`` straight from Friedmann (exact, not differenced)."""
        return self._W(self.Phi(t, r), r)

    def Phi_r(self, t, r):
        r"""Analytic radial derivative (smooth spline of the exact ``Phi_,r``)."""
        return float(self._phi_r_spline(t, r, dx=0, dy=0)[0, 0])

    def Phi_tr(self, t, r):
        return float(self._phi_r_spline(t, r, dx=1, dy=0)[0, 0])

    def Phi_rr(self, t, r):
        return float(self._phi_r_spline(t, r, dx=0, dy=1)[0, 0])

    def phi_t_exact(self, t, r):
        r"""Alias of :meth:`Phi_t` (kept for the cross-check naming in tests)."""
        return self._W(self.Phi(t, r), r)

    # ------------------------------------------------------------------
    #  Density  (G-free; see module docstring)
    # ------------------------------------------------------------------
    def rho(self, t, r, p, q):
        r"""Rest-mass density ``4 pi rho = (M_,r - 3 M E_,r/E)/[Phi^2 (Phi_,r - Phi E_,r/E)]``."""
        Phi = self.Phi(t, r)
        Phi_r = self.Phi_r(t, r)
        Mr = self.M(r)
        Mr_r = self.dM(r)
        Er_over_E = self.E_ratio_r(r, p, q)
        num = Mr_r - 3.0 * Mr * Er_over_E
        den = Phi * Phi * (Phi_r - Phi * Er_over_E)
        return num / (4.0 * np.pi * den)

    # ------------------------------------------------------------------
    #  Shell-crossing / regularity guard
    # ------------------------------------------------------------------
    def proper_radial_factor(self, t, r, p, q):
        r"""The factor ``Phi_,r - Phi E_,r/E`` whose vanishing signals a shell crossing.

        Returns the (signed) value; the metric ``g_rr`` is its square over
        ``epsilon - k``.  A sign change or near-zero crossing along a ray means a
        shell crossing / coordinate breakdown (cf. brief sec. 4.4).
        """
        return self.Phi_r(t, r) - self.Phi(t, r) * self.E_ratio_r(r, p, q)


# ----------------------------------------------------------------------
#  Convenience builders
# ----------------------------------------------------------------------
def flrw_equivalent(cosmology, *, r_min=1e-4, r_max=1.0, n_r=200, n_t=400,
                    t_min=None, t_max=None):
    r"""Build a Szekeres model that is *exactly* FLRW (dipole off).

    With ``Phi = a(t) r`` the Friedmann--LTB equation reduces to the FLRW
    Friedmann equation provided

        M(r) = M0 r^3,   k(r) = k0 r^2,   t_B = const,   S,P,Q = const,

    with ``2 G M0 = H0^2 Omega_m``, ``-k0 c^2 = H0^2 Omega_k`` and
    ``Lambda = 3 H0^2 Omega_Lambda / c^2``.  This is the FLRW-limit oracle for
    the Szekeres test suite (sec. 6, tests 1-2).

    Parameters
    ----------
    cosmology : LCDM_Cosmology
        Provides ``H0`` (s^-1 / active units) and the density parameters.

    Returns
    -------
    SzekeresModel
    """
    H0 = cosmology.H0
    Om = cosmology.Omega_m
    Ok = cosmology.Omega_k
    OL = cosmology.Omega_lambda

    M0 = H0 * H0 * Om / (2.0 * G)
    k0 = -H0 * H0 * Ok / (c * c)
    Lambda = 3.0 * H0 * H0 * OL / (c * c)

    free_derivs = {
        "dM": lambda r: 3.0 * M0 * r * r,
        "dk": lambda r: 2.0 * k0 * r,
        "dS": lambda r: 0.0,
        "dP": lambda r: 0.0,
        "dQ": lambda r: 0.0,
        "dt_B": lambda r: 0.0,
        "ddS": lambda r: 0.0,
        "ddP": lambda r: 0.0,
        "ddQ": lambda r: 0.0,
    }

    return SzekeresModel(
        M=lambda r: M0 * r ** 3,
        k=lambda r: k0 * r ** 2,
        S=lambda r: 1.0,
        P=lambda r: 0.0,
        Q=lambda r: 0.0,
        t_B=lambda r: 0.0,
        Lambda=Lambda,
        epsilon=1,
        r_min=r_min, r_max=r_max, n_r=n_r, n_t=n_t,
        t_min=t_min, t_max=t_max,
        free_derivs=free_derivs,
    )
