"""
Quasi-spherical Szekeres metric (QSS, epsilon = +1) -- readable reference.

This is the Phase-1 (correctness-first) implementation, mirroring the structure
of :mod:`excalibur.metrics.perturbed_flrw_metric`.  It builds the *diagonal*
Szekeres metric and assembles the Christoffel symbols directly from the metric
and its derivatives -- so every factor of ``c`` falls out automatically and the
geodesic RHS is correct-by-construction in any consistent unit system.  Setting
``c = G = 1`` reduces everything to the geometric form of Celerier (2024),
eqs. 7-10, which is used purely as a cross-check (see the test suite).

State / coordinates
-------------------
``x^mu = (t, r, p, q)`` comoving synchronous (``t`` = cosmic time, index 0).
Geodesic state ``[t, r, p, q, k^t, k^r, k^p, k^q]`` with ``k^mu = dx^mu/ds``.

Metric (signature ``(-, +, +, +)``, ``g_tt = -c^2``)::

    ds^2 = -c^2 dt^2 + H^2 dr^2 + F^2 (dp^2 + dq^2)

    H^2 = (Phi_,r - Phi E_,r/E)^2 / (epsilon - k)        (= g_rr)
    F^2 = Phi^2 / E^2                                     (= g_pp = g_qq)

All ``Phi`` / ``E`` quantities come from a :class:`SzekeresModel`.

Pitfall (brief sec. 1.6): Szekeres null geodesics are **not** radial -- the
``E_,r/E`` dipole couples ``r`` to ``(p, q)``.  Always initialise ``k^p, k^q``
from the null condition + intended sky direction, never fix ``(p, q)`` by hand.
"""
import numpy as np

from .base_metric import Metric
from excalibur.core.constants import c


class SzekeresMetric(Metric):
    r"""Diagonal quasi-spherical Szekeres metric driven by a :class:`SzekeresModel`.

    Parameters
    ----------
    model : excalibur.core.szekeres_model.SzekeresModel
        Provides ``Phi`` (and t/r derivatives), ``E`` (and derivatives),
        ``k(r)``, ``k_,r(r)`` and ``epsilon``.
    """

    # No coordinate conversion: native coords are the integration coords.
    enable_lensing = False

    def __init__(self, model):
        self.model = model
        self.epsilon = model.epsilon

    # ------------------------------------------------------------------
    #  Local bundle of all background quantities needed at a point
    # ------------------------------------------------------------------
    def _local(self, t, r, p, q):
        m = self.model
        Phi = m.Phi(t, r)
        Phi_t = m.Phi_t(t, r)
        Phi_r = m.Phi_r(t, r)
        Phi_tr = m.Phi_tr(t, r)
        Phi_rr = m.Phi_rr(t, r)

        E = m.E(r, p, q)
        E_p = m.E_p(r, p, q)
        E_q = m.E_q(r, p, q)
        E_r = m.E_r(r, p, q)
        E_rr = m.E_rr(r, p, q)
        E_rp = m.E_rp(r, p, q)
        E_rq = m.E_rq(r, p, q)

        k = m.k(r)
        k_r = m.dk(r)
        return dict(
            Phi=Phi, Phi_t=Phi_t, Phi_r=Phi_r, Phi_tr=Phi_tr, Phi_rr=Phi_rr,
            E=E, E_p=E_p, E_q=E_q, E_r=E_r, E_rr=E_rr, E_rp=E_rp, E_rq=E_rq,
            k=k, k_r=k_r,
        )

    # ------------------------------------------------------------------
    #  Diagonal metric  g_mu_nu
    # ------------------------------------------------------------------
    def _diag(self, L):
        """Return the four diagonal components ``(g_tt, g_rr, g_pp, g_qq)``."""
        ER = L["E_r"] / L["E"]
        D = L["Phi_r"] - L["Phi"] * ER          # proper radial factor
        denom_k = self.epsilon - L["k"]
        g_tt = -c * c
        g_rr = D * D / denom_k
        F2 = (L["Phi"] * L["Phi"]) / (L["E"] * L["E"])
        return g_tt, g_rr, F2, F2

    def metric_tensor(self, x):
        t, r, p, q = x[0], x[1], x[2], x[3]
        L = self._local(t, r, p, q)
        g_tt, g_rr, g_pp, g_qq = self._diag(L)
        g = np.zeros((4, 4))
        g[0, 0] = g_tt
        g[1, 1] = g_rr
        g[2, 2] = g_pp
        g[3, 3] = g_qq
        return g

    # ------------------------------------------------------------------
    #  Derivatives of the diagonal components -> dg[mu, a] = d g_aa / d x^mu
    # ------------------------------------------------------------------
    def _metric_diag_and_grad(self, L):
        r"""Return ``(g_diag, dg)`` with ``g_diag[a] = g_aa`` and ``dg[mu, a] = d_mu g_aa``.

        Only diagonal metric components are non-zero, so we only track their
        gradients.  Index map: 0=t, 1=r, 2=p, 3=q.
        """
        Phi = L["Phi"]; Phi_t = L["Phi_t"]; Phi_r = L["Phi_r"]
        Phi_tr = L["Phi_tr"]; Phi_rr = L["Phi_rr"]
        E = L["E"]; E_p = L["E_p"]; E_q = L["E_q"]; E_r = L["E_r"]
        E_rr = L["E_rr"]; E_rp = L["E_rp"]; E_rq = L["E_rq"]
        k = L["k"]; k_r = L["k_r"]

        ER = E_r / E                                  # E_,r / E
        D = Phi_r - Phi * ER                          # proper radial factor
        denom_k = self.epsilon - k

        # --- derivatives of D = Phi_,r - Phi (E_,r/E) ---
        # d_r(E_,r/E) = E_,rr/E - (E_,r/E)^2
        dER_dr = E_rr / E - ER * ER
        # d_p(E_,r/E) = E_,rp/E - (E_,r/E)(E_,p/E)
        dER_dp = E_rp / E - ER * (E_p / E)
        dER_dq = E_rq / E - ER * (E_q / E)

        Dt = Phi_tr - Phi_t * ER
        Dr = Phi_rr - Phi_r * ER - Phi * dER_dr
        Dp = -Phi * dER_dp
        Dq = -Phi * dER_dq

        # --- g_rr = D^2 / (epsilon - k) ---
        grr = D * D / denom_k
        dgrr_dt = 2.0 * D * Dt / denom_k
        # d_r includes the explicit k_,r from the denominator
        dgrr_dr = 2.0 * D * Dr / denom_k + D * D * k_r / (denom_k * denom_k)
        dgrr_dp = 2.0 * D * Dp / denom_k
        dgrr_dq = 2.0 * D * Dq / denom_k

        # --- g_pp = g_qq = Phi^2 / E^2 ---
        E2 = E * E
        E3 = E2 * E
        gpp = Phi * Phi / E2
        dgpp_dt = 2.0 * Phi * Phi_t / E2
        dgpp_dr = 2.0 * Phi * Phi_r / E2 - 2.0 * Phi * Phi * E_r / E3
        dgpp_dp = -2.0 * Phi * Phi * E_p / E3
        dgpp_dq = -2.0 * Phi * Phi * E_q / E3

        g_diag = np.array([-c * c, grr, gpp, gpp])

        dg = np.zeros((4, 4))            # dg[mu, a]
        # a = 0 (t): g_tt = -c^2 constant -> zero gradient (already zeros)
        # a = 1 (r):
        dg[0, 1] = dgrr_dt
        dg[1, 1] = dgrr_dr
        dg[2, 1] = dgrr_dp
        dg[3, 1] = dgrr_dq
        # a = 2 (p):
        dg[0, 2] = dgpp_dt
        dg[1, 2] = dgpp_dr
        dg[2, 2] = dgpp_dp
        dg[3, 2] = dgpp_dq
        # a = 3 (q): identical functional form to g_pp
        dg[0, 3] = dgpp_dt
        dg[1, 3] = dgpp_dr
        dg[2, 3] = dgpp_dp
        dg[3, 3] = dgpp_dq

        return g_diag, dg

    def christoffel(self, x):
        r"""Christoffel symbols ``Gamma^a_{bc}`` from the diagonal metric.

        For a diagonal metric,

            Gamma^a_{bc} = (1 / 2 g_aa)
                           [ delta_{ac} d_b g_aa + delta_{ab} d_c g_aa
                             - delta_{bc} d_a g_bb ]
        """
        t, r, p, q = x[0], x[1], x[2], x[3]
        L = self._local(t, r, p, q)
        g_diag, dg = self._metric_diag_and_grad(L)

        G3 = np.zeros((4, 4, 4))
        for a in range(4):
            inv = 0.5 / g_diag[a]
            for b in range(4):
                for cc in range(4):
                    term = 0.0
                    if a == cc:
                        term += dg[b, a]
                    if a == b:
                        term += dg[cc, a]
                    if b == cc:
                        term -= dg[a, b]
                    if term != 0.0:
                        G3[a, b, cc] = inv * term
        return G3

    # ------------------------------------------------------------------
    #  Geodesic equations  d/ds [x^mu, k^mu] = [k^mu, -Gamma^mu_ab k^a k^b]
    # ------------------------------------------------------------------
    def geodesic_equations(self, state):
        x, u = state[:4], state[4:8]
        chris = self.christoffel(x)
        du = -np.einsum('abc,b,c->a', chris, u, u)
        return np.concatenate([u, du])

    # ------------------------------------------------------------------
    #  Diagnostics
    # ------------------------------------------------------------------
    def null_constraint(self, state):
        r"""First integral ``g_mu_nu k^mu k^nu`` (should stay ~0 along the ray).

        In cosmo coordinates this reads
        ``-(c k^t)^2 + H^2 (k^r)^2 + F^2 [(k^p)^2 + (k^q)^2]``.
        """
        x, u = state[:4], state[4:8]
        g = self.metric_tensor(x)
        return float(u @ (g @ u))

    def null_constraint_relative(self, state):
        """Scale-free null-condition error |g k k| / sum|g_aa (k^a)^2|."""
        x, u = state[:4], state[4:8]
        g = self.metric_tensor(x)
        norm = float(u @ (g @ u))
        denom = sum(abs(g[i, i] * u[i] * u[i]) for i in range(4))
        return abs(norm) / denom if denom > 0 else abs(norm)

    def solve_null_kt(self, x, k_spatial, *, future_pointing=False):
        r"""Return ``k^t`` making ``k^mu`` null at position ``x``.

        ``(c k^t)^2 = H^2 (k^r)^2 + F^2 [(k^p)^2 + (k^q)^2]``.  By default we
        return the **past-pointing** root (``k^t < 0``) used for backward
        ray-tracing from the observer.
        """
        t, r, p, q = x[0], x[1], x[2], x[3]
        L = self._local(t, r, p, q)
        _, g_rr, g_pp, g_qq = self._diag(L)
        kr, kp, kq = k_spatial
        spatial = g_rr * kr * kr + g_pp * kp * kp + g_qq * kq * kq
        kt = np.sqrt(spatial) / c            # from -c^2 (k^t)^2 + spatial = 0
        return kt if future_pointing else -kt

    def redshift_from_kt(self, kt_emit, kt_obs):
        r"""``1 + z = k^t_emit / k^t_obs`` (ratio, unit-independent; Celerier 2024 eq. 12)."""
        return kt_emit / kt_obs

    # ------------------------------------------------------------------
    #  Reference implementations from Celerier (2024) -- cross-checks
    # ------------------------------------------------------------------
    def geodesic_rhs_paper(self, state):
        r"""Geodesic RHS transcribed from Celerier (2024) eqs. 7-10.

        Independent of :meth:`christoffel`; used purely to cross-check the
        Christoffel-assembled :meth:`geodesic_equations`.  The paper writes the
        equations in geometric units (``g_tt = -1``); restoring ``g_tt = -c^2``
        only multiplies the ``d^2 t/ds^2`` equation by ``1/c^2`` (its Christoffels
        carry ``g^{tt} = -1/c^2``); eqs. 8-10 are unchanged.  Setting ``c = 1``
        reproduces the paper term by term.
        """
        t, r, p, q = state[0], state[1], state[2], state[3]
        ut, ur, up, uq = state[4], state[5], state[6], state[7]
        L = self._local(t, r, p, q)
        Phi = L["Phi"]; Phi_t = L["Phi_t"]; Phi_r = L["Phi_r"]
        Phi_tr = L["Phi_tr"]; Phi_rr = L["Phi_rr"]
        E = L["E"]; E_p = L["E_p"]; E_q = L["E_q"]; E_r = L["E_r"]
        E_rr = L["E_rr"]; E_rp = L["E_rp"]; E_rq = L["E_rq"]
        k = L["k"]; k_r = L["k_r"]
        eps = self.epsilon

        ER = E_r / E
        D = Phi_r - Phi * ER
        ek = eps - k
        E2 = E * E

        # eq 7  (d^2 t/ds^2), with the 1/c^2 from g^{tt}
        A = ((Phi_tr - Phi_t * ER) / ek) * D
        B = Phi * Phi_t / E2
        du_t = -(1.0 / (c * c)) * (A * ur * ur + B * (up * up + uq * uq))

        # eq 8  (d^2 r/ds^2)
        c1 = (Phi_tr - Phi_t * ER) / D
        c2 = (Phi_rr - Phi * E_rr / E) / D - ER + k_r / (2.0 * ek)
        c3 = (Phi / E2) * (E_r * E_p - E * E_rp) / D
        c4 = (Phi / E2) * (E_r * E_q - E * E_rq) / D
        c5 = (Phi / E2) * ek / D
        du_r = -(2.0 * c1 * ut * ur + c2 * ur * ur
                 + 2.0 * c3 * ur * up + 2.0 * c4 * ur * uq
                 - c5 * (up * up + uq * uq))

        # eq 9  (d^2 p/ds^2)
        d1 = Phi_t / Phi
        d3 = D / Phi
        dp2 = (D / (Phi * ek)) * (E_r * E_p - E * E_rp)
        du_p = -(2.0 * d1 * ut * up - dp2 * ur * ur + 2.0 * d3 * ur * up
                 - 2.0 * (E_q / E) * up * uq
                 + (E_p / E) * (-up * up + uq * uq))

        # eq 10  (d^2 q/ds^2)  (p <-> q symmetric)
        dq2 = (D / (Phi * ek)) * (E_r * E_q - E * E_rq)
        du_q = -(2.0 * d1 * ut * uq - dq2 * ur * ur + 2.0 * d3 * ur * uq
                 - 2.0 * (E_p / E) * up * uq
                 + (E_q / E) * (up * up - uq * uq))

        return np.array([ut, ur, up, uq, du_t, du_r, du_p, du_q])

    def bondi_dlnz_ds(self, state):
        r"""``d ln(1+z)/ds`` from the Bondi wave-crest method (Celerier 2024 eq. 20).

        Independent ODE for the redshift, integrated alongside the geodesic as a
        cross-check of the ``k^t``-ratio method.  Restoring ``g_tt = -c^2`` adds a
        ``1/c^2`` factor (same origin as eq. 7).
        """
        t, r, p, q = state[0], state[1], state[2], state[3]
        ut, ur, up, uq = state[4], state[5], state[6], state[7]
        L = self._local(t, r, p, q)
        Phi = L["Phi"]; Phi_t = L["Phi_t"]; Phi_r = L["Phi_r"]; Phi_tr = L["Phi_tr"]
        E = L["E"]; E_r = L["E_r"]; k = L["k"]
        ER = E_r / E
        ek = self.epsilon - k
        num = (Phi_tr * Phi_r + Phi * Phi_t * ER * ER
               - (Phi_t * Phi_r + Phi * Phi_tr) * ER)
        bracket = num / ek * ur * ur + (Phi * Phi_t / (E * E)) * (up * up + uq * uq)
        return -(1.0 / (c * c * ut)) * bracket

    # ------------------------------------------------------------------
    #  Recording hook (called by Integrator with the 4-position)
    # ------------------------------------------------------------------
    def metric_physical_quantities(self, x):
        t, r, p, q = x[0], x[1], x[2], x[3]
        m = self.model
        Phi = m.Phi(t, r)
        Phi_t = m.Phi_t(t, r)
        Phi_r = m.Phi_r(t, r)
        E = m.E(r, p, q)
        rho = m.rho(t, r, p, q)
        return np.array([Phi, Phi_t, Phi_r, E, rho])
