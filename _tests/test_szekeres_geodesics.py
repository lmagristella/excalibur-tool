#!/usr/bin/env python3
r"""
Validation of :mod:`excalibur.metrics.szekeres_metric` (QSS null geodesics).

Tests (brief sec. 6, tests 1, 4, 5, 8):
    * Christoffel symbols (analytic, diagonal-metric assembly) vs a finite
      difference of ``metric_tensor`` -> the curvature/RHS is *physically*
      correct, not merely self-consistent.
    * Null first integral ``g_mu_nu k^mu k^nu`` preserved along a ray (radial
      FLRW limit: machine precision; non-radial through a dipole: ~1e-6).
    * Einstein--de Sitter limit: backward radial ray reproduces the FLRW
      redshift  ``1 + z = (t_obs / t_emit)^{2/3}``  (= a_obs/a_emit).

Everything runs in the active (SI) unit system but at **Gpc / Gyr scales**, so
that ``c`` is not numerically overwhelming and a light ray actually accrues
cosmological redshift across the box.

Run directly:  ``python _tests/test_szekeres_geodesics.py``
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from excalibur.core.constants import G, c, one_Gpc, one_Gyr
from excalibur.core.szekeres_model import SzekeresModel
from excalibur.metrics.szekeres_metric import SzekeresMetric
from excalibur.integration.integrator import RK4

L = one_Gpc        # length scale (1 Gpc, in active units)
T = one_Gyr        # time scale   (1 Gyr, in active units)
M0 = 2.0 / (9.0 * G)   # makes the EdS closed form Phi = r t^{2/3} exact


# ----------------------------------------------------------------------
#  Model builders
# ----------------------------------------------------------------------
def _eds_radial_model():
    """Flat dust (k=0, Lambda=0), dipole off -> exact EdS, Phi = r t^{2/3}."""
    return SzekeresModel(
        M=lambda r: M0 * r ** 3, k=lambda r: 0.0,
        S=lambda r: 1.0, P=lambda r: 0.0, Q=lambda r: 0.0,
        t_B=lambda r: 0.0, Lambda=0.0, epsilon=1,
        r_min=0.3 * L, r_max=4.0 * L, n_r=120, n_t=300,
        t_min=2.0 * T, t_max=20.0 * T,
        free_derivs={"dM": lambda r: 3.0 * M0 * r * r, "dk": lambda r: 0.0,
                     "dS": lambda r: 0.0, "dP": lambda r: 0.0, "dQ": lambda r: 0.0,
                     "dt_B": lambda r: 0.0, "ddS": lambda r: 0.0,
                     "ddP": lambda r: 0.0, "ddQ": lambda r: 0.0},
    )


def _dipole_model():
    """Genuine quasi-spherical Szekeres: S, P, Q, M, k all vary with r."""
    return SzekeresModel(
        M=lambda r: M0 * r ** 3 * (1.0 + 0.15 * np.sin(r / L)),
        k=lambda r: 0.05 * (r / L) ** 2 * np.exp(-(r / L)),
        S=lambda r: 0.8 + 0.2 * (r / L),
        P=lambda r: 0.1 * np.sin(0.5 * r / L),
        Q=lambda r: -0.08 * (r / L),
        t_B=lambda r: 0.0, Lambda=0.0, epsilon=1,
        r_min=0.3 * L, r_max=4.0 * L, n_r=140, n_t=300,
        t_min=2.0 * T, t_max=20.0 * T,
    )


# ----------------------------------------------------------------------
#  1.  Christoffel vs finite difference of the metric
# ----------------------------------------------------------------------
def _christoffel_fd(met, x):
    h = np.array([1e-6 * T, 1e-6 * L, 1e-6, 1e-6])
    g = met.metric_tensor(x)
    ginv = np.linalg.inv(g)
    dg = np.zeros((4, 4, 4))                  # dg[mu, a, b] = d_mu g_ab
    for mu in range(4):
        xp = x.copy(); xp[mu] += h[mu]
        xm = x.copy(); xm[mu] -= h[mu]
        dg[mu] = (met.metric_tensor(xp) - met.metric_tensor(xm)) / (2 * h[mu])
    G3 = np.zeros((4, 4, 4))
    for a in range(4):
        for b in range(4):
            for cc in range(4):
                s = sum(ginv[a, d] * (dg[b, d, cc] + dg[cc, d, b] - dg[d, b, cc])
                        for d in range(4))
                G3[a, b, cc] = 0.5 * s
    return G3


def test_christoffel_matches_finite_difference():
    met = SzekeresMetric(_dipole_model())
    rng = np.random.default_rng(1)
    max_err = 0.0
    for _ in range(10):
        x = np.array([rng.uniform(4, 18) * T, rng.uniform(0.6, 3.5) * L,
                      rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4)])
        Ga = met.christoffel(x)
        Gf = _christoffel_fd(met, x)
        scale = np.abs(Ga).max() + 1e-300
        max_err = max(max_err, np.abs(Ga - Gf).max() / scale)
    assert max_err < 1e-4, f"Christoffel vs FD rel-err {max_err:.2e}"
    print(f"[ok] Christoffel = FD(metric)  (max rel-err {max_err:.2e})")


def test_geodesic_rhs_matches_paper_eqs_7_10():
    """The Christoffel-assembled RHS must equal Celerier (2024) eqs. 7-10."""
    met = SzekeresMetric(_dipole_model())
    rng = np.random.default_rng(3)
    max_err = 0.0
    for _ in range(12):
        x = np.array([rng.uniform(4, 18) * T, rng.uniform(0.6, 3.5) * L,
                      rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4)])
        k_spatial = [rng.uniform(-1, 1), 2e-26 * rng.uniform(-1, 1),
                     2e-26 * rng.uniform(-1, 1)]
        kt = met.solve_null_kt(x, k_spatial)
        state = np.concatenate([x, [kt, *k_spatial]])
        a = met.geodesic_equations(state)[4:]
        b = met.geodesic_rhs_paper(state)[4:]
        scale = np.abs(a).max() + 1e-300
        max_err = max(max_err, np.abs(a - b).max() / scale)
    assert max_err < 1e-10, f"RHS vs paper eqs 7-10 rel-err {max_err:.2e}"
    print(f"[ok] geodesic RHS = Celerier eqs 7-10  (max rel-err {max_err:.2e})")


# ----------------------------------------------------------------------
#  helper: simple fixed-step RK4 ray tracer
# ----------------------------------------------------------------------
def _trace(met, x0, k_spatial, *, n_steps=4000, span_t=10.0 * T,
           stop=None, record=None):
    kt = met.solve_null_kt(x0, k_spatial)         # past-pointing root (k^t < 0)
    state = np.concatenate([x0, [kt, *k_spatial]])
    rk = RK4()
    ds = abs(span_t / (kt * n_steps))
    out = []
    for _ in range(n_steps):
        state, _, _ = rk.step(met, state, ds)
        if stop is not None and stop(state):
            break
        if record is not None:
            record(state, out)
    return state, kt, out


# ----------------------------------------------------------------------
#  2.  Null constraint preservation
# ----------------------------------------------------------------------
def test_null_preserved_radial_eds():
    met = SzekeresMetric(_eds_radial_model())
    x0 = np.array([18.0 * T, 3.5 * L, 0.0, 0.0])
    worst = {"v": 0.0}

    def rec(state, _):
        worst["v"] = max(worst["v"], met.null_constraint_relative(state))

    def stop(state):
        return state[0] < 2.2 * T or state[1] < 0.35 * L or state[1] > 3.9 * L

    _trace(met, x0, [-1.0, 0.0, 0.0], stop=stop, record=rec)
    assert worst["v"] < 1e-10, f"radial null drift {worst['v']:.2e}"
    print(f"[ok] null preserved (radial EdS): max rel {worst['v']:.2e}")


def test_null_preserved_dipole_nonradial():
    met = SzekeresMetric(_dipole_model())
    x0 = np.array([17.0 * T, 3.0 * L, 0.15, -0.1])
    # g_pp ~ Phi^2 is enormous, so transverse momenta must be tiny to stay null.
    k_spatial = [-1.0, 2e-26, -1.5e-26]
    worst = {"v": 0.0}

    def rec(state, _):
        worst["v"] = max(worst["v"], met.null_constraint_relative(state))

    def stop(state):
        return state[0] < 2.2 * T or state[1] < 0.35 * L or state[1] > 3.9 * L

    _trace(met, x0, k_spatial, stop=stop, record=rec)
    assert worst["v"] < 1e-5, f"dipole null drift {worst['v']:.2e}"
    print(f"[ok] null preserved (dipole, non-radial): max rel {worst['v']:.2e}")


# ----------------------------------------------------------------------
#  3.  EdS redshift  1 + z = (t_obs / t_emit)^{2/3}
# ----------------------------------------------------------------------
def test_eds_radial_redshift():
    met = SzekeresMetric(_eds_radial_model())
    t_obs = 18.0 * T
    x0 = np.array([t_obs, 3.5 * L, 0.0, 0.0])

    res = {"max": 0.0, "zmax": 0.0}

    def rec(state, _):
        t = state[0]
        z_num = state[4] / kt_obs - 1.0
        z_eds = (t_obs / t) ** (2.0 / 3.0) - 1.0
        if z_eds > 1e-2:
            res["max"] = max(res["max"], abs(z_num - z_eds) / z_eds)
        res["zmax"] = max(res["zmax"], z_eds)

    def stop(state):
        return state[0] < 2.2 * T or state[1] < 0.35 * L or state[1] > 3.9 * L

    kt_obs = met.solve_null_kt(x0, [-1.0, 0.0, 0.0])
    _trace(met, x0, [-1.0, 0.0, 0.0], stop=stop, record=rec)

    assert res["zmax"] > 1.0, f"ray did not reach cosmological z (zmax={res['zmax']:.2f})"
    assert res["max"] < 1e-3, f"redshift rel-err {res['max']:.2e}"
    print(f"[ok] EdS redshift 1+z=(t_o/t_e)^(2/3) up to z={res['zmax']:.2f}  "
          f"(max rel-err {res['max']:.2e})")


def test_bondi_redshift_matches_kt_method():
    """Bondi wave-crest ODE (eq. 20) must agree with the k^t-ratio redshift."""
    met = SzekeresMetric(_eds_radial_model())
    t_obs = 18.0 * T
    x0 = np.array([t_obs, 3.5 * L, 0.0, 0.0])
    kt0 = met.solve_null_kt(x0, [-1.0, 0.0, 0.0])
    state = np.concatenate([x0, [kt0, -1.0, 0.0, 0.0]])

    rk = RK4()
    ds = abs(10.0 * T / (kt0 * 5000))
    lnz = 0.0
    worst = 0.0
    z_kt = 0.0
    for _ in range(5000):
        lnz += met.bondi_dlnz_ds(state) * ds          # forward accumulation
        state, _, _ = rk.step(met, state, ds)
        if state[0] < 2.2 * T or state[1] < 0.35 * L or state[1] > 3.9 * L:
            break
        z_bondi = np.exp(lnz) - 1.0
        z_kt = state[4] / kt0 - 1.0
        if z_kt > 1e-2:
            worst = max(worst, abs(z_bondi - z_kt) / z_kt)
    assert z_kt > 1.0, f"ray did not reach cosmological z (z={z_kt:.2f})"
    assert worst < 2e-3, f"Bondi vs k^t redshift rel-err {worst:.2e}"
    print(f"[ok] Bondi redshift = k^t method up to z={z_kt:.2f}  "
          f"(max rel-err {worst:.2e})")


if __name__ == "__main__":
    test_christoffel_matches_finite_difference()
    test_geodesic_rhs_matches_paper_eqs_7_10()
    test_null_preserved_radial_eds()
    test_null_preserved_dipole_nonradial()
    test_eds_radial_redshift()
    test_bondi_redshift_matches_kt_method()
    print("\nAll Szekeres geodesic tests passed.")
