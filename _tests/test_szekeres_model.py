#!/usr/bin/env python3
r"""
Validation of :mod:`excalibur.core.szekeres_model` (QSS background + dipole).

Strategy (brief sec. 6, tests 1, 6, 8):
    * E-derivative closed forms vs finite differences of E.
    * Einstein--de Sitter closed-form oracle (flat dust, k=0, Lambda=0):
          Phi(t, r)   = r t^{2/3}            (with M0 = 2/(9 G))
          Phi_,t      = (2/3) r t^{-1/3}
          rho(t)      = 1 / (6 pi G t^2)
      which exercises the integral inversion, the spline derivatives and the
      Friedmann cross-check Phi_,t == W, plus the (G-free) density equation.

Run directly:  ``python _tests/test_szekeres_model.py``
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from excalibur.core.constants import G
from excalibur.core.szekeres_model import SzekeresModel


# ----------------------------------------------------------------------
#  1.  E and its derivatives (closed form vs finite differences)
# ----------------------------------------------------------------------
def test_E_derivatives_match_finite_differences():
    rng = np.random.default_rng(0)
    # Non-trivial dipole: S, P, Q vary with r.
    model = SzekeresModel(
        M=lambda r: 1.0 * r ** 3,
        k=lambda r: 0.0,
        S=lambda r: 0.7 + 0.3 * r + 0.1 * r * r,
        P=lambda r: 0.2 * r - 0.05 * r * r,
        Q=lambda r: -0.1 * r + 0.04 * r * r,
        t_B=lambda r: 0.0,
        Lambda=0.0,
        epsilon=1,
        r_min=0.05, r_max=1.0, n_r=60, n_t=60,
        t_min=0.3, t_max=2.0,
    )

    h = 1e-6
    for _ in range(20):
        r = rng.uniform(0.2, 0.9)
        p = rng.uniform(-0.5, 0.5)
        q = rng.uniform(-0.5, 0.5)

        # E_,p, E_,q
        Ep_fd = (model.E(r, p + h, q) - model.E(r, p - h, q)) / (2 * h)
        Eq_fd = (model.E(r, p, q + h) - model.E(r, p, q - h)) / (2 * h)
        assert abs(model.E_p(r, p, q) - Ep_fd) < 1e-6
        assert abs(model.E_q(r, p, q) - Eq_fd) < 1e-6

        # E_,r
        Er_fd = (model.E(r + h, p, q) - model.E(r - h, p, q)) / (2 * h)
        assert abs(model.E_r(r, p, q) - Er_fd) < 1e-5

        # E_,rp, E_,rq  (FD of analytic E_,p / E_,q in r)
        Erp_fd = (model.E_p(r + h, p, q) - model.E_p(r - h, p, q)) / (2 * h)
        Erq_fd = (model.E_q(r + h, p, q) - model.E_q(r - h, p, q)) / (2 * h)
        assert abs(model.E_rp(r, p, q) - Erp_fd) < 1e-5
        assert abs(model.E_rq(r, p, q) - Erq_fd) < 1e-5

        # E_,rr  (second FD of E in r)
        Err_fd = (model.E(r + h, p, q) - 2 * model.E(r, p, q)
                  + model.E(r - h, p, q)) / (h * h)
        assert abs(model.E_rr(r, p, q) - Err_fd) < 1e-3
    print("[ok] E derivatives match finite differences")


# ----------------------------------------------------------------------
#  Einstein--de Sitter Szekeres model (dipole off) -> closed form
# ----------------------------------------------------------------------
def _eds_model():
    M0 = 2.0 / (9.0 * G)             # makes Phi = r t^{2/3} exactly
    return SzekeresModel(
        M=lambda r: M0 * r ** 3,
        k=lambda r: 0.0,
        S=lambda r: 1.0,
        P=lambda r: 0.0,
        Q=lambda r: 0.0,
        t_B=lambda r: 0.0,
        Lambda=0.0,
        epsilon=1,
        r_min=0.05, r_max=1.1, n_r=200, n_t=500,
        t_min=0.3, t_max=6.0,
        free_derivs={
            "dM": lambda r: 3.0 * M0 * r * r, "dk": lambda r: 0.0,
            "dS": lambda r: 0.0, "dP": lambda r: 0.0, "dQ": lambda r: 0.0,
            "dt_B": lambda r: 0.0, "ddS": lambda r: 0.0,
            "ddP": lambda r: 0.0, "ddQ": lambda r: 0.0,
        },
    )


def test_eds_areal_radius_closed_form():
    model = _eds_model()
    ts = np.linspace(0.6, 5.0, 9)
    rs = np.linspace(0.15, 1.0, 7)
    max_rel = 0.0
    for t in ts:
        for r in rs:
            exact = r * t ** (2.0 / 3.0)
            got = model.Phi(t, r)
            max_rel = max(max_rel, abs(got - exact) / exact)
    # Phi=u^2 substitution + Simpson + PCHIP inversion -> ~1e-10 (was ~1e-5).
    assert max_rel < 1e-6, f"Phi rel-err {max_rel:.2e}"
    print(f"[ok] EdS Phi = r t^(2/3)  (max rel-err {max_rel:.2e})")


def test_eds_phi_t_matches_closed_form():
    model = _eds_model()
    # Phi_,t = W(Phi) is taken exactly from Friedmann; this checks Phi accuracy.
    ts = np.linspace(0.6, 5.0, 9)
    rs = np.linspace(0.15, 1.0, 7)
    max_rel = 0.0
    for t in ts:
        for r in rs:
            exact = (2.0 / 3.0) * r * t ** (-1.0 / 3.0)
            got = model.Phi_t(t, r)
            max_rel = max(max_rel, abs(got - exact) / abs(exact))
    assert max_rel < 5e-3, f"Phi_t vs closed form {max_rel:.2e}"
    print(f"[ok] EdS Phi_,t = (2/3) r t^(-1/3)  (max rel-err {max_rel:.2e})")


def test_eds_phi_r_analytic_and_crosscheck():
    model = _eds_model()
    # (a) analytic Phi_,r vs closed form t^{2/3}
    # (b) analytic Phi_,r vs an *independent* estimate: the raw r-derivative of
    #     the Phi table spline (validates the I_r machinery against the table).
    ts = np.linspace(0.6, 5.0, 9)
    rs = np.linspace(0.2, 1.0, 7)
    max_cf = 0.0
    max_cross = 0.0
    for t in ts:
        for r in rs:
            exact = t ** (2.0 / 3.0)
            analytic = model.Phi_r(t, r)
            raw_spline = float(model._phi_spline(t, r, dx=0, dy=1)[0, 0])
            max_cf = max(max_cf, abs(analytic - exact) / exact)
            max_cross = max(max_cross, abs(analytic - raw_spline) / exact)
    assert max_cf < 5e-3, f"Phi_,r vs closed form {max_cf:.2e}"
    assert max_cross < 5e-3, f"analytic vs raw-spline Phi_,r {max_cross:.2e}"
    print(f"[ok] EdS Phi_,r = t^(2/3)  (vs closed form {max_cf:.2e}, "
          f"vs raw spline {max_cross:.2e})")


def test_eds_homogeneity_and_radial_curvature():
    model = _eds_model()
    ts = np.linspace(0.6, 5.0, 6)
    rs = np.linspace(0.2, 1.0, 6)
    for t in ts:
        # Phi/r must be r-independent (homogeneity); Phi_,rr ~ 0 (linear in r).
        a_vals = np.array([model.Phi(t, r) / r for r in rs])
        assert np.ptp(a_vals) / np.mean(a_vals) < 3e-3
        for r in rs:
            assert abs(model.Phi_rr(t, r)) < 1e-3 * t ** (2.0 / 3.0)
    print("[ok] EdS homogeneity: Phi/r is r-independent, Phi_,rr ~ 0")


def test_eds_density_closed_form():
    model = _eds_model()
    # EdS dust density: rho = 1/(6 pi G t^2), independent of r and (p,q).
    ts = np.linspace(0.8, 5.0, 7)
    rs = np.linspace(0.2, 1.0, 5)
    max_rel = 0.0
    for t in ts:
        rho_exact = 1.0 / (6.0 * np.pi * G * t * t)
        for r in rs:
            got = model.rho(t, r, 0.1, -0.2)
            max_rel = max(max_rel, abs(got - rho_exact) / rho_exact)
    assert max_rel < 1e-2, f"rho rel-err {max_rel:.2e}"
    print(f"[ok] EdS density rho = 1/(6 pi G t^2)  (max rel-err {max_rel:.2e})")


if __name__ == "__main__":
    test_E_derivatives_match_finite_differences()
    test_eds_areal_radius_closed_form()
    test_eds_phi_t_matches_closed_form()
    test_eds_phi_r_analytic_and_crosscheck()
    test_eds_homogeneity_and_radial_curvature()
    test_eds_density_closed_form()
    print("\nAll Szekeres model tests passed.")
