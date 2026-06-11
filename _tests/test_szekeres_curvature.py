#!/usr/bin/env python3
r"""
Validation of :mod:`excalibur.observables.riemann_szekeres`.

Covered (brief sec. 6, tests 6, 8):
    * analytic Ricci focusing (Appendix A, regenerated symbolically) equals the
      exact dust result  ``R_{ab} k^a k^b = 8 pi G rho (k^t)^2``  for a null k;
    * analytic Ricci & screen tidal tensor agree with the independent numerical
      Riemann (finite-difference of the analytic Christoffels);
    * the numerical Riemann itself reproduces the Ricci focusing.

The Szekeres curvature is **badly conditioned in SI** (the unnormalised scale
factor a ~ 1e11 makes H, F differ by ~1e26, so the H,F-form curvature loses all
precision to catastrophic cancellation).  We therefore force ``EXCALIBUR_UNITS``
to cosmo (c, a ~ O(1)) *before* importing excalibur, and use an O(1) model.

Run directly:  ``python _tests/test_szekeres_curvature.py``
"""
import os
import sys

# Must be set before importing excalibur.core.constants.
os.environ.setdefault("EXCALIBUR_UNITS", "cosmo")

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from excalibur.core.constants import G, c, unit_system
from excalibur.core.szekeres_model import SzekeresModel
from excalibur.metrics.szekeres_metric import SzekeresMetric
from excalibur.observables import riemann_szekeres as rs

assert c < 1e3, (
    f"Szekeres curvature tests need a well-conditioned unit system (c~O(1)); "
    f"got c={c:.3e} (unit_system={unit_system!r}).  Run standalone so "
    f"EXCALIBUR_UNITS=cosmo is picked up at import."
)

M0 = 2.0 / (9.0 * G)


def _eds():
    fd = {"dM": lambda r: 3.0 * M0 * r * r, "dk": lambda r: 0.0,
          "dS": lambda r: 0.0, "dP": lambda r: 0.0, "dQ": lambda r: 0.0,
          "dt_B": lambda r: 0.0, "ddS": lambda r: 0.0,
          "ddP": lambda r: 0.0, "ddQ": lambda r: 0.0}
    return SzekeresModel(M=lambda r: M0 * r ** 3, k=lambda r: 0.0,
                         S=lambda r: 1.0, P=lambda r: 0.0, Q=lambda r: 0.0,
                         t_B=lambda r: 0.0, Lambda=0.0, epsilon=1,
                         r_min=0.3, r_max=4.0, n_r=160, n_t=350,
                         t_min=2.0, t_max=20.0, free_derivs=fd)


def _dipole():
    return SzekeresModel(
        M=lambda r: M0 * r ** 3 * (1.0 + 0.15 * np.sin(r)),
        k=lambda r: 0.05 * r ** 2 * np.exp(-r),
        S=lambda r: 0.8 + 0.2 * r,
        P=lambda r: 0.1 * np.sin(0.5 * r),
        Q=lambda r: -0.08 * r,
        t_B=lambda r: 0.0, Lambda=0.0, epsilon=1,
        r_min=0.3, r_max=4.0, n_r=160, n_t=350, t_min=2.0, t_max=20.0)


def _samples(met, n, rng):
    out = []
    for _ in range(n):
        x = np.array([rng.uniform(6, 15), rng.uniform(0.9, 3.0),
                      rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3)])
        ks = [rng.uniform(-1, 1), rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3)]
        kt = met.solve_null_kt(x, ks)
        out.append((x, np.array([kt, *ks])))
    return out


def test_analytic_ricci_focusing_equals_dust_source():
    for label, model in [("EdS", _eds()), ("dipole", _dipole())]:
        met = SzekeresMetric(model)
        err = 0.0
        for x, k in _samples(met, 8, np.random.default_rng(5)):
            an = rs.ricci_focusing_appendix(model, x, k)
            ex = rs.ricci_focusing_analytic(model, x, k[0])   # 8 pi G rho (k^t)^2
            err = max(err, abs(an - ex) / abs(ex))
        assert err < 1e-3, f"{label}: analytic Ricci focusing vs 8piGrho {err:.2e}"
        print(f"[ok] analytic Ricci focusing = 8 pi G rho (k^t)^2  ({label}, {err:.2e})")


def test_numerical_ricci_focusing_equals_dust_source():
    model = _dipole()
    met = SzekeresMetric(model)
    err = 0.0
    for x, k in _samples(met, 8, np.random.default_rng(7)):
        num = rs.ricci_focusing(met, x, k)
        ex = rs.ricci_focusing_analytic(model, x, k[0])
        err = max(err, abs(num - ex) / abs(ex))
    assert err < 1e-2, f"numerical Ricci focusing vs 8piGrho {err:.2e}"
    print(f"[ok] numerical Ricci focusing = 8 pi G rho (k^t)^2  (dipole, {err:.2e})")


def test_analytic_tidal_tensor_matches_numerical_riemann():
    model = _dipole()
    met = SzekeresMetric(model)
    conv = np.array([1.0 / c, 1.0, 1.0, 1.0])     # cosmo -> geometric (t index)
    err = 0.0
    for x, k in _samples(met, 8, np.random.default_rng(9)):
        Tan = rs.tidal_tensor_analytic(model, x, k)
        Rlow = rs.riemann_tensor_lower(met, x)
        Tnum = np.einsum('manb,a,b->mn', Rlow, k, k) * conv[:, None] * conv[None, :]
        err = max(err, np.abs(Tan - Tnum).max() / (np.abs(Tan).max() + 1e-300))
    assert err < 1e-2, f"analytic vs numerical tidal tensor {err:.2e}"
    print(f"[ok] analytic tidal tensor = numerical Riemann  (dipole, {err:.2e})")


if __name__ == "__main__":
    test_analytic_ricci_focusing_equals_dust_source()
    test_numerical_ricci_focusing_equals_dust_source()
    test_analytic_tidal_tensor_matches_numerical_riemann()
    print("\nAll Szekeres curvature tests passed.")
