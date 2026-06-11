#!/usr/bin/env python3
r"""
Validation of the Numba (Phase-2) Szekeres geodesic backend.

Checks (brief sec. 4.3: "validate Phase 2 == Phase 1 before optimising further"):
    * the compiled geodesic reproduces the readable-Python trajectory and
      redshift to interpolation precision (~1e-5);
    * the Einstein--de Sitter redshift closed form is recovered;
    * a coarse speed sanity check (compiled run is far faster than Phyton RK4).

Run standalone (cosmo units):  ``python _tests/test_szekeres_fast.py``
"""
import os
import sys
import time

os.environ.setdefault("EXCALIBUR_UNITS", "cosmo")

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from excalibur.core.constants import G, c
from excalibur.core.szekeres_model import SzekeresModel
from excalibur.metrics.szekeres_metric import SzekeresMetric
from excalibur.metrics.szekeres_metric_fast import (
    FastSzekeres, integrate_geodesic_fast, integrate_distance_fast)
from excalibur.observables.szekeres_distances import integrate_area_distance
from excalibur.integration.integrator import RK4

M0 = 2.0 / (9.0 * G)


def _eds():
    fd = {"dM": lambda r: 3.0 * M0 * r * r, "dk": lambda r: 0.0,
          "dS": lambda r: 0.0, "dP": lambda r: 0.0, "dQ": lambda r: 0.0,
          "dt_B": lambda r: 0.0, "ddS": lambda r: 0.0,
          "ddP": lambda r: 0.0, "ddQ": lambda r: 0.0}
    return SzekeresModel(M=lambda r: M0 * r ** 3, k=lambda r: 0.0,
                         S=lambda r: 1.0, P=lambda r: 0.0, Q=lambda r: 0.0,
                         t_B=lambda r: 0.0, Lambda=0.0, epsilon=1,
                         r_min=0.2, r_max=4.5, n_r=220, n_t=440,
                         t_min=1.0, t_max=22.0, free_derivs=fd)


def _dipole():
    return SzekeresModel(
        M=lambda r: M0 * r ** 3 * (1.0 + 0.2 * np.exp(-((r - 2.0) / 0.8) ** 2)),
        k=lambda r: 0.04 * r ** 2 * np.exp(-((r - 2.0) / 0.8) ** 2),
        S=lambda r: 0.9 + 0.1 * r, P=lambda r: 0.15 * np.sin(0.6 * r),
        Q=lambda r: -0.1 * r, t_B=lambda r: 0.0, Lambda=0.0, epsilon=1,
        r_min=0.2, r_max=4.5, n_r=220, n_t=440, t_min=1.0, t_max=22.0)


def _phase1_traj(model, x0, ksp, n_steps, span_t):
    met = SzekeresMetric(model)
    kt = met.solve_null_kt(np.asarray(x0, float), ksp)
    st = np.concatenate([x0, [kt, *ksp]])
    rk = RK4()
    ds = abs(span_t / (kt * n_steps))
    rec = []
    for _ in range(n_steps):
        st, _, _ = rk.step(met, st, ds)
        if st[0] < 1.3 or st[1] < 0.25 or st[1] > 4.4:
            break
        rec.append((st[0], st[1], st[4] / kt - 1.0))   # t, r, z
    return np.array(rec)


def test_fast_matches_phase1():
    model = _dipole()
    x0, ksp = [18.0, 3.8, 0.0, 0.0], [-1.0, 0.02, -0.015]
    p1 = _phase1_traj(model, x0, ksp, 6000, 11.0)
    fast = FastSzekeres(model)
    f = integrate_geodesic_fast(fast, x0, ksp, n_steps=6000, span_t=11.0)

    zc = min(p1[-1, 2], f["z"][-1]) - 0.05
    m = (p1[:, 2] > 0.05) & (p1[:, 2] < zc)
    r_fast = np.interp(p1[m, 2], f["z"], f["r"])
    t_fast = np.interp(p1[m, 2], f["z"], f["t"])
    dr = np.max(np.abs(r_fast - p1[m, 1]))
    dt = np.max(np.abs(t_fast - p1[m, 0]))
    assert dr < 1e-4 and dt < 1e-3, f"Phase2 vs Phase1: dr={dr:.2e}, dt={dt:.2e}"
    print(f"[ok] Numba geodesic == Phase-1 trajectory: max|dr|={dr:.2e}, max|dt|={dt:.2e}")


def test_fast_eds_redshift():
    model = _eds()
    fast = FastSzekeres(model)
    f = integrate_geodesic_fast(fast, [18.0, 3.8, 0, 0], [-1.0, 0, 0],
                                n_steps=6000, span_t=11.0)
    z, t = f["z"], f["t"]
    m = z > 0.05
    z_exact = (18.0 / t) ** (2.0 / 3.0) - 1.0
    rel = np.max(np.abs(z[m] - z_exact[m]) / z_exact[m])
    assert z[-1] > 1.5 and rel < 1e-3, f"EdS redshift rel-err {rel:.2e}"
    print(f"[ok] Numba EdS redshift = (t_o/t_e)^(2/3) up to z={z[-1]:.2f} "
          f"(max rel-err {rel:.2e})")


def test_fast_is_faster():
    model = _eds()
    x0, ksp = [18.0, 3.8, 0, 0], [-1.0, 0, 0]
    fast = FastSzekeres(model)
    integrate_geodesic_fast(fast, x0, ksp, n_steps=2000, span_t=11.0)  # JIT warmup
    t0 = time.time()
    integrate_geodesic_fast(fast, x0, ksp, n_steps=6000, span_t=11.0)
    t_fast = time.time() - t0
    t0 = time.time()
    _phase1_traj(model, x0, ksp, 1500, 11.0)       # quarter-length Python ref
    t_p1_quarter = time.time() - t0
    speedup = (4.0 * t_p1_quarter) / max(t_fast, 1e-6)
    assert speedup > 20.0, f"speedup only x{speedup:.0f}"
    print(f"[ok] Numba backend ~x{speedup:.0f} faster than Python RK4 "
          f"(fast {t_fast*1e3:.1f}ms / 6000 steps)")


def test_fast_distance_matches_phase1():
    """The 24-comp Numba distance path reproduces Phase-1 D_A (incl. dipole)."""
    model = _dipole()
    x0, ksp = [18.0, 3.8, 0, 0], [-1.0, 0.02, -0.015]
    fast = FastSzekeres(model)
    rf = integrate_distance_fast(fast, x0, ksp, n_steps=6000, span_t=11.0)
    r1 = integrate_area_distance(SzekeresMetric(model), model, x0, ksp,
                                 n_steps=6000, span_t=11.0,
                                 stop=lambda s: s[0] < 1.3 or s[1] < 0.25 or s[1] > 4.4)
    zc = min(rf["z"][-1], r1["z"][-1]) - 0.05
    m = (r1["z"] > 0.05) & (r1["z"] < zc)
    DA_f = np.interp(r1["z"][m], rf["z"], rf["D_A"])
    rel = np.max(np.abs(DA_f - r1["D_A"][m]) / r1["D_A"][m])
    assert rel < 1e-4, f"fast vs Phase1 D_A {rel:.2e}"
    print(f"[ok] Numba D_A == Phase-1 (dipole): max rel-err {rel:.2e}")


def test_fast_eds_distance_oracle():
    """Numba D_A matches the EdS closed-form oracle a(t_e)|Delta r|."""
    model = _eds()
    fast = FastSzekeres(model)
    rf = integrate_distance_fast(fast, [18.0, 3.8, 0, 0], [-1.0, 0, 0],
                                 n_steps=6000, span_t=11.0)
    oracle = rf["t"] ** (2.0 / 3.0) * np.abs(3.8 - rf["r"])
    m = rf["z"] > 0.05
    rel = np.max(np.abs(rf["D_A"][m] - oracle[m]) / oracle[m])
    # reciprocity D_L = (1+z)^2 D_A
    relL = np.max(np.abs(rf["D_L"][m] - (1 + rf["z"][m]) ** 2 * rf["D_A"][m])
                  / rf["D_L"][m])
    assert rel < 1e-3 and relL < 1e-12
    print(f"[ok] Numba EdS D_A = a(t_e)|Delta r| (rel {rel:.2e}), "
          f"D_L=(1+z)^2 D_A (rel {relL:.2e})")


if __name__ == "__main__":
    test_fast_matches_phase1()
    test_fast_eds_redshift()
    test_fast_distance_matches_phase1()
    test_fast_eds_distance_oracle()
    test_fast_is_faster()
    print("\nAll Szekeres fast-backend tests passed.")
