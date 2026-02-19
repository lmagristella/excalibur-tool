#!/usr/bin/env python3
"""
test_schwarzschild_fast.py

Goal:
- Verify that trajectory outputs from SchwarzschildMetric (reference)
  and SchwarzschildMetricFast (optimized) are numerically identical (within tolerance)
- Measure speedup.

Assumptions:
- Both metrics expose .geodesic_equations(state)->(8,) derivative in INTERNAL spherical form
  when coords="spherical".
- Your reference class lives at metrics/schwarzschild_metric.py
- Your fast class lives at metrics/schwarzschild_metric_fast.py (or adjust import below)

Run:
  python test_schwarzschild_fast.py
"""

from __future__ import annotations

import time
import numpy as np

from excalibur.core.constants import *
from excalibur.metrics.schwarzschild_metric import SchwarzschildMetric
from excalibur.metrics.schwarzschild_metric_fast import SchwarzschildMetricFast


def rk4_step(metric, y: np.ndarray, h: float) -> np.ndarray:
    """Classic RK4 for y' = f(y)."""
    k1 = metric.geodesic_equations(y)
    k2 = metric.geodesic_equations(y + 0.5 * h * k1)
    k3 = metric.geodesic_equations(y + 0.5 * h * k2)
    k4 = metric.geodesic_equations(y + h * k3)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate(metric, y0: np.ndarray, h: float, n_steps: int) -> np.ndarray:
    """Return trajectory array shape (n_steps+1, 8)."""
    traj = np.empty((n_steps + 1, y0.size), dtype=float)
    traj[0] = y0
    y = y0.copy()
    for i in range(1, n_steps + 1):
        y = rk4_step(metric, y, h)
        traj[i] = y
    return traj


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def rel_err(a: np.ndarray, b: np.ndarray, eps: float = 1e-30) -> float:
    denom = np.maximum(np.abs(a), np.abs(b))
    denom = np.maximum(denom, eps)
    return float(np.max(np.abs(a - b) / denom))


def make_initial_state_spherical(
    *,
    r: float,
    theta: float,
    phi: float,
    dtdl: float,
    drdl: float,
    dthetadl: float,
    dphidl: float,
) -> np.ndarray:
    """
    State convention: (t, r, theta, phi, dt/dλ, dr/dλ, dθ/dλ, dφ/dλ)
    """
    return np.array([0.0, r, theta, phi, dtdl, drdl, dthetadl, dphidl], dtype=float)


def run_correctness_check(
    ref_metric,
    fast_metric,
    y0: np.ndarray,
    h: float,
    n_steps: int,
    atol: float = 1e-10,
    rtol: float = 1e-10,
) -> None:
    traj_ref = integrate(ref_metric, y0, h, n_steps)
    traj_fast = integrate(fast_metric, y0, h, n_steps)

    abs_max = max_abs_diff(traj_ref, traj_fast)
    rel_max = rel_err(traj_ref, traj_fast)

    ok = (abs_max <= atol) or (rel_max <= rtol)

    print("=== Correctness ===")
    print(f"steps={n_steps}  h={h:g}")
    print(f"max |Δ|     = {abs_max:.3e}")
    print(f"max rel err = {rel_max:.3e}")
    print(f"PASS? {ok}")

    if not ok:
        # Show where it diverges most
        idx = np.unravel_index(np.argmax(np.abs(traj_ref - traj_fast)), traj_ref.shape)
        i, j = idx
        labels = ["t", "r", "theta", "phi", "u0", "u1", "u2", "u3"]
        print("\nWorst component:")
        print(f" step={i}, comp={labels[j]}")
        print(f" ref = {traj_ref[i, j]:.16e}")
        print(f" fast= {traj_fast[i, j]:.16e}")
        print(f" diff= {traj_ref[i, j] - traj_fast[i, j]:.16e}")


def bench(
    metric,
    y0: np.ndarray,
    h: float,
    n_steps: int,
    n_runs: int,
    warmup: int = 2,
) -> float:
    """
    Return median runtime (seconds) over n_runs after warmup.
    """
    # warmup (important if numba is involved)
    for _ in range(warmup):
        integrate(metric, y0, h, max(10, n_steps // 10))

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        integrate(metric, y0, h, n_steps)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return float(np.median(times))


def main():
    # ------------------ Config ------------------
    # Pick parameters that are safe (not too close to horizon, not at poles)
    mass = 1.0e15  # kg? (match whatever your code expects; keep consistent for both)
    radius = 3 * one_pc  # same unit convention as your SchwarzschildMetric
    center = np.array([0.0, 0.0, 0.0], dtype=float)

    # Integration parameters
    n_steps = 40000
    h = 1e-3
    n_runs = 7

    # Correctness tolerances (tighten if you expect bitwise identity)
    atol = 1e-10
    rtol = 1e-10

    # ------------------ Metrics ------------------
    # Keep settings identical (critical for dt/dλ handling)
    analytical_geodesics = False
    free_time_geodesic = True

    ref = SchwarzschildMetric(
        mass=mass,
        radius=radius,
        center=center,
        analytical_geodesics=analytical_geodesics,
        free_time_geodesic=free_time_geodesic,
        coords="spherical",
    )
    fast = SchwarzschildMetricFast(
        mass=mass,
        radius=radius,
        center=center,
        analytical_geodesics=analytical_geodesics,
        free_time_geodesic=free_time_geodesic,
        coords="spherical",
    )

    # ------------------ Initial state ------------------
    # Spherical initial state: choose a mildly non-trivial orbit/bending case.
    # You can also create multiple y0 and loop them.
    y0 = make_initial_state_spherical(
        r=20.0,                 # far enough from rs for stability
        theta=1.1,              # avoid poles
        phi=0.2,
        dtdl=1.0,               # keep same convention for both
        drdl=-0.02,
        dthetadl=0.0,
        dphidl=0.03,
    )

    # ------------------ Correctness ------------------
    run_correctness_check(ref, fast, y0, h, n_steps, atol=atol, rtol=rtol)

    # ------------------ Benchmark ------------------
    t_ref = bench(ref, y0, h, n_steps, n_runs=n_runs)
    t_fast = bench(fast, y0, h, n_steps, n_runs=n_runs)

    speedup = t_ref / t_fast if t_fast > 0 else np.inf

    print("\n=== Speed ===")
    print(f"reference median: {t_ref:.6f} s")
    print(f"fast      median: {t_fast:.6f} s")
    print(f"speedup         : {speedup:.2f}×")

    # Optional: sanity check final state only (useful when you scale to many photons)
    traj_ref = integrate(ref, y0, h, n_steps)
    traj_fast = integrate(fast, y0, h, n_steps)
    print("\nFinal state diff (abs max):", max_abs_diff(traj_ref[-1], traj_fast[-1]))


if __name__ == "__main__":
    main()
