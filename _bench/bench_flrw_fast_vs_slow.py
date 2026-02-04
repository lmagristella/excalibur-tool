"""Micro-benchmark: FLRW fast vs slow metric RHS.

Goal
----
Measure the wall time spent evaluating the geodesic RHS (8D) for the same state.
This isolates the acceleration computation + interpolator overhead.

Notes
-----
- This script is intentionally tiny and prints a median over multiple reps.
- First call includes numba compilation for the fast metric; we do an explicit warmup.
"""

from __future__ import annotations

import time
import numpy as np

from excalibur.core.constants import c, one_Mpc
from excalibur.core.cosmology import LCDM_Cosmology
from excalibur.grid.grid import Grid
from excalibur.grid.interpolator_fast import InterpolatorFast
from excalibur.metrics.perturbed_flrw_metric import PerturbedFLRWMetric
from excalibur.metrics.perturbed_flrw_metric_fast import PerturbedFLRWMetricFast
from excalibur.objects.spherical_mass import spherical_mass


def _median_time(fn, reps: int) -> float:
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def main() -> None:
    cosmology = LCDM_Cosmology(70, Omega_m=0.3, Omega_r=0.0, Omega_lambda=0.7)
    _ = cosmology.a_of_eta(1e18)
    eta = cosmology._eta_at_a1

    N = 32
    grid_size = 50 * one_Mpc
    spacing = (grid_size / N, grid_size / N, grid_size / N)
    origin3 = np.array([0.0, 0.0, 0.0])

    grid = Grid((N, N, N), spacing, origin3)
    # ------------
    # Case A: Phi=0 (cheap + deterministic)
    # ------------
    grid.add_field("Phi", np.zeros((N, N, N), dtype=float))

    interp = InterpolatorFast(grid)

    def make_metrics():
        slow = PerturbedFLRWMetric(cosmology=cosmology, grid=grid, interpolator=interp)
        fast = PerturbedFLRWMetricFast(
            a_of_eta=cosmology.a_of_eta,
            adot_of_eta=cosmology.adot_of_eta,
            grid=grid,
            interpolator=interp,
            analytical_geodesics=False,
        )
        return slow, fast

    slow, fast = make_metrics()

    x = np.array([eta, 0.2 * grid_size, 0.3 * grid_size, 0.4 * grid_size], float)
    state = np.zeros(8)
    state[:4] = x
    state[4:] = np.array([-0.4, 0.1 * c, 0.2 * c, 0.3 * c])

    # Warmups (numba compile etc.)
    _ = slow.geodesic_equations_tensor(state)
    _ = fast.geodesic_equations(state)
    for _ in range(5):
        _ = fast.geodesic_equations(state)

    reps = 200

    t_slow = _median_time(lambda: slow.geodesic_equations_tensor(state), reps=reps)
    t_fast = _median_time(lambda: fast.geodesic_equations(state), reps=reps)

    print("RHS benchmark (median over", reps, "calls) — Phi=0")
    print("  slow:", t_slow, "s")
    print("  fast:", t_fast, "s")
    if t_fast > 0:
        print("  speedup:", t_slow / t_fast, "x")

    # ------------
    # Case B: spherical_mass potential (non-zero Phi)
    # ------------
    from excalibur.core.constants import one_Msun
    M = 1e15 * one_Msun
    R = 5 * one_Mpc
    center = np.array([0.5, 0.5, 0.5]) * grid_size
    halo = spherical_mass(mass=M, radius=R, center=center)

    # Sample Phi on grid nodes (SI units) and overwrite the field
    xs = origin3[0] + np.arange(N) * spacing[0]
    ys = origin3[1] + np.arange(N) * spacing[1]
    zs = origin3[2] + np.arange(N) * spacing[2]
    grid.fields["Phi"][:] = halo.potential(xs, ys, zs)

    slow2, fast2 = make_metrics()
    _ = slow2.geodesic_equations_tensor(state)
    _ = fast2.geodesic_equations(state)
    for _ in range(5):
        _ = fast2.geodesic_equations(state)

    t_slow2 = _median_time(lambda: slow2.geodesic_equations_tensor(state), reps=reps)
    t_fast2 = _median_time(lambda: fast2.geodesic_equations(state), reps=reps)

    print("RHS benchmark (median over", reps, "calls) — spherical_mass")
    print("  slow:", t_slow2, "s")
    print("  fast:", t_fast2, "s")
    if t_fast2 > 0:
        print("  speedup:", t_slow2 / t_fast2, "x")


if __name__ == "__main__":
    main()
