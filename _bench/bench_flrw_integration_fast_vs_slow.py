"""Benchmarks: FLRW fast vs slow integration.

This benchmarks full integration cost (RK4) rather than just RHS evaluation.

We run two scenarios:
  A) Phi = 0
  B) Phi from a uniform spherical_mass

And for each scenario, two workloads:
  1) 1 photon x 50k steps
  2) 200 photons x 2000 steps

Notes
-----
- This is a wall-clock benchmark. Results depend on your CPU and Python/Numba.
- We warm up the fast metric to include numba compilation outside timings.
- We keep everything sequential for reproducibility.
"""

from __future__ import annotations

import time
import numpy as np

from excalibur.core.constants import c, one_Mpc, one_Msun
from excalibur.core.cosmology import LCDM_Cosmology
from excalibur.grid.grid import Grid
from excalibur.grid.interpolator_fast import InterpolatorFast
from excalibur.integration.integrator import Integrator
from excalibur.metrics.perturbed_flrw_metric import PerturbedFLRWMetric
from excalibur.metrics.perturbed_flrw_metric_fast import PerturbedFLRWMetricFast
from excalibur.objects.spherical_mass import spherical_mass
from excalibur.photon.photon import Photon


def _make_grid_and_interp(N: int, grid_size: float):
    spacing = (grid_size / N, grid_size / N, grid_size / N)
    origin3 = np.array([0.0, 0.0, 0.0])
    grid = Grid((N, N, N), spacing, origin3)
    grid.add_field("Phi", np.zeros((N, N, N), dtype=float))
    interp = InterpolatorFast(grid)
    return grid, interp


def _make_metrics(cosmology, grid, interp):
    slow = PerturbedFLRWMetric(cosmology=cosmology, grid=grid, interpolator=interp)
    fast = PerturbedFLRWMetricFast(
        a_of_eta=cosmology.a_of_eta,
        adot_of_eta=cosmology.adot_of_eta,
        grid=grid,
        interpolator=interp,
        analytical_geodesics=False,
    )
    return slow, fast


def _set_phi0(grid):
    grid.fields["Phi"][:] = 0.0


def _set_phi_spherical_mass(grid, grid_size: float):
    center = np.array([0.5, 0.5, 0.5]) * grid_size
    M = 1e15 * one_Msun
    R = 5 * one_Mpc
    halo = spherical_mass(mass=M, radius=R, center=center)

    N = int(grid.shape[0])
    spacing = grid.spacing
    origin3 = grid.origin

    xs = origin3[0] + np.arange(N) * spacing[0]
    ys = origin3[1] + np.arange(N) * spacing[1]
    zs = origin3[2] + np.arange(N) * spacing[2]
    grid.fields["Phi"][:] = halo.potential(xs, ys, zs)


def _make_photon(metric, x0, dir3_unit, vfrac=0.2):
    u_spatial = dir3_unit * (vfrac * c)
    g = metric.metric_tensor(x0)
    g00 = g[0, 0]
    spatial_term = g[1, 1] * u_spatial[0] ** 2 + g[2, 2] * u_spatial[1] ** 2 + g[3, 3] * u_spatial[2] ** 2
    u0 = -np.sqrt(abs(-spatial_term / g00))
    return Photon(x0.copy(), np.array([u0, *u_spatial], dtype=float))


def _bench_one_photon(metric, x0, dir3_unit, n_steps: int, dlambda: float) -> float:
    photon = _make_photon(metric, x0, dir3_unit)
    integrator = Integrator(metric=metric, dt=dlambda, integrator="rk4", mode="sequential")

    t0 = time.perf_counter()
    integrator.integrate_single(photon, stop_mode="steps", stop_value=n_steps, record_every=0)
    return time.perf_counter() - t0


def _bench_batch(metric, x0, dir3_unit, n_photons: int, n_steps: int, dlambda: float) -> float:
    integrator = Integrator(metric=metric, dt=dlambda, integrator="rk4", mode="sequential")

    photons = [
        _make_photon(metric, x0, dir3_unit)
        for _ in range(n_photons)
    ]

    t0 = time.perf_counter()
    for p in photons:
        integrator.integrate_single(p, stop_mode="steps", stop_value=n_steps, record_every=0)
    return time.perf_counter() - t0


def _warmup_fast(fast_metric, x0, dir3_unit):
    # Warm up: numba compile + a few RHS calls
    p = _make_photon(fast_metric, x0, dir3_unit)
    state = np.concatenate([p.x, p.u])
    for _ in range(10):
        _ = fast_metric.geodesic_equations(state)


def _run_case(name: str, set_phi_fn, cosmology, eta, grid, interp, grid_size):
    set_phi_fn(grid)
    slow, fast = _make_metrics(cosmology, grid, interp)

    # Choose an interior point (avoid interpolator boundary stencil issues)
    x0 = np.array([eta, 0.2 * grid_size, 0.3 * grid_size, 0.4 * grid_size], dtype=float)
    center = np.array([0.5, 0.5, 0.5]) * grid_size
    dir3 = center - x0[1:]
    dir3_unit = dir3 / np.linalg.norm(dir3)

    _warmup_fast(fast, x0, dir3_unit)

    print("\n===", name, "===")

    # Workload 1
    n_steps_1 = 50_000
    dlambda_1 = 5e-4
    t_slow_1 = _bench_one_photon(slow, x0, dir3_unit, n_steps_1, dlambda_1)
    t_fast_1 = _bench_one_photon(fast, x0, dir3_unit, n_steps_1, dlambda_1)
    print("1 photon x 50k steps")
    print("  slow:", t_slow_1, "s")
    print("  fast:", t_fast_1, "s")
    if t_fast_1 > 0:
        print("  speedup:", t_slow_1 / t_fast_1, "x")

    # Workload 2
    n_photons_2 = 200
    n_steps_2 = 2_000
    dlambda_2 = 5e-4
    t_slow_2 = _bench_batch(slow, x0, dir3_unit, n_photons_2, n_steps_2, dlambda_2)
    t_fast_2 = _bench_batch(fast, x0, dir3_unit, n_photons_2, n_steps_2, dlambda_2)
    print("200 photons x 2000 steps")
    print("  slow:", t_slow_2, "s")
    print("  fast:", t_fast_2, "s")
    if t_fast_2 > 0:
        print("  speedup:", t_slow_2 / t_fast_2, "x")


def main():
    cosmology = LCDM_Cosmology(70, Omega_m=0.3, Omega_r=0.0, Omega_lambda=0.7)
    _ = cosmology.a_of_eta(1e18)
    eta = cosmology._eta_at_a1

    # Moderate grid: big enough to keep interpolation stable, small enough to be quick.
    N = 48
    grid_size = 80 * one_Mpc
    grid, interp = _make_grid_and_interp(N=N, grid_size=grid_size)

    _run_case("Phi=0", lambda g: _set_phi0(g), cosmology, eta, grid, interp, grid_size)
    _run_case("spherical_mass", lambda g: _set_phi_spherical_mass(g, grid_size), cosmology, eta, grid, interp, grid_size)


if __name__ == "__main__":
    main()
