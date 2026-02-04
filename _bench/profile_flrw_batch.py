"""Profile: where does time go in batch integration.

This is meant to answer scaling questions like:
  - why 200x2000 is much slower than expected from 1x50k

It runs a medium workload by default (to keep profiling time reasonable)
but you can tweak N_PHOTONS and N_STEPS.

Usage:
  python3 _bench/profile_flrw_batch.py

It will write a .prof file next to this script.
"""

from __future__ import annotations

import cProfile
import pstats
import time
import numpy as np

from excalibur.core.constants import c, one_Mpc, one_Msun
from excalibur.core.cosmology import LCDM_Cosmology
from excalibur.grid.grid import Grid
from excalibur.grid.interpolator_fast import InterpolatorFast
from excalibur.integration.integrator import Integrator
from excalibur.metrics.perturbed_flrw_metric import PerturbedFLRWMetric
from excalibur.objects.spherical_mass import spherical_mass
from excalibur.photon.photon import Photon


def main():
    # Keep it moderate; profiling 200x2000 can take a while.
    N_PHOTONS = 50
    N_STEPS = 2000
    DLAMBDA = 5e-4

    cosmology = LCDM_Cosmology(70, Omega_m=0.3, Omega_r=0.0, Omega_lambda=0.7)
    _ = cosmology.a_of_eta(1e18)
    eta = cosmology._eta_at_a1

    N = 48
    grid_size = 80 * one_Mpc
    spacing = (grid_size / N, grid_size / N, grid_size / N)
    origin3 = np.array([0.0, 0.0, 0.0])

    grid = Grid((N, N, N), spacing, origin3)
    grid.add_field("Phi", np.zeros((N, N, N), dtype=float))

    # spherical_mass potential
    center = np.array([0.5, 0.5, 0.5]) * grid_size
    M = 1e15 * one_Msun
    R = 5 * one_Mpc
    halo = spherical_mass(mass=M, radius=R, center=center)

    xs = origin3[0] + np.arange(N) * spacing[0]
    ys = origin3[1] + np.arange(N) * spacing[1]
    zs = origin3[2] + np.arange(N) * spacing[2]
    grid.fields["Phi"][:] = halo.potential(xs, ys, zs)

    interp = InterpolatorFast(grid)
    metric = PerturbedFLRWMetric(cosmology=cosmology, grid=grid, interpolator=interp)

    # Initial conditions
    x0 = np.array([eta, 0.2 * grid_size, 0.3 * grid_size, 0.4 * grid_size], dtype=float)
    dir3 = center - x0[1:]
    dir3 = dir3 / np.linalg.norm(dir3)
    u_spatial = dir3 * (0.2 * c)

    g = metric.metric_tensor(x0)
    g00 = g[0, 0]
    spatial_term = g[1, 1] * u_spatial[0] ** 2 + g[2, 2] * u_spatial[1] ** 2 + g[3, 3] * u_spatial[2] ** 2
    u0 = -np.sqrt(abs(-spatial_term / g00))

    integrator = Integrator(metric=metric, dt=DLAMBDA, integrator="rk4", mode="sequential")

    photons = [Photon(x0.copy(), np.array([u0, *u_spatial], dtype=float)) for _ in range(N_PHOTONS)]

    prof_path = "/home/magri/excalibur_project/_bench/profile_flrw_batch.prof"

    pr = cProfile.Profile()
    pr.enable()
    t0 = time.perf_counter()
    for p in photons:
        integrator.integrate_single(p, stop_mode="steps", stop_value=N_STEPS)
    elapsed = time.perf_counter() - t0
    pr.disable()

    pr.dump_stats(prof_path)

    stats = pstats.Stats(pr)
    stats.strip_dirs().sort_stats("tottime")
    print("Elapsed:", elapsed, "s")
    print("Wrote profile to:", prof_path)
    print("Top 20 by tottime:")
    stats.print_stats(20)


if __name__ == "__main__":
    main()
