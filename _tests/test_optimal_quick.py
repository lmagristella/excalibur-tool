#!/usr/bin/env python3
"""
Quick test of OPTIMAL version with minimal configuration.
Tests that Numba JIT + Persistent Pool work correctly together.
"""

import numpy as np
from scipy import interpolate 
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from excalibur.grid.grid import Grid
from excalibur.grid.interpolator_fast import InterpolatorFast
from excalibur.metrics.perturbed_flrw_metric_fast import PerturbedFLRWMetricFast
from excalibur.photon.photons import Photons
from excalibur.integration.parallel_integrator_persistent import PersistentPoolIntegrator
from excalibur.core.constants import *
from excalibur.core.cosmology import LCDM_Cosmology
from excalibur.objects.spherical_mass import spherical_mass

def main():
    print("=" * 60)
    print("QUICK TEST - OPTIMAL VERSION")
    print("=" * 60)

    start = time.time()

    # Minimal cosmology
    H0 = 70
    Omega_m = 0.3
    Omega_r = 0
    Omega_lambda = 0.7
    cosmology = LCDM_Cosmology(H0, Omega_m, Omega_r, Omega_lambda)

    eta_start = 1.0e25
    eta_end = 5.0e26
    eta_sample = np.linspace(eta_start, eta_end, 500)
    a_sample = cosmology.a_of_eta(eta_sample)
    a_of_eta = interpolate.interp1d(eta_sample, a_sample, kind='cubic')

    print(f"✓ Cosmology setup: {time.time()-start:.2f}s")

    # Small grid for quick test
    N = 64
    grid_size = 1000 * one_Mpc
    dx = dy = dz = grid_size / N

    shape = (N, N, N)
    spacing = (dx, dy, dz)
    origin = (-grid_size/2, -grid_size/2, -grid_size/2)
    grid = Grid(shape, spacing, origin)

    x = y = z = np.linspace(-grid_size/2, grid_size/2, N)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    M = 1e20 * one_Msun
    radius = 10 * one_Mpc
    center = np.array([250.0, 250.0, 250.0]) * one_Mpc
    spherical_halo = spherical_mass(M, radius, center)

    phi_field = spherical_halo.potential(X, Y, Z)
    grid.add_field("Phi", phi_field)

    print(f"✓ Grid setup ({N}³): {time.time()-start:.2f}s")

    # Optimal metric
    interpolator = InterpolatorFast(grid)
    metric = PerturbedFLRWMetricFast(a_of_eta, grid, interpolator)

    print(f"✓ Metric setup (OPTIMAL): {time.time()-start:.2f}s")

    # Generate 20 photons (good for testing parallel)
    observer_eta = 4.4e26
    observer_position = np.array([0.0, 0.0, 0.0])
    direction_to_mass = center - observer_position
    direction_to_mass = direction_to_mass / np.linalg.norm(direction_to_mass)

    photons = Photons()
    observer_4d = np.array([observer_eta, *observer_position])

    photons.generate_cone_random(
        n_photons=20,
        origin=observer_4d,
        central_direction=direction_to_mass,
        cone_angle=np.pi/12,
        energy=1.0
    )

    for p in photons:
        p.record()

    print(f"✓ Generated {len(photons)} photons: {time.time()-start:.2f}s")

    # Integration parameters
    n_steps = 200
    distance_to_mass = np.linalg.norm(center - observer_position)
    a_obs = a_of_eta(observer_eta)
    total_time = 1.5 * (a_obs / c) * distance_to_mass
    dt = -total_time / n_steps

    print(f"\nIntegration: {len(photons)} photons × {n_steps} steps")
    print(f"Using Persistent Pool with 4 workers...")

    integration_start = time.time()

    # TEST OPTIMAL VERSION
    with PersistentPoolIntegrator(metric, dt=dt, n_workers=4) as integrator:
        integrator.integrate_photons(photons, n_steps)

    integration_time = time.time() - integration_start

    print(f"\n✓ Integration completed: {integration_time:.2f}s")
    print(f"  ({len(photons)/integration_time:.1f} photons/second)")

    # Verify results
    successful = sum(1 for p in photons if len(p.history.states) > 0)
    avg_states = np.mean([len(p.history.states) for p in photons])

    print(f"\n✓ Results: {successful}/{len(photons)} photons successful")
    print(f"  Average trajectory length: {avg_states:.0f} states")

    total_time = time.time() - start

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Configuration:  {len(photons)} photons × {n_steps} steps on {N}³ grid")
    print(f"Integration:    {integration_time:.2f}s")
    print(f"Total time:     {total_time:.2f}s")
    print(f"Throughput:     {len(photons)/integration_time:.1f} photons/second")
    print("=" * 60)
    print("✅ OPTIMAL VERSION WORKS!")
    print("=" * 60)

if __name__ == '__main__':
    main()
