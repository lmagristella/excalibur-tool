#!/usr/bin/env python3
"""
Test parallel speedup with SHARED MEMORY to avoid copying overhead on Windows.

This version should show real speedup because the 1GB grid is NOT copied to each worker.
"""

import numpy as np
from scipy import interpolate 
import os
import sys
import time
from multiprocessing import cpu_count

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from excalibur.grid.grid import Grid
from excalibur.grid.interpolator_fast import InterpolatorFast
from excalibur.metrics.perturbed_flrw_metric_fast import PerturbedFLRWMetricFast
from excalibur.photon.photons import Photons
from excalibur.integration.integrator import Integrator
from excalibur.integration.parallel_integrator_sharedmem import ParallelIntegratorSharedMem
from excalibur.core.constants import *
from excalibur.core.cosmology import LCDM_Cosmology
from excalibur.objects.spherical_mass import spherical_mass


def run_sharedmem_test():
    """Test parallel performance with shared memory."""
    print("=== PARALLEL SPEEDUP TEST - SHARED MEMORY VERSION ===\n")
    print(f"CPU cores available: {cpu_count()}\n")

    # Setup
    N = 64
    grid_size = 2000 * one_Mpc
    dx = dy = dz = grid_size / N
    shape = (N, N, N)
    spacing = (dx, dy, dz)
    origin = (-grid_size/2, -grid_size/2, -grid_size/2)
    grid = Grid(shape, spacing, origin)

    x = y = z = np.linspace(-grid_size/2, grid_size/2, N)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    M = 1e20 * one_Msun
    radius = 10 * one_Mpc
    center = np.array([500.0, 500.0, 500.0]) * one_Mpc
    spherical_halo = spherical_mass(M, radius, center)

    print("Computing potential field...")
    phi_field = spherical_halo.potential(X, Y, Z)
    grid.add_field("Phi", phi_field)
    
    grid_size_mb = phi_field.nbytes / 1e6
    print(f"Grid size: {grid_size_mb:.1f} MB")
    print(f"With shared memory: Grid is stored ONCE (not copied to workers)\n")

    H0 = 70
    cosmology = LCDM_Cosmology(H0, Omega_m=0.3, Omega_r=0, Omega_lambda=0.7)
    eta_sample = np.linspace(1.0e25, 5.0e26, 1000)
    a_sample = cosmology.a_of_eta(eta_sample)
    a_of_eta = interpolate.interp1d(eta_sample, a_sample, kind='cubic', fill_value="extrapolate")

    interpolator = InterpolatorFast(grid)
    metric = PerturbedFLRWMetricFast(a_of_eta, grid, interpolator)

    # Setup integration parameters
    observer_eta = 4.4e26
    observer_position = np.array([0.0, 0.0, 0.0])
    direction_to_mass = center - observer_position
    direction_to_mass = direction_to_mass / np.linalg.norm(direction_to_mass)

    a_obs = a_of_eta(observer_eta)
    distance_to_mass = np.linalg.norm(center - observer_position)
    time_to_reach_mass = (a_obs / c) * distance_to_mass
    total_time = 1.5 * time_to_reach_mass
    dt_magnitude = (grid_size / 50000) / c
    n_steps = min(200, int(total_time / dt_magnitude))
    dt = -total_time / n_steps

    # Generate test photons
    n_test_photons = 20
    print(f"Generating {n_test_photons} test photons...")
    observer_4d = np.array([observer_eta, *observer_position])

    photons_sets = []
    worker_counts = [1, 2, 4, min(8, cpu_count()-1)]
    
    for n_workers in worker_counts:
        photons = Photons()
        for _ in range(n_test_photons):
            photons.generate_cone_random(1, observer_4d, direction_to_mass, np.pi/12, 1.0)
        for p in photons:
            p.record()
        photons_sets.append((n_workers, photons))

    print(f"Integration: {n_steps} steps with dt={dt:.2e}s\n")

    # Test different worker counts
    results = []

    for n_workers, photons in photons_sets:
        print(f"--- Testing with {n_workers} worker(s) (SHARED MEMORY) ---")
        
        if n_workers == 1:
            # Sequential baseline
            integrator = Integrator(metric, dt=dt)
            start = time.time()
            success = 0
            for photon in photons:
                try:
                    integrator.integrate(photon, n_steps)
                    success += 1
                except:
                    pass
            elapsed = time.time() - start
        else:
            # Parallel with SHARED MEMORY
            parallel_integrator = ParallelIntegratorSharedMem(
                metric=metric,
                dt=dt,
                n_workers=n_workers
            )
            
            start = time.time()
            success = parallel_integrator.integrate_photons_sharedmem(
                photons, n_steps, verbose=False
            )
            elapsed = time.time() - start
        
        rate = n_test_photons * n_steps / elapsed
        results.append((n_workers, elapsed, rate, success))
        
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Success: {success}/{n_test_photons}")
        print(f"  Performance: {rate:.0f} step-evals/sec")
        
        if len(results) > 1:
            speedup = results[0][1] / elapsed
            efficiency = speedup / n_workers * 100
            print(f"  Speedup vs 1 worker: {speedup:.2f}x")
            print(f"  Parallel efficiency: {efficiency:.1f}%")
        print()

    # Summary
    print("="*60)
    print("PARALLEL SPEEDUP SUMMARY - SHARED MEMORY")
    print("="*60)
    print(f"{'Workers':<10} {'Time (s)':<12} {'Speedup':<10} {'Efficiency':<12}")
    print("-"*60)
    baseline_time = results[0][1]
    for n_workers, elapsed, rate, success in results:
        speedup = baseline_time / elapsed
        efficiency = speedup / n_workers * 100 if n_workers > 1 else 100
        print(f"{n_workers:<10} {elapsed:<12.3f} {speedup:<10.2f}x {efficiency:<12.1f}%")
    print("="*60)

    # Analysis
    print("\nAnalysis:")
    best_speedup = max(r[1]/results[0][1] for r in results)
    print(f"  Best speedup achieved: {best_speedup:.2f}x")
    print(f"  Theoretical maximum: {max(r[0] for r in results)}x")
    
    if best_speedup > 1.5:
        print(f"  ✅ Shared memory works! Real speedup achieved")
        overhead = (1 - best_speedup/max(r[0] for r in results[1:])) * 100
        print(f"  Overhead: ~{overhead:.1f}%")
    else:
        print(f"  ⚠️  Still no speedup - may need more photons or longer integration")


if __name__ == '__main__':
    run_sharedmem_test()
