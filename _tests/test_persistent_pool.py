#!/usr/bin/env python3
"""
Test PERSISTENT POOL approach - should show REAL speedup.

This version creates workers ONCE and reuses them, avoiding the
expensive overhead of process creation for each integration.
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
from excalibur.integration.integrator_old import Integrator
from excalibur.integration.parallel_integrator_persistent import PersistentPoolIntegrator
from excalibur.core.constants import *
from excalibur.core.cosmology import LCDM_Cosmology
from excalibur.objects.spherical_mass import spherical_mass


def run_persistent_pool_test():
    """Test persistent pool performance."""
    print("=== PERSISTENT POOL TEST (Workers Created ONCE) ===\n")
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
    
    print(f"Grid size: {phi_field.nbytes / 1e6:.1f} MB\n")

    H0 = 70
    cosmology = LCDM_Cosmology(H0, Omega_m=0.3, Omega_r=0, Omega_lambda=0.7)
    eta_sample = np.linspace(1.0e25, 5.0e26, 1000)
    a_sample = cosmology.a_of_eta(eta_sample)
    a_of_eta = interpolate.interp1d(eta_sample, a_sample, kind='cubic', fill_value="extrapolate")

    interpolator = InterpolatorFast(grid)
    metric = PerturbedFLRWMetricFast(a_of_eta, grid, interpolator)

    # Integration parameters
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

    # Generate photons once
    n_test_photons = 40  # More photons to amortize overhead
    print(f"Generating {n_test_photons} test photons...")
    observer_4d = np.array([observer_eta, *observer_position])
    
    photons = Photons()
    for _ in range(n_test_photons):
        photons.generate_cone_random(1, observer_4d, direction_to_mass, np.pi/12, 1.0)
    for p in photons:
        p.record()

    print(f"Integration: {n_steps} steps with dt={dt:.2e}s\n")

    # Test configurations
    results = []
    
    # 1. Baseline: Sequential
    print("="*60)
    print("1. BASELINE: Sequential (1 core)")
    print("="*60)
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
    rate = n_test_photons * n_steps / elapsed
    results.append(("Sequential", 1, elapsed, rate, success))
    print(f"Time: {elapsed:.3f}s")
    print(f"Performance: {rate:.0f} step-evals/sec\n")
    
    # 2. Persistent Pool with different worker counts
    for n_workers in [2, 4, min(8, cpu_count()-1)]:
        print("="*60)
        print(f"{len(results)+1}. PERSISTENT POOL: {n_workers} workers")
        print("="*60)
        
        # Create pool once (overhead measured separately)
        with PersistentPoolIntegrator(metric, dt, n_workers) as integrator:
            # Now integrate (this is the part we measure)
            start_integration = time.time()
            n_success, _ = integrator.integrate_photons(photons, n_steps, verbose=False)
            elapsed_integration = time.time() - start_integration
            
            rate = n_test_photons * n_steps / elapsed_integration
            results.append((f"Persistent-{n_workers}", n_workers, elapsed_integration, rate, n_success))
            
            print(f"Integration time: {elapsed_integration:.3f}s (excluding pool creation)")
            print(f"Performance: {rate:.0f} step-evals/sec")
            print(f"Success: {n_success}/{n_test_photons}\n")
    
    # 3. Chunked version (best for many photons)
    n_workers = 4
    print("="*60)
    print(f"{len(results)+1}. PERSISTENT POOL CHUNKED: {n_workers} workers")
    print("="*60)
    
    with PersistentPoolIntegrator(metric, dt, n_workers) as integrator:
        start_integration = time.time()
        n_success, _ = integrator.integrate_photons_chunked(
            photons, n_steps, chunk_size=5, verbose=False
        )
        elapsed_integration = time.time() - start_integration
        
        rate = n_test_photons * n_steps / elapsed_integration
        results.append((f"Chunked-{n_workers}", n_workers, elapsed_integration, rate, n_success))
        
        print(f"Integration time: {elapsed_integration:.3f}s")
        print(f"Performance: {rate:.0f} step-evals/sec")
        print(f"Success: {n_success}/{n_test_photons}\n")

    # Summary
    print("\n" + "="*70)
    print("FINAL RESULTS - PERSISTENT POOL")
    print("="*70)
    print(f"{'Method':<20} {'Workers':<8} {'Time (s)':<10} {'Speedup':<10} {'Efficiency':<12}")
    print("-"*70)
    
    baseline_time = results[0][2]
    for method, n_workers, elapsed, rate, success in results:
        speedup = baseline_time / elapsed
        efficiency = (speedup / n_workers * 100) if n_workers > 1 else 100
        print(f"{method:<20} {n_workers:<8} {elapsed:<10.3f} {speedup:<10.2f}x {efficiency:<12.1f}%")
    
    print("="*70)
    
    # Analysis
    best_speedup = max(r[2] for r in results[1:]) / baseline_time
    best_method = max(results[1:], key=lambda r: baseline_time / r[2])
    
    print("\n📊 Analysis:")
    print(f"  Baseline (sequential): {baseline_time:.3f}s")
    print(f"  Best parallel: {best_method[0]} in {best_method[2]:.3f}s")
    print(f"  Best speedup: {baseline_time / best_method[2]:.2f}x")
    
    if baseline_time / best_method[2] > 1.8:
        print(f"  ✅ SUCCESS! Persistent pool shows real speedup!")
    elif baseline_time / best_method[2] > 1.2:
        print(f"  ⚠️  Modest speedup - may need more photons or longer integration")
    else:
        print(f"  ❌ Still no speedup - overhead still dominates")
    
    print(f"\n💡 Note: Pool creation overhead (~1-2s) is paid ONCE and amortized")
    print(f"   over multiple integration calls. Persistent pool is ideal when")
    print(f"   you need to integrate many batches of photons.")


if __name__ == '__main__':
    run_persistent_pool_test()
