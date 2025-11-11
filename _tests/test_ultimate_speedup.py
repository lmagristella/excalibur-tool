#!/usr/bin/env python3
"""
Test FINAL : Mesure du speedup combiné Numba JIT + Persistent Pool.

Compare 3 versions :
1. Standard (baseline)
2. Numba JIT seul (single-core)
3. Numba JIT + Persistent Pool (4 workers)

But : Mesurer le speedup TOTAL de toutes les optimisations.
"""

import numpy as np
from scipy import interpolate 
import sys
import os
import time

# Ajouter le chemin du projet
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from excalibur.grid.grid import Grid
from excalibur.grid.interpolator import Interpolator
from excalibur.grid.interpolator_fast import InterpolatorFast
from excalibur.metrics.perturbed_flrw_metric import PerturbedFLRWMetric
from excalibur.metrics.perturbed_flrw_metric_fast import PerturbedFLRWMetricFast
from excalibur.photon.photons import Photons
from excalibur.integration.integrator import Integrator
from excalibur.integration.parallel_integrator_persistent import PersistentPoolIntegrator
from excalibur.core.constants import *
from excalibur.core.cosmology import LCDM_Cosmology
from excalibur.objects.spherical_mass import spherical_mass


def setup_environment():
    """Setup commun pour tous les tests."""
    print("Setting up test environment...")
    
    # Grid setup
    N = 64
    grid_size = 2000 * one_Mpc
    dx = dy = dz = grid_size / N
    shape = (N, N, N)
    spacing = (dx, dy, dz)
    origin = (-grid_size/2, -grid_size/2, -grid_size/2)
    grid = Grid(shape, spacing, origin)

    x = y = z = np.linspace(-grid_size/2, grid_size/2, N)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Mass setup
    M = 1e20 * one_Msun
    radius = 10 * one_Mpc
    center = np.array([500.0, 500.0, 500.0]) * one_Mpc
    spherical_halo = spherical_mass(M, radius, center)

    phi_field = spherical_halo.potential(X, Y, Z)
    grid.add_field("Phi", phi_field)

    # Cosmology
    H0 = 70
    cosmology = LCDM_Cosmology(H0, Omega_m=0.3, Omega_r=0, Omega_lambda=0.7)
    eta_sample = np.linspace(1.0e25, 5.0e26, 1000)
    a_sample = cosmology.a_of_eta(eta_sample)
    a_of_eta = interpolate.interp1d(eta_sample, a_sample, kind='cubic', fill_value="extrapolate")

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

    return {
        'grid': grid,
        'a_of_eta': a_of_eta,
        'observer_eta': observer_eta,
        'observer_position': observer_position,
        'direction_to_mass': direction_to_mass,
        'center': center,
        'dt': dt,
        'n_steps': n_steps
    }


def generate_photons(params, n_photons):
    """Générer photons pour test."""
    photons = Photons()
    observer_4d = np.array([params['observer_eta'], *params['observer_position']])
    
    for _ in range(n_photons):
        photons.generate_cone_random(1, observer_4d, params['direction_to_mass'], np.pi/12, 1.0)
    
    for p in photons:
        p.record()
    
    return photons


def test_standard(params, photons, n_steps):
    """Test version STANDARD (baseline)."""
    print("\n" + "="*70)
    print("TEST 1/3: VERSION STANDARD (Baseline)")
    print("="*70)
    
    # Créer interpolator et metric STANDARD
    interpolator = Interpolator(params['grid'])
    metric = PerturbedFLRWMetric(params['a_of_eta'], params['grid'], interpolator)
    integrator = Integrator(metric, dt=params['dt'])
    
    # Warm-up (compile Numba si besoin)
    test_photon = photons[0]
    try:
        integrator.integrate(test_photon, 1)
    except:
        pass
    
    # Test réel
    start = time.time()
    success = 0
    for photon in photons:
        try:
            integrator.integrate(photon, n_steps)
            success += 1
        except:
            pass
    elapsed = time.time() - start
    
    rate = len(photons) * n_steps / elapsed
    
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Success: {success}/{len(photons)}")
    print(f"  Performance: {rate:.0f} step-evals/sec")
    
    return elapsed, rate


def test_numba_only(params, photons, n_steps):
    """Test version NUMBA JIT seul (single-core)."""
    print("\n" + "="*70)
    print("TEST 2/3: NUMBA JIT OPTIMIZED (Single-core)")
    print("="*70)
    
    # Créer interpolator et metric OPTIMIZED
    interpolator = InterpolatorFast(params['grid'])
    metric = PerturbedFLRWMetricFast(params['a_of_eta'], params['grid'], interpolator)
    integrator = Integrator(metric, dt=params['dt'])
    
    # Warm-up (compile Numba)
    print("  Warming up Numba JIT...")
    test_photon = photons[0]
    try:
        integrator.integrate(test_photon, 10)
    except:
        pass
    
    # Test réel
    print("  Running benchmark...")
    start = time.time()
    success = 0
    for photon in photons:
        try:
            integrator.integrate(photon, n_steps)
            success += 1
        except:
            pass
    elapsed = time.time() - start
    
    rate = len(photons) * n_steps / elapsed
    
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Success: {success}/{len(photons)}")
    print(f"  Performance: {rate:.0f} step-evals/sec")
    
    return elapsed, rate


def test_numba_plus_parallel(params, photons, n_steps, n_workers=4):
    """Test version NUMBA JIT + PERSISTENT POOL."""
    print("\n" + "="*70)
    print(f"TEST 3/3: NUMBA JIT + PERSISTENT POOL ({n_workers} workers)")
    print("="*70)
    
    # Créer interpolator et metric OPTIMIZED
    interpolator = InterpolatorFast(params['grid'])
    metric = PerturbedFLRWMetricFast(params['a_of_eta'], params['grid'], interpolator)
    
    # Warm-up Numba dans le process principal
    print("  Warming up Numba JIT...")
    test_integrator = Integrator(metric, dt=params['dt'])
    test_photon = photons[0]
    try:
        test_integrator.integrate(test_photon, 10)
    except:
        pass
    
    # Créer persistent pool (overhead mesuré mais amorti)
    print(f"  Creating persistent pool with {n_workers} workers...")
    pool_start = time.time()
    integrator = PersistentPoolIntegrator(metric, params['dt'], n_workers=n_workers)
    pool_creation_time = time.time() - pool_start
    print(f"  Pool created in {pool_creation_time:.3f}s (one-time overhead)")
    
    # Test réel (sans overhead de création)
    print("  Running benchmark...")
    start = time.time()
    n_success, results = integrator.integrate_photons(photons, n_steps, verbose=False)
    elapsed = time.time() - start
    
    integrator.close()
    
    rate = len(photons) * n_steps / elapsed
    
    print(f"  Integration time: {elapsed:.3f}s (excluding pool creation)")
    print(f"  Total time: {elapsed + pool_creation_time:.3f}s (including pool creation)")
    print(f"  Success: {n_success}/{len(photons)}")
    print(f"  Performance: {rate:.0f} step-evals/sec")
    
    return elapsed, rate, pool_creation_time


def main():
    """Test principal."""
    print("="*70)
    print("ULTIMATE SPEEDUP TEST - Numba JIT + Parallel")
    print("="*70)
    print("\nMeasuring combined speedup from ALL optimizations:")
    print("  - Numba JIT compilation")
    print("  - Cached scale factor")
    print("  - Persistent worker pool")
    print("  - Fast interpolation")
    print()
    
    # Setup
    params = setup_environment()
    
    # Test parameters
    n_photons = 40  # Assez pour voir l'effet du parallèle
    n_steps = params['n_steps']
    
    print(f"\nTest configuration:")
    print(f"  Photons: {n_photons}")
    print(f"  Integration steps: {n_steps}")
    print(f"  Total step-evaluations: {n_photons * n_steps}")
    print(f"  Grid: 64³ cells")
    
    # Générer photons TROIS fois (un set pour chaque test)
    print("\nGenerating photons for tests...")
    photons_standard = generate_photons(params, n_photons)
    photons_numba = generate_photons(params, n_photons)
    photons_parallel = generate_photons(params, n_photons)
    
    # Test 1: Standard
    time_standard, rate_standard = test_standard(params, photons_standard, n_steps)
    
    # Test 2: Numba only
    time_numba, rate_numba = test_numba_only(params, photons_numba, n_steps)
    
    # Test 3: Numba + Parallel
    time_parallel, rate_parallel, pool_overhead = test_numba_plus_parallel(
        params, photons_parallel, n_steps, n_workers=4
    )
    
    # RESULTS
    print("\n" + "="*70)
    print("FINAL RESULTS - SPEEDUP SUMMARY")
    print("="*70)
    
    speedup_numba = time_standard / time_numba
    speedup_parallel = time_standard / time_parallel
    speedup_total_with_overhead = time_standard / (time_parallel + pool_overhead)
    
    print(f"\n{'Configuration':<30} {'Time':<12} {'Speedup':<12} {'Efficiency':<12}")
    print("-"*70)
    print(f"{'1. Standard (baseline)':<30} {time_standard:<12.3f}s {1.0:<12.2f}x {100.0:<12.1f}%")
    print(f"{'2. Numba JIT (1 core)':<30} {time_numba:<12.3f}s {speedup_numba:<12.2f}x {speedup_numba*100:<12.1f}%")
    print(f"{'3. Numba + Parallel (4 cores)':<30} {time_parallel:<12.3f}s {speedup_parallel:<12.2f}x {speedup_parallel/4*100:<12.1f}%")
    print(f"{'   (with pool overhead)':<30} {time_parallel+pool_overhead:<12.3f}s {speedup_total_with_overhead:<12.2f}x {speedup_total_with_overhead/4*100:<12.1f}%")
    print("="*70)
    
    # Analysis
    print("\n📊 ANALYSIS:")
    print(f"\n1. Numba JIT speedup: {speedup_numba:.2f}x")
    print(f"   - Single-core optimization")
    print(f"   - Compiled interpolation + cached metric")
    
    additional_speedup = speedup_parallel / speedup_numba
    print(f"\n2. Additional speedup from parallelization: {additional_speedup:.2f}x")
    print(f"   - On top of Numba JIT")
    print(f"   - Using 4 workers")
    print(f"   - Efficiency: {additional_speedup/4*100:.1f}%")
    
    print(f"\n3. TOTAL COMBINED SPEEDUP: {speedup_parallel:.2f}x")
    print(f"   - From: {time_standard:.3f}s")
    print(f"   - To:   {time_parallel:.3f}s")
    print(f"   - Gain: {time_standard - time_parallel:.3f}s saved")
    
    print(f"\n4. Pool creation overhead: {pool_overhead:.3f}s")
    print(f"   - One-time cost (amortized over multiple batches)")
    print(f"   - Total speedup including overhead: {speedup_total_with_overhead:.2f}x")
    
    # Projections
    print(f"\n🚀 PERFORMANCE PROJECTION FOR PRODUCTION:")
    print(f"\n   Configuration: 50 photons × 5000 steps")
    time_prod_standard = (50 * 5000) / rate_standard
    time_prod_numba = (50 * 5000) / rate_numba
    time_prod_parallel = (50 * 5000) / rate_parallel
    
    print(f"   Standard:        {time_prod_standard:.1f}s ({time_prod_standard/60:.1f} min)")
    print(f"   Numba JIT:       {time_prod_numba:.1f}s")
    print(f"   Numba + Parallel: {time_prod_parallel:.1f}s")
    print(f"\n   Total time saved: {time_prod_standard - time_prod_parallel:.1f}s ({(time_prod_standard - time_prod_parallel)/60:.1f} min)")
    
    # Recommendation
    print(f"\n💡 RECOMMENDATION:")
    if speedup_parallel > 30:
        print(f"   ✅ EXCELLENT! Combined optimizations give {speedup_parallel:.0f}x speedup")
        print(f"   ✅ Use Numba + Persistent Pool for all production runs")
    elif speedup_parallel > 20:
        print(f"   ✅ VERY GOOD! Combined optimizations give {speedup_parallel:.0f}x speedup")
        print(f"   ✅ Recommended for production with >40 photons")
    elif speedup_parallel > 10:
        print(f"   ✅ GOOD! Combined optimizations give {speedup_parallel:.0f}x speedup")
        print(f"   ⚠️  Consider Numba-only for small runs (<20 photons)")
    else:
        print(f"   ⚠️  Moderate speedup: {speedup_parallel:.1f}x")
        print(f"   💡 Use Numba-only (simpler) unless running many batches")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
