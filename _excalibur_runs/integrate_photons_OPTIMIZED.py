#!/usr/bin/env python3
"""
OPTIMIZED Backward ray tracing simulation using the excalibur library.

Optimizations applied:
1. Numba JIT compilation for critical functions (10-100x speedup)
2. Fast interpolator with compiled trilinear interpolation
3. Optimized metric with caching and inlined Christoffel calculations
4. Parallel integration using multiprocessing (N-core speedup)
5. Reduced memory allocations in tight loops

Expected performance: 50-200x faster than original version
"""

import numpy as np
from scipy import interpolate 
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

### excalibur imports ###
from excalibur.grid.grid import Grid
from excalibur.grid.interpolator_fast import InterpolatorFast  # OPTIMIZED
from excalibur.metrics.perturbed_flrw_metric_fast import PerturbedFLRWMetricFast  # OPTIMIZED
from excalibur.photon.photons import Photons
from excalibur.integration.integrator_old import Integrator
from excalibur.integration.parallel_integrator import ParallelIntegratorChunked  # OPTIMIZED
from excalibur.core.constants import *
from excalibur.core.cosmology import LCDM_Cosmology
from excalibur.objects.spherical_mass import spherical_mass
##########################

def main():
    """Main function for optimized backward ray tracing simulation."""
    
    print("=== OPTIMIZED Backward Ray Tracing with Excalibur ===\n")
    print("Performance optimizations:")
    print("  ✓ Numba JIT compilation")
    print("  ✓ Fast interpolation")
    print("  ✓ Metric caching")
    print("  ✓ Parallel integration\n")
    
    overall_start = time.time()
    
    # =============================================================================
    # 1. COSMOLOGICAL SETUP
    # =============================================================================
    print("1. Setting up cosmology...")
    setup_start = time.time()
    
    # Define ΛCDM cosmology
    H0 = 70  # km/s/Mpc
    Omega_m = 0.3
    Omega_r = 0
    Omega_lambda = 0.7
    cosmology = LCDM_Cosmology(H0, Omega_m=Omega_m, Omega_r=Omega_r, Omega_lambda=Omega_lambda)
    
    # Create scale factor interpolation
    eta_start = 1.0e25
    eta_end = 5.0e26
    eta_sample = np.linspace(eta_start, eta_end, 1000)
    a_sample = cosmology.a_of_eta(eta_sample)
    a_of_eta = interpolate.interp1d(eta_sample, a_sample, kind='cubic', fill_value="extrapolate")
    
    print(f"   ✓ Cosmology setup complete ({time.time() - setup_start:.2f}s)")
    
    # =============================================================================
    # 2. GRID AND MASS DISTRIBUTION SETUP
    # =============================================================================
    print("\n2. Setting up grid and mass distribution...")
    grid_start = time.time()
    
    # Grid parameters  
    N = 512  # Grid resolution
    grid_size = 2000 * one_Mpc
    dx = dy = dz = grid_size / N
    
    shape = (N, N, N)
    spacing = (dx, dy, dz)
    origin = (-grid_size/2, -grid_size/2, -grid_size/2)
    grid = Grid(shape, spacing, origin)
    
    # Create coordinate arrays
    x = y = z = np.linspace(-grid_size/2, grid_size/2, N)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Define spherical mass
    M = 1e20 * one_Msun  # Adjusted mass
    radius = 10 * one_Mpc
    center = np.array([500.0, 500.0, 500.0]) * one_Mpc
    spherical_halo = spherical_mass(M, radius, center)
    
    # Compute potential field
    print(f"   Computing potential field on {N}³ grid...")
    pot_start = time.time()
    phi_field = spherical_halo.potential(X, Y, Z)
    grid.add_field("Phi", phi_field)
    print(f"   ✓ Potential computed ({time.time() - pot_start:.2f}s)")
    
    print(f"   ✓ Grid setup complete ({time.time() - grid_start:.2f}s)")
    
    # =============================================================================
    # 3. INTERPOLATOR AND METRIC SETUP (OPTIMIZED)
    # =============================================================================
    print("\n3. Setting up OPTIMIZED spacetime metric...")
    metric_start = time.time()
    
    # Use FAST interpolator with Numba compilation
    interpolator = InterpolatorFast(grid)
    
    # Use FAST metric with caching and optimized calculations
    metric = PerturbedFLRWMetricFast(a_of_eta, grid, interpolator)
    
    print(f"   ✓ Optimized metric initialized ({time.time() - metric_start:.2f}s)")
    
    # =============================================================================
    # 4. BACKWARD RAY TRACING SETUP
    # =============================================================================
    print("\n4. Setting up backward ray tracing...")
    
    # Observer configuration
    observer_eta = 4.4e26
    observer_position = np.array([0.0, 0.0, 0.0])
    
    # Direction towards the mass
    direction_to_mass = center - observer_position
    direction_to_mass = direction_to_mass / np.linalg.norm(direction_to_mass)
    
    # Cone parameters
    n_photons = 50
    cone_angle = np.pi / 12
    energy = 1.0
    
    print(f"   Observer at origin, looking toward mass at [{center[0]/one_Mpc:.0f}, {center[1]/one_Mpc:.0f}, {center[2]/one_Mpc:.0f}] Mpc")
    print(f"   Cone: {n_photons} photons, {cone_angle*180/np.pi:.1f}° half-angle")
    
    # =============================================================================
    # 5. PHOTON GENERATION
    # =============================================================================
    print("\n5. Generating photons...")
    photon_start = time.time()
    
    photons = Photons()
    observer_4d_position = np.array([observer_eta, *observer_position])
    
    photons.generate_cone_random(
        n_photons=n_photons,
        origin=observer_4d_position,
        central_direction=direction_to_mass,
        cone_angle=cone_angle,
        energy=energy
    )
    
    # Record initial states
    for photon in photons:
        photon.record()
    
    print(f"   ✓ Generated {len(photons)} photons ({time.time() - photon_start:.2f}s)")
    
    # =============================================================================
    # 6. OPTIMIZED BACKWARD INTEGRATION
    # =============================================================================
    print("\n6. Performing OPTIMIZED backward ray tracing integration...")
    integration_start = time.time()
    
    # Calculate integration parameters
    a_obs = a_of_eta(observer_eta)
    distance_to_mass = np.linalg.norm(center - observer_position)
    time_to_reach_mass = (a_obs / c) * distance_to_mass
    total_integration_time = 1.5 * time_to_reach_mass
    
    # Time step calculation
    target_displacement_per_step = grid_size / 50000
    dt_from_displacement = target_displacement_per_step / c
    
    typical_acceleration = 1e6
    max_velocity_change = 0.01 * c
    dt_from_acceleration = max_velocity_change / typical_acceleration
    
    dt_magnitude = min(dt_from_displacement, dt_from_acceleration)
    n_steps = int(np.ceil(abs(total_integration_time / dt_magnitude)))
    n_steps = max(500, min(n_steps, 50000))
    
    dt = -total_integration_time / n_steps  # Negative for backward tracing
    
    print(f"   Integration parameters:")
    print(f"     Steps: {n_steps}")
    print(f"     dt: {dt:.2e} s")
    print(f"     Total time: {n_steps * abs(dt):.2e} s")
    print(f"     Expected displacement per step: {c * abs(dt)/one_Mpc:.4f} Mpc")
    
    # PARALLEL INTEGRATION
    from multiprocessing import cpu_count
    n_cores = cpu_count()
    print(f"   Using PARALLEL integration with {n_cores-1} workers")
    
    parallel_integrator = ParallelIntegratorChunked(
        metric=metric,
        dt=dt,
        n_workers=n_cores-1,
        chunk_size=max(1, n_photons // (n_cores-1))
    )
    
    success_count = parallel_integrator.integrate_photons_chunked(
        photons, n_steps, verbose=True
    )
    
    integration_time = time.time() - integration_start
    print(f"   ✓ Integration complete in {integration_time:.2f}s")
    print(f"   Performance: {n_photons * n_steps / integration_time:.0f} step-evals/sec")
    
    # =============================================================================
    # 7. ANALYSIS AND RESULTS
    # =============================================================================
    print("\n7. Analyzing results...")
    
    trajectory_lengths = [len(photon.history.states) for photon in photons]
    avg_length = np.mean(trajectory_lengths)
    
    final_positions = []
    final_times = []
    
    for photon in photons:
        if len(photon.history.states) > 0:
            final_state = photon.history.states[-1]
            final_times.append(final_state[0])
            final_positions.append(final_state[1:4])
    
    if final_positions:
        final_positions = np.array(final_positions)
        final_times = np.array(final_times)
        
        distances_from_mass = [np.linalg.norm(pos - center) for pos in final_positions]
        avg_distance_from_mass = np.mean(distances_from_mass)
        min_distance_from_mass = np.min(distances_from_mass)
        
        print(f"   Average trajectory length: {avg_length:.1f} states")
        print(f"   Time range: eta in [{final_times.min():.2e}, {final_times.max():.2e}] s")
        print(f"   Distance from mass: avg = {avg_distance_from_mass/one_Mpc:.2f} Mpc, min = {min_distance_from_mass/one_Mpc:.2f} Mpc")
    
    # =============================================================================
    # 8. SAVE RESULTS
    # =============================================================================
    print("\n8. Saving trajectories...")
    save_start = time.time()
    
    mass_x = center[0] / one_Mpc
    mass_y = center[1] / one_Mpc
    mass_z = center[2] / one_Mpc
    output_filename = f"backward_raytracing_OPTIMIZED_mass_{mass_x:.0f}_{mass_y:.0f}_{mass_z:.0f}_Mpc.h5"
    
    try:
        photons.save_all_histories(output_filename)
        file_size = os.path.getsize(output_filename)
        print(f"   ✓ Saved to {output_filename} ({file_size/1024:.1f} KB)")
    except Exception as e:
        print(f"   ✗ Error saving: {e}")
    
    print(f"   Save time: {time.time() - save_start:.2f}s")
    
    # =============================================================================
    # 9. PERFORMANCE SUMMARY
    # =============================================================================
    total_time = time.time() - overall_start
    
    print("\n" + "="*70)
    print("OPTIMIZED BACKWARD RAY TRACING - PERFORMANCE SUMMARY")
    print("="*70)
    print(f"Total execution time:     {total_time:.2f} seconds")
    print(f"  - Setup:                {grid_start - overall_start:.2f}s")
    print(f"  - Grid/potential:       {time.time() - grid_start - integration_time:.2f}s")
    print(f"  - Integration:          {integration_time:.2f}s ({integration_time/total_time*100:.1f}%)")
    print(f"Integration performance:  {n_photons * n_steps / integration_time:.0f} step-evaluations/sec")
    print(f"Photons:                  {n_photons} photons, {n_steps} steps each")
    print(f"Success rate:             {success_count}/{n_photons} ({success_count/n_photons*100:.1f}%)")
    print(f"Output:                   {output_filename}")
    print("="*70)
    print("✓ OPTIMIZED ray tracing complete!")
    

if __name__ == "__main__":
    main()
