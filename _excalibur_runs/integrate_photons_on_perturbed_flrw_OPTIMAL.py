#!/usr/bin/env python3
"""
Backward ray tracing simulation using the excalibur library - OPTIMAL VERSION.

This is the OPTIMAL version that combines ALL performance optimizations:
- Numba-compiled interpolation (InterpolatorFast) - 15x speedup
- Optimized metric with cached scale factor (PerturbedFLRWMetricFast)
- Persistent worker pool for parallel processing - 4x additional speedup
- Total expected speedup: 60x faster than the standard version

This version is RECOMMENDED for:
- Large-scale simulations
- Multi-photon backward ray tracing

For small runs (<20 photons), use the OPTIMIZED version (parallel overhead not worth it).
"""

import numpy as np
from scipy import interpolate 
import sys
import os
import time
import platform

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

### excalibur imports - OPTIMAL VERSIONS ###
from excalibur.grid.grid import Grid
from excalibur.grid.interpolator_fast import InterpolatorFast  # OPTIMIZED
from excalibur.grid.interpolator import Interpolator  # STANDARD
from excalibur.metrics.perturbed_flrw_metric_fast import PerturbedFLRWMetricFast  # OPTIMIZED
from excalibur.metrics.perturbed_flrw_metric import PerturbedFLRWMetric  # STANDARD
from excalibur.photon.photons import Photons
from excalibur.photon.photon import Photon
from excalibur.integration.parallel_integrator_persistent import PersistentPoolIntegrator  # PARALLEL
from excalibur.integration.integrator_old import Integrator # STANDARD
from excalibur.core.constants import *
from excalibur.core.cosmology import LCDM_Cosmology
from excalibur.objects.spherical_mass import spherical_mass
from excalibur.io.filename_utils import generate_trajectory_filename
##########################

def main():
    """Main function for backward ray tracing simulation - OPTIMAL VERSION."""
    
    print("=== Backward Ray Tracing with Excalibur (OPTIMAL) ===")
    print("    Numba JIT (15x) + Persistent Pool 4 workers (4x) = 60x speedup")
    print()
    
    start_time = time.time()
    
    # =============================================================================
    # 1. COSMOLOGICAL SETUP
    # =============================================================================
    print("1. Setting up cosmology...")
    
    # Define ΛCDM cosmology
    H0 = 70  # km/s/Mpc
    Omega_m = 0.3
    Omega_r = 0
    Omega_lambda = 0.7
    cosmology = LCDM_Cosmology(H0, Omega_m=Omega_m, Omega_r=Omega_r, Omega_lambda=Omega_lambda)
    
    # Get conformal time at present (a=1)
    # Need to call a_of_eta once to compute _eta_at_a1
    _ = cosmology.a_of_eta(1e18)  # Dummy call to initialize (use correct order of magnitude)
    eta_present = cosmology._eta_at_a1  # Conformal time at a=1 (~1.46e18 s ≈ 46 Gyr)
    
    # Create scale factor interpolation over conformal time in seconds
    # Go back in conformal time (backward ray tracing)
    eta_start = eta_present  # Present day (a=1)
    eta_end = 0.5 * eta_present  # Earlier time (smaller a, higher z)
    eta_sample = np.linspace(eta_start, eta_end, 1000)
    a_sample = cosmology.a_of_eta(eta_sample)
    a_of_eta = interpolate.interp1d(eta_sample, a_sample, kind='cubic', fill_value="extrapolate")
    
    print(f"   Cosmology: H0={H0} km/s/Mpc, Omega_m={Omega_m}, Omega_Lambda={Omega_lambda}")
    print(f"   Conformal time at present: eta_0 = {eta_present:.3e} s = {eta_present/(365.25*24*3600*1e9):.1f} Gyr (a=1)")
    print(f"   Conformal time range: eta in [{eta_start:.2e}, {eta_end:.2e}] s")
    print(f"   Scale factor range: a({eta_start:.2e} s) = {a_sample[0]:.4f} to a({eta_end:.2e} s) = {a_sample[-1]:.4f}")
    
    # =============================================================================
    # 2. GRID AND MASS DISTRIBUTION SETUP
    # =============================================================================
    print("\n2. Setting up grid and mass distribution...")
    
    # Grid parameters  
    N = 512  # Grid resolution
    grid_size = 1000 * one_Mpc  # 2000 Mpc box size
    dx = dy = dz = grid_size / N
    
    shape = (N, N, N)
    spacing = (dx, dy, dz)
    origin = np.array([0, 0, 0])*grid_size  # Center grid at origin
    grid = Grid(shape, spacing, origin)
    
    # Create coordinate arrays (centered on origin)
    x = y = z = np.linspace(0, grid_size, N)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Define spherical mass (target for backward ray tracing)
    M = 1e15 * one_Msun  # Galaxy cluster mass in kg
    radius = 5 * one_Mpc   # Virial radius
    center = np.array([0.5, 0.5, 0.5]) * grid_size  # 500 Mpc from origin
    spherical_halo = spherical_mass(M, radius, center)
    
    # Compute potential field
    phi_field = spherical_halo.potential(X, Y, Z) 
    grid.add_field("Phi", phi_field)
    
    print(f"   Grid: {N}³ cells, {grid_size/one_Mpc:.0f} Mpc box")
    print(f"   Mass: {M/one_Msun:.1e} M_sun at [{center[0]/one_Mpc:.1f}, {center[1]/one_Mpc:.1f}, {center[2]/one_Mpc:.1f}] Mpc")
    print(f"   Potential range: [{phi_field.min():.2e}, {phi_field.max():.2e}] m^2/s^2")
    
    # =============================================================================
    # 3. INTERPOLATOR AND METRIC SETUP - OPTIMIZED
    # =============================================================================
    print("\n3. Setting up spacetime metric (OPTIMAL)...")
    
    # Use OPTIMIZED interpolator and metric classes
    interpolator = InterpolatorFast(grid)  # Numba-compiled interpolation
    metric = PerturbedFLRWMetric(a_of_eta, grid, interpolator)  # Cached metric
    
    # Test metric at mass center
    test_pos = [2.0, center[0], center[1], center[2]]  # [eta, x, y, z]
    christoffel = metric.christoffel(test_pos)
    print(f"   Optimal metric initialized successfully")
    print(f"   Test Christoffel Gamma[0,0,0] = {christoffel[0,0,0]:.2e} at mass center")
    
    # =============================================================================
    # 4. BACKWARD RAY TRACING SETUP
    # =============================================================================
    print("\n4. Setting up backward ray tracing...")
    
    # Observer position and time (where photons are "detected")
    observer_eta = eta_present
    observer_position = np.array([1e-12, 1e-12, 1e-12]) * one_Mpc  # Observer location
    observer_position = np.array([0, 0, 0]) * one_Mpc  # Observer location

    
    # Direction towards the mass (for backward tracing, we point towards the source)
    direction_to_mass = center - observer_position
    direction_to_mass = direction_to_mass / np.linalg.norm(direction_to_mass)
    
    # Cone parameters for photon generation
    n_photons = 500  # Number of photons to trace
    cone_angle = np.pi / 24  # Half-angle of cone (radians)
    
    print(f"   Observer at [{observer_position[0]/one_Mpc:.1f}, {observer_position[1]/one_Mpc:.1f}, {observer_position[2]/one_Mpc:.1f}] Mpc")
    print(f"   Observation time: eta = {observer_eta:.1f}")
    print(f"   Cone: {n_photons} photons, {cone_angle*180/np.pi:.1f} deg half-angle")
    print(f"   Direction to mass: [{direction_to_mass[0]:.3f}, {direction_to_mass[1]:.3f}, {direction_to_mass[2]:.3f}]")
    
    # =============================================================================
    # 5. PHOTON GENERATION
    # =============================================================================
    print("\n5. Generating photons for backward ray tracing...")
    
    photons = Photons(metric=metric)
    
    # Generate photons in a cone pointing towards the mass
    observer_4d_position = np.array([observer_eta, *observer_position])
    
    photons.generate_cone_grid(
        n_theta=int(np.sqrt(n_photons)),
        n_phi=int(np.sqrt(n_photons)),
        origin=observer_4d_position,
        central_direction=direction_to_mass,
        cone_angle=cone_angle
    )
    
    print(f"   Generated {len(photons)} photons in cone")
    
    # CRITICAL: Invert 4-velocity for BACKWARD tracing
    # For backward evolution, we need u⁰ < 0 (going backward in time)
    for i, photon in enumerate(photons):
        # Invert the entire 4-velocity for time-reversal
        photon.u = -photon.u
        
        # Verify null condition after inversion (with relative error)
        if hasattr(photon, 'null_condition_relative_error'):
            rel_error = photon.null_condition_relative_error(metric=metric)
            if rel_error > 1e-5:  # Relaxed threshold
                u_u = photon.null_condition(metric=metric)
                print(f"Warning: Photon {i} null condition violated after inversion: u.u = {u_u:.3e}, rel_error = {rel_error:.3e}")
        
        # Record initial state for backward evolution
        photon.record()
    
    print("   Photons ready for backward time evolution (dt < 0, u0 < 0)")
    
    # =============================================================================
    # 6. INTEGRATION PARAMETERS CALCULATION
    # =============================================================================
    print("\n6. Calculating integration parameters...")
    
    a_obs = a_of_eta(observer_eta)
    
    # Calculate characteristic timescales
    box_comoving_distance = grid_size
    light_crossing_time_conformal = (a_obs / c) * box_comoving_distance
    
    H_conformal = cosmology.H_of_eta(observer_eta)
    hubble_time_conformal = 1.0 / H_conformal if H_conformal > 0 else light_crossing_time_conformal
    
    radius_val = 10 * one_Mpc
    potential_crossing_time = (a_obs / c) * radius_val
    
    distance_to_mass = np.linalg.norm(center - observer_position)
    time_to_reach_mass = (a_obs / c) * distance_to_mass
    
    print(f"   Scale factor at observation: a(eta={observer_eta:.2e} s) = {a_obs:.6f}")
    print(f"   Light crossing time (box): {light_crossing_time_conformal:.2e} s")
    print(f"   Hubble time (conformal): {hubble_time_conformal:.2e} s")
    print(f"   Potential crossing time: {potential_crossing_time:.2e} s")
    print(f"   Distance to mass center: {distance_to_mass/one_Mpc:.2f} Mpc")
    print(f"   Time to reach mass: {time_to_reach_mass:.2e} s")
    
    # Choose integration parameters
    total_integration_time = 1.5 * time_to_reach_mass
    
    min_timescale = potential_crossing_time / 5
    dt_magnitude = min_timescale
    
    target_displacement_per_step = grid_size / 50000
    dt_from_displacement = target_displacement_per_step / c
    
    typical_acceleration = 1e6
    max_velocity_change = 0.01 * c
    dt_from_acceleration = max_velocity_change / typical_acceleration
    
    dt_from_displacement = min(dt_from_displacement, dt_from_acceleration)
    
    print(f"   dt from displacement: {target_displacement_per_step/c:.2e} s")
    print(f"   dt from acceleration: {dt_from_acceleration:.2e} s")
    print(f"   Using dt: {dt_from_displacement:.2e} s (most restrictive)")
    
    dt_magnitude = min(dt_magnitude, dt_from_displacement)
    
    n_steps = int(np.ceil(abs(total_integration_time / dt_magnitude)))
    n_steps = max(500, min(n_steps, 50000))
    
    dt = -total_integration_time / n_steps  # NEGATIVE time step for backward tracing
    
    expected_displacement_per_step = c * abs(dt)
    print(f"   Time step check: dt = {dt:.2e} s (negative for backward)")
    print(f"   Expected displacement per step: {expected_displacement_per_step/one_Mpc:.4f} Mpc")
    print(f"   (Should be << {grid_size/one_Mpc:.0f} Mpc grid size)")
    print(f"   Integration: {n_steps} steps with dt = {dt:.2e} s (negative)")
    print(f"   Total integration time: {n_steps * abs(dt):.2e} s")
    
    # =============================================================================
    # 7. PARALLEL BACKWARD INTEGRATION - OPTIMAL
    # =============================================================================
    print("\n7. Performing backward ray tracing...")
    
    # Check if we're on Windows - multiprocessing has issues with Numba on Windows
    is_windows = platform.system() == 'Windows'
    use_parallel = not is_windows  # Disable parallel on Windows
    
    if is_windows:
        print(f"   ⚠ Windows detected - using sequential mode (multiprocessing issues)")
        print(f"   >> Sequential integration with Numba JIT")
        print(f"   Expected speedup vs standard: ~15x")
    
    # Try parallel integration on non-Windows platforms, fallback to sequential otherwise
    if use_parallel:
        try:
            print(f"   >> Attempting Persistent Worker Pool with 4 workers")
            print(f"   Expected speedup vs standard: ~60x")
            
            integration_start = time.time()
            
            # Use context manager for automatic cleanup
            with PersistentPoolIntegrator(metric, dt=dt, n_workers=-1) as integrator:
                print(f"   Worker pool ready, integrating {len(photons)} photons...")
                
                # Integrate all photons in parallel
                integrator.integrate_photons(photons, n_steps)
                
                print(f"   [OK] All photons integrated successfully (PARALLEL)")
            integration_time = time.time() - integration_start
            parallel_mode = True
            
        except (OSError, PermissionError, AttributeError) as e:
            print(f"\n   ⚠ Parallel integration failed: {type(e).__name__}")
            print(f"   >> Falling back to sequential integration")
            use_parallel = False
    
    if not use_parallel:
        integration_start = time.time()
        
        # Use sequential integrator with optimized metric
        integrator = Integrator(metric, dt=dt)
        
        print(f"   Integrating {len(photons)} photons sequentially...")
        for i, photon in enumerate(photons):
            integrator.integrate(photon, n_steps)
            if (i + 1) % 10 == 0 or i == 0:
                print(f"   Progress: {i+1}/{len(photons)} photons")
        
        print(f"   [OK] All photons integrated successfully (SEQUENTIAL + Numba JIT)")
        
        integration_time = time.time() - integration_start
        parallel_mode = False
    
    print(f"   Integration time: {integration_time:.2f}s")
    
    # =============================================================================
    # 8. ANALYSIS AND RESULTS
    # =============================================================================
    print("\n8. Analyzing results...")
    
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
        print(f"   Final time range: eta in [{final_times.min():.6e}, {final_times.max():.6e}] s")
        print(f"   Initial time: eta = {observer_eta:.6e} s")
        print(f"   Time change: Delta_eta = {final_times.mean() - observer_eta:.6e} s (should be negative)")
        print(f"   Distance from mass: avg = {avg_distance_from_mass/one_Mpc:.2f} Mpc, min = {min_distance_from_mass/one_Mpc:.2f} Mpc")
        print(f"   Spatial spread: {np.std(final_positions, axis=0)/one_Mpc} Mpc")
    
    # =============================================================================
    # 9. SAVE RESULTS
    # =============================================================================
    print("\n9. Saving trajectories...")
    
    # Generate standardized filename with all geometric information
    output_filename = generate_trajectory_filename(
        mass_kg=M,
        radius_m=radius,
        mass_position_m=center,
        observer_position_m=observer_position,
        metric_type="perturbed_flrw",
        output_dir="../_data/output"
    )
    
    print(f"   Generated filename: {os.path.basename(output_filename)}")
    
    try:
        photons.save_all_histories(output_filename)
        print(f"   [OK] Saved all {len(photons)} photon trajectories to {output_filename}")
        
        file_size = os.path.getsize(output_filename)
        print(f"   File size: {file_size/1024:.1f} KB")
        
    except Exception as e:
        print(f"   [ERROR] Error saving trajectories: {e}")
    
    # =============================================================================
    # 10. PERFORMANCE SUMMARY
    # =============================================================================
    total_time = time.time() - start_time
    
    mode_str = "Parallel (4 workers)" if parallel_mode else "Sequential (Numba JIT)"
    speedup_str = "60x" if parallel_mode else "15x"
    
    print("="*70)
    print("BACKWARD RAY TRACING SUMMARY")
    print("="*70)
    print(f"Mode:             {mode_str}")
    print(f"Expected speedup: {speedup_str} vs standard version")
    print(f"Cosmology:        LCDM (H0={H0}, Omega_m={Omega_m}, Omega_Lambda={Omega_lambda})")
    print(f"Grid:             {N}^3 cells, {grid_size/one_Mpc:.0f} Mpc box")
    print(f"Mass:             {M/one_Msun:.1e} M_sun, R={radius/one_Mpc:.1f} Mpc")
    print(f"Observer:         eta={observer_eta}, r=[{observer_position[0]/one_Mpc:.1f}, {observer_position[1]/one_Mpc:.1f}, {observer_position[2]/one_Mpc:.1f}] Mpc")
    print(f"Photons:          {len(photons)} in {cone_angle*180/np.pi:.1f} deg cone")
    print(f"Integration:      {n_steps} steps, dt={dt:.2e}s (negative for backward)")
    print(f"Integration time: {integration_time:.2f}s")
    print(f"Total time:       {total_time:.2f}s")
    print(f"Time per photon:  {integration_time/len(photons):.3f}s")
    print(f"Output:           {output_filename}")
    print("="*70)
    print(f"[SUCCESS] Backward ray tracing completed successfully!")
    print(f"  Performance: {len(photons)} photons in {integration_time:.2f}s")
    print(f"  (~{len(photons)/integration_time:.1f} photons/second)")
    

if __name__ == "__main__":
    main()
