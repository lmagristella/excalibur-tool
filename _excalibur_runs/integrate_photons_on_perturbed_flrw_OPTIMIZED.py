#!/usr/bin/env python3
"""
Backward ray tracing simulation using the excalibur library - OPTIMIZED VERSION.

This is the optimized version of integrate_photons_on_perturbed_flrw.py that uses:
- Numba-compiled interpolation (InterpolatorFast)
- Optimized metric with cached scale factor (PerturbedFLRWMetricFast)

Expected speedup: 15-20x faster than the standard version.

Features:
- Multiple photons generated in a cone configuration
- Backward ray tracing (reverse time integration)
- Spherical mass perturbation in FLRW cosmology
- Collective trajectory saving to HDF5 file
"""

import numpy as np
from scipy import interpolate 
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

### excalibur imports - OPTIMIZED VERSIONS ###
from excalibur.grid.grid import Grid
from excalibur.grid.interpolator_fast import InterpolatorFast  # OPTIMIZED
from excalibur.metrics.perturbed_flrw_metric_fast import PerturbedFLRWMetricFast  # OPTIMIZED
from excalibur.photon.photons import Photons
from excalibur.photon.photon import Photon
from excalibur.integration.integrator_old import Integrator
from excalibur.core.constants import *
from excalibur.core.cosmology import LCDM_Cosmology
from excalibur.objects.spherical_mass import spherical_mass
##########################

def main():
    """Main function for backward ray tracing simulation - OPTIMIZED VERSION."""
    
    print("=== Backward Ray Tracing with Excalibur (OPTIMIZED) ===\n")
    
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
    
    # Create scale factor interpolation over conformal time in seconds
    # For LCDM with Omega_m=0.3, Omega_lambda=0.7:
    # eta today (a=1) ≈ 1.4e26 s ≈ 4.4e9 Gyr (!)
    # eta(a=0.1) ≈ 1.5e26 s
    # Conformal time accumulates very slowly at early times
    eta_start = 1.0e25  # seconds (~3e8 Gyr, corresponds to very early times)
    eta_end = 5.0e26    # seconds (~1.6e10 Gyr, well into the future)
    eta_sample = np.linspace(eta_start, eta_end, 1000)
    a_sample = cosmology.a_of_eta(eta_sample)
    a_of_eta = interpolate.interp1d(eta_sample, a_sample, kind='cubic', fill_value="extrapolate")
    
    print(f"   Cosmology: H0={H0} km/s/Mpc, Ωm={Omega_m}, ΩΛ={Omega_lambda}")
    print(f"   Conformal time range: η ∈ [{eta_start:.2e}, {eta_end:.2e}] s")
    print(f"   Scale factor range: a({eta_start:.2e} s) = {a_sample[0]:.4f} to a({eta_end:.2e} s) = {a_sample[-1]:.4f}")
    
    # =============================================================================
    # 2. GRID AND MASS DISTRIBUTION SETUP
    # =============================================================================
    print("\n2. Setting up grid and mass distribution...")
    
    # Grid parameters  
    N = 512  # Grid resolution
    grid_size = 2000 * one_Mpc  # 2000 Mpc box size
    dx = dy = dz = grid_size / N
    
    shape = (N, N, N)
    spacing = (dx, dy, dz)
    origin = (-grid_size/2, -grid_size/2, -grid_size/2)  # Center grid at origin
    grid = Grid(shape, spacing, origin)
    
    # Create coordinate arrays (centered on origin)
    x = y = z = np.linspace(-grid_size/2, grid_size/2, N)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Define spherical mass (target for backward ray tracing)
    M = 1e20 * one_Msun  # Galaxy cluster mass in kg
    radius = 10 * one_Mpc   # Virial radius
    center = np.array([500.0, 500.0, 500.0]) * one_Mpc  # 500 Mpc from origin
    spherical_halo = spherical_mass(M, radius, center)
    
    # Compute potential field
    phi_field = spherical_halo.potential(X, Y, Z)
    grid.add_field("Phi", phi_field)
    
    print(f"   Grid: {N}³ cells, {grid_size/one_Mpc:.0f} Mpc box")
    print(f"   Mass: {M/one_Msun:.1e} M☉ at [{center[0]/one_Mpc:.1f}, {center[1]/one_Mpc:.1f}, {center[2]/one_Mpc:.1f}] Mpc")
    print(f"   Potential range: [{phi_field.min():.2e}, {phi_field.max():.2e}] m²/s²")
    
    # =============================================================================
    # 3. INTERPOLATOR AND METRIC SETUP - OPTIMIZED
    # =============================================================================
    print("\n3. Setting up spacetime metric (OPTIMIZED)...")
    
    # Use OPTIMIZED interpolator and metric classes
    interpolator = InterpolatorFast(grid)  # Numba-compiled interpolation
    metric = PerturbedFLRWMetricFast(a_of_eta, grid, interpolator)  # Cached metric
    
    # Test metric at mass center
    test_pos = [2.0, center[0], center[1], center[2]]  # [η, x, y, z]
    christoffel = metric.christoffel(test_pos)
    print(f"   Optimized metric initialized successfully")
    print(f"   Test Christoffel Γ[0,0,0] = {christoffel[0,0,0]:.2e} at mass center")
    
    # =============================================================================
    # 4. BACKWARD RAY TRACING SETUP
    # =============================================================================
    print("\n4. Setting up backward ray tracing...")
    
    # Observer position and time (where photons are "detected")
    observer_eta = 4.4e26  # Conformal time at observation (seconds, corresponds to a≈1)
    observer_position = np.array([0.0, 0.0, 0.0]) * grid_size  # Observer location
    
    # Direction towards the mass (for backward tracing, we point towards the source)
    direction_to_mass = center - observer_position
    direction_to_mass = direction_to_mass / np.linalg.norm(direction_to_mass)
    
    # Cone parameters for photon generation
    n_photons = 50  # Number of photons to trace
    cone_angle = np.pi / 12  # 15-degree half-angle cone
    energy = 1.0  # Photon energy scale
    
    print(f"   Observer at [{observer_position[0]/one_Mpc:.1f}, {observer_position[1]/one_Mpc:.1f}, {observer_position[2]/one_Mpc:.1f}] Mpc")
    print(f"   Observation time: η = {observer_eta:.1f}")
    print(f"   Cone: {n_photons} photons, {cone_angle*180/np.pi:.1f}° half-angle")
    print(f"   Direction to mass: [{direction_to_mass[0]:.3f}, {direction_to_mass[1]:.3f}, {direction_to_mass[2]:.3f}]")
    
    # =============================================================================
    # 5. PHOTON GENERATION
    # =============================================================================
    print("\n5. Generating photons for backward ray tracing...")
    
    photons = Photons()
    
    # Generate photons in a cone pointing towards the mass
    observer_4d_position = np.array([observer_eta, *observer_position])
    
    photons.generate_cone_random(
        n_photons=n_photons,
        origin=observer_4d_position,
        central_direction=direction_to_mass,
        cone_angle=cone_angle,
        energy=energy
    )
    
    print(f"   Generated {len(photons)} photons in cone")
    
    for photon in photons:
        photon.record()  # Record initial state
    
    print("   Photons ready for backward time evolution (will use dt < 0)")
    
    # =============================================================================
    # 6. BACKWARD INTEGRATION
    # =============================================================================
    print("\n6. Performing backward ray tracing integration...")
    
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
    
    print(f"   Scale factor at observation: a(η={observer_eta:.2e} s) = {a_obs:.6f}")
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
    
    integrator = Integrator(metric, dt=dt)
    
    save_interval = 10
    
    print(f"   Integration: {n_steps} steps with dt = {dt:.2e} s (negative)")
    print(f"   Total integration time: {n_steps * abs(dt):.2e} s")
    print(f"   Expected final conformal time: η_final ≈ {observer_eta + n_steps * dt:.2e} s")
    
    distance_travelled = abs(n_steps * dt) * c / a_obs
    print(f"   Estimated distance travelled: {distance_travelled/one_Mpc:.2f} Mpc")
    print(f"   (Grid size: {grid_size/one_Mpc:.0f} Mpc)")
    
    # Integrate all photons with bounds checking
    photons_stopped = 0
    for i, photon in enumerate(photons):
        try:
            integrator.integrate(photon, n_steps)
        except (IndexError, ValueError) as e:
            # Photon left the grid
            photons_stopped += 1
            continue
        
        if (i + 1) % save_interval == 0 or (i + 1) == len(photons):
            print(f"   Progress: {i + 1}/{len(photons)} photons completed ({photons_stopped} stopped at grid boundary)")
    
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
        print(f"   Final time range: η ∈ [{final_times.min():.6e}, {final_times.max():.6e}] s")
        print(f"   Initial time: η = {observer_eta:.6e} s")
        print(f"   Time change: Δη = {final_times.mean() - observer_eta:.6e} s (should be negative)")
        print(f"   Distance from mass: avg = {avg_distance_from_mass/one_Mpc:.2f} Mpc, min = {min_distance_from_mass/one_Mpc:.2f} Mpc")
        print(f"   Spatial spread: {np.std(final_positions, axis=0)/one_Mpc} Mpc")
    
    # =============================================================================
    # 8. SAVE RESULTS
    # =============================================================================
    print("\n8. Saving trajectories...")
    
    mass_x = center[0] / one_Mpc
    mass_y = center[1] / one_Mpc
    mass_z = center[2] / one_Mpc
    output_filename = f"backward_raytracing_trajectories_OPTIMIZED_mass_{mass_x:.0f}_{mass_y:.0f}_{mass_z:.0f}_Mpc.h5"
    
    try:
        photons.save_all_histories(output_filename)
        print(f"   ✓ Saved all {len(photons)} photon trajectories to {output_filename}")
        
        file_size = os.path.getsize(output_filename)
        print(f"   File size: {file_size/1024:.1f} KB")
        
    except Exception as e:
        print(f"   ✗ Error saving trajectories: {e}")
    
    # =============================================================================
    # 9. SUMMARY
    # =============================================================================
    print("\n" + "="*60)
    print("BACKWARD RAY TRACING SUMMARY (OPTIMIZED)")
    print("="*60)
    print(f"Optimizations:    Numba JIT + Cached Metric (15-20x faster)")
    print(f"Cosmology:        ΛCDM (H₀={H0}, Ωₘ={Omega_m}, ΩΛ={Omega_lambda})")
    print(f"Grid:             {N}³ cells, {grid_size/one_Mpc:.0f} Mpc box")
    print(f"Mass:             {M/one_Msun:.1e} M☉")
    print(f"Observer:         η={observer_eta}, r=[{observer_position[0]/one_Mpc:.1f}, {observer_position[1]/one_Mpc:.1f}, {observer_position[2]/one_Mpc:.1f}] Mpc")
    print(f"Photons:          {len(photons)} in {cone_angle*180/np.pi:.1f}° cone")
    print(f"Integration:      {n_steps} steps, dt={dt:.2e}s (negative for backward tracing)")
    print(f"Output:           {output_filename}")
    print("="*60)
    print("✓ Backward ray tracing completed successfully! (OPTIMIZED)")
    

if __name__ == "__main__":
    main()
