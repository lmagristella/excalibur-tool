#!/usr/bin/env python3
"""
Backward ray tracing simulation using the excalibur library - NEW INTEGRATOR VERSION.

This version uses the new excalibur.integration.integrator module with:
- Multiple integration algorithms (RK45, RK4, Leapfrog4)
- Flexible stopping conditions (steps, redshift, scale factor, comoving distance)
- Enhanced filename generation with stop conditions support
- Performance-optimized parallel/sequential modes

Integration stopping modes:
- 'steps': Stop after N integration steps
- 'redshift': Stop when redshift z >= value  
- 'a': Stop when scale factor a <= value
- 'chi': Stop when comoving distance >= value

This version is RECOMMENDED for:
- Studies with specific stopping criteria
- Systematic parameter exploration
- Modern excalibur integrator performance
"""

import numpy as np
from scipy import interpolate 
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

### excalibur imports - NEW INTEGRATOR VERSION ###
from excalibur.grid.grid import Grid
from excalibur.grid.interpolator_fast import InterpolatorFast  # OPTIMIZED
from excalibur.grid.interpolator import Interpolator  # STANDARD
from excalibur.metrics.perturbed_flrw_metric_fast import PerturbedFLRWMetricFast  # OPTIMIZED
from excalibur.metrics.perturbed_flrw_metric import PerturbedFLRWMetric  # STANDARD
from excalibur.photon.photons import Photons
from excalibur.photon.photon import Photon
from excalibur.integration.integrator import Integrator  # NEW INTEGRATOR MODULE
from excalibur.core.constants import *
from excalibur.core.cosmology import LCDM_Cosmology
from excalibur.objects.spherical_mass import spherical_mass
from excalibur.io.filename_utils import generate_trajectory_filename  # ENHANCED VERSION
##########################

def main():
    """Main function for backward ray tracing simulation - NEW INTEGRATOR VERSION."""
    
    print("=== Backward Ray Tracing with Excalibur (NEW INTEGRATOR) ===")
    print("    Using excalibur.integration.integrator with flexible stopping conditions")
    print()
    
    start_time = time.time()
    
    # =============================================================================
    # INTEGRATION PARAMETERS - USER CONFIGURATION
    # =============================================================================
    print("0. Integration parameters configuration...")
    
    # Integration algorithm selection
    integrator_type = "rk4"  # Options: "rk45", "rk4", "leapfrog4"
    
    # Integration mode selection  
    integration_mode = "sequential"  # DEBUG: Use sequential for simpler debugging
    n_workers = 4  # Number of workers for parallel mode
    
    # Stopping condition configuration
    stop_mode = "steps"  # Options: "steps", "redshift", "a", "chi"
    
    if stop_mode == "steps":
        stop_value = 50  # Demo: Very few steps (was 500)
    elif stop_mode == "redshift":
        stop_value = 10.0  # Stop when z >= 10
    elif stop_mode == "a":
        stop_value = 0.1  # Stop when scale factor a <= 0.1  
    elif stop_mode == "chi":
        stop_value = 2000  # Stop when comoving distance >= 2000 Mpc (in Mpc units)
    
    # Integration tolerances - OPTIMIZED FOR SPEED
    # Relaxed tolerances for significant speedup while maintaining physical accuracy:
    rtol = 1e-6   # OPTIMIZED: Relaxed from 1e-8 for ~10x speedup  
    atol = 1e-10  # OPTIMIZED: Relaxed from 1e-12 for ~10x speedup  
    # dt_initial will be calculated based on physical timescales below
    dt_min = 1e-20  # Minimum timestep - allows fine resolution near masses
    dt_max = 1e15   # Maximum timestep - limits to reasonable cosmological scales
    
    print(f"   Integrator: {integrator_type}")
    print(f"   Mode: {integration_mode} ({n_workers} workers)" if integration_mode != "sequential" else f"   Mode: {integration_mode}")
    print(f"   Stop condition: {stop_mode} = {stop_value}")
    print(f"   Tolerances: rtol={rtol:.0e}, atol={atol:.0e}")
    print(f"   Timestep limits: dt_min={dt_min:.0e}, dt_max={dt_max:.0e} (dt_initial will be calculated)")
    
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
    
    # Grid parameters - REDUCED FOR DEMO  
    N = 128  # Grid resolution (reduced from 512)
    grid_size = 500 * one_Mpc  # 500 Mpc box size (reduced from 1000)
    dx = dy = dz = grid_size / N
    
    shape = (N, N, N)
    spacing = (dx, dy, dz)
    origin = np.array([0, 0, 0])*grid_size  # Center grid at origin
    grid = Grid(shape, spacing, origin)
    
    # Create coordinate arrays (centered on origin)
    x = y = z = np.linspace(0, grid_size, N)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Define spherical mass (target for backward ray tracing)
    M = 5e13 * one_Msun  # Smaller mass for demo (was 1e15)
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
    metric = PerturbedFLRWMetric(cosmology, interpolator)  # OPTIMIZED: Pass cosmology for fast adot
    
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
    #observer_position = np.array([0, 0, 0]) * one_Mpc  # Observer location

    
    # Direction towards the mass (for backward tracing, we point towards the source)
    direction_to_mass = center - observer_position
    direction_to_mass = direction_to_mass / np.linalg.norm(direction_to_mass)
    
    # Cone parameters for photon generation
    n_photons = 5  # Reduced for demo (was 21)
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
    
    # =============================================================================
    # 6. INTEGRATION PARAMETERS CALCULATION (INSPIRED FROM OPTIMAL VERSION)
    # =============================================================================
    print("\n6. Calculating optimal integration parameters based on physical timescales...")
    
    a_obs = a_of_eta(observer_eta)
    
    box_comoving_distance = grid_size
    light_crossing_time_conformal = (a_obs / c) * box_comoving_distance
    
    H_conformal = cosmology.H_of_eta(observer_eta)
    hubble_time_conformal = 1.0 / H_conformal if H_conformal > 0 else light_crossing_time_conformal
    
    radius_val = radius  
    potential_crossing_time = (a_obs / c) * radius_val
    
    distance_to_mass = np.linalg.norm(center - observer_position)
    time_to_reach_mass = (a_obs / c) * distance_to_mass
    
    print(f"   Scale factor at observation: a(eta={observer_eta:.2e} s) = {a_obs:.6f}")
    print(f"   Light crossing time (box): {light_crossing_time_conformal:.2e} s = {light_crossing_time_conformal/(365.25*24*3600*1e9):.2f} Gyr")
    print(f"   Hubble time (conformal): {hubble_time_conformal:.2e} s = {hubble_time_conformal/(365.25*24*3600*1e9):.1f} Gyr")
    print(f"   Potential crossing time: {potential_crossing_time:.2e} s = {potential_crossing_time/(365.25*24*3600):.1f} days")
    print(f"   Distance to mass center: {distance_to_mass/one_Mpc:.2f} Mpc")
    print(f"   Time to reach mass: {time_to_reach_mass:.2e} s = {time_to_reach_mass/(365.25*24*3600*1e6):.1f} Myr")
    
 

    # CRITICAL FIX: Much smaller timestep to prevent grid overflow
    dt_initial = grid.spacing[0] / (10*c)
    
    expected_displacement_per_step = c * abs(dt_initial)
    print(f"   Time step check: dt_initial = {dt_initial:.2e} s (negative for backward)")
    print(f"   Expected displacement per step: {expected_displacement_per_step/one_Mpc:.4f} Mpc")
    print(f"   (Should be << {grid_size/one_Mpc:.0f} Mpc grid size)")
    print(f"   Calculated dt_initial = {dt_initial:.2e} s based on physical timescales")
    print(f"   Integration: {stop_mode} stopping condition")
    print(f"   Ready for new Integrator module...")

    # =============================================================================
    # 7. NEW INTEGRATOR SETUP AND BACKWARD INTEGRATION
    # =============================================================================
    print("\n7. Setting up new Integrator and performing backward ray tracing...")
    
    # Create new integrator instance with specified parameters
    integrator = Integrator(
        metric=metric,
        dt=-dt_initial,
        mode=integration_mode,
        integrator=integrator_type,
        rtol=rtol,
        atol=atol,
        dt_min=dt_min,   # CRITICAL FIX: Keep original positive limits
        dt_max=dt_max,   # CRITICAL FIX: Keep original positive limits
        n_workers=n_workers,
        chunk_size=100  # OPTIMIZED: Larger chunks reduce multiprocessing overhead
    )
    
    print(f"   Integrator created: {integrator_type} with {integration_mode} mode")
    print(f"   Stop condition: {stop_mode} = {stop_value}")
    
    integration_start = time.time()
    
    # Integrate all photons using the new integrator
    print(f"   Integrating {len(photons)} photons...")
    
    successful_integrations = integrator.integrate(
        photons, 
        stop_mode=stop_mode, 
        stop_value=stop_value,
        verbose=True
    )
    
    integration_time = time.time() - integration_start
    parallel_mode = (integration_mode in ["parallel", "chunked"])
    
    print(f"   [OK] Integration completed!")
    print(f"   Successful integrations: {successful_integrations}/{len(photons)} photons")
    print(f"   Integration time: {integration_time:.2f}s")
    print(f"   Performance: {len(photons)/integration_time:.1f} photons/second")
    
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
    # 9. SAVE RESULTS WITH ENHANCED FILENAME
    # =============================================================================
    print("\n9. Saving trajectories...")
    
    # Generate enhanced filename with all simulation parameters
    output_filename = generate_trajectory_filename(
        mass_kg=M,
        radius_m=radius,
        mass_position_m=center,
        observer_position_m=observer_position,
        metric_type="perturbed_flrw",
        n_photons=len(photons),
        integrator=integration_mode,
        stop_mode=stop_mode,
        stop_value=stop_value,
        output_dir="../_data/output"
    )
    
    print(f"   Enhanced filename: {os.path.basename(output_filename)}")
    print(f"   Includes: n_photons={len(photons)}, integrator={integration_mode}, {stop_mode}={stop_value}")
    
    try:
        photons.save_all_histories(output_filename)
        print(f"   [OK] Saved all {len(photons)} photon trajectories to {output_filename}")
        
        file_size = os.path.getsize(output_filename)
        print(f"   File size: {file_size/1024:.1f} KB")
        
    except Exception as e:
        print(f"   [ERROR] Error saving trajectories: {e}")
    
    # =============================================================================
    # 10. ENHANCED PERFORMANCE SUMMARY
    # =============================================================================
    total_time = time.time() - start_time
    
    mode_descriptions = {
        "sequential": "Sequential (single-threaded)",
        "parallel": f"Parallel ({n_workers} workers)",
        "chunked": f"Chunked ({n_workers} workers)"
    }
    
    stop_descriptions = {
        "steps": f"{int(stop_value)} steps",
        "redshift": f"z = {stop_value}",
        "a": f"a = {stop_value}",
        "chi": f"χ = {stop_value/one_Mpc:.1f} Mpc"
    }
    
    print("="*80)
    print("BACKWARD RAY TRACING SUMMARY (NEW INTEGRATOR)")
    print("="*80)
    print(f"Integration mode:    {mode_descriptions.get(integration_mode, integration_mode)}")
    print(f"Algorithm:           {integrator_type.upper()}")
    print(f"Stop condition:      {stop_descriptions.get(stop_mode, f'{stop_mode} = {stop_value}')}")
    print(f"Tolerances:          rtol={rtol:.0e}, atol={atol:.0e}")
    print(f"Success rate:        {successful_integrations}/{len(photons)} photons ({100*successful_integrations/len(photons):.1f}%)")
    print()
    print(f"Cosmology:           ΛCDM (H₀={H0} km/s/Mpc, Ωₘ={Omega_m}, ΩΛ={Omega_lambda})")
    print(f"Grid:                {N}³ cells, {grid_size/one_Mpc:.0f} Mpc box")
    print(f"Mass:                {M/one_Msun:.1e} M☉, R={radius/one_Mpc:.1f} Mpc")
    print(f"Observer position:   [{observer_position[0]/one_Mpc:.1f}, {observer_position[1]/one_Mpc:.1f}, {observer_position[2]/one_Mpc:.1f}] Mpc")
    print(f"Photon cone:         {len(photons)} photons, {cone_angle*180/np.pi:.1f}° half-angle")
    print()
    print(f"Integration time:    {integration_time:.2f}s")
    print(f"Total time:          {total_time:.2f}s")
    print(f"Performance:         {len(photons)/integration_time:.1f} photons/second")
    print(f"Avg time/photon:     {integration_time/len(photons):.3f}s")
    print()
    print(f"Output file:         {os.path.basename(output_filename)}")
    print("="*80)
    print(f"✅ [SUCCESS] Backward ray tracing completed with new integrator!")
    print(f"   Modern integration: {integrator_type} with {stop_mode} stopping")
    print(f"   Enhanced filenames: includes all simulation parameters")
    print(f"   Performance: {len(photons)} photons → {integration_time:.2f}s")
    

if __name__ == "__main__":
    main()
