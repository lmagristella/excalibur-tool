#!/usr/bin/env python3
"""
Backward ray tracing in PURE FLRW (no perturbations) - For validation.

This script generates trajectories in a homogeneous FLRW universe WITHOUT
any perturbations (Phi = 0 everywhere). This serves as a reference to 
validate the effects of perturbations in the perturbed simulations.

Key differences from perturbed version:
- NO mass distribution (M = 0)
- NO gravitational potential field (Phi = 0)
- Pure homogeneous expansion only

Expected results:
- z_H should match FLRW formula exactly: 1 + z = a_obs / a_em
- z_SW = 0 (no potential difference)
- z_ISW = 0 (no time-varying potential)
- z_total = z_H (only Hubble redshift)
"""

import numpy as np
from scipy import interpolate 
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

### excalibur imports ###
from excalibur.grid.grid import Grid
from excalibur.grid.interpolator_fast import InterpolatorFast
from excalibur.metrics.perturbed_flrw_metric import PerturbedFLRWMetric
from excalibur.photon.photons import Photons
from excalibur.photon.photon import Photon
from excalibur.integration.parallel_integrator_persistent import PersistentPoolIntegrator
from excalibur.core.constants import *
from excalibur.core.cosmology import LCDM_Cosmology
##########################

def main():
    """Main function for PURE FLRW backward ray tracing (NO PERTURBATIONS)."""
    
    print("=== Backward Ray Tracing in PURE FLRW (No Perturbations) ===")
    print("    This generates reference trajectories for validation")
    print()
    
    start_time = time.time()
    
    # =============================================================================
    # 1. COSMOLOGICAL SETUP
    # =============================================================================
    print("1. Setting up cosmology...")
    
    # Define ΛCDM cosmology (SAME as perturbed version)
    H0 = 70  # km/s/Mpc
    Omega_m = 0.3
    Omega_r = 0
    Omega_lambda = 0.7
    cosmology = LCDM_Cosmology(H0, Omega_m=Omega_m, Omega_r=Omega_r, Omega_lambda=Omega_lambda)
    
    # Get conformal time at present (a=1)
    _ = cosmology.a_of_eta(1e18)
    eta_present = cosmology._eta_at_a1
    
    # Create scale factor interpolation (SAME as perturbed version)
    eta_start = eta_present
    eta_end = 0.5 * eta_present
    eta_sample = np.linspace(eta_start, eta_end, 1000)
    a_sample = cosmology.a_of_eta(eta_sample)
    a_of_eta = interpolate.interp1d(eta_sample, a_sample, kind='cubic', fill_value="extrapolate")
    
    print(f"   Cosmology: H0={H0} km/s/Mpc, Omega_m={Omega_m}, Omega_Lambda={Omega_lambda}")
    print(f"   Conformal time at present: eta_0 = {eta_present:.3e} s")
    print(f"   Conformal time range: eta in [{eta_start:.2e}, {eta_end:.2e}] s")
    print(f"   Scale factor range: a = {a_sample[0]:.4f} to {a_sample[-1]:.4f}")
    
    # =============================================================================
    # 2. GRID SETUP (NO MASS DISTRIBUTION)
    # =============================================================================
    print("\n2. Setting up grid (NO mass, Phi = 0 everywhere)...")
    
    # Grid parameters (SAME as perturbed version for consistency)
    N = 512
    grid_size = 1000 * one_Mpc
    dx = dy = dz = grid_size / N
    
    shape = (N, N, N)
    spacing = (dx, dy, dz)
    origin = np.array([0, 0, 0]) * grid_size
    grid = Grid(shape, spacing, origin)
    
    # Create coordinate arrays
    x = y = z = np.linspace(0, grid_size, N)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # *** KEY DIFFERENCE: NO MASS - Phi = 0 everywhere ***
    print("   ⚠️  NO MASS DISTRIBUTION - Pure FLRW (Phi = 0)")
    Phi = np.zeros((N, N, N))  # Zero potential everywhere
    
    # Add zero field to grid
    grid.add_field("Phi", Phi)
    
    print(f"   Grid: {N}^3 = {N**3:,} points")
    print(f"   Grid size: {grid_size/one_Mpc:.0f} Mpc")
    print(f"   Resolution: dx = {dx/one_Mpc:.3f} Mpc")
    print(f"   Phi statistics:")
    print(f"      min  = {np.min(Phi):.6e} m²/s²")
    print(f"      max  = {np.max(Phi):.6e} m²/s²")
    print(f"      mean = {np.mean(Phi):.6e} m²/s²")
    print(f"      std  = {np.std(Phi):.6e} m²/s²")
    
    # =============================================================================
    # 3. INTERPOLATOR AND METRIC
    # =============================================================================
    print("\n3. Setting up interpolator and metric...")
    
    # Create fast interpolator (even though Phi=0, keep same structure)
    interpolator = InterpolatorFast(grid)
    print("   ✓ Using InterpolatorFast (Numba JIT compilation)")
    
    # Create metric with ZERO potential
    metric = PerturbedFLRWMetric(a_of_eta, grid, interpolator)
    print("   ✓ Using PerturbedFLRWMetric (but Phi=0 everywhere)")
    
    # =============================================================================
    # 4. PHOTON INITIAL CONDITIONS (EXACT SAME AS PERTURBED)
    # =============================================================================
    print("\n4. Setting up photon initial conditions (SAME as perturbed)...")
    
    # *** CRITICAL: Use EXACT SAME configuration as perturbed simulation ***
    
    # Observer position and time (SAME)
    observer_eta = eta_start
    observer_position = np.array([0, 0, 0]) * one_Mpc  # SAME: origin
    
    # Direction towards where the mass WOULD BE (even though Phi=0)
    # This ensures photons travel in the SAME directions as perturbed case
    center = np.array([0.5, 0.5, 0.5]) * grid_size  # SAME: 500 Mpc from origin
    direction_to_mass = center - observer_position
    direction_to_mass = direction_to_mass / np.linalg.norm(direction_to_mass)
    
    # Cone parameters (SAME)
    n_photons = 50  # SAME number
    cone_angle = np.pi / 24  # SAME half-angle (7.5 deg)
    
    print(f"   Observer at [{observer_position[0]/one_Mpc:.1f}, {observer_position[1]/one_Mpc:.1f}, {observer_position[2]/one_Mpc:.1f}] Mpc")
    print(f"   Observation time: eta = {observer_eta:.3e} s")
    print(f"   Cone: {n_photons} photons, {cone_angle*180/np.pi:.1f} deg half-angle")
    print(f"   Direction (towards absent mass): [{direction_to_mass[0]:.3f}, {direction_to_mass[1]:.3f}, {direction_to_mass[2]:.3f}]")
    
    # Create Photons collection
    photons = Photons(metric=metric)
    
    # Generate photons in cone (SAME method)
    observer_4d_position = np.array([observer_eta, *observer_position])
    
    photons.generate_cone_grid(
        n_theta=int(np.sqrt(n_photons)),
        n_phi=int(np.sqrt(n_photons)),
        origin=observer_4d_position,
        central_direction=direction_to_mass,
        cone_angle=cone_angle
    )
    
    print(f"   Generated {len(photons)} photons")
    
    # CRITICAL: Invert 4-velocity for BACKWARD tracing (SAME)
    for i, photon in enumerate(photons):
        photon.u = -photon.u  # Time-reversal
        
        # Verify null condition (SAME check)
        if hasattr(photon, 'null_condition_relative_error'):
            rel_error = photon.null_condition_relative_error(metric=metric)
            if rel_error > 1e-5:
                u_u = photon.null_condition(metric=metric)
                print(f"Warning: Photon {i} null condition violated: u.u = {u_u:.3e}, rel_error = {rel_error:.3e}")
        
        # Record initial state
        photon.record()
    
    print(f"   ✓ {len(photons)} photons ready (IDENTICAL configuration to perturbed case)")
    
    # =============================================================================
    # 5. INTEGRATION PARAMETERS (EXACT SAME CALCULATION AS PERTURBED)
    # =============================================================================
    print("\n5. Calculating integration parameters (SAME as perturbed)...")
    
    a_obs = a_of_eta(observer_eta)
    
    # Calculate characteristic timescales (SAME)
    box_comoving_distance = grid_size
    light_crossing_time_conformal = (a_obs / c) * box_comoving_distance
    
    H_conformal = cosmology.H_of_eta(observer_eta)
    hubble_time_conformal = 1.0 / H_conformal if H_conformal > 0 else light_crossing_time_conformal
    
    radius_val = 10 * one_Mpc
    potential_crossing_time = (a_obs / c) * radius_val
    
    distance_to_mass = np.linalg.norm(center - observer_position)
    time_to_reach_mass = (a_obs / c) * distance_to_mass
    
    print(f"   Scale factor at observation: a = {a_obs:.6f}")
    print(f"   Distance to (absent) mass center: {distance_to_mass/one_Mpc:.2f} Mpc")
    print(f"   Time to reach target: {time_to_reach_mass:.2e} s")
    
    # Choose integration parameters (SAME logic)
    total_integration_time = 1.5 * time_to_reach_mass
    
    min_timescale = potential_crossing_time / 5
    dt_magnitude = min_timescale
    
    target_displacement_per_step = grid_size / 50000
    dt_from_displacement = target_displacement_per_step / c
    
    typical_acceleration = 1e6
    max_velocity_change = 0.01 * c
    dt_from_acceleration = max_velocity_change / typical_acceleration
    
    dt_from_displacement = min(dt_from_displacement, dt_from_acceleration)
    
    dt_magnitude = min(dt_magnitude, dt_from_displacement)
    
    n_steps = int(np.ceil(abs(total_integration_time / dt_magnitude)))
    n_steps = max(500, min(n_steps, 50000))
    
    dt = -total_integration_time / n_steps  # NEGATIVE time step for backward tracing
    
    expected_displacement_per_step = c * abs(dt)
    print(f"   dt = {dt:.2e} s (negative for backward)")
    print(f"   n_steps = {n_steps:,}")
    print(f"   Expected displacement per step: {expected_displacement_per_step/one_Mpc:.4f} Mpc")
    print(f"   Total integration time: {n_steps * abs(dt):.2e} s")
    
    # =============================================================================
    # 6. INTEGRATION
    # =============================================================================
    print("\n6. Backward ray tracing integration...")
    integration_start = time.time()
    
    # Use context manager for automatic cleanup
    with PersistentPoolIntegrator(metric, dt=dt, n_workers=4) as integrator:
        print(f"   Worker pool ready, integrating {len(photons)} photons...")
        
        # Integrate all photons in parallel
        integrator.integrate_photons(photons, n_steps)
        
        print(f"   [OK] All photons integrated successfully")
    
    integration_time = time.time() - integration_start
    print(f"\n   ✓ Integration completed in {integration_time:.1f} seconds")
    
    # =============================================================================
    # 6. SAVE RESULTS
    # =============================================================================
    print("\n6. Saving results...")
    
    output_file = "backward_raytracing_trajectories_FLRW_pure.h5"
    
    try:
        photons.save_all_histories(output_file)
        print(f"   ✓ Saved to: {output_file}")
        
        # Print file size
        file_size = os.path.getsize(output_file) / (1024**2)
        print(f"   File size: {file_size:.1f} MB")
    except Exception as e:
        print(f"   ❌ Error saving: {e}")
    
    # =============================================================================
    # 7. SUMMARY
    # =============================================================================
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("PURE FLRW SIMULATION SUMMARY")
    print("="*70)
    print(f"Total runtime: {total_time:.1f} seconds")
    print(f"Integration time: {integration_time:.1f} seconds")
    print(f"Photons: {n_photons}")
    print(f"Steps per photon: {n_steps:,}")
    print(f"Total steps: {n_photons * n_steps:,}")
    print(f"Speed: {n_photons * n_steps / integration_time:,.0f} steps/second")
    print(f"\nOutput: {output_file} ({file_size:.1f} MB)")
    print("\n⚠️  NOTE: This is a PURE FLRW simulation (Phi = 0)")
    print("   Expected results:")
    print("   - z_total = z_H (only Hubble redshift)")
    print("   - z_SW = 0 (no Sachs-Wolfe effect)")
    print("   - z_ISW = 0 (no ISW effect)")
    print("="*70)


if __name__ == "__main__":
    main()
