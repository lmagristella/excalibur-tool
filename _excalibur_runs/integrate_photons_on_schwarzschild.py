#!/usr/bin/env python3
"""
Backward ray tracing simulation with Schwarzschild metric.

This script demonstrates backward ray tracing in Schwarzschild spacetime
around a point mass. Photons are manually initialized to satisfy the
null condition in Schwarzschild coordinates.

Key differences from FLRW:
- No grid needed (analytical metric)
- No scale factor evolution
- Proper Schwarzschild null condition initialization
"""

import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

### excalibur imports ###
from excalibur.metrics.schwarzschild_metric_cartesian import SchwarzschildMetricCartesian
from excalibur.photon.photons import Photons
from excalibur.photon.photon import Photon
from excalibur.integration.parallel_integrator import ParallelIntegrator
from excalibur.core.constants import *
##########################


def generate_photons_schwarzschild_cone(metric, n_photons, origin, central_direction, cone_angle):
    """
    Generate photons in a cone for Schwarzschild metric in CARTESIAN coordinates.
    
    Args:
        metric: SchwarzschildMetricCartesian instance
        n_photons: number of photons to generate
        origin: 4D position [t, x, y, z] in CARTESIAN COORDINATES
        central_direction: 3D direction vector (will be normalized) in CARTESIAN
        cone_angle: half-angle of cone in radians
    
    Returns:
        list of Photon objects
    """
    photons = []
    
    # Normalize central direction
    central_dir = np.asarray(central_direction, dtype=float)
    central_dir = central_dir / np.linalg.norm(central_dir)
    
    # Create grid in cone
    n_theta = int(np.sqrt(n_photons))
    n_phi = int(np.sqrt(n_photons))
    
    theta_values = np.linspace(0, cone_angle, n_theta)
    phi_values = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    
    for theta_cone in theta_values:
        for phi_cone in phi_values:
            # Skip duplicate central photons
            if theta_cone == 0 and phi_cone != phi_values[0]:
                continue
            
            # Direction in cone coordinates
            direction_cone = np.array([
                np.sin(theta_cone) * np.cos(phi_cone),
                np.sin(theta_cone) * np.sin(phi_cone),
                np.cos(theta_cone)
            ])
            
            # Rotate to align with central direction
            direction_3d = _rotate_to_direction(direction_cone, central_dir)
            direction_3d = direction_3d / np.linalg.norm(direction_3d)
            
            # Debug: verify norm
            norm_check = np.linalg.norm(direction_3d)
            if abs(norm_check - 1.0) > 1e-6:
                print(f"   Warning: direction norm = {norm_check}, should be 1")
            
            # Get metric at this position
            g = metric.metric_tensor(origin)
            
            # For a photon moving in direction_3d:
            # Set spatial velocity u^i = c * direction_3d[i] (in m/s)
            u_spatial = direction_3d * c
            
            # Compute spatial part of null condition: g_ij u^i u^j
            spatial_norm_sq = 0.0
            for i in range(3):
                for j in range(3):
                    spatial_norm_sq += g[i+1, j+1] * u_spatial[i] * u_spatial[j]
            

            
            if g[0, 0] < 0:  # Ensure timelike signature
                u0_squared = -spatial_norm_sq / g[0, 0]
                if u0_squared > 0:
                    u0 = np.sqrt(u0_squared)
                else:
                    print(f"         Warning: u0_squared = {u0_squared:.3e} < 0, using fallback")
                    u0 = 1.0  # Fallback - dimensionless
            else:
                u0 = 1.0  # Fallback
            
            # Build full 4-velocity
            full_4_velocity = np.array([u0, *u_spatial], dtype=float)
            
            # Create photon
            photon = Photon(position=origin, direction=full_4_velocity)
            
            photons.append(photon)
    
    return photons


def _rotate_to_direction(v, target):
    """
    Rotate vector v to align z-axis with target direction.
    
    Args:
        v: vector in original frame
        target: target direction (normalized)
    
    Returns:
        rotated vector
    """
    z_axis = np.array([0, 0, 1])
    
    # If target is already z-axis, no rotation needed
    if np.allclose(target, z_axis):
        return v
    
    # If target is opposite to z-axis
    if np.allclose(target, -z_axis):
        return np.array([v[0], v[1], -v[2]])
    
    # Rotation axis: z × target
    axis = np.cross(z_axis, target)
    axis = axis / np.linalg.norm(axis)
    
    # Rotation angle
    angle = np.arccos(np.clip(np.dot(z_axis, target), -1.0, 1.0))
    
    # Rodrigues' rotation formula
    v_rot = (v * np.cos(angle) + 
             np.cross(axis, v) * np.sin(angle) +
             axis * np.dot(axis, v) * (1 - np.cos(angle)))
    
    return v_rot


def main():
    """Main function for Schwarzschild backward ray tracing."""
    
    print("=== Backward Ray Tracing with Schwarzschild Metric ===")
    print("    Using analytical Schwarzschild solution")
    print()
    
    start_time = time.time()
    
    # =============================================================================
    # 1. SCHWARZSCHILD METRIC SETUP
    # =============================================================================
    print("1. Setting up Schwarzschild metric...")
    
    # Mass parameters
    M = 1e15 * one_Msun  # Galaxy cluster mass in kg
    radius = 5 * one_Mpc   # Characteristic radius
    center = np.array([500, 500, 500]) * one_Mpc  # Center in Cartesian coordinates
    
    # Create Schwarzschild metric (Cartesian version)
    metric = SchwarzschildMetricCartesian(M, radius, center)
    
    # Schwarzschild radius
    r_s = 2 * G * M / c**2
    
    print(f"   Mass: {M/one_Msun:.1e} M_sun")
    print(f"   Schwarzschild radius: {r_s/one_Mpc:.2e} Mpc ({r_s/1e3:.2f} km)")
    print(f"   Center: [{center[0]/one_Mpc:.1f}, {center[1]/one_Mpc:.1f}, {center[2]/one_Mpc:.1f}] Mpc")
    
    # Test metric at a distance from center
    test_pos = [0.0, center[0], center[1], center[2] + 100*one_Mpc]  # [t, x, y, z]
    christoffel = metric.christoffel(test_pos)
    print(f"   Test Christoffel Gamma[0,0,0] = {christoffel[0,0,0]:.2e} at r=100 Mpc")
    
    # =============================================================================
    # 2. BACKWARD RAY TRACING SETUP
    # =============================================================================
    print("\n2. Setting up backward ray tracing...")
    
    # Observer position and time (where photons are "detected")
    observer_t = 0.0  # Coordinate time at observation (Schwarzschild time)
    observer_position = np.array([1, 1, 1]) * one_Mpc  # Observer location (near origin)
    
    # Direction towards the mass (for backward tracing, we point towards the source)
    direction_to_mass = center - observer_position
    direction_to_mass = direction_to_mass / np.linalg.norm(direction_to_mass)
    
    # Cone parameters for photon generation
    n_photons = 43  # Number of photons in grid
    cone_angle = np.pi / 24  # Half-angle of cone (7.5 degrees)
    
    print(f"   Observer at [{observer_position[0]/one_Mpc:.1f}, {observer_position[1]/one_Mpc:.1f}, {observer_position[2]/one_Mpc:.1f}] Mpc")
    print(f"   Observation time: t = {observer_t:.1f} s")
    print(f"   Cone: {n_photons} photons, {cone_angle*180/np.pi:.1f} deg half-angle")
    print(f"   Direction to mass: [{direction_to_mass[0]:.3f}, {direction_to_mass[1]:.3f}, {direction_to_mass[2]:.3f}]")
    
    # =============================================================================
    # 3. PHOTON GENERATION (SCHWARZSCHILD-SPECIFIC)
    # =============================================================================
    print("\n3. Generating photons with Schwarzschild null condition...")
    
    # Create observer 4D position [t, x, y, z]
    observer_4d_position = np.array([observer_t, *observer_position])
    
    # Generate photons using custom Schwarzschild initialization
    photon_list = generate_photons_schwarzschild_cone(
        metric=metric,
        n_photons=n_photons,
        origin=observer_4d_position,
        central_direction=direction_to_mass,
        cone_angle=cone_angle
    )
    
    photons = Photons(metric=metric)
    for photon in photon_list:
        photons.add_photon(photon)
    
    print(f"   Generated {len(photons)} photons in cone")
    
    # Verify null conditions
    print("\n   Checking null conditions...")
    null_violations = []
    for i, photon in enumerate(photons):
        u_u = photon.null_condition(metric=metric)
        null_violations.append(abs(u_u))
        if abs(u_u) > 1e-5:
            print(f"   Warning: Photon {i} null condition: u.u = {u_u:.3e}")
            # Debug: print details
            if i == 0:
                print(f"      Position: {photon.x}")
                print(f"      Velocity: {photon.u}")
                g = metric.metric_tensor(photon.x)
                print(f"      g_00 = {g[0,0]:.3e}, g_11 = {g[1,1]:.3e}")
    
    print(f"   Null condition range: [{min(null_violations):.2e}, {max(null_violations):.2e}]")
    
    # CRITICAL: Invert 4-velocity for BACKWARD tracing
    print("\n   Inverting velocities for backward time evolution...")
    for photon in photons:
        photon.u = -photon.u
        photon.record()
    
    print("   Photons ready for backward time evolution (dt < 0, u^t < 0)")
    
    # =============================================================================
    # 4. INTEGRATION PARAMETERS
    # =============================================================================
    print("\n4. Calculating integration parameters...")
    
    # Calculate distance to mass and characteristic time
    distance_to_mass = np.linalg.norm(center - observer_position)
    time_to_reach_mass = distance_to_mass / c
    
    print(f"   Distance to mass center: {distance_to_mass/one_Mpc:.2f} Mpc")
    print(f"   Light travel time: {time_to_reach_mass:.2e} s ({time_to_reach_mass/(365.25*24*3600):.2f} years)")
    
    # Choose integration parameters
    # We want the photon to trace back ~1.5x the distance to the mass
    total_integration_time = 1.5 * time_to_reach_mass
    
    # Time step: CRITICAL - must be small enough for numerical stability
    # For geodesic integration, we want Δx ~ 0.01 * characteristic_length per step
    # With photon moving at c: Δx = c * Δt, so Δt = Δx / c
    
    # Use conservative step size: ~0.01 Mpc per step
    target_step_size = 0.01 * one_Mpc  # meters
    dt_magnitude = target_step_size / c  # seconds
    
    # Calculate number of steps
    n_steps = int(np.ceil(abs(total_integration_time / dt_magnitude)))
    n_steps = max(100, min(n_steps, 10000))  # Up to 10k steps
    
    # NEGATIVE time step for backward tracing
    dt = -total_integration_time / n_steps
    
    expected_displacement_per_step = c * abs(dt)
    print(f"   Time step: dt = {dt:.2e} s (negative for backward)")
    print(f"   Expected displacement per step: {expected_displacement_per_step/one_Mpc:.4f} Mpc")
    print(f"   Integration: {n_steps} steps over {abs(total_integration_time):.2e} s")
    
    # =============================================================================
    # 5. PARALLEL BACKWARD INTEGRATION
    # =============================================================================
    print("\n5. Performing backward ray tracing...")
    
    # First, test with a single photon to see any errors
    print(f"   Testing with first photon...")
    from excalibur.integration.integrator_old import Integrator
    test_integrator = Integrator(metric, dt=dt)
    
    try:
        test_integrator.integrate(photons.photons[0], min(10, n_steps))
        print(f"   Test integration successful! Photon moved.")
        print(f"      History length: {len(photons.photons[0].history.states)}")
        if len(photons.photons[0].history.states) > 1:
            print(f"      Initial pos: {photons.photons[0].history.states[0][1:4] / one_Mpc} Mpc")
            print(f"      Final pos: {photons.photons[0].history.states[-1][1:4] / one_Mpc} Mpc")
    except Exception as e:
        print(f"   ERROR in test integration: {e}")
        import traceback
        traceback.print_exc()
        return
    
    
    integration_start = time.time()
    
    # Use parallel integrator for better performance
    parallel_integrator = ParallelIntegrator(metric, dt=dt, n_workers=4)
    print(f"   Integrating {len(photons)} photons in parallel with 4 workers...")
    
    # Integrate all photons in parallel
    n_success = parallel_integrator.integrate_photons(photons, n_steps)
    
    print(f"\n   [OK] {n_success}/{len(photons)} photons integrated successfully")
    
    integration_time = time.time() - integration_start
    print(f"   Integration time: {integration_time:.2f}s")
    
    # =============================================================================
    # 6. ANALYSIS AND RESULTS
    # =============================================================================
    print("\n6. Analyzing results...")
    
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
        print(f"   Final time range: t in [{final_times.min():.6e}, {final_times.max():.6e}] s")
        print(f"   Initial time: t = {observer_t:.6e} s")
        print(f"   Time change: Delta_t = {final_times.mean() - observer_t:.6e} s (should be negative)")
        print(f"   Distance from mass: avg = {avg_distance_from_mass/one_Mpc:.2f} Mpc, min = {min_distance_from_mass/one_Mpc:.2f} Mpc")
        print(f"   Spatial spread: {np.std(final_positions, axis=0)/one_Mpc} Mpc")
    
    # =============================================================================
    # 7. SAVE RESULTS
    # =============================================================================
    print("\n7. Saving trajectories...")
    
    mass_x = center[0] / one_Mpc
    mass_y = center[1] / one_Mpc
    mass_z = center[2] / one_Mpc
    output_filename = f"backward_raytracing_schwarzschild_mass_{mass_x:.0f}_{mass_y:.0f}_{mass_z:.0f}_Mpc.h5"
    
    try:
        photons.save_all_histories(output_filename)
        print(f"   [OK] Saved all {len(photons)} photon trajectories to {output_filename}")
        
        file_size = os.path.getsize(output_filename)
        print(f"   File size: {file_size/1024:.1f} KB")
        
    except Exception as e:
        print(f"   [ERROR] Error saving trajectories: {e}")
    
    # =============================================================================
    # 8. SUMMARY
    # =============================================================================
    total_time = time.time() - start_time
    
    print("="*70)
    print("SCHWARZSCHILD BACKWARD RAY TRACING SUMMARY")
    print("="*70)
    print(f"Metric:           Schwarzschild analytical")
    print(f"Mass:             {M/one_Msun:.1e} M_sun")
    print(f"Center:           [{center[0]/one_Mpc:.1f}, {center[1]/one_Mpc:.1f}, {center[2]/one_Mpc:.1f}] Mpc")
    print(f"Observer:         t={observer_t}, r=[{observer_position[0]/one_Mpc:.1f}, {observer_position[1]/one_Mpc:.1f}, {observer_position[2]/one_Mpc:.1f}] Mpc")
    print(f"Photons:          {len(photons)} in {cone_angle*180/np.pi:.1f} deg cone")
    print(f"Integration:      {n_steps} steps, dt={dt:.2e}s (negative for backward)")
    print(f"Workers:          4 parallel workers")
    print(f"Integration time: {integration_time:.2f}s")
    print(f"Total time:       {total_time:.2f}s")
    print(f"Time per photon:  {integration_time/len(photons):.3f}s")
    print(f"Output:           {output_filename}")
    print("="*70)
    print("[SUCCESS] Schwarzschild backward ray tracing completed!")
    print(f"  Performance: {len(photons)} photons in {integration_time:.2f}s")
    print(f"  (~{len(photons)/integration_time:.1f} photons/second)")
    

if __name__ == "__main__":
    main()
