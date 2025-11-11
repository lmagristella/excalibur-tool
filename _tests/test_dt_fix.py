#!/usr/bin/env python3
"""
Quick test script to verify the dt fix without waiting for large grid computation.
"""

import numpy as np
from scipy import interpolate 
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from excalibur.grid.grid import Grid
from excalibur.grid.interpolator import Interpolator
from excalibur.metrics.perturbed_flrw_metric import PerturbedFLRWMetric
from excalibur.photon.photons import Photons
from excalibur.integration.integrator import Integrator
from excalibur.core.constants import *
from excalibur.core.cosmology import LCDM_Cosmology
from excalibur.objects.spherical_mass import spherical_mass

print("=== Quick dt Fix Test ===\n")

# Small grid for fast testing
N = 64  # Much smaller grid
grid_size = 2000 * one_Mpc
dx = dy = dz = grid_size / N

shape = (N, N, N)
spacing = (dx, dy, dz)
origin = (-grid_size/2, -grid_size/2, -grid_size/2)
grid = Grid(shape, spacing, origin)

print(f"Grid: {N}³ cells, {grid_size/one_Mpc:.0f} Mpc box")
print(f"Cell size: {dx/one_Mpc:.1f} Mpc")

# Quick potential field
x = y = z = np.linspace(-grid_size/2, grid_size/2, N)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

M = 1e30 * one_Msun
radius = 10 * one_Mpc
center = np.array([500.0, 500.0, 500.0]) * one_Mpc
spherical_halo = spherical_mass(M, radius, center)

print("Computing potential field...")
phi_field = spherical_halo.potential(X, Y, Z)
grid.add_field("Phi", phi_field)
print(f"✓ Potential computed")

# Setup cosmology
H0 = 70
Omega_m = 0.3
Omega_lambda = 0.7
cosmology = LCDM_Cosmology(H0, Omega_m=Omega_m, Omega_r=0, Omega_lambda=Omega_lambda)

eta_start = 1.0e25
eta_end = 5.0e26
eta_sample = np.linspace(eta_start, eta_end, 1000)
a_sample = cosmology.a_of_eta(eta_sample)
a_of_eta = interpolate.interp1d(eta_sample, a_sample, kind='cubic', fill_value="extrapolate")

# Setup metric
interpolator = Interpolator(grid)
metric = PerturbedFLRWMetric(a_of_eta, grid, interpolator)
print("✓ Metric initialized")

# Generate photons
observer_eta = 4.4e26
observer_position = np.array([0.0, 0.0, 0.0])
direction_to_mass = center - observer_position
direction_to_mass = direction_to_mass / np.linalg.norm(direction_to_mass)

photons = Photons()
observer_4d_position = np.array([observer_eta, *observer_position])

# Just 5 photons for quick test
photons.generate_cone_random(
    n_photons=5,
    origin=observer_4d_position,
    central_direction=direction_to_mass,
    cone_angle=np.pi / 12,
    energy=1.0
)

# Reverse velocities for backward tracing
# NO! This was the bug - don't reverse spatial velocities
# Just use negative dt instead
for photon in photons:
    # photon.u = -photon.u  # WRONG: sends photons opposite direction
    photon.record()

print(f"✓ Generated {len(photons)} test photons (will use dt < 0)")

# Calculate dt with the fix
a_obs = a_of_eta(observer_eta)
distance_to_mass = np.linalg.norm(center - observer_position)
time_to_reach_mass = (a_obs / c) * distance_to_mass
total_integration_time = 1.5 * time_to_reach_mass

# NEW CALCULATION
target_displacement_per_step = grid_size / 50000
dt_from_displacement = target_displacement_per_step / c

print(f"\n--- dt Calculation ---")
print(f"Distance to mass: {distance_to_mass/one_Mpc:.2f} Mpc")
print(f"Time to reach mass: {time_to_reach_mass:.2e} s")
print(f"Total integration time target: {total_integration_time:.2e} s")
print(f"Target displacement per step: {target_displacement_per_step/one_Mpc:.4f} Mpc")
print(f"dt from displacement: {dt_from_displacement:.2e} s")

n_steps = int(np.ceil(abs(total_integration_time / dt_from_displacement)))
n_steps = max(500, min(n_steps, 50000))
dt = -total_integration_time / n_steps  # NEGATIVE for backward tracing

expected_displacement_per_step = c * abs(dt)
print(f"Number of steps: {n_steps}")
print(f"Final dt: {dt:.2e} s (negative for backward)")
print(f"Expected displacement per step: {expected_displacement_per_step/one_Mpc:.4f} Mpc")
print(f"Grid size: {grid_size/one_Mpc:.0f} Mpc")
print(f"Ratio displacement/grid: {expected_displacement_per_step/grid_size:.6f}")

# Check initial photon state
photon = photons[0]
print(f"\n--- First Photon State ---")
print(f"Position: {photon.x}")
print(f"Velocity: {photon.u}")
print(f"Spatial velocity magnitude: {np.linalg.norm(photon.u[1:]):.2e} m/s")

# Try one integration step
integrator = Integrator(metric, dt=dt)

print(f"\n--- Testing Integration ---")
print(f"Attempting to integrate {len(photons)} photons for 50 steps...")

success_count = 0
for i, photon in enumerate(photons):
    try:
        integrator.integrate(photon, 50)
        success_count += 1
        final_pos = photon.history.states[-1][1:4]
        initial_pos = photon.history.states[0][1:4]
        displacement = np.linalg.norm(final_pos - initial_pos)
        print(f"  Photon {i}: SUCCESS - {len(photon.history.states)} states, displaced {displacement/one_Mpc:.2f} Mpc")
    except Exception as e:
        print(f"  Photon {i}: FAILED - {e}")

print(f"\n{success_count}/{len(photons)} photons successfully integrated")

if success_count > 0:
    print("\n✓ FIX SUCCESSFUL! Photons no longer exit grid immediately")
else:
    print("\n✗ FIX FAILED! Photons still exit grid")
