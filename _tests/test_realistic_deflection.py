#!/usr/bin/env python3
"""
Test with a REALISTIC mass (10^16 solar masses - galaxy cluster scale)
to verify deflection works correctly.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from excalibur.grid.grid import Grid
from excalibur.grid.interpolator_fast import InterpolatorFast
from excalibur.metrics.perturbed_flrw_metric_fast import PerturbedFLRWMetricFast
from excalibur.photon.photon import Photon
from excalibur.integration.integrator import Integrator
from excalibur.core.constants import *
from excalibur.core.cosmology import LCDM_Cosmology
from excalibur.objects.spherical_mass import spherical_mass
from scipy import interpolate

print("="*70)
print("TEST: Photon Deflection with Realistic Mass")
print("="*70)

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

# Setup grid with REALISTIC mass
N = 128
grid_size = 1000 * one_Mpc
dx = dy = dz = grid_size / N

shape = (N, N, N)
spacing = (dx, dy, dz)
origin = np.array([0, 0, 0]) * grid_size
grid = Grid(shape, spacing, origin)

x = y = z = np.linspace(0, grid_size, N)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# REALISTIC MASS - 10^16 solar masses (large galaxy cluster)
M_realistic = 1e16 * one_Msun
radius = 10 * one_Mpc
center = np.array([0.5, 0.5, 0.5]) * grid_size

print("\n1. SETUP:")
print(f"   Mass: {M_realistic/one_Msun:.2e} M_sun (galaxy cluster)")
print(f"   Center: {center/one_Mpc} Mpc")

# Compute potential field
spherical_halo = spherical_mass(M_realistic, radius, center)
phi_field = spherical_halo.potential(X, Y, Z)
grid.add_field("Phi", phi_field)

print(f"   Potential range: [{phi_field.min():.2e}, {phi_field.max():.2e}] m^2/s^2")

# Check weak field validity
test_pos_50Mpc = center + np.array([50*one_Mpc, 0, 0])
phi_50 = spherical_halo.potential(test_pos_50Mpc[0], test_pos_50Mpc[1], test_pos_50Mpc[2])
if isinstance(phi_50, np.ndarray):
    phi_50 = phi_50.item() if phi_50.size == 1 else phi_50[0]
phi_50_norm = phi_50 / (c**2)
print(f"   At 50 Mpc: phi/c^2 = {phi_50_norm:.6f}")
if abs(phi_50_norm) < 0.01:
    print(f"   [OK] Weak field approximation VALID")
else:
    print(f"   [WARNING] Weak field approximation marginal")

# Setup metric
interpolator = InterpolatorFast(grid)
metric = PerturbedFLRWMetricFast(a_of_eta, grid, interpolator)

# Create photon that passes near the mass
observer_eta = 4.4e26
# Start photon 100 Mpc away, aimed to pass 50 Mpc from mass
start_pos = center + np.array([-100*one_Mpc, 0, 0])
# Direction: towards +x (will pass near mass)
direction = np.array([1.0, 0.0, 0.0])

print("\n2. PHOTON SETUP:")
print(f"   Starting position: {(start_pos - center)/one_Mpc} Mpc from mass")
print(f"   Initial direction: {direction}")
print(f"   Impact parameter: should pass ~50 Mpc from mass center")

# Initialize photon
u_spatial = direction * c
u0 = np.linalg.norm(u_spatial) / c
u = np.array([u0, u_spatial[0], u_spatial[1], u_spatial[2]])

start_4d = np.array([observer_eta, *start_pos])
photon = Photon(start_4d, u)
photon.record()

# Check initial null condition
rel_error_init = photon.null_condition_relative_error(metric=metric)
print(f"   Initial null condition error: {rel_error_init:.3e}")

# Integrate photon
print("\n3. INTEGRATING PHOTON:")
dt = 1e12  # 1e12 seconds per step
n_steps = 2000
integrator = Integrator(metric, dt=dt)

integrator.integrate(photon, n_steps)

print(f"   Integrated {n_steps} steps with dt={dt:.2e} s")
print(f"   Total trajectory points: {len(photon.history.states)}")

# Analyze trajectory
if len(photon.history.states) > 1:
    initial_state = photon.history.states[0]
    final_state = photon.history.states[-1]
    
    initial_pos = initial_state[1:4]
    final_pos = final_state[1:4]
    initial_vel = initial_state[4:7]
    final_vel = final_state[4:7]
    
    # Find closest approach to mass
    min_dist = float('inf')
    min_dist_state = None
    for state in photon.history.states:
        pos = state[1:4]
        dist = np.linalg.norm(pos - center)
        if dist < min_dist:
            min_dist = dist
            min_dist_state = state
    
    print("\n4. RESULTS:")
    print(f"   Initial position from mass: {np.linalg.norm(initial_pos - center)/one_Mpc:.1f} Mpc")
    print(f"   Closest approach: {min_dist/one_Mpc:.1f} Mpc")
    print(f"   Final position from mass: {np.linalg.norm(final_pos - center)/one_Mpc:.1f} Mpc")
    
    # Check deflection
    initial_dir = initial_vel / np.linalg.norm(initial_vel)
    final_dir = final_vel / np.linalg.norm(final_vel)
    deflection_angle = np.arccos(np.clip(np.dot(initial_dir, final_dir), -1, 1))
    
    print(f"\n5. DEFLECTION ANALYSIS:")
    print(f"   Initial direction: {initial_dir}")
    print(f"   Final direction: {final_dir}")
    print(f"   Deflection angle: {deflection_angle*180/np.pi:.6f} degrees")
    print(f"                     {deflection_angle*3600*180/np.pi:.3f} arcseconds")
    
    # Theoretical deflection (Einstein formula for weak field)
    # delta_theta ≈ 4GM / (b*c^2)
    b = min_dist  # impact parameter
    theoretical_deflection = 4 * G * M_realistic / (b * c**2)
    print(f"\n   Theoretical deflection (Einstein): {theoretical_deflection*180/np.pi:.6f} degrees")
    print(f"                                       {theoretical_deflection*3600*180/np.pi:.3f} arcseconds")
    
    ratio = deflection_angle / theoretical_deflection if theoretical_deflection > 0 else 0
    print(f"   Ratio (computed/theoretical): {ratio:.3f}")
    
    if 0.5 < ratio < 2.0:
        print(f"   [OK] Deflection matches theory within factor of 2")
    elif ratio > 0.1:
        print(f"   [WARNING] Deflection differs from theory")
    else:
        print(f"   [ERROR] No significant deflection detected!")

else:
    print("   [ERROR] Integration failed - only initial state recorded")

print("="*70)
