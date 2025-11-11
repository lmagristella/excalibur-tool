#!/usr/bin/env python3
"""
Test script to investigate strong field issues:
1. Why null condition is violated for strong masses
2. Why photons are not deflected despite strong potential
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from excalibur.grid.grid import Grid
from excalibur.grid.interpolator_fast import InterpolatorFast
from excalibur.metrics.perturbed_flrw_metric_fast import PerturbedFLRWMetricFast
from excalibur.photon.photon import Photon
from excalibur.core.constants import *
from excalibur.core.cosmology import LCDM_Cosmology
from excalibur.objects.spherical_mass import spherical_mass
from scipy import interpolate

print("="*70)
print("INVESTIGATION: Strong Field Issues")
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

# Setup grid with STRONG mass
N = 64  # Smaller for faster test
grid_size = 1000 * one_Mpc
dx = dy = dz = grid_size / N

shape = (N, N, N)
spacing = (dx, dy, dz)
origin = np.array([0, 0, 0]) * grid_size
grid = Grid(shape, spacing, origin)

x = y = z = np.linspace(0, grid_size, N)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# STRONG MASS - 10^20 solar masses
M_strong = 1e20 * one_Msun
radius = 10 * one_Mpc
center = np.array([0.5, 0.5, 0.5]) * grid_size

print("\n1. SETUP:")
print(f"   Mass: {M_strong/one_Msun:.2e} M_sun")
print(f"   Center: {center/one_Mpc} Mpc")

# Compute potential field
spherical_halo = spherical_mass(M_strong, radius, center)
phi_field = spherical_halo.potential(X, Y, Z)
grid.add_field("Phi", phi_field)

print(f"   Potential range: [{phi_field.min():.2e}, {phi_field.max():.2e}] m^2/s^2")

# Check potential at various distances
test_distances = [10, 50, 100, 500]  # Mpc
print("\n2. POTENTIAL ANALYSIS:")
for dist_mpc in test_distances:
    test_pos = center + np.array([dist_mpc * one_Mpc, 0, 0])
    # Use the potential method which accepts scalar/array inputs
    phi_value = spherical_halo.potential(test_pos[0], test_pos[1], test_pos[2])
    # If it returns an array, extract scalar
    if isinstance(phi_value, np.ndarray):
        phi_value = phi_value.item() if phi_value.size == 1 else phi_value[0]
    phi_normalized = phi_value / (c**2)
    print(f"   At r={dist_mpc} Mpc: Phi={phi_value:.2e} m^2/s^2, phi/c^2={phi_normalized:.3f}")
    if abs(phi_normalized) > 0.1:
        print(f"      WARNING: |phi/c^2| > 0.1, weak field approximation INVALID!")

# Setup metric
interpolator = InterpolatorFast(grid)
metric = PerturbedFLRWMetricFast(a_of_eta, grid, interpolator)

# Test photon at observer position
observer_eta = 4.4e26
observer_position = np.array([1e-12, 1e-12, 1e-12]) * one_Mpc
direction_to_mass = (center - observer_position) / np.linalg.norm(center - observer_position)

print("\n3. PHOTON INITIALIZATION TEST:")
print(f"   Observer at: {observer_position/one_Mpc} Mpc")
print(f"   Direction to mass: {direction_to_mass}")

# Initialize photon with corrected formula
u_spatial = direction_to_mass * c
u0 = np.linalg.norm(u_spatial) / c

print(f"   u0 = {u0:.3e}")
print(f"   |u_spatial| = {np.linalg.norm(u_spatial):.3e}")

# Check at observer position
observer_4d = np.array([observer_eta, *observer_position])
g = metric.metric_tensor(observer_4d)
phi_obs, grad_phi_obs, _ = interpolator.value_gradient_and_time_derivative(observer_position, "Phi", observer_eta)
phi_obs_normalized = phi_obs / (c**2)

print(f"   Potential at observer: Phi={phi_obs:.2e}, phi/c^2={phi_obs_normalized:.6f}")
print(f"   Gradient at observer: {grad_phi_obs}")

# Check metric components
a_obs = a_of_eta(observer_eta)
print(f"   Scale factor a = {a_obs:.6f}")
print(f"   g_00 = {g[0,0]:.3e} (should be ~ -a^2*c^2 = {-(a_obs*c)**2:.3e})")
print(f"   g_11 = {g[1,1]:.3e} (should be ~ a^2 = {a_obs**2:.3e})")

# Test null condition
u = np.array([u0, u_spatial[0], u_spatial[1], u_spatial[2]])
photon = Photon(observer_4d, u)
u_u = photon.null_condition(metric=metric)
rel_error = photon.null_condition_relative_error(metric=metric)

print(f"\n4. NULL CONDITION CHECK:")
print(f"   g_munu u^mu u^nu = {u_u:.3e}")
print(f"   Relative error = {rel_error:.3e}")

# Manual calculation
g00_term = g[0,0] * u[0]**2
g_spatial_term = g[1,1] * (u[1]**2 + u[2]**2 + u[3]**2)
print(f"   g_00 (u^0)^2 = {g00_term:.3e}")
print(f"   g_ii (u^i)^2 = {g_spatial_term:.3e}")
print(f"   Sum = {g00_term + g_spatial_term:.3e}")

# Now check at a point closer to the mass
print("\n5. TESTING NEAR MASS CENTER:")
near_center = center + np.array([50*one_Mpc, 0, 0])
test_4d = np.array([observer_eta, *near_center])

phi_near, grad_phi_near, _ = interpolator.value_gradient_and_time_derivative(near_center, "Phi", observer_eta)
phi_near_normalized = phi_near / (c**2)

print(f"   Position: 50 Mpc from mass center")
print(f"   Potential: Phi={phi_near:.2e}, phi/c^2={phi_near_normalized:.3f}")
print(f"   Gradient magnitude: {np.linalg.norm(grad_phi_near):.3e} m/s^2")

if abs(phi_near_normalized) > 0.1:
    print(f"   ERROR: Weak field approximation breaks down!")
    print(f"   The metric perturbation theory assumes |phi/c^2| << 1")
    print(f"   With |phi/c^2| = {abs(phi_near_normalized):.2f}, you need full GR!")

# Check Christoffel symbols
print("\n6. CHRISTOFFEL SYMBOLS CHECK:")
christoffel = metric.christoffel(test_4d)
print(f"   Gamma^1_00 = {christoffel[1,0,0]:.3e} (deflection term)")
print(f"   Gamma^0_11 = {christoffel[0,1,1]:.3e}")

# Expected from gradient: Gamma^i_00 ~ grad_phi / a^2
expected_gamma_100 = grad_phi_near[0] / (c**2) / (a_obs**2)
print(f"   Expected Gamma^1_00 ~ grad_phi_x/c^2/a^2 = {expected_gamma_100:.3e}")

print("\n7. GEODESIC ACCELERATION TEST:")
# Test geodesic equations
state = np.concatenate([test_4d, u])
derivs = metric.geodesic_equations(state)
du = derivs[4:]

print(f"   du^0/dlambda = {du[0]:.3e}")
print(f"   du^1/dlambda = {du[1]:.3e} (x-acceleration, should be non-zero!)")
print(f"   du^2/dlambda = {du[2]:.3e}")
print(f"   du^3/dlambda = {du[3]:.3e}")

if abs(du[1]) < 1e-10:
    print(f"   WARNING: Spatial acceleration is nearly ZERO!")
    print(f"   Photons will NOT be deflected!")

print("\n" + "="*70)
print("DIAGNOSIS SUMMARY:")
print("="*70)

issues = []

if abs(phi_near_normalized) > 0.1:
    issues.append("1. Weak field approximation INVALID for this mass")
    issues.append("   Solution: Use smaller mass or full GR metric")

if abs(du[1]) < abs(grad_phi_near[0]) * 1e-10:
    issues.append("2. Geodesic equations NOT producing deflection")
    issues.append("   Possible bug in Christoffel calculation")

if rel_error > 1e-3:
    issues.append("3. Null condition violated significantly")
    issues.append("   Check photon initialization or metric calculation")

if issues:
    print("ISSUES FOUND:")
    for issue in issues:
        print(f"   {issue}")
else:
    print("No major issues detected.")

print("="*70)
