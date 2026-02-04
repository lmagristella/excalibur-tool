#!/usr/bin/env python3
"""
Debug null condition problem in OPTIMAL version.
"""

import numpy as np
from scipy import interpolate 
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from excalibur.grid.grid import Grid
from excalibur.grid.interpolator_fast import InterpolatorFast
from excalibur.metrics.perturbed_flrw_metric_fast import PerturbedFLRWMetricFast
from excalibur.metrics.perturbed_flrw_metric import PerturbedFLRWMetric
from excalibur.photon.photon import Photon
from excalibur.core.constants import *
from excalibur.core.cosmology import LCDM_Cosmology
from excalibur.objects.spherical_mass import spherical_mass

print("="*70)
print("NULL CONDITION DEBUG")
print("="*70)

# Setup cosmology
H0 = 70
Omega_m = 0.3
Omega_r = 0
Omega_lambda = 0.7
cosmology = LCDM_Cosmology(H0, Omega_m, Omega_r, Omega_lambda)

eta_start = 1.0 * one_Gyr
eta_end = 70.0 * one_Gyr
eta_sample = np.linspace(eta_start, eta_end, 1000)
a_sample = cosmology.a_of_eta(eta_sample)
a_of_eta = interpolate.interp1d(eta_sample, a_sample, kind='cubic')

print(f"\n1. Cosmology check")
# Use a realistic eta value within the interpolation range
# eta_test should be between eta_start and eta_end
eta_test = 46.0 * one_Gyr  # 50 Gyr, within [1, 70] Gyr range
a_test = a_of_eta(eta_test)
print(f"   η = {eta_test:.2e} s = {eta_test/one_Gyr:.1f} Gyr")
print(f"   a(η) = {a_test:.6f}")
print(f"   a² = {a_test**2:.3e}")
print(f"   Interpolation range: [{eta_start/one_Gyr:.1f}, {eta_end/one_Gyr:.1f}] Gyr")

# Small grid
N = 64
grid_size = 1000 * one_Mpc
dx = dy = dz = grid_size / N

shape = (N, N, N)
spacing = (dx, dy, dz)
origin = (-grid_size/2, -grid_size/2, -grid_size/2)
grid = Grid(shape, spacing, origin)

x = y = z = np.linspace(-grid_size/2, grid_size/2, N)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

M = 1e15 * one_Msun
radius = 10 * one_Mpc
center = np.array([0.5, 0.5, 0.5]) * grid_size
spherical_halo = spherical_mass(M, radius, center)

phi_field = spherical_halo.potential(X, Y, Z)
grid.add_field("Phi", phi_field/c**2)

print(f"\n2. Grid check")
print(f"   Grid: {N}³")
print(f"   Phi range: [{phi_field.min():.2e}, {phi_field.max():.2e}] m²/s²")

# Metric
interpolator = InterpolatorFast(grid)
metric = PerturbedFLRWMetric(a_of_eta, grid, interpolator)

print(f"\n3. Metric tensor check at observer")

# Observer position
observer_eta = 46 * one_Gyr
observer_position = np.array([0.0, 0.0, 0.0])
x_observer = np.array([observer_eta, *observer_position])

# Get metric
g = metric.metric_tensor(x_observer)

# Also get phi
phi_obs, _, _ = interpolator.value_gradient_and_time_derivative(observer_position, "Phi", observer_eta)
psi_obs = phi_obs

print(f"   Position: η={observer_eta:.2e}, (0, 0, 0)")
print(f"   a(η) = {a_of_eta(observer_eta):.6f}")
print(f"   Φ(observer) = {phi_obs:.3e} m²/s²")
print(f"   Ψ = Φ = {psi_obs:.3e}")
print(f"   1 + 2Ψ = {1 + 2*psi_obs:.3e}")
print(f"\n   Metric tensor:")
print(f"   g[0,0] = {g[0,0]:.6e}  (should be negative!)")
print(f"   g[1,1] = {g[1,1]:.6e}")
print(f"   g[2,2] = {g[2,2]:.6e}")
print(f"   g[3,3] = {g[3,3]:.6e}")

# Calculate what g[0,0] SHOULD be
a_obs = a_of_eta(observer_eta)
phi_obs, _, _ = interpolator.value_gradient_and_time_derivative(observer_position, "Phi", observer_eta)
psi_obs = phi_obs

g00_expected = -a_obs**2 * (1 + 2*psi_obs) * c**2
g00_current = -a_obs**2 * (1 + 2*psi_obs) / c**2  # BUG

print(f"\n   PROBLEM IDENTIFIED:")
print(f"   g[0,0] expected = -a²(1+2ψ)c² = {g00_expected:.6e}")
print(f"   g[0,0] current  = -a²(1+2ψ)/c² = {g00_current:.6e}")
print(f"   Ratio = {g00_current / g00_expected:.2e}")
print(f"   ERROR: Factor c⁴ = {c**4:.2e} wrong!")

# Test photon
print(f"\n4. Photon null condition test")

direction_to_mass = center - observer_position
direction_to_mass = direction_to_mass / np.linalg.norm(direction_to_mass)

# Create photon with proper null initialization
# For null geodesic: -c²(u⁰)² + a²(u_spatial)² = 0
# So: u⁰ = ±(a/c) * |u_spatial|

u_spatial_magnitude = c  # Spatial 3-velocity magnitude
u_spatial = direction_to_mass * u_spatial_magnitude

# Temporal component (positive for backward tracing)
u0 = -(a_obs / c) * u_spatial_magnitude  # Negative for backward
# Calculate u⁰ from null condition: g_μν u^μ u^ν = 0
# With the current (buggy) metric: g[0,0] = -a²(1+2ψ)/c²
# Null condition: g[0,0](u⁰)² + g[i,i](u^i)² = 0
# So: (u⁰)² = -g[i,i](u^i)² / g[0,0]

u_spatial_norm_sq = np.sum(u_spatial**2)
u0_squared = -g[1,1] * u_spatial_norm_sq / g[0,0]
u0_from_null = -np.sqrt(u0_squared)  # Negative for backward tracing

print(f"   u⁰ from null condition = {u0_from_null:.3e}")
print(f"   u⁰ from formula        = {u0:.3e}")
print(f"   Difference = {abs(u0 - u0_from_null):.3e}")

# Use the null-derived value
u0 = u0_from_null

u_full = np.array([u0, *u_spatial])

print(f"   Direction: {direction_to_mass}")
print(f"   u⁰ = {u0:.3e}")
print(f"   u¹ = {u_spatial[0]:.3e}")
print(f"   u² = {u_spatial[1]:.3e}")
print(f"   u³ = {u_spatial[2]:.3e}")

photon = Photon(x_observer, u_full)

# Check null condition
null_cond = photon.null_condition(metric)

print(f"\n   Null condition check:")
print(f"   u·u = g_μν u^μ u^ν = {null_cond:.3e}")
print(f"   Expected: ≈ 0")
print(f"   Status: {'✅ OK' if abs(null_cond) < 1e-10 else '❌ WRONG'}")

print(f"Is the gravitational field over c**2 << 1 ?")
print(f"   Φ/c² = {phi_obs:.3e}")

# Manual calculation
g00_u0_sq = g[0,0] * u0**2
gii_uspatial = g[1,1] * (u_spatial[0]**2 + u_spatial[1]**2 + u_spatial[2]**2)
manual_norm = g00_u0_sq + gii_uspatial

print(f"\n   Breakdown:")
print(f"   g₀₀(u⁰)² = {g00_u0_sq:.3e}")
print(f"   gᵢᵢ(uⁱ)² = {gii_uspatial:.3e}")
print(f"   Sum      = {manual_norm:.3e}")

# Expected with correct metric
g00_correct = -a_obs**2 * (1 + 2*psi_obs) * c**2
g00_correct_contribution = g00_correct * u0**2
correct_norm = g00_correct_contribution + gii_uspatial

print(f"\n   If g[0,0] was correct ({g00_correct:.3e}):")
print(f"   g₀₀(u⁰)² = {g00_correct_contribution:.3e}")
print(f"   Sum      = {correct_norm:.3e}")
print(f"   Status: {'✅ NULL' if abs(correct_norm) < 1e-10 else '❌ NOT NULL'}")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("BUG IN perturbed_flrw_metric_fast.py line 86:")
print("   CURRENT:  g[0,0] = -a**2 * (1 + 2*psi) / (c**2)")
print("   CORRECT:  g[0,0] = -a**2 * (1 + 2*psi) * (c**2)")
print("\nFactor c⁴ error causes null condition to fail!")
print("="*70)
