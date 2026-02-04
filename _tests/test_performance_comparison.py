#!/usr/bin/env python3
"""
Quick performance test comparing optimized vs standard implementations.
"""

import numpy as np
from scipy import interpolate 
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from excalibur.grid.grid import Grid
from excalibur.grid.interpolator import Interpolator
from excalibur.grid.interpolator_fast import InterpolatorFast
from excalibur.metrics.perturbed_flrw_metric import PerturbedFLRWMetric
from excalibur.metrics.perturbed_flrw_metric_fast import PerturbedFLRWMetricFast
from excalibur.photon.photons import Photons
from excalibur.integration.integrator_old import Integrator
from excalibur.core.constants import *
from excalibur.core.cosmology import LCDM_Cosmology
from excalibur.objects.spherical_mass import spherical_mass

print("=== Performance Comparison Test ===\n")

# Small grid for quick testing
N = 64
grid_size = 2000 * one_Mpc
dx = dy = dz = grid_size / N
shape = (N, N, N)
spacing = (dx, dy, dz)
origin = (-grid_size/2, -grid_size/2, -grid_size/2)
grid = Grid(shape, spacing, origin)

x = y = z = np.linspace(-grid_size/2, grid_size/2, N)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

M = 1e20 * one_Msun
radius = 10 * one_Mpc
center = np.array([500.0, 500.0, 500.0]) * one_Mpc
spherical_halo = spherical_mass(M, radius, center)

print("Computing potential field...")
phi_field = spherical_halo.potential(X, Y, Z)
grid.add_field("Phi", phi_field)

# Setup cosmology
H0 = 70
cosmology = LCDM_Cosmology(H0, Omega_m=0.3, Omega_r=0, Omega_lambda=0.7)
eta_sample = np.linspace(1.0e25, 5.0e26, 1000)
a_sample = cosmology.a_of_eta(eta_sample)
a_of_eta = interpolate.interp1d(eta_sample, a_sample, kind='cubic', fill_value="extrapolate")

# Generate test photons
observer_eta = 4.4e26
observer_position = np.array([0.0, 0.0, 0.0])
direction_to_mass = center - observer_position
direction_to_mass = direction_to_mass / np.linalg.norm(direction_to_mass)

photons_standard = Photons()
photons_optimized = Photons()
observer_4d = np.array([observer_eta, *observer_position])

n_test_photons = 5
for _ in range(n_test_photons):
    photons_standard.generate_cone_random(1, observer_4d, direction_to_mass, np.pi/12, 1.0)
    photons_optimized.generate_cone_random(1, observer_4d, direction_to_mass, np.pi/12, 1.0)

for p in photons_standard:
    p.record()
for p in photons_optimized:
    p.record()

print(f"Generated {n_test_photons} test photons\n")

# Test parameters
a_obs = a_of_eta(observer_eta)
distance_to_mass = np.linalg.norm(center - observer_position)
time_to_reach_mass = (a_obs / c) * distance_to_mass
total_time = 1.5 * time_to_reach_mass
dt_magnitude = (grid_size / 50000) / c
n_steps = min(100, int(total_time / dt_magnitude))  # Limit to 100 steps for quick test
dt = -total_time / n_steps

print(f"Test integration: {n_steps} steps with dt={dt:.2e}s\n")

# ===== TEST 1: STANDARD IMPLEMENTATION =====
print("--- STANDARD Implementation ---")
interpolator_std = Interpolator(grid)
metric_std = PerturbedFLRWMetric(a_of_eta, grid, interpolator_std)
integrator_std = Integrator(metric_std, dt=dt)

start = time.time()
success_std = 0
for photon in photons_standard:
    try:
        integrator_std.integrate(photon, n_steps)
        success_std += 1
    except:
        pass
time_std = time.time() - start

print(f"Time: {time_std:.3f}s")
print(f"Success: {success_std}/{n_test_photons}")
print(f"Performance: {n_test_photons * n_steps / time_std:.0f} step-evals/sec\n")

# ===== TEST 2: OPTIMIZED IMPLEMENTATION =====
print("--- OPTIMIZED Implementation ---")
interpolator_opt = InterpolatorFast(grid)
metric_opt = PerturbedFLRWMetricFast(a_of_eta, grid, interpolator_opt)
integrator_opt = Integrator(metric_opt, dt=dt)

# Warm-up JIT compilation
print("Warming up JIT...")
test_photon = photons_optimized[0]
test_photon_copy = Photons()
test_photon_copy.add_photon(test_photon)
try:
    integrator_opt.integrate(test_photon, 5)
except:
    pass
print("JIT warmed up\n")

start = time.time()
success_opt = 0
for photon in photons_optimized:
    try:
        integrator_opt.integrate(photon, n_steps)
        success_opt += 1
    except:
        pass
time_opt = time.time() - start

print(f"Time: {time_opt:.3f}s")
print(f"Success: {success_opt}/{n_test_photons}")
print(f"Performance: {n_test_photons * n_steps / time_opt:.0f} step-evals/sec\n")

# ===== SUMMARY =====
speedup = time_std / time_opt if time_opt > 0 else 0
print("="*50)
print(f"SPEEDUP: {speedup:.2f}x faster with optimizations")
print("="*50)
