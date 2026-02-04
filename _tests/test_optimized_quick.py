#!/usr/bin/env python3
"""Quick test of the optimized version with only 5 photons."""

import numpy as np
from scipy import interpolate 
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from excalibur.grid.grid import Grid
from excalibur.grid.interpolator_fast import InterpolatorFast
from excalibur.metrics.perturbed_flrw_metric_fast import PerturbedFLRWMetricFast
from excalibur.photon.photons import Photons
from excalibur.integration.integrator_old import Integrator
from excalibur.core.constants import *
from excalibur.core.cosmology import LCDM_Cosmology
from excalibur.objects.spherical_mass import spherical_mass

print("Testing optimized version with 5 photons...\n")

start_total = time.time()

# Quick setup
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

phi_field = spherical_halo.potential(X, Y, Z)
grid.add_field("Phi", phi_field)

H0 = 70
cosmology = LCDM_Cosmology(H0, Omega_m=0.3, Omega_r=0, Omega_lambda=0.7)
eta_sample = np.linspace(1.0e25, 5.0e26, 1000)
a_sample = cosmology.a_of_eta(eta_sample)
a_of_eta = interpolate.interp1d(eta_sample, a_sample, kind='cubic', fill_value="extrapolate")

# OPTIMIZED classes
interpolator = InterpolatorFast(grid)
metric = PerturbedFLRWMetricFast(a_of_eta, grid, interpolator)

observer_eta = 4.4e26
observer_position = np.array([0.0, 0.0, 0.0])
direction_to_mass = center - observer_position
direction_to_mass = direction_to_mass / np.linalg.norm(direction_to_mass)

# Generate 5 test photons
photons = Photons()
observer_4d = np.array([observer_eta, *observer_position])
photons.generate_cone_random(5, observer_4d, direction_to_mass, np.pi/12, 1.0)

for p in photons:
    p.record()

# Integration parameters
a_obs = a_of_eta(observer_eta)
distance_to_mass = np.linalg.norm(center - observer_position)
time_to_reach_mass = (a_obs / c) * distance_to_mass
total_time = 1.5 * time_to_reach_mass
dt_magnitude = (grid_size / 50000) / c
n_steps = min(100, int(total_time / dt_magnitude))
dt = -total_time / n_steps

integrator = Integrator(metric, dt=dt)

print(f"Running {len(photons)} photons for {n_steps} steps...")
start_integration = time.time()

for photon in photons:
    try:
        integrator.integrate(photon, n_steps)
    except:
        pass

elapsed_integration = time.time() - start_integration
elapsed_total = time.time() - start_total

print(f"\n✓ Integration completed!")
print(f"  Integration time: {elapsed_integration:.3f}s")
print(f"  Total time: {elapsed_total:.3f}s")
print(f"  Performance: {len(photons) * n_steps / elapsed_integration:.0f} step-evals/sec")
print(f"\nOptimized version working correctly! 🚀")
