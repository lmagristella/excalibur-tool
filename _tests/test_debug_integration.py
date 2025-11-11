#!/usr/bin/env python3
"""
Debug script to understand what's happening in the first integration step.
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
from excalibur.core.constants import *
from excalibur.core.cosmology import LCDM_Cosmology
from excalibur.objects.spherical_mass import spherical_mass

print("=== Debug Integration ===\n")

# Small grid
N = 64
grid_size = 2000 * one_Mpc
dx = dy = dz = grid_size / N

shape = (N, N, N)
spacing = (dx, dy, dz)
origin = (-grid_size/2, -grid_size/2, -grid_size/2)
grid = Grid(shape, spacing, origin)

# Potential field
x = y = z = np.linspace(-grid_size/2, grid_size/2, N)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

M = 1e30 * one_Msun
radius = 10 * one_Mpc
center = np.array([500.0, 500.0, 500.0]) * one_Mpc
spherical_halo = spherical_mass(M, radius, center)

phi_field = spherical_halo.potential(X, Y, Z)
grid.add_field("Phi", phi_field)

# Cosmology
H0 = 70
cosmology = LCDM_Cosmology(H0, Omega_m=0.3, Omega_r=0, Omega_lambda=0.7)
eta_sample = np.linspace(1.0e25, 5.0e26, 1000)
a_sample = cosmology.a_of_eta(eta_sample)
a_of_eta = interpolate.interp1d(eta_sample, a_sample, kind='cubic', fill_value="extrapolate")

# Metric
interpolator = Interpolator(grid)
metric = PerturbedFLRWMetric(a_of_eta, grid, interpolator)

# Create one photon manually
observer_eta = 4.4e26
observer_position = np.array([0.0, 0.0, 0.0])
direction_to_mass = (center - observer_position) / np.linalg.norm(center - observer_position)

# Manual photon state
energy = 1.0
spatial_magnitude = np.linalg.norm(direction_to_mass)
u0 = -energy * spatial_magnitude
u_spatial = energy * c * direction_to_mass

state = np.array([observer_eta, 0.0, 0.0, 0.0, u0, *u_spatial])

print("Initial state:")
print(f"  eta = {state[0]:.2e} s")
print(f"  x = {state[1]:.2e} m ({state[1]/one_Mpc:.2f} Mpc)")
print(f"  y = {state[2]:.2e} m")
print(f"  z = {state[3]:.2e} m")
print(f"  u0 = {state[4]:.6f}")
print(f"  u1 = {state[5]:.2e} m/s")
print(f"  u2 = {state[6]:.2e} m/s")
print(f"  u3 = {state[7]:.2e} m/s")

# Calculate dt
distance_to_mass = np.linalg.norm(center)
a_obs = a_of_eta(observer_eta)
time_to_reach_mass = (a_obs / c) * distance_to_mass
total_integration_time = 1.5 * time_to_reach_mass
target_displacement_per_step = grid_size / 50000
dt_from_displacement = target_displacement_per_step / c
n_steps = int(np.ceil(abs(total_integration_time / dt_from_displacement)))
n_steps = max(500, min(n_steps, 50000))
dt = -total_integration_time / n_steps

print(f"\ndt = {dt:.2e} s")

# Manual RK4 step
print("\n--- RK4 Step Calculation ---")

k1 = metric.geodesic_equations(state)
print(f"k1[0:4] (velocities) = {k1[:4]}")
print(f"k1[4:8] (accelerations) = {k1[4:]}")

state_k2 = state + 0.5 * dt * k1
print(f"\nstate + 0.5*dt*k1:")
print(f"  eta = {state_k2[0]:.2e} s (Delta_eta = {state_k2[0] - state[0]:.2e})")
print(f"  x = {state_k2[1]:.2e} m ({state_k2[1]/one_Mpc:.2f} Mpc, Delta_x = {(state_k2[1] - state[1])/one_Mpc:.2f} Mpc)")
print(f"  y = {state_k2[2]:.2e} m")
print(f"  z = {state_k2[3]:.2e} m")

# Try to evaluate k2
print(f"\nTrying to evaluate k2 at position x=[{state_k2[1]:.2e}, {state_k2[2]:.2e}, {state_k2[3]:.2e}] m")
print(f"Grid bounds: [{origin[0]/one_Mpc:.1f}, {(origin[0] + grid_size)/one_Mpc:.1f}] Mpc")
print(f"Position in Mpc: [{state_k2[1]/one_Mpc:.2f}, {state_k2[2]/one_Mpc:.2f}, {state_k2[3]/one_Mpc:.2f}]")

# Check if in bounds
in_bounds_x = origin[0] <= state_k2[1] <= origin[0] + grid_size
in_bounds_y = origin[1] <= state_k2[2] <= origin[1] + grid_size
in_bounds_z = origin[2] <= state_k2[3] <= origin[2] + grid_size

print(f"In bounds: x={in_bounds_x}, y={in_bounds_y}, z={in_bounds_z}")

try:
    k2 = metric.geodesic_equations(state_k2)
    print(f"✓ k2 computed successfully")
    
    # Continue with k3
    state_k3 = state + 0.5 * dt * k2
    print(f"\nstate + 0.5*dt*k2:")
    print(f"  Full state_k3:")
    print(f"    eta = {state_k3[0]:.2e}")
    print(f"    x = {state_k3[1]:.2e} m ({state_k3[1]/one_Mpc:.4f} Mpc)")
    print(f"    y = {state_k3[2]:.2e} m")
    print(f"    z = {state_k3[3]:.2e} m")
    print(f"    u0 = {state_k3[4]:.6f}")
    print(f"    u1 = {state_k3[5]:.2e}")
    print(f"    u2 = {state_k3[6]:.2e}")
    print(f"    u3 = {state_k3[7]:.2e}")
    
    k3 = metric.geodesic_equations(state_k3)
    print(f"✓ k3 computed successfully")
    print(f"k3[0:4] (velocities) = {k3[:4]}")
    print(f"k3[4:8] (accelerations) = {k3[4:]}")
    
    # Continue with k4
    state_k4 = state + dt * k3
    print(f"\nstate + dt*k3:")
    print(f"  Δx from k3: dt * k3[1] = {dt * k3[1]:.2e} m = {dt * k3[1] / one_Mpc:.2f} Mpc")
    print(f"  Position in Mpc: [{state_k4[1]/one_Mpc:.4f}, {state_k4[2]/one_Mpc:.4f}, {state_k4[3]/one_Mpc:.4f}]")
    
    k4 = metric.geodesic_equations(state_k4)
    print(f"✓ k4 computed successfully")
    
    # Final update
    state_new = state + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    print(f"\nFinal state after full RK4 step:")
    print(f"  eta = {state_new[0]:.2e} s (Delta_eta = {state_new[0] - state[0]:.2e})")
    print(f"  Position in Mpc: [{state_new[1]/one_Mpc:.4f}, {state_new[2]/one_Mpc:.4f}, {state_new[3]/one_Mpc:.4f}]")
    print(f"  Displacement: {np.linalg.norm(state_new[1:4])/one_Mpc:.4f} Mpc")
    
    print("\n✓ FULL RK4 STEP SUCCESSFUL!")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
