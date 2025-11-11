#!/usr/bin/env python3
"""
Debug Christoffel symbols to find the problem.
"""

import numpy as np
from scipy import interpolate 
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from excalibur.grid.grid import Grid
from excalibur.grid.interpolator import Interpolator
from excalibur.metrics.perturbed_flrw_metric import PerturbedFLRWMetric
from excalibur.core.constants import *
from excalibur.core.cosmology import LCDM_Cosmology
from excalibur.objects.spherical_mass import spherical_mass

# Setup (same as before)
N = 64
grid_size = 2000 * one_Mpc
dx = dy = dz = grid_size / N
shape = (N, N, N)
spacing = (dx, dy, dz)
origin = (-grid_size/2, -grid_size/2, -grid_size/2)
grid = Grid(shape, spacing, origin)

x = y = z = np.linspace(-grid_size/2, grid_size/2, N)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

M = 1e30 * one_Msun
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

interpolator = Interpolator(grid)
metric = PerturbedFLRWMetric(a_of_eta, grid, interpolator)

observer_eta = 4.4e26
energy = 1.0
direction = np.array([1., 1., 1.]) / np.sqrt(3)
u0 = -1.0
u_spatial = energy * c * direction

state = np.array([observer_eta, 0.0, 0.0, 0.0, u0, *u_spatial])

dt = -4.12e+12

# Compute k1, k2
k1 = metric.geodesic_equations(state)
state_k2 = state + 0.5 * dt * k1
k2 = metric.geodesic_equations(state_k2)

print("\n=== k2 values (velocities and accelerations) ===")
print(f"k2[0:4] (velocities) = {k2[:4]}")
print(f"k2[4:8] (accelerations) = {k2[4:]}")
print(f"dt = {dt:.2e}")
print(f"0.5 * dt * k2[5] = {0.5 * dt * k2[5]:.2e} (change in u1)")

# Compute state_k3 where the problem occurs
state_k3 = state + 0.5 * dt * k2

print("=== State k3 (where velocities explode) ===")
print(f"Position: [{state_k3[1]/one_Mpc:.4f}, {state_k3[2]/one_Mpc:.4f}, {state_k3[3]/one_Mpc:.4f}] Mpc")
print(f"Velocities: u0={state_k3[4]:.2e}, u1={state_k3[5]:.2e}, u2={state_k3[6]:.2e}, u3={state_k3[7]:.2e}")

# Compute Christoffel symbols at this state
x_k3 = state_k3[:4]
Gamma = metric.christoffel(x_k3)

print("\n=== Christoffel symbols Gamma^mu_nu_rho ===")
print(f"Gamma^0_00 = {Gamma[0,0,0]:.2e}")
print(f"Gamma^1_00 = {Gamma[1,0,0]:.2e}")
print(f"Gamma^0_11 = {Gamma[0,1,1]:.2e}")
print(f"Gamma^1_11 = {Gamma[1,1,1]:.2e}")
print(f"Gamma^1_01 = {Gamma[1,0,1]:.2e}")

# Compute geodesic acceleration manually
u = state_k3[4:]
du = np.zeros(4)
for mu in range(4):
    du[mu] = -np.einsum('ij,i,j->', Gamma[mu,:,:], u, u)

print("\n=== Accelerations du/dlambda ===")
print(f"du0/dlambda = {du[0]:.2e}")
print(f"du1/dlambda = {du[1]:.2e}")
print(f"du2/dlambda = {du[2]:.2e}")
print(f"du3/dlambda = {du[3]:.2e}")

print("\n=== Analysis ===")
print(f"u·u contraction: {np.einsum('ij,i,j->', Gamma[1,:,:], u, u):.2e}")
print(f"Expected acceleration scale: ~ Gamma * u^2 ~ {Gamma[1,0,0]} * {u[0]**2}")

# Check the interpolated potential and gradient at this position
pos_spatial = state_k3[1:4]
eta_k3 = state_k3[0]
phi_val, grad_phi, phi_dot = interpolator.value_gradient_and_time_derivative(pos_spatial, "Phi", eta_k3)

print("\n=== Potential and gradient ===")
print(f"Phi = {phi_val:.2e} m^2/s^2")
print(f"grad_Phi = [{grad_phi[0]:.2e}, {grad_phi[1]:.2e}, {grad_phi[2]:.2e}] m/s^2")
print(f"phi_dot = {phi_dot:.2e}")
