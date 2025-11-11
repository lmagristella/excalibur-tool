#!/usr/bin/env python3
"""
Simple test of Schwarzschild metric in Cartesian coordinates.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from excalibur.metrics.schwarzschild_metric_cartesian import SchwarzschildMetricCartesian
from excalibur.photon.photon import Photon
from excalibur.integration.integrator import Integrator
from excalibur.core.constants import *

print("=== Simple Schwarzschild Test ===\n")

# Setup
M = 1e16 * one_Msun
radius = 20 * one_Mpc
center = np.array([500, 500, 500]) * one_Mpc

metric = SchwarzschildMetricCartesian(M, radius, center)
print(f"Mass: {M/one_Msun:.1e} M_sun")
print(f"Center: [{center[0]/one_Mpc:.0f}, {center[1]/one_Mpc:.0f}, {center[2]/one_Mpc:.0f}] Mpc\n")

# Create a single photon
t0 = 0.0
pos0 = np.array([1, 1, 1]) * one_Mpc
direction = (center - pos0) / np.linalg.norm(center - pos0)

# Compute proper u^0
origin = np.array([t0, *pos0])
g = metric.metric_tensor(origin)

u_spatial = direction * c

# Debug: compute spatial norm more carefully
spatial_norm_sq = 0.0
for i in range(3):
    for j in range(3):
        spatial_norm_sq += g[i+1, j+1] * u_spatial[i] * u_spatial[j]

print(f"\nDebug calculation:")
print(f"  g_00 = {g[0,0]:.6e}")
print(f"  g_11 = {g[1,1]:.6e}, g_22 = {g[2,2]:.6e}, g_33 = {g[3,3]:.6e}")
print(f"  u_spatial = {u_spatial}")
print(f"  g_ij u^i u^j = {spatial_norm_sq:.6e}")

if g[0, 0] < 0:
    u0_squared = -spatial_norm_sq / g[0, 0]
    u0 = np.sqrt(u0_squared) if u0_squared > 0 else 1.0
    print(f"  -g_ij u^i u^j / g_00 = {u0_squared:.6e}")
    print(f"  u^0 = {u0:.6e}")
else:
    u0 = 1.0

u_full = np.array([u0, *u_spatial])

print(f"\nInitial position: {pos0/one_Mpc} Mpc")
print(f"Direction: {direction}")
print(f"u = [{u0:.3e}, {u_spatial[0]:.3e}, {u_spatial[1]:.3e}, {u_spatial[2]:.3e}]")

# Create photon
photon = Photon(position=origin, direction=u_full)

# Check position and velocity
print(f"\nPhoton state after creation:")
print(f"  photon.x = {photon.x}")
print(f"  photon.u = {photon.u}")

# Manually compute null condition with same metric
g_check = metric.metric_tensor(photon.x)
null_manual = 0.0
for mu in range(4):
    for nu in range(4):
        null_manual += g_check[mu, nu] * photon.u[mu] * photon.u[nu]

print(f"\nManual null calculation:")
print(f"  g from metric_tensor(photon.x) = ")
print(f"    g_00 = {g_check[0,0]:.6e}")
print(f"    g_11 = {g_check[1,1]:.6e}")
print(f"  Check symmetry: g_01 = {g_check[0,1]:.6e}, g_10 = {g_check[1,0]:.6e}")
print(f"  g matrix:")
print(g_check)
print(f"  Sum g_munu u^mu u^nu = {null_manual:.6e}")

# Now do it the numpy way like photon.null_condition()
g_dot_u = np.dot(g_check, photon.u)
null_numpy = np.dot(photon.u, g_dot_u)
print(f"  Using numpy (like photon.null_condition()): {null_numpy:.6e}")
print(f"  g·u = {g_dot_u}")

# Check BEFORE inversion using photon method
null_before = photon.null_condition(metric)
print(f"\nUsing photon.null_condition(): {null_before:.3e}")

# Invert for backward
photon.u = -photon.u
photon.record()

# Check AFTER inversion  
null_after = photon.null_condition(metric)
print(f"\nAfter inversion:")
print(f"  photon.u = {photon.u}")
print(f"  null condition = {null_after:.3e}")

# Verify null condition manually after inversion
null_val_manual = g[0,0] * photon.u[0]**2
for i in range(3):
    for j in range(3):
        null_val_manual += g[i+1,j+1] * photon.u[i+1] * photon.u[j+1]

print(f"\nManual null check after inversion: g_μν u^μ u^ν = {null_val_manual:.3e}")
print(f"  g_00 (-u^0)² = {g[0,0] * photon.u[0]**2:.3e}")
spatial_contrib = 0
for i in range(3):
    for j in range(3):
        spatial_contrib += g[i+1,j+1] * photon.u[i+1] * photon.u[j+1]
print(f"  g_ij (-u^i)(-u^j) = {spatial_contrib:.3e}")

# Integrate
# KEY: Need dt such that displacement is reasonable
# Want: Δx = u_spatial * dt * n_steps ~ 10 Mpc
# With u_spatial ~ 1.7e8 m/s, n_steps = 100:
# dt = 10 * one_Mpc / (u_spatial * n_steps) = 10 * 3e22 / (1.7e8 * 100) ≈ 1.8e13 s
#
# But also need Δt = u^0 * dt to be reasonable
# u^0 * dt = 9e16 * 1.8e13 = 1.6e30 s (too large!)
#
# Let's use dt = 1e10 s as compromise
dt = -1e10  # 10 billion seconds ≈ 317 years
n_steps = 100

print(f"Integrating {n_steps} steps with dt = {dt:.2e} s...")
print(f"Expected time change: Δt = u^0 * dt * n_steps = {u0 * dt * n_steps:.2e} s")
print(f"Expected spatial displacement: Δx ≈ {1.7e8 * abs(dt) * n_steps / one_Mpc:.2f} Mpc")
integrator = Integrator(metric, dt=dt)

try:
    integrator.integrate(photon, n_steps)
    print(f"  SUCCESS! Trajectory has {len(photon.history.states)} states")
    if len(photon.history.states) > 1:
        initial_state = photon.history.states[0]
        final_state = photon.history.states[-1]
        initial_pos = initial_state[1:4]
        final_pos = final_state[1:4]
        dist_moved = np.linalg.norm(final_pos - initial_pos) / one_Mpc
        dist_from_center = np.linalg.norm(final_pos - center) / one_Mpc
        print(f"  Initial position: {initial_pos/one_Mpc} Mpc")
        print(f"  Final position: {final_pos/one_Mpc} Mpc")
        print(f"  Distance moved: {dist_moved:.4f} Mpc")
        print(f"  Distance from center: {dist_from_center:.2f} Mpc")
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n[DONE]")
