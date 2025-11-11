#!/usr/bin/env python3
"""
Test script to verify integration parameters before running the full simulation.
"""

import numpy as np
from scipy import interpolate
from excalibur.core.constants import *
from excalibur.core.cosmology import LCDM_Cosmology

print("="*70)
print("TEST DES PARAMETRES D'INTEGRATION")
print("="*70)

# Setup cosmology
H0 = 70
Omega_m = 0.3
Omega_r = 0
Omega_lambda = 0.7
cosmology = LCDM_Cosmology(H0, Omega_m=Omega_m, Omega_r=Omega_r, Omega_lambda=Omega_lambda)

# Create interpolation
eta_start = 1.0e25
eta_end = 5.0e26
eta_sample = np.linspace(eta_start, eta_end, 1000)
a_sample = cosmology.a_of_eta(eta_sample)
a_of_eta = interpolate.interp1d(eta_sample, a_sample, kind='cubic', fill_value="extrapolate")

# Simulation parameters
observer_eta = 4.4e26
grid_size = 1000 * one_Mpc
radius = 10 * one_Mpc
observer_position = np.array([0.0, 0.0, 0.0]) * grid_size
center = np.array([0.5, 0.5, 0.5]) * grid_size

# Get scale factor
a_obs = a_of_eta(observer_eta)

# Calculate parameters
distance_to_mass = np.linalg.norm(center - observer_position)
time_to_reach_mass = (a_obs / c) * distance_to_mass
total_integration_time = 1.5 * time_to_reach_mass
potential_crossing_time = (a_obs / c) * radius
min_timescale = potential_crossing_time / 5

n_steps = int(np.ceil(abs(total_integration_time / min_timescale)))
n_steps = max(500, min(n_steps, 20000))
dt = total_integration_time / n_steps  # POSITIVE dt, backward motion from reversed velocities

distance_travelled = abs(n_steps * dt) * c / a_obs

print(f"\n1. COSMOLOGIE:")
print(f"   H0 = {H0} km/s/Mpc")
print(f"   a(η={observer_eta:.2e} s) = {a_obs:.6f}")

print(f"\n2. GEOMETRIE:")
print(f"   Grille: {grid_size/one_Mpc:.0f} Mpc")
print(f"   Observateur: [0.0, 0.0, 0.0] Mpc")
print(f"   Centre de masse: [{center[0]/one_Mpc:.1f}, {center[1]/one_Mpc:.1f}, {center[2]/one_Mpc:.1f}] Mpc")
print(f"   Distance à la masse: {distance_to_mass/one_Mpc:.2f} Mpc")

print(f"\n3. PARAMETRES D'INTEGRATION:")
print(f"   Nombre de pas: {n_steps}")
print(f"   Time step: dt = {dt:.2e} s (POSITIF)")
print(f"   Total integration time: {n_steps * dt:.2e} s")
print(f"   Note: Backward motion via reversed 4-velocities, not negative dt")

print(f"\n4. DISTANCE PARCOURUE:")
print(f"   Distance parcourue: {distance_travelled/one_Mpc:.2f} Mpc")
print(f"   Distance à la masse: {distance_to_mass/one_Mpc:.2f} Mpc")
print(f"   Ratio: {distance_travelled/distance_to_mass:.2f}x")

print(f"\n5. TEMPS CONFORME:")
print(f"   η initial: {observer_eta:.6e} s")
print(f"   Integration time: {n_steps * dt:.6e} s")
print(f"   η final attendu: {observer_eta - n_steps * dt:.6e} s (minus car backward)")
print(f"   Changement absolu: {abs(n_steps * dt):.6e} s")
print(f"   Changement relatif: {abs(n_steps * dt) / observer_eta * 100:.4f}%")

print("\n" + "="*70)
if dt > 0:
    print("✓ dt est POSITIF - backward tracing via reversed velocities!")
else:
    print("✗ ERREUR: dt devrait etre positif dans cette approche!")

if distance_travelled > distance_to_mass:
    print(f"✓ Distance parcourue ({distance_travelled/one_Mpc:.1f} Mpc) > Distance à la masse ({distance_to_mass/one_Mpc:.1f} Mpc)")
    print("  Les photons devraient atteindre la masse!")
else:
    print(f"✗ ERREUR: Distance insuffisante!")
    print(f"  Distance parcourue: {distance_travelled/one_Mpc:.1f} Mpc")
    print(f"  Distance à la masse: {distance_to_mass/one_Mpc:.1f} Mpc")

print("="*70)
