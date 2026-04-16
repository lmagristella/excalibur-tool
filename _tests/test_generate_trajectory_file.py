#!/usr/bin/env python3
"""
Quick test to generate a small trajectory file with new filename format.
"""

import numpy as np
import h5py
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from excalibur.core.constants import one_Mpc, one_Msun
from excalibur.io.filename_utils import generate_trajectory_filename

print("="*70)
print("QUICK TEST: Generate trajectory file with new format")
print("="*70)

observer_pos = np.array([0, 0, 0]) * one_Mpc
mass_pos = np.array([500, 500, 500]) * one_Mpc

# Generate filename with new format
mass_kg = 1e15 * one_Msun
radius_m = 5.0 * one_Mpc

filename = generate_trajectory_filename(
    mass_kg=mass_kg,
    radius_m=radius_m,
    mass_position_m=mass_pos,
    observer_position_m=observer_pos,
    metric_type="perturbed_flrw",
    version="TEST",
    output_dir="_data"
)

print(f"\nGenerating: {os.path.basename(filename)}")

# Create HDF5 file with 5 photon trajectories
n_photons = 5
n_points = 10

with h5py.File(filename, 'w') as f:
    for i in range(n_photons):
        # Create photon group
        photon_id = f"photon_{i}"
        grp = f.create_group(photon_id)
        
        # Simple straight-line trajectory
        angle = 2 * np.pi * i / n_photons
        direction = np.array([np.cos(angle), np.sin(angle), 0.5])
        direction = direction / np.linalg.norm(direction)
        
        # Create fake trajectory (10 points)
        distances = np.linspace(0, 1000*one_Mpc, n_points)
        states = np.zeros((n_points, 7))  # [eta, x, y, z, vx, vy, vz]
        
        for j, dist in enumerate(distances):
            pos = observer_pos + direction * dist
            time = 1.5e18 - j * 1e16  # Backward in time
            velocity = direction * 3e8  # c
            
            states[j, 0] = time
            states[j, 1:4] = pos
            states[j, 4:7] = velocity
        
        # Save states
        grp.create_dataset('states', data=states)
        
        # Add fake redshift data
        grp.create_dataset('redshift_total', data=0.3 + 0.01 * i)
        grp.create_dataset('redshift_H', data=0.3)
        grp.create_dataset('redshift_SW', data=0.01 * i)
        grp.create_dataset('redshift_ISW', data=0.0)

file_size = os.path.getsize(filename) / 1024
print(f"[ok] Saved {n_photons} trajectories ({file_size:.1f} KB)")

print("\n" + "="*70)
print("Test file generated successfully!")
print(f"File: {filename}")
print("="*70)
