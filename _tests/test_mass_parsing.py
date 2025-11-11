#!/usr/bin/env python3
"""
Test script to verify mass position parsing from filename.
"""

import re
import numpy as np

# Test filenames
test_filenames = [
    "backward_raytracing_trajectories_mass_500_500_500_Mpc.h5",
    "backward_raytracing_trajectories_mass_100_200_300_Mpc.h5",
    "backward_raytracing_trajectories.h5",  # No mass info
    "other_file_mass_50_60_70_Mpc.h5"
]

print("="*70)
print("TEST DU PARSING DE LA POSITION DE LA MASSE")
print("="*70)

pattern = r'mass_(\d+)_(\d+)_(\d+)_Mpc'

for filename in test_filenames:
    print(f"\nFilename: {filename}")
    match = re.search(pattern, filename)
    
    if match:
        mass_position = np.array([
            float(match.group(1)),
            float(match.group(2)),
            float(match.group(3))
        ])
        print(f"  ✓ Mass position parsed: [{mass_position[0]:.0f}, {mass_position[1]:.0f}, {mass_position[2]:.0f}] Mpc")
    else:
        print(f"  ✗ No mass position found in filename")

print("\n" + "="*70)
print("Test terminé!")
print("="*70)
