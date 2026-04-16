#!/usr/bin/env python3
"""
Test the filename utilities for trajectory data.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from excalibur.io.filename_utils import (
    generate_trajectory_filename,
    parse_trajectory_filename,
    format_simulation_info
)
from excalibur.core.constants import one_Mpc, one_Msun
import numpy as np

print("="*70)
print("TESTING FILENAME UTILITIES")
print("="*70)

# Test 1: Generate filename
print("\n1. Testing filename generation...")
print("-"*70)

mass_kg = 1e15 * one_Msun
radius_m = 5.0 * one_Mpc
mass_position_m = np.array([500, 500, 500]) * one_Mpc
observer_position_m = np.array([0, 0, 0]) * one_Mpc

filename = generate_trajectory_filename(
    mass_kg=mass_kg,
    radius_m=radius_m,
    mass_position_m=mass_position_m,
    observer_position_m=observer_position_m,
    metric_type="perturbed_flrw",
    version="OPTIMAL",
    output_dir="_data"
)

print(f"Generated filename:\n  {filename}")
print(f"\nBasename:\n  {os.path.basename(filename)}")

# Test 2: Parse filename
print("\n2. Testing filename parsing...")
print("-"*70)

info = parse_trajectory_filename(filename)

if info:
    print("\nParsed successfully!")
    print(f"\nMetric type: {info['metric_type']}")
    print(f"Version: {info['version']}")
    print(f"Mass: {info['mass_msun']:.2e} M_sun = {info['mass_kg']:.2e} kg")
    print(f"Radius: {info['radius_mpc']:.1f} Mpc = {info['radius_m']:.2e} m")
    print(f"Mass position: {info['mass_position_mpc']} Mpc")
    print(f"Observer position: {info['observer_position_mpc']} Mpc")
    print(f"Distance: {info['distance_mpc']:.2f} Mpc")
    
    # Verify round-trip
    print("\n3. Verifying round-trip consistency...")
    print("-"*70)
    
    errors = []
    if not np.isclose(info['mass_kg'], mass_kg, rtol=1e-6):
        errors.append(f"Mass mismatch: {info['mass_kg']} vs {mass_kg}")
    if not np.isclose(info['radius_m'], radius_m, rtol=1e-6):
        errors.append(f"Radius mismatch: {info['radius_m']} vs {radius_m}")
    if not np.allclose(info['mass_position_m'], mass_position_m, rtol=0.01):
        errors.append(f"Mass position mismatch")
    if not np.allclose(info['observer_position_m'], observer_position_m, rtol=0.01):
        errors.append(f"Observer position mismatch")
    
    if errors:
        print("[FAIL] Round-trip FAILED:")
        for err in errors:
            print(f"  - {err}")
    else:
        print("[ok] Round-trip successful! All values match.")
else:
    print("[FAIL] Parsing failed!")

# Test 3: Format info
print("\n4. Testing formatted output...")
print("-"*70)
print(format_simulation_info(filename))

# Test 4: Parse existing files
print("\n5. Testing with existing files...")
print("-"*70)

import glob
existing_files = glob.glob("_data/backward_raytracing*.h5")

if existing_files:
    print(f"\nFound {len(existing_files)} existing trajectory files:")
    for f in existing_files[:3]:  # Show first 3
        print(f"\nFile: {os.path.basename(f)}")
        info = parse_trajectory_filename(f)
        if info:
            if info['mass_msun'] is not None:
                print(f"  [ok] New format - M={info['mass_msun']:.1e} M_sun, "
                      f"R={info['radius_mpc']:.1f} Mpc, "
                      f"d={info['distance_mpc']:.1f} Mpc")
            else:
                print(f"  WARNING: Legacy format - Mass pos: {info['mass_position_mpc']} Mpc")
        else:
            print(f"  [FAIL] Could not parse")
else:
    print("No existing files found in _data/")

# Test 5: Different metric types
print("\n6. Testing different configurations...")
print("-"*70)

configs = [
    {"metric_type": "schwarzschild", "version": "standard", "mass": 1e14*one_Msun, "radius": 2*one_Mpc},
    {"metric_type": "perturbed_flrw", "version": "OPTIMIZED", "mass": 5e15*one_Msun, "radius": 10*one_Mpc},
]

for i, config in enumerate(configs, 1):
    fname = generate_trajectory_filename(
        mass_kg=config["mass"],
        radius_m=config["radius"],
        mass_position_m=np.array([250, 250, 250])*one_Mpc,
        observer_position_m=np.array([50, 50, 50])*one_Mpc,
        metric_type=config["metric_type"],
        version=config["version"],
        output_dir="_data"
    )
    print(f"\nConfig {i}: {config['metric_type']}, {config['version']}")
    print(f"  {os.path.basename(fname)}")
    
    # Parse back
    info = parse_trajectory_filename(fname)
    if info and info['metric_type'] == config['metric_type']:
        print(f"  [ok] Parse OK")
    else:
        print(f"  [FAIL] Parse FAILED")

print("\n" + "="*70)
print("TESTING COMPLETE")
print("="*70)
