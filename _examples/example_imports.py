#!/usr/bin/env python3
"""
Different ways to import and use the excalibur project modules.

This file demonstrates various import patterns you can use depending on
how you want to structure your external scripts.
"""

import numpy as np
import os

print("=== Import Pattern Examples ===\n")

# METHOD 1: Using PYTHONPATH (recommended for external scripts)
print("Method 1: Using PYTHONPATH environment variable")
print("   Run this script with:")
print("   PYTHONPATH=/home/magri/excalibur_project python example_imports.py")
print()

# METHOD 2: Programmatic path manipulation
print("Method 2: Adding to sys.path in the script")
print("   import sys")
print("   sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))")
print("   from excalibur.grid.grid import Grid")
print()

# METHOD 3: Using the module as a package (when running from parent directory)
print("Method 3: Running as module from parent directory")
print("   cd /home/magri/excalibur_project")
print("   python -m your_script_name")
print()

print("=== Example Import Statements ===\n")

print("# Core components:")
print("from excalibur.grid.grid import Grid")
print("from excalibur.grid.interpolator import Interpolator")
print("from excalibur.metrics.base_metric import Metric")
print("from excalibur.metrics.perturbed_flrw_metric import PerturbedFLRWMetric")
print("from excalibur.photon.photon import Photon")
print("from excalibur.photon.photon_history import PhotonHistory")
print("from excalibur.integration.integrator import Integrator")
print()

print("# Or import entire modules:")
print("from excalibur import grid, metrics, photon, integration")
print("# Then use: grid.Grid(), metrics.PerturbedFLRWMetric(), etc.")
print()

print("=== Typical Usage Pattern ===\n")

typical_usage = '''
# 1. Set up the environment
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 2. Import what you need
from excalibur.grid.grid import Grid
from excalibur.grid.interpolator import Interpolator
from excalibur.metrics.perturbed_flrw_metric import PerturbedFLRWMetric
from excalibur.photon.photon import Photon
from excalibur.integration.integrator import Integrator

# 3. Create your simulation
def run_simulation():
    # Create grid with fields
    grid = Grid(shape=(64, 64, 64), spacing=(0.1, 0.1, 0.1))
    
    # Add perturbation fields
    psi = np.random.normal(0, 1e-5, (64, 64, 64))
    phi = np.random.normal(0, 1e-5, (64, 64, 64))
    grid.add_field("Psi", psi)
    grid.add_field("Phi", phi)
    
    # Set up interpolation and metric
    interp = Interpolator(grid)
    metric = PerturbedFLRWMetric(lambda eta: eta, grid, interp)
    
    # Create and evolve photon
    photon = Photon([1.0, 0, 0, 0], [1.0, 1, 0, 0])
    integrator = Integrator(metric)
    integrator.integrate(photon, steps=1000)
    
    return photon.history

# 4. Run your analysis
if __name__ == "__main__":
    history = run_simulation()
    print(f"Simulated {len(history.states)} steps")
'''

print(typical_usage)

if __name__ == "__main__":
    print("\n=== Testing simple imports ===")
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from excalibur.grid.grid import Grid
        print("✓ Successfully imported Grid")
        
        from excalibur.metrics.base_metric import Metric
        print("✓ Successfully imported Metric")
        
        print("\n✓ All imports successful!")
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        print("Make sure to run with PYTHONPATH set or from the correct directory")