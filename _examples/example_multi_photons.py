#!/usr/bin/env python3
"""
Example usage of the multi-photon system (Photons class).
This demonstrates how to generate multiple photons with different sampling strategies
and save all their trajectories together.
"""

import numpy as np
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from excalibur.photon.photons import Photons
from excalibur.photon.photon import Photon
from excalibur.grid.grid import Grid
from excalibur.grid.interpolator import Interpolator
from excalibur.metrics.perturbed_flrw_metric import PerturbedFLRWMetric
from excalibur.objects.spherical_mass import spherical_mass
from excalibur.integration.integrator import Integrator

def example_scale_factor(eta):
    """Simple scale factor function for testing."""
    return max(eta, 1e-6)

def main():
    print("=== Multi-Photon System Example ===\n")
    
    # 1. Setup the same environment as before
    print("1. Setting up physics environment...")
    
    # Create grid
    n_cells = 32  # Smaller for faster testing
    box_size = 3.086e25  # 1 Gpc in meters
    
    # Create coordinate arrays
    dx = dy = dz = box_size / n_cells
    shape = (n_cells, n_cells, n_cells)
    spacing = (dx, dy, dz)
    origin = (0, 0, 0)
    grid = Grid(shape, spacing, origin)
    
    # Create meshgrid for potential calculation
    x = y = z = np.linspace(0, box_size, n_cells)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    print(f"   Grid: {n_cells}³ cells, {box_size:.2e} m box size")
    spherical_halo = spherical_mass(
        mass=1e15,  # Solar masses  
        radius=box_size/10,
        center=np.array([box_size/2, box_size/2, box_size/2])
    )
    phi_field = spherical_halo.potential(X, Y, Z)
    grid.add_field("Phi", phi_field)
    
    # Setup interpolator and metric
    interpolator = Interpolator(grid)
    metric = PerturbedFLRWMetric(example_scale_factor, grid, interpolator)
    
    print("   Physics environment ready")
    
    # 2. Create multi-photon system
    print("\n2. Creating multi-photon systems...")
    
    # Test 1: Random sampling in a cone
    photons_random = Photons()
    origin = np.array([1.0, 0.0, 0.0, 0.0])  # [eta, x, y, z]
    central_direction = np.array([1.0, 0.0, 0.0])  # pointing in +x direction
    cone_angle = np.pi / 6  # 30 degree half-angle
    
    photons_random.generate_cone_random(
        n_photons=10,
        origin=origin,
        central_direction=central_direction,
        cone_angle=cone_angle,
        energy=1.0
    )
    print(f"   Generated {len(photons_random)} photons with random cone sampling")
    
    # Test 2: Grid sampling in a cone
    photons_grid = Photons()
    photons_grid.generate_cone_grid(
        n_theta=3,
        n_phi=4,
        origin=origin,
        central_direction=central_direction,
        cone_angle=cone_angle,
        energy=1.0
    )
    print(f"   Generated {len(photons_grid)} photons with grid cone sampling")
    
    # 3. Integrate photon trajectories
    print("\n3. Integrating photon trajectories...")
    
    integrator = Integrator(metric, dt=0.001)
    n_steps = 50  # Shorter for testing
    
    # Integrate random sample photons
    print(f"   Integrating {len(photons_random)} random photons...")
    for i, photon in enumerate(photons_random):
        integrator.integrate(photon, n_steps)
        if (i + 1) % 5 == 0:
            print(f"     Completed {i + 1}/{len(photons_random)}")
    
    # Integrate grid sample photons  
    print(f"   Integrating {len(photons_grid)} grid photons...")
    for i, photon in enumerate(photons_grid):
        integrator.integrate(photon, n_steps)
        if (i + 1) % 5 == 0:
            print(f"     Completed {i + 1}/{len(photons_grid)}")
    
    # 4. Save results
    print("\n4. Saving trajectories...")
    
    try:
        photons_random.save_all_histories("multi_photons_random.h5")
        print("   ✓ Saved random photons to multi_photons_random.h5")
    except Exception as e:
        print(f"   ✗ Failed to save random photons: {e}")
    
    try:
        photons_grid.save_all_histories("multi_photons_grid.h5")
        print("   ✓ Saved grid photons to multi_photons_grid.h5")
    except Exception as e:
        print(f"   ✗ Failed to save grid photons: {e}")
    
    # 5. Test loading
    print("\n5. Testing data loading...")
    
    try:
        # Test loading the random photons
        loaded_photons = Photons()
        loaded_photons.load_from_hdf5("multi_photons_random.h5")
        print(f"   ✓ Loaded {len(loaded_photons)} photons from file")
        
        # Verify data integrity
        if len(loaded_photons) == len(photons_random):
            print("   ✓ Photon count matches original")
        else:
            print(f"   ✗ Photon count mismatch: {len(loaded_photons)} vs {len(photons_random)}")
        
        # Check that histories are preserved
        total_states_original = sum(len(p.history.states) for p in photons_random)
        total_states_loaded = sum(len(p.history.states) for p in loaded_photons)
        
        if total_states_original == total_states_loaded:
            print("   ✓ History states count matches original")
        else:
            print(f"   ✗ States count mismatch: {total_states_loaded} vs {total_states_original}")
            
    except Exception as e:
        print(f"   ✗ Failed to load photons: {e}")
    
    # 6. Summary statistics
    print("\n6. Summary:")
    print(f"   Random photons: {len(photons_random)} photons")
    print(f"   Grid photons: {len(photons_grid)} photons") 
    
    if len(photons_random) > 0:
        avg_states_random = np.mean([len(p.history.states) for p in photons_random])
        print(f"   Average trajectory length (random): {avg_states_random:.1f} states")
    
    if len(photons_grid) > 0:
        avg_states_grid = np.mean([len(p.history.states) for p in photons_grid])
        print(f"   Average trajectory length (grid): {avg_states_grid:.1f} states")
    
    print("\n=== Multi-photon example completed successfully! ===")

if __name__ == "__main__":
    main()