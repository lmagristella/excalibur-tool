#!/usr/bin/env python3
"""
Example script showing how to use the excalibur project modules from an external script.

This script demonstrates the proper import patterns and basic usage of:
- Grid and Interpolator for field management
- PerturbedFLRWMetric for spacetime geometry
- Photon for particle representation
- Integrator for geodesic integration
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

### excalibur imports ###
from excalibur.grid.grid import Grid
from excalibur.grid.interpolator import Interpolator
from excalibur.metrics.perturbed_flrw_metric import PerturbedFLRWMetric
from excalibur.photon.photon import Photon
from excalibur.integration.integrator import Integrator
from excalibur.core.constants import *
from excalibur.objects.spherical_mass import spherical_mass
##########################

def example_scale_factor(eta):
    return max(eta, 1e-6)  


def main():
    print("=== Excalibur Project Usage Example ===\n")
    
    # 1. Create a Grid
    print("1. Setting up grid...")
    
    N = 2**7
    grid_size = 1000 * one_Mpc #meters
    dx = dy = dz = grid_size/N

    shape = (N, N, N)
    spacing = (dx,dy,dz)
    origin = (0, 0, 0)
    grid = Grid(shape, spacing, origin)
    
    # Add some example fields (Psi and Phi potentials)
    x = y = z = np.linspace(0, grid_size, N)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Example perturbation fields
    M = 10**20 * one_Msun #halo mass in kg
    radius = 2 * dx #halo radius in meters
    center = np.array([0.5,0.5,0.5]) * grid_size
    spherical_halo = spherical_mass(M, radius, center)
    
    phi_field = spherical_halo.potential(X, Y, Z) #Potential of the halo in m²/s²

    grid.add_field("Phi", phi_field)
    print(f"   Grid shape: {grid.shape}")
    print(f"   Grid spacing: {grid.spacing}")
    print(f"   Fields: {list(grid.fields.keys())}")
    
    # 2. Create an Interpolator
    print("\n2. Setting up interpolator...")
    interpolator = Interpolator(grid)
    
    # Test interpolation at a point
    test_pos = np.array([0.6, 0.45, 0.55]) * grid_size
    psi_val, psi_grad = interpolator.value_and_gradient(test_pos, "Phi")
    print(f"   Psi at {test_pos/grid_size}: {psi_val:.2e}")
    print(f"   ∇Psi at {test_pos/grid_size}: [{psi_grad[0]:.2e}, {psi_grad[1]:.2e}, {psi_grad[2]:.2e}]")
    
    # 3. Create a Metric
    print("\n3. Setting up perturbed FLRW metric...")
    metric = PerturbedFLRWMetric(example_scale_factor, grid, interpolator)
    
    # Test metric evaluation
    test_spacetime_pos = [1.0, 0.5, 0.5, 0.5]  # [η, x, y, z]
    christoffel = metric.christoffel(test_spacetime_pos)
    print(f"   Christoffel symbols computed for position {test_spacetime_pos}")
    print(f"   Γ[0,0,0] = {christoffel[1,1,1]:.2e}")
    
    # 4. Create a Photon
    print("\n4. Setting up photon...")
    initial_position = [1.0, -1.0, 0.0, 0.0]  # [η, x, y, z]
    initial_direction = [1.0, 1.0, 0.0, 0.0]   # [u^η, u^x, u^y, u^z]
    
    # Normalize the 4-velocity (null geodesic condition)
    photon = Photon(initial_position, initial_direction, weight=1.0)
    print(f"   Initial position: {photon.x}")
    print(f"   Initial 4-velocity: {photon.u}")
    print(f"   Initial state: {photon.state}")
    
    # 5. Create an Integrator and evolve the photon
    print("\n5. Integrating photon geodesic...")
    integrator = Integrator(metric, dt=1e-3)
    
    # Record initial state
    photon.record()
    
    # Integrate for a few steps
    steps = 100
    integrator.integrate(photon, steps)
    
    print(f"   Integrated {steps} steps")
    print(f"   Final position: {photon.x}")
    print(f"   Final 4-velocity: {photon.u}")
    print(f"   History length: {len(photon.history.states)}")
    
    # 6. Save results
    print("\n6. Saving photon history...")
    try:
        photon.history.save_to_hdf5("photon_trajectory.h5")
        print("   Saved to photon_trajectory.h5")
    except Exception as e:
        print(f"   Warning: Could not save HDF5 file: {e}")
    
    print("\n=== Example completed successfully! ===")


if __name__ == "__main__":
    main()