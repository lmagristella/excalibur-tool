#!/usr/bin/env python3
"""
Compute redshift map for gravitational lensing.

This script analyzes backward ray tracing results to compute the various
contributions to the observed redshift in perturbed FLRW cosmology:
- Homogeneous (expansion) redshift
- Sachs-Wolfe effect
- Integrated Sachs-Wolfe effect
- Doppler shift from peculiar velocities

The output is a 2D sky map showing how each redshift component varies
across the field of view.
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from excalibur.core.constants import *
from excalibur.observables.redshift import compute_redshift_components
from scipy.interpolate import griddata


def load_trajectories(filename):
    """Load photon trajectories from HDF5 file."""
    trajectories = []
    metadata = {}
    
    with h5py.File(filename, 'r') as f:
        n_photons = f.attrs['n_photons']
        metadata['n_photons'] = n_photons
        
        if 'photon_info' in f:
            photon_info = f['photon_info'][:]
            metadata['photon_info'] = photon_info
        
        for i in range(n_photons):
            dataset_name = f"photon_{i}_states"
            if dataset_name in f:
                states = f[dataset_name][:]
                valid_mask = ~np.isnan(states).any(axis=1)
                clean_states = states[valid_mask]
                trajectories.append(clean_states)
    
    return trajectories, metadata


def extract_sky_positions(trajectories):
    """
    Extract initial sky positions for each photon.
    Projects onto plane perpendicular to mean direction.
    """
    directions = []
    for traj in trajectories:
        if len(traj) < 1:
            continue
        u_spatial = traj[0, 5:8]
        direction = u_spatial / np.linalg.norm(u_spatial)
        directions.append(direction)
    
    directions = np.array(directions)
    mean_direction = np.mean(directions, axis=0)
    mean_direction = mean_direction / np.linalg.norm(mean_direction)
    
    # Create orthonormal basis
    if abs(mean_direction[2]) < 0.9:
        up = np.array([0, 0, 1])
    else:
        up = np.array([1, 0, 0])
    
    basis_x = up - np.dot(up, mean_direction) * mean_direction
    basis_x = basis_x / np.linalg.norm(basis_x)
    basis_y = np.cross(mean_direction, basis_x)
    
    # Project
    x_proj = np.zeros(len(directions))
    y_proj = np.zeros(len(directions))
    
    for i, direction in enumerate(directions):
        cos_theta = np.dot(direction, mean_direction)
        theta = np.arccos(np.clip(cos_theta, -1, 1))
        
        if theta > 1e-10:
            in_plane = direction - cos_theta * mean_direction
            in_plane = in_plane / np.linalg.norm(in_plane)
            x_proj[i] = np.dot(in_plane, basis_x) * theta
            y_proj[i] = np.dot(in_plane, basis_y) * theta
    
    return x_proj, y_proj


def compute_redshift_map(trajectories, a_of_eta=None, potential_func=None, 
                         velocity_func=None, verbose=False):
    """
    Compute redshift components for all photons.
    
    **NEW**: If trajectories contain pre-computed quantities (14+ columns),
    no need to provide a_of_eta or potential_func - they will be extracted
    automatically from the trajectory data.
    
    Returns:
    --------
    redshift_data : dict
        Dictionary with arrays for each redshift component
    sky_positions : tuple
        (x_proj, y_proj) sky coordinates
    """
    n_photons = len(trajectories)
    
    # Check if trajectories have pre-computed quantities
    has_quantities = (trajectories[0].shape[1] >= 14 if len(trajectories) > 0 else False)
    
    if has_quantities:
        print("\n✓ Trajectories contain pre-computed quantities (a, phi, grad_phi, phi_dot)")
        print("  No need for a_of_eta or potential_func callables")
    else:
        if a_of_eta is None or potential_func is None:
            raise ValueError(
                "Trajectories do not contain pre-computed quantities, so a_of_eta "
                "and potential_func must be provided"
            )
        print("\n  Using provided a_of_eta and potential_func callables")
    
    # Arrays to store results
    z_H_array = np.zeros(n_photons)
    z_SW_array = np.zeros(n_photons)
    z_ISW_array = np.zeros(n_photons)
    z_D_array = np.zeros(n_photons)
    z_total_array = np.zeros(n_photons)
    
    print(f"\nComputing redshift for {n_photons} photons...")
    
    for i, traj in enumerate(trajectories):
        if len(traj) < 2:
            continue
        
        try:
            results = compute_redshift_components(
                traj, a_of_eta, potential_func, velocity_func, 
                verbose=(verbose and i == 0)  # Verbose for first photon only
            )
            
            z_H_array[i] = results['z_H']
            z_SW_array[i] = results['z_SW']
            z_ISW_array[i] = results['z_ISW']
            z_D_array[i] = results['z_D']
            z_total_array[i] = results['z_total']
            
        except Exception as e:
            print(f"Warning: Error computing redshift for photon {i}: {e}")
            continue
        
        if (i + 1) % 10 == 0:
            print(f"   Progress: {i+1}/{n_photons}")
    
    # Get sky positions
    x_proj, y_proj = extract_sky_positions(trajectories)
    
    redshift_data = {
        'z_H': z_H_array,
        'z_SW': z_SW_array,
        'z_ISW': z_ISW_array,
        'z_D': z_D_array,
        'z_total': z_total_array
    }
    
    return redshift_data, (x_proj, y_proj)


def plot_redshift_maps(redshift_data, sky_positions, output_filename=None):
    """
    Create multi-panel plot showing all redshift components.
    """
    x_proj, y_proj = sky_positions
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    components = [
        ('z_H', 'Homogeneous (Expansion)', 'z_H'),
        ('z_SW', 'Sachs-Wolfe', 'z_SW'),
        ('z_ISW', 'Integrated Sachs-Wolfe', 'z_ISW'),
        ('z_D', 'Doppler', 'z_D'),
        ('z_total', 'Total Redshift', 'z_total'),
    ]
    
    for idx, (key, title, label) in enumerate(components):
        ax = axes[idx]
        data = redshift_data[key]
        
        # Determine if we need diverging or sequential colormap
        if key in ['z_SW', 'z_ISW', 'z_D']:
            # Perturbations can be positive or negative
            vmax = np.nanmax(np.abs(data))
            if vmax > 0:
                norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
                cmap = 'RdBu_r'
            else:
                # All zeros - use sequential
                norm = None
                cmap = 'viridis'
        else:
            # Expansion and total are always positive
            norm = None
            cmap = 'viridis'
        
        scatter = ax.scatter(x_proj, y_proj, c=data, s=100, 
                           edgecolors='k', linewidth=0.5,
                           cmap=cmap, norm=norm)
        
        ax.set_xlabel('Sky Position X (radians)')
        ax.set_ylabel('Sky Position Y (radians)')
        ax.set_title(title)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        
        # Force square limits
        x_center = (x_proj.min() + x_proj.max()) / 2
        y_center = (y_proj.min() + y_proj.max()) / 2
        max_range = max(x_proj.max() - x_proj.min(), 
                       y_proj.max() - y_proj.min()) * 1.1
        ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
        ax.set_ylim(y_center - max_range/2, y_center + max_range/2)
        
        cbar = plt.colorbar(scatter, ax=ax, label=label)
    
    # Hide unused subplot
    axes[-1].axis('off')
    
    plt.tight_layout()
    
    if output_filename:
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        print(f"\n   Saved redshift maps to: {output_filename}")
    
    plt.show()


def print_statistics(redshift_data):
    """Print statistics for each redshift component."""
    print("\n" + "="*70)
    print("REDSHIFT STATISTICS")
    print("="*70)
    
    for key in ['z_H', 'z_SW', 'z_ISW', 'z_D', 'z_total']:
        data = redshift_data[key]
        print(f"\n{key}:")
        print(f"  Mean:   {np.mean(data):.6e}")
        print(f"  Std:    {np.std(data):.6e}")
        print(f"  Min:    {np.min(data):.6e}")
        print(f"  Max:    {np.max(data):.6e}")
        print(f"  Median: {np.median(data):.6e}")
    
    print("="*70)


def main(filename, potential_func=None, a_of_eta=None, velocity_func=None):
    """
    Main function.
    
    **NEW**: If trajectories contain pre-computed quantities (14+ columns),
    potential_func and a_of_eta are optional and will be extracted from data.
    
    Parameters:
    -----------
    filename : str
        HDF5 file with photon trajectories
    potential_func : callable, optional
        Function Φ(x, y, z, η) returning gravitational potential (required if no quantities)
    a_of_eta : callable, optional
        Function a(η) returning scale factor (required if no quantities)
    velocity_func : callable, optional
        Function v(x, y, z, η) returning peculiar velocity
    """
    print("="*70)
    print("REDSHIFT MAP COMPUTATION")
    print("="*70)
    print(f"\nInput file: {filename}")
    
    # Load trajectories
    print("\n1. Loading trajectories...")
    trajectories, metadata = load_trajectories(filename)
    print(f"   Loaded {len(trajectories)} photon trajectories")
    
    # Compute redshifts
    print("\n2. Computing redshift components...")
    redshift_data, sky_positions = compute_redshift_map(
        trajectories, a_of_eta, potential_func, velocity_func, verbose=True
    )
    
    # Print statistics
    print_statistics(redshift_data)
    
    # Plot
    print("\n3. Creating redshift maps...")
    output_filename = filename.replace('.h5', '_redshift_map.png')
    plot_redshift_maps(redshift_data, sky_positions, output_filename)
    
    print("\n" + "="*70)
    print("DONE!")
    print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compute redshift maps from photon trajectories"
    )
    parser.add_argument(
        'filename',
        type=str,
        nargs='?',
        default='backward_raytracing_trajectories_OPTIMAL_mass_500_500_500_Mpc.h5',
        help='HDF5 file with photon trajectories'
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.filename):
        print(f"Error: File not found: {args.filename}")
        sys.exit(1)
    
    # Run with automatic detection of pre-computed quantities
    # No need to provide potential_func/a_of_eta if quantities are in trajectory
    print("\n" + "="*70)
    print("AUTO-DETECTION MODE")
    print("="*70)
    print("If trajectories contain pre-computed quantities (14+ columns),")
    print("they will be used automatically. Otherwise, you need to modify")
    print("this script to provide potential_func and a_of_eta callables.")
    print("="*70 + "\n")
    
    main(args.filename)
