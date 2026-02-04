#!/usr/bin/env python3
"""
Advanced redshift and time delay map visualization with geometric information.

This script creates sky maps (observer's view) showing:
- Redshift distribution across the field of view
- Time delay distribution
- Projected mass radius overlay
- Observer-mass geometry indicators
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import h5py
import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from excalibur.core.constants import one_Mpc, c
from excalibur.io.filename_utils import parse_trajectory_filename, format_simulation_info


def load_trajectories(filename):
    """Load photon trajectories from HDF5 file."""
    trajectories = []
    
    with h5py.File(filename, 'r') as f:
        # Get all keys - could be photon_X_states format or photon_X group format
        keys = list(f.keys())
        
        # Detect format
        if keys and 'states' in keys[0]:
            # Old format: photon_X_states as datasets directly
            photon_keys = [k for k in keys if '_states' in k]
            
            for key in photon_keys:
                states = f[key][:]
                
                # Extract photon ID
                photon_id = key.replace('_states', '')
                
                # Extract key information
                initial_pos = states[0, 1:4]
                final_pos = states[-1, 1:4]
                initial_time = states[0, 0]
                final_time = states[-1, 0]
                
                # Check for redshift datasets
                z_total_key = key.replace('_states', '_redshift_total')
                z_H_key = key.replace('_states', '_redshift_H')
                z_SW_key = key.replace('_states', '_redshift_SW')
                z_ISW_key = key.replace('_states', '_redshift_ISW')
                
                z_total = f[z_total_key][()] if z_total_key in f else None
                z_H = f[z_H_key][()] if z_H_key in f else None
                z_SW = f[z_SW_key][()] if z_SW_key in f else None
                z_ISW = f[z_ISW_key][()] if z_ISW_key in f else None
                
                trajectories.append({
                    'id': photon_id,
                    'initial_pos': initial_pos,
                    'final_pos': final_pos,
                    'initial_time': initial_time,
                    'final_time': final_time,
                    'time_delay': final_time - initial_time,  # ✓ CORRECTION: valeur absolue pour raytracing rétrograde
                    'distance': np.linalg.norm(final_pos - initial_pos),
                    'z_total': z_total,
                    'z_H': z_H,
                    'z_SW': z_SW,
                    'z_ISW': z_ISW,
                    'states': states
                })
        else:
            # New format: photon_X groups with states inside
            photon_ids = [k for k in keys if k.startswith('photon_')]
            
            for photon_id in photon_ids:
                grp = f[photon_id]
                
                # Load trajectory data
                states = grp['states'][:]
                
                # Extract key information
                initial_pos = states[0, 1:4]
                final_pos = states[-1, 1:4]
                initial_time = states[0, 0]
                final_time = states[-1, 0]
                
                # Load redshift if available
                if 'redshift_total' in grp:
                    z_total = grp['redshift_total'][()]
                    z_H = grp['redshift_H'][()] if 'redshift_H' in grp else None
                    z_SW = grp['redshift_SW'][()] if 'redshift_SW' in grp else None
                    z_ISW = grp['redshift_ISW'][()] if 'redshift_ISW' in grp else None
                else:
                    z_total = z_H = z_SW = z_ISW = None
                
                trajectories.append({
                    'id': photon_id,
                    'initial_pos': initial_pos,
                    'final_pos': final_pos,
                    'initial_time': initial_time,
                    'final_time': final_time,
                    'time_delay': abs(final_time - initial_time),  # ✓ CORRECTION: valeur absolue pour raytracing rétrograde
                    'distance': np.linalg.norm(final_pos - initial_pos),
                    'z_total': z_total,
                    'z_H': z_H,
                    'z_SW': z_SW,
                    'z_ISW': z_ISW,
                    'states': states
                })
    
    return trajectories


def compute_sky_coordinates(positions, observer_pos, direction):
    """
    Compute angular sky coordinates relative to observer's line of sight.
    
    Parameters
    ----------
    positions : array-like, shape (N, 3)
        3D positions in space
    observer_pos : array-like, shape (3,)
        Observer position
    direction : array-like, shape (3,)
        Observer's pointing direction (will be normalized)
    
    Returns
    -------
    theta : array-like, shape (N,)
        Angular offset from center (radians)
    phi : array-like, shape (N,)
        Azimuthal angle around line of sight (radians)
    """
    # Normalize direction
    direction = np.array(direction) / np.linalg.norm(direction)
    
    # Compute vectors from observer to positions
    vectors = positions - observer_pos
    
    # Distance from observer
    distances = np.linalg.norm(vectors, axis=1)
    
    # Normalize vectors
    unit_vectors = vectors / distances[:, np.newaxis]
    
    # Compute angular offset from center (theta)
    cos_theta = np.dot(unit_vectors, direction)
    cos_theta = np.clip(cos_theta, -1, 1)  # Numerical safety
    theta = np.arccos(cos_theta)
    
    # Compute azimuthal angle (phi) - need to define a reference axis perpendicular to direction
    # Use arbitrary perpendicular vector
    if abs(direction[2]) < 0.9:
        perp1 = np.cross(direction, [0, 0, 1])
    else:
        perp1 = np.cross(direction, [1, 0, 0])
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(direction, perp1)
    perp2 = perp2 / np.linalg.norm(perp2)
    
    # Project onto perpendicular plane
    proj1 = np.dot(unit_vectors, perp1)
    proj2 = np.dot(unit_vectors, perp2)
    phi = np.arctan2(proj2, proj1)
    
    return theta, phi


def plot_redshift_sky_map(trajectories, geometry_info, save_path=None):
    """
    Create sky map showing redshift distribution with mass geometry overlay.
    
    Parameters
    ----------
    trajectories : list
        List of trajectory dictionaries
    geometry_info : dict
        Geometric information from filename parsing
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle("Sky Maps: Redshift Components with Mass Geometry", fontsize=16, fontweight='bold')
    
    # Extract data
    n_traj = len(trajectories)
    final_positions = np.array([t['final_pos'] for t in trajectories])
    
    # Get observer position and mass position from geometry
    if geometry_info['observer_position_m'] is not None:
        observer_pos = geometry_info['observer_position_m']
    else:
        # Fallback: use first initial position as observer
        # ⚠️ WARNING: Assuming positions in file are in meters (should be verified)
        observer_pos = trajectories[0]['initial_pos']
        print(f"⚠️  Warning: Using trajectory initial position as observer reference")
        print(f"   Position: {observer_pos/one_Mpc} Mpc (assuming meters in file)")
    
    if geometry_info['mass_position_m'] is not None:
        mass_pos = geometry_info['mass_position_m']
        mass_radius = geometry_info['radius_m']
        has_geometry = True
    else:
        has_geometry = False
    
    # Compute direction (average direction to emission points)
    directions = final_positions - observer_pos
    avg_direction = np.mean(directions, axis=0)
    avg_direction = avg_direction / np.linalg.norm(avg_direction)
    
    # Compute sky coordinates for emission points
    theta, phi = compute_sky_coordinates(final_positions, observer_pos, avg_direction)
    
    # Convert to degrees and to x-y plane coordinates (stereographic-like projection)
    theta_deg = np.degrees(theta)
    x_sky = theta_deg * np.cos(phi)
    y_sky = theta_deg * np.sin(phi)
    
    # Compute mass projection if geometry available
    if has_geometry:
        # Compute angular position of mass center
        mass_direction = mass_pos - observer_pos
        mass_distance = np.linalg.norm(mass_direction)
        mass_theta, mass_phi = compute_sky_coordinates(
            mass_pos[np.newaxis, :], observer_pos, avg_direction
        )
        mass_theta_deg = np.degrees(mass_theta[0])
        mass_x_sky = mass_theta_deg * np.cos(mass_phi[0])
        mass_y_sky = mass_theta_deg * np.sin(mass_phi[0])
        
        # Compute projected angular radius
        angular_radius_rad = mass_radius / mass_distance
        angular_radius_deg = np.degrees(angular_radius_rad)
        
        info_text = (
            f"Mass: {geometry_info['mass_msun']:.2e} M☉\n"
            f"Radius: {geometry_info['radius_mpc']:.1f} Mpc\n"
            f"Distance: {geometry_info['distance_mpc']:.1f} Mpc\n"
            f"Angular size: {angular_radius_deg:.3f}°"
        )
    else:
        info_text = "Geometry info not available"
    
    # Prepare redshift components
    z_components = {
        'Total Redshift (z_total)': [t['z_total'] for t in trajectories],
        'Homogeneous (z_H)': [t['z_H'] for t in trajectories],
        'Sachs-Wolfe (z_SW)': [t['z_SW'] for t in trajectories],
        'Integrated SW (z_ISW)': [t['z_ISW'] for t in trajectories]
    }
    
    axes = axes.flatten()
    
    for idx, (title, z_values) in enumerate(z_components.items()):
        ax = axes[idx]
        
        # Filter out None values
        valid = [i for i, z in enumerate(z_values) if z is not None]
        if not valid:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(title)
            continue
        
        z_plot = np.array([z_values[i] for i in valid])
        x_plot = x_sky[valid]
        y_plot = y_sky[valid]
        
        # Create scatter plot
        scatter = ax.scatter(x_plot, y_plot, c=z_plot, s=50, cmap='RdYlBu_r',
                           edgecolors='black', linewidth=0.5, alpha=0.8)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(title.split('(')[1].rstrip(')'), fontsize=10)
        
        # Overlay mass geometry if available
        if has_geometry:
            # Draw mass center
            ax.plot(mass_x_sky, mass_y_sky, 'k+', markersize=15, markeredgewidth=2,
                   label='Mass center')
            
            # Draw mass radius circle
            circle = plt.Circle((mass_x_sky, mass_y_sky), angular_radius_deg,
                              fill=False, edgecolor='black', linewidth=2,
                              linestyle='--', label=f'Mass radius ({mass_radius/one_Mpc:.1f} Mpc)')
            ax.add_patch(circle)
            
            # Add legend only on first plot
            if idx == 0:
                ax.legend(loc='upper right', fontsize=9)
        
        ax.set_xlabel('Angular offset RA (degrees)', fontsize=11)
        ax.set_ylabel('Angular offset Dec (degrees)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Set equal limits for all subplots
        max_extent = max(abs(x_sky).max(), abs(y_sky).max()) * 1.1
        ax.set_xlim(-max_extent, max_extent)
        ax.set_ylim(-max_extent, max_extent)
    
    # Add info text box
    fig.text(0.99, 0.01, info_text, fontsize=10, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    if save_path:
        output_dir = "../_visualizations"
        os.makedirs(output_dir, exist_ok=True)
        full_path = os.path.join(output_dir, save_path)
        plt.savefig(full_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved sky map to: {full_path}")
    
    plt.show()


def plot_time_delay_map(trajectories, geometry_info, save_path=None):
    """
    Create sky map showing time delay distribution.
    
    Parameters
    ----------
    trajectories : list
        List of trajectory dictionaries
    geometry_info : dict
        Geometric information from filename parsing
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 9))
    fig.suptitle("Time Delay Sky Map with Mass Geometry", fontsize=14, fontweight='bold')
    
    # Extract data
    final_positions = np.array([t['final_pos'] for t in trajectories])
    time_delays = np.array([t['time_delay'] for t in trajectories])
    
    # Get observer and mass positions
    if geometry_info['observer_position_m'] is not None:
        observer_pos = geometry_info['observer_position_m']
    else:
        observer_pos = trajectories[0]['initial_pos']
    
    has_geometry = geometry_info['mass_position_m'] is not None
    
    # Compute direction
    directions = final_positions - observer_pos
    avg_direction = np.mean(directions, axis=0)
    avg_direction = avg_direction / np.linalg.norm(avg_direction)
    
    # Compute sky coordinates
    theta, phi = compute_sky_coordinates(final_positions, observer_pos, avg_direction)
    theta_deg = np.degrees(theta)
    x_sky = theta_deg * np.cos(phi)
    y_sky = theta_deg * np.sin(phi)
    
    # Plot time delays (convert to Myr for readability)
    # Note: time_delay est déjà en valeur absolue, converti de secondes vers Myr
    time_delays_Myr = time_delays #/ (1e6 * 365.25 * 24 * 3600)
    
    scatter = ax.scatter(x_sky, y_sky, c=time_delays_Myr, s=80, cmap='viridis',
                        edgecolors='black', linewidth=0.5, alpha=0.8)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Time Delay (s)', fontsize=11)
    
    # Overlay geometry
    if has_geometry:
        mass_pos = geometry_info['mass_position_m']
        mass_radius = geometry_info['radius_m']
        mass_distance = geometry_info['distance_mpc'] * one_Mpc
        
        mass_theta, mass_phi = compute_sky_coordinates(
            mass_pos[np.newaxis, :], observer_pos, avg_direction
        )
        mass_theta_deg = np.degrees(mass_theta[0])
        mass_x_sky = mass_theta_deg * np.cos(mass_phi[0])
        mass_y_sky = mass_theta_deg * np.sin(mass_phi[0])
        
        angular_radius_deg = np.degrees(mass_radius / mass_distance)
        
        ax.plot(mass_x_sky, mass_y_sky, 'r+', markersize=20, markeredgewidth=3,
               label='Mass center')
        
        circle = plt.Circle((mass_x_sky, mass_y_sky), angular_radius_deg,
                          fill=False, edgecolor='red', linewidth=2.5,
                          linestyle='--', label=f'Mass radius')
        ax.add_patch(circle)
        
        ax.legend(loc='upper right', fontsize=10)
        
        info_text = (
            f"Mass: {geometry_info['mass_msun']:.2e} M☉\n"
            f"Radius: {geometry_info['radius_mpc']:.1f} Mpc\n"
            f"Distance: {geometry_info['distance_mpc']:.1f} Mpc\n"
            f"Angular size: {angular_radius_deg:.3f}°"
        )
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=10, va='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    ax.set_xlabel('Angular offset RA (degrees)', fontsize=12)
    ax.set_ylabel('Angular offset Dec (degrees)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    max_extent = max(abs(x_sky).max(), abs(y_sky).max()) * 1.1
    ax.set_xlim(-max_extent, max_extent)
    ax.set_ylim(-max_extent, max_extent)
    
    plt.tight_layout()
    
    if save_path:
        output_dir = "../_visualizations"
        os.makedirs(output_dir, exist_ok=True)
        full_path = os.path.join(output_dir, save_path)
        plt.savefig(full_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved time delay map to: {full_path}")
    
    plt.show()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Create advanced redshift and time delay sky maps with geometric overlays'
    )
    parser.add_argument('filename', type=str, nargs='?', default=None,
                       help='Path to trajectory HDF5 file')
    parser.add_argument('--no-time-delay', action='store_true',
                       help='Skip time delay map')
    
    args = parser.parse_args()
    
    print("="*70)
    print("REDSHIFT & TIME DELAY SKY MAPS WITH GEOMETRY")
    print("="*70)
    
    # Find file
    if args.filename is None:
        import glob
        files = glob.glob("/home/magri/_data/output/excalibur_run_perturbed_flrw_M1.0e15_R5.0_mass500_500_500_obs0_0_0_N463_parallel_S5120_Mpc.h5")
        
        if not files:
            print("\n❌ Error: No trajectory files found in ../_data/output/")
            print("Usage: python plot_redshift_maps_with_geometry.py [filename.h5]")
            return
        
        # Use most recent
        filename = sorted(files)[-1]
        print(f"\nAuto-detected file: {filename}")
    else:
        filename = args.filename
        if not os.path.exists(filename):
            print(f"\n❌ Error: File not found: {filename}")
            return
    
    # Parse filename for geometry info
    print("\n" + "-"*70)
    geometry_info = parse_trajectory_filename(filename)
    
    if geometry_info:
        print("\nParsed simulation geometry:")
        print(format_simulation_info(filename))
    else:
        print("\n⚠ Warning: Could not parse geometry from filename")
        print("  (Using legacy format or non-standard name)")
        geometry_info = {
            'mass_position_m': None,
            'observer_position_m': None,
            'radius_m': None,
            'mass_msun': None,
            'radius_mpc': None,
            'distance_mpc': None
        }
    
    # Load trajectories
    print("\n" + "-"*70)
    print(f"\nLoading trajectories from: {os.path.basename(filename)}")
    trajectories = load_trajectories(filename)
    print(f"  Loaded {len(trajectories)} photon trajectories")
    
    # Check if redshift data available
    has_redshift = any(t['z_total'] is not None for t in trajectories)
    
    if not has_redshift:
        print("\n⚠ Warning: No redshift data found in trajectory file")
        print("  Only time delay map will be created")
    
    # Create visualizations
    print("\n" + "-"*70)
    print("\nCreating visualizations...")
    
    if has_redshift:
        print("\n1. Redshift sky maps...")
        plot_redshift_sky_map(trajectories, geometry_info, 
                            save_path="redshift_sky_map_with_geometry.png")
    
    if not args.no_time_delay:
        print("\n2. Time delay sky map...")
        plot_time_delay_map(trajectories, geometry_info,
                           save_path="time_delay_sky_map_with_geometry.png")
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print("\nGenerated files in ../_visualizations/:")
    if has_redshift:
        print("  - redshift_sky_map_with_geometry.png")
    if not args.no_time_delay:
        print("  - time_delay_sky_map_with_geometry.png")


if __name__ == "__main__":
    main()
