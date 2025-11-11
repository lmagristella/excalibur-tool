#!/usr/bin/env python3
"""
Compute 2D time delay map for gravitational lensing.

This script analyzes backward ray tracing results to compute gravitational
time delays - the extra time photons take traveling through curved spacetime
compared to straight-line propagation in flat space.

Process:
1. Loads photon trajectories from backward ray tracing HDF5 file
2. For each photon:
   - Integrates actual path length along curved geodesic
   - Computes straight-line distance between endpoints
   - Calculates time delay: Δt = (path_curved - path_straight) / c
3. Projects photons onto 2D sky map using their initial directions
4. Creates interpolated 2D map of time delays
5. Visualizes spatial pattern of gravitational time delays

The time delay includes:
- Shapiro delay: Light slowing down in gravitational potential
- Geometric delay: Extra path length due to deflection

Units are automatically selected (days, hours, minutes, seconds, etc.)
based on the magnitude of the delays.
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from excalibur.core.constants import *


def load_trajectories(filename):
    """
    Load photon trajectories from HDF5 file.
    
    Returns:
        trajectories: list of arrays, each shape (n_steps, 8+)
                      [eta, x, y, z, u0, u1, u2, u3, ...]
        metadata: dict with photon info
    """
    trajectories = []
    metadata = {}
    
    with h5py.File(filename, 'r') as f:
        n_photons = f.attrs['n_photons']
        metadata['n_photons'] = n_photons
        
        # Load metadata if available
        if 'photon_info' in f:
            photon_info = f['photon_info'][:]
            metadata['photon_info'] = photon_info
        
        # Load each photon trajectory
        for i in range(n_photons):
            dataset_name = f"photon_{i}_states"
            if dataset_name in f:
                states = f[dataset_name][:]
                # Remove NaN padding
                valid_mask = ~np.isnan(states).any(axis=1)
                clean_states = states[valid_mask]
                trajectories.append(clean_states)
    
    return trajectories, metadata


def compute_travel_time_deflected(trajectory):
    """
    Compute actual path length and travel time along deflected geodesic.
    
    We integrate the spatial distance along the trajectory to get the actual
    path length, which may be longer than the straight-line distance due to
    gravitational deflection.
    
    Args:
        trajectory: array (n_steps, 8+) with [eta, x, y, z, u0, u1, u2, u3, ...]
    
    Returns:
        delta_t_physical: travel time (path_length / c) in seconds
        path_length: spatial distance traveled along curved path (meters)
    """
    if len(trajectory) < 2:
        return 0.0, 0.0
    
    # Integrate path length by summing distances between consecutive points
    positions = trajectory[:, 1:4]  # [x, y, z] coordinates
    
    path_length = 0.0
    for i in range(len(positions) - 1):
        segment_length = np.linalg.norm(positions[i+1] - positions[i])
        path_length += segment_length
    
    # Travel time = path_length / c (for light)
    delta_t_physical = path_length / c
    
    return delta_t_physical, path_length


def compute_travel_time_straight(pos_initial, direction, target_comoving_distance, c_light):
    """
    Compute travel time for straight line path (no deflection) in a given direction.
    
    This computes how long it would take light to travel a specific comoving distance
    in flat space along the initial direction, without any gravitational deflection.
    
    This ensures we're comparing "apples to apples": both trajectories (deflected and
    straight) travel the same comoving distance from the observer, but the deflected
    one takes a longer path due to curvature.
    
    Args:
        pos_initial: initial position [x, y, z] in meters
        direction: unit direction vector [dx, dy, dz]
        target_comoving_distance: comoving distance to travel (meters)
        c_light: speed of light in m/s
    
    Returns:
        delta_t_straight: travel time (distance / c) in seconds
        distance: straight-line distance traveled (meters)
    """
    # In flat space, light travels in a straight line
    # The distance traveled equals the target comoving distance
    distance = target_comoving_distance
    delta_t_straight = distance / c_light
    
    return delta_t_straight, distance


def extract_initial_directions(trajectories):
    """
    Extract initial sky positions (angular coordinates) for each photon.
    
    Projects photon directions onto a plane perpendicular to the mean direction
    to avoid elliptical distortion when viewing direction is not along z-axis.
    
    Returns:
        x_proj, y_proj: projected positions on sky plane
        theta, phi: spherical angles (radians) from mean direction
    """
    n_photons = len(trajectories)
    
    if n_photons == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    # Get observer position (should be same for all)
    observer_pos = trajectories[0][0, 1:4]
    
    # For each photon, get initial direction
    directions = []
    for traj in trajectories:
        if len(traj) < 1:
            continue
        # Initial velocity direction
        u_spatial = traj[0, 5:8]  # [u1, u2, u3]
        direction = u_spatial / np.linalg.norm(u_spatial)
        directions.append(direction)
    
    directions = np.array(directions)
    
    # Compute mean direction (central direction of cone)
    mean_direction = np.mean(directions, axis=0)
    mean_direction = mean_direction / np.linalg.norm(mean_direction)
    
    # Create orthonormal basis for projection plane perpendicular to mean_direction
    # Choose arbitrary perpendicular vector
    if abs(mean_direction[2]) < 0.9:
        # mean_direction not close to z-axis
        up = np.array([0, 0, 1])
    else:
        # mean_direction close to z-axis, use x-axis as reference
        up = np.array([1, 0, 0])
    
    # Gram-Schmidt orthogonalization
    # basis_x: first basis vector in the plane (perpendicular to mean_direction)
    basis_x = up - np.dot(up, mean_direction) * mean_direction
    basis_x = basis_x / np.linalg.norm(basis_x)
    
    # basis_y: second basis vector (perpendicular to both mean_direction and basis_x)
    basis_y = np.cross(mean_direction, basis_x)
    basis_y = basis_y / np.linalg.norm(basis_y)
    
    # Project each direction onto the plane
    x_proj = np.zeros(len(directions))
    y_proj = np.zeros(len(directions))
    theta = np.zeros(len(directions))
    phi = np.zeros(len(directions))
    
    for i, direction in enumerate(directions):
        # Angle from mean direction
        cos_theta = np.dot(direction, mean_direction)
        theta[i] = np.arccos(np.clip(cos_theta, -1, 1))
        
        # Project onto plane
        # Component in the plane = direction - (direction · mean_direction) * mean_direction
        in_plane = direction - cos_theta * mean_direction
        
        # If theta is very small, direction ≈ mean_direction, projection is at origin
        if theta[i] > 1e-10:
            in_plane = in_plane / np.linalg.norm(in_plane)  # normalize
            
            # Get coordinates in the (basis_x, basis_y) plane
            x_proj[i] = np.dot(in_plane, basis_x) * theta[i]
            y_proj[i] = np.dot(in_plane, basis_y) * theta[i]
            
            # Azimuthal angle in the plane
            phi[i] = np.arctan2(y_proj[i], x_proj[i])
        else:
            x_proj[i] = 0
            y_proj[i] = 0
            phi[i] = 0
    
    return x_proj, y_proj, theta, phi


def compute_time_delays(trajectories, metadata):
    """
    Compute time delay for each photon trajectory.
    
    The time delay is computed by comparing:
    1. Actual path traveled along curved geodesic (deflected by gravity)
    2. Straight-line path in flat space covering the same comoving distance
    
    This ensures both paths travel the same radial distance from the observer,
    making the comparison physically meaningful for gravitational lensing.
    
    Returns:
        time_delays: array of time delays in seconds
        distances_deflected: array of distances traveled (deflected path)
        distances_straight: array of straight-line distances
        positions_2d: (x_proj, y_proj) sky positions
    """
    n_photons = len(trajectories)
    time_delays = []
    distances_deflected = []
    distances_straight = []
    
    for i, traj in enumerate(trajectories):
        if len(traj) < 2:
            # No trajectory, skip
            time_delays.append(0.0)
            distances_deflected.append(0.0)
            distances_straight.append(0.0)
            continue
        
        # Deflected path (actual geodesic with gravitational deflection)
        delta_t_deflected, path_deflected = compute_travel_time_deflected(traj)
        
        # Get initial position and direction
        pos_initial = traj[0, 1:4]
        pos_final = traj[-1, 1:4]
        
        # Compute comoving distance traveled (radial distance from observer)
        displacement = pos_final - pos_initial
        comoving_distance = np.linalg.norm(displacement)
        
        # Get initial direction (unit vector)
        u_spatial_initial = traj[0, 5:8]
        direction = u_spatial_initial / np.linalg.norm(u_spatial_initial)
        
        # Straight path: same initial direction, same comoving distance
        # This is what would happen in flat space with no deflection
        delta_t_straight, path_straight = compute_travel_time_straight(
            pos_initial, direction, comoving_distance, c
        )
        
        # Time delay = difference in travel times
        # Positive delay means deflected path took longer (longer path due to curvature)
        time_delay = delta_t_deflected - delta_t_straight
        
        time_delays.append(time_delay)
        distances_deflected.append(path_deflected)
        distances_straight.append(path_straight)
    
    time_delays = np.array(time_delays)
    distances_deflected = np.array(distances_deflected)
    distances_straight = np.array(distances_straight)
    
    # Get sky positions
    x_proj, y_proj, theta, phi = extract_initial_directions(trajectories)
    
    return time_delays, distances_deflected, distances_straight, (x_proj, y_proj)


def create_2d_map(x_proj, y_proj, time_delays, grid_resolution=50):
    """
    Create a 2D interpolated map of time delays.
    
    Args:
        x_proj, y_proj: projected sky positions (radians or dimensionless)
        time_delays: time delay values (seconds)
        grid_resolution: number of grid points per axis
    
    Returns:
        X_grid, Y_grid: meshgrid for plotting
        time_delay_grid: 2D array of interpolated time delays
    """
    from scipy.interpolate import griddata
    
    # Create regular grid with SQUARE aspect ratio for circular symmetry
    x_min, x_max = x_proj.min(), x_proj.max()
    y_min, y_max = y_proj.min(), y_proj.max()
    
    # Use the maximum range to create a square grid
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    max_range = max(x_max - x_min, y_max - y_min)
    
    # Add padding (10%)
    max_range *= 1.2
    half_range = max_range / 2
    
    x_min = x_center - half_range
    x_max = x_center + half_range
    y_min = y_center - half_range
    y_max = y_center + half_range
    
    x_grid = np.linspace(x_min, x_max, grid_resolution)
    y_grid = np.linspace(y_min, y_max, grid_resolution)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    
    # Interpolate time delays onto grid
    points = np.column_stack([x_proj, y_proj])
    time_delay_grid = griddata(
        points, time_delays, (X_grid, Y_grid),
        method='cubic', fill_value=np.nan
    )
    
    return X_grid, Y_grid, time_delay_grid


def plot_time_delay_map(X_grid, Y_grid, time_delay_grid, 
                         x_proj, y_proj, time_delays,
                         output_filename=None):
    """
    Visualize the 2D time delay map.
    """
    # Use square subplots to avoid elliptical distortion
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Convert to convenient units
    # Try days, hours, minutes, seconds, milliseconds, microseconds
    time_unit = 86400.0  # days (default)
    time_label = "days"
    
    max_delay = np.nanmax(np.abs(time_delay_grid))
    
    if max_delay < 86400:  # Less than 1 day
        time_unit = 3600.0  # hours
        time_label = "hours"
    
    if max_delay < 3600:  # Less than 1 hour
        time_unit = 60.0  # minutes
        time_label = "minutes"
    
    if max_delay < 60:  # Less than 1 minute
        time_unit = 1.0  # seconds
        time_label = "s"
    
    if max_delay < 1e-3:
        time_unit = 1e6  # microseconds
        time_label = "μs"
    elif np.nanmax(np.abs(time_delay_grid)) < 1.0:
        time_unit = 1e3  # milliseconds
        time_label = "ms"
    
    time_delay_grid_scaled = time_delay_grid / time_unit
    time_delays_scaled = time_delays / time_unit
    
    # Plot 1: Interpolated 2D map
    ax1 = axes[0]
    
    # Use diverging colormap centered at zero
    vmax = np.nanmax(np.abs(time_delay_grid_scaled))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    
    im = ax1.contourf(X_grid, Y_grid, time_delay_grid_scaled, 
                      levels=20, cmap='RdBu_r', norm=norm)
    ax1.contour(X_grid, Y_grid, time_delay_grid_scaled, 
                levels=10, colors='k', alpha=0.3, linewidths=0.5)
    
    # Overlay photon positions
    scatter = ax1.scatter(x_proj, y_proj, c=time_delays_scaled, 
                          s=30, edgecolors='k', linewidth=0.5,
                          cmap='RdBu_r', norm=norm)
    
    ax1.set_xlabel('Sky Position X (radians)')
    ax1.set_ylabel('Sky Position Y (radians)')
    ax1.set_title('Time Delay Map (Interpolated)')
    ax1.set_aspect('equal', adjustable='box')
    
    # Force square limits
    x_center_plot = (x_proj.min() + x_proj.max()) / 2
    y_center_plot = (y_proj.min() + y_proj.max()) / 2
    max_range_plot = max(x_proj.max() - x_proj.min(), y_proj.max() - y_proj.min()) * 1.1
    ax1.set_xlim(x_center_plot - max_range_plot/2, x_center_plot + max_range_plot/2)
    ax1.set_ylim(y_center_plot - max_range_plot/2, y_center_plot + max_range_plot/2)
    
    ax1.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(im, ax=ax1, label=f'Time Delay Δt ({time_label})')
    
    # Plot 2: Scatter plot with values
    ax2 = axes[1]
    scatter2 = ax2.scatter(x_proj, y_proj, c=time_delays_scaled, 
                           s=100, edgecolors='k', linewidth=1,
                           cmap='RdBu_r', norm=norm)
    
    # Annotate with values
    for i, (x, y, dt) in enumerate(zip(x_proj, y_proj, time_delays_scaled)):
        if i % 3 == 0:  # Label every 3rd point to avoid crowding
            ax2.annotate(f'{dt:.2f}', (x, y), 
                        fontsize=7, ha='center', va='bottom')
    
    ax2.set_xlabel('Sky Position X (radians)')
    ax2.set_ylabel('Sky Position Y (radians)')
    ax2.set_title('Time Delay (Scatter Points)')
    ax2.set_aspect('equal', adjustable='box')
    
    # Force same square limits as ax1
    ax2.set_xlim(x_center_plot - max_range_plot/2, x_center_plot + max_range_plot/2)
    ax2.set_ylim(y_center_plot - max_range_plot/2, y_center_plot + max_range_plot/2)
    
    ax2.grid(True, alpha=0.3)
    
    cbar2 = plt.colorbar(scatter2, ax=ax2, label=f'Time Delay Δt ({time_label})')
    
    plt.tight_layout()
    
    if output_filename:
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        print(f"   Saved time delay map to: {output_filename}")
    
    plt.show()


def print_statistics(time_delays, distances_deflected, distances_straight, distance_unit='auto'):
    """
    Print statistics about time delays.
    
    Args:
        time_delays: array of time delays in seconds
        distances_deflected: array of deflected path lengths in meters
        distances_straight: array of straight path lengths in meters
        distance_unit: unit for displaying distances
            - 'auto': automatically choose based on magnitude
            - 'Mpc': Megaparsecs (cosmological scales)
            - 'kpc': kiloparsecs
            - 'pc': parsecs
            - 'uMpc' or 'μMpc': micro-Megaparsecs (for small deflections)
    """
    print("\n" + "="*70)
    print("TIME DELAY STATISTICS")
    print("="*70)
    
    # Convert to convenient units (prefer days for cosmological scales)
    max_delay = np.max(np.abs(time_delays))
    
    if max_delay >= 86400:  # 1 day or more
        time_unit = 86400.0
        time_label = "days"
    elif max_delay >= 3600:  # 1 hour or more
        time_unit = 3600.0
        time_label = "hours"
    elif max_delay >= 60:  # 1 minute or more
        time_unit = 60.0
        time_label = "minutes"
    elif max_delay >= 1.0:
        time_unit = 1.0
        time_label = "s"
    elif max_delay >= 1e-3:
        time_unit = 1e3
        time_label = "ms"
    else:
        time_unit = 1e6
        time_label = "μs"
    
    print(f"Number of photons: {len(time_delays)}")
    print(f"Mean time delay: {np.mean(time_delays)/time_unit:.3f} {time_label}")
    print(f"Std time delay: {np.std(time_delays)/time_unit:.3f} {time_label}")
    print(f"Min time delay: {np.min(time_delays)/time_unit:.3f} {time_label}")
    print(f"Max time delay: {np.max(time_delays)/time_unit:.3f} {time_label}")
    
    # Choose distance unit
    path_diffs = distances_deflected - distances_straight
    mean_path_diff = np.mean(np.abs(path_diffs))
    
    if distance_unit == 'auto':
        # Auto-select based on magnitude of deflections
        if mean_path_diff > 0.1 * one_Mpc:
            dist_scale = one_Mpc
            dist_label = "Mpc"
            decimals = 2
        elif mean_path_diff > 100 * one_pc:
            dist_scale = 1000 * one_pc  # kpc
            dist_label = "kpc"
            decimals = 2
        elif mean_path_diff > 0.001 * one_Mpc:
            dist_scale = one_pc
            dist_label = "pc"
            decimals = 1
        else:
            dist_scale = 1e-6 * one_Mpc  # micro-Mpc
            dist_label = "μMpc"
            decimals = 3
    elif distance_unit in ['uMpc', 'μMpc', 'micro-Mpc']:
        dist_scale = 1e-6 * one_Mpc
        dist_label = "μMpc"
        decimals = 3
    elif distance_unit == 'kpc':
        dist_scale = 1000 * one_pc
        dist_label = "kpc"
        decimals = 2
    elif distance_unit == 'pc':
        dist_scale = one_pc
        dist_label = "pc"
        decimals = 1
    else:  # 'Mpc' or default
        dist_scale = one_Mpc
        dist_label = "Mpc"
        decimals = 2
    
    print(f"\nDistance statistics (in {dist_label}):")
    print(f"Mean deflected path: {np.mean(distances_deflected)/dist_scale:.{decimals}f} {dist_label}")
    print(f"Mean straight path: {np.mean(distances_straight)/dist_scale:.{decimals}f} {dist_label}")
    print(f"Mean path difference: {np.mean(path_diffs)/dist_scale:.{decimals}f} {dist_label}")
    print(f"Min path difference: {np.min(path_diffs)/dist_scale:.{decimals}f} {dist_label}")
    print(f"Max path difference: {np.max(path_diffs)/dist_scale:.{decimals}f} {dist_label}")
    
    # Fractional difference
    frac_diff = path_diffs / distances_straight
    print(f"\nMean fractional path difference: {np.mean(frac_diff)*100:.6f}%")
    print(f"Max fractional path difference: {np.max(frac_diff)*100:.6f}%")
    
    print("="*70)


def main(filename, distance_unit='auto'):
    """
    Main function.
    
    Args:
        filename: HDF5 file with photon trajectories
        distance_unit: unit for displaying distances ('auto', 'Mpc', 'kpc', 'pc', 'μMpc')
    """
    print("="*70)
    print("TIME DELAY MAP COMPUTATION")
    print("="*70)
    print(f"\nInput file: {filename}")
    
    # Load trajectories
    print("\n1. Loading trajectories...")
    trajectories, metadata = load_trajectories(filename)
    print(f"   Loaded {len(trajectories)} photon trajectories")
    
    # Compute time delays
    print("\n2. Computing time delays...")
    time_delays, dist_def, dist_straight, (x_proj, y_proj) = compute_time_delays(
        trajectories, metadata
    )
    print(f"   Computed time delays for {len(time_delays)} photons")
    
    # Print statistics
    print_statistics(time_delays, dist_def, dist_straight, distance_unit=distance_unit)
    
    # Create 2D map
    print("\n3. Creating 2D interpolated map...")
    X_grid, Y_grid, time_delay_grid = create_2d_map(
        x_proj, y_proj, time_delays, grid_resolution=50
    )
    print(f"   Created {X_grid.shape[0]}x{X_grid.shape[1]} grid")
    
    # Plot
    print("\n4. Plotting time delay map...")
    output_filename = filename.replace('.h5', '_time_delay_map.png')
    plot_time_delay_map(X_grid, Y_grid, time_delay_grid,
                        x_proj, y_proj, time_delays,
                        output_filename=output_filename)
    
    print("\n" + "="*70)
    print("DONE!")
    print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compute 2D time delay map from photon trajectories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Distance units:
  auto    : Automatically select based on deflection magnitude (default)
  Mpc     : Megaparsecs (for large cosmological deflections)
  kpc     : kiloparsecs
  pc      : parsecs
  μMpc    : micro-Megaparsecs (for small deflections, ~0.1 Mpc)
  
Examples:
  python compute_time_delay_map.py trajectories.h5
  python compute_time_delay_map.py trajectories.h5 --distance-unit μMpc
  python compute_time_delay_map.py trajectories.h5 --distance-unit Mpc
        """
    )
    parser.add_argument(
        'filename',
        type=str,
        nargs='?',
        default='backward_raytracing_trajectories_OPTIMAL_mass_500_500_500_Mpc.h5',
        help='HDF5 file with photon trajectories'
    )
    parser.add_argument(
        '--distance-unit', '-d',
        type=str,
        default='auto',
        choices=['auto', 'Mpc', 'kpc', 'pc', 'μMpc', 'uMpc'],
        help='Unit for displaying path length differences (default: auto)'
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.filename):
        print(f"Error: File not found: {args.filename}")
        print("\nUsage:")
        print(f"  python {os.path.basename(__file__)} <trajectory_file.h5>")
        sys.exit(1)
    
    main(args.filename, distance_unit=args.distance_unit)
