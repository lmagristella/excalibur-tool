"""
Utilities for standardized trajectory filename generation and parsing.

This module provides functions to create descriptive filenames for trajectory
data that encode key simulation parameters (mass, radius, positions) and parse
this information back from filenames.
"""

import os
import re
import numpy as np
from excalibur.core.constants import one_Mpc, one_Msun


def generate_trajectory_filename(
    mass_kg,
    radius_m,
    mass_position_m,
    observer_position_m,
    metric_type="perturbed_flrw",
    version="OPTIMAL",
    output_dir="_data"
):
    """
    Generate a standardized filename for trajectory data.
    
    Parameters
    ----------
    mass_kg : float
        Mass in kilograms
    radius_m : float
        Mass radius in meters
    mass_position_m : array-like
        Mass center position [x, y, z] in meters
    observer_position_m : array-like
        Observer position [x, y, z] in meters
    metric_type : str, optional
        Type of metric ("perturbed_flrw", "schwarzschild", etc.)
    version : str, optional
        Version tag ("OPTIMAL", "OPTIMIZED", "standard", etc.)
    output_dir : str, optional
        Output directory path (default: "_data")
    
    Returns
    -------
    str
        Full path to output file with standardized naming convention
        
    Examples
    --------
    >>> filename = generate_trajectory_filename(
    ...     mass_kg=1e15*one_Msun,
    ...     radius_m=5*one_Mpc,
    ...     mass_position_m=[500*one_Mpc, 500*one_Mpc, 500*one_Mpc],
    ...     observer_position_m=[0, 0, 0]
    ... )
    >>> print(filename)
    _data/backward_raytracing_perturbed_flrw_OPTIMAL_M1.0e+15_R5.0_mass500_500_500_obs0_0_0_Mpc.h5
    """
    # Convert to standard units
    mass_msun = mass_kg / one_Msun
    radius_mpc = radius_m / one_Mpc
    mass_pos_mpc = np.array(mass_position_m) / one_Mpc
    obs_pos_mpc = np.array(observer_position_m) / one_Mpc
    
    # Format mass in scientific notation (compact)
    mass_str = f"M{mass_msun:.1e}".replace("+", "")
    
    # Format radius
    radius_str = f"R{radius_mpc:.1f}"
    
    # Format positions (rounded to integers for filename brevity)
    mass_pos_str = f"mass{int(round(mass_pos_mpc[0]))}_{int(round(mass_pos_mpc[1]))}_{int(round(mass_pos_mpc[2]))}"
    obs_pos_str = f"obs{int(round(obs_pos_mpc[0]))}_{int(round(obs_pos_mpc[1]))}_{int(round(obs_pos_mpc[2]))}"
    
    # Build filename
    filename = (
        f"backward_raytracing_{metric_type}_{version}_"
        f"{mass_str}_{radius_str}_"
        f"{mass_pos_str}_"
        f"{obs_pos_str}_Mpc.h5"
    )
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    return os.path.join(output_dir, filename)


def parse_trajectory_filename(filename):
    """
    Parse standardized trajectory filename to extract simulation parameters.
    
    Parameters
    ----------
    filename : str
        Trajectory filename (with or without path)
    
    Returns
    -------
    dict
        Dictionary containing parsed parameters:
        - 'metric_type': str
        - 'version': str
        - 'mass_msun': float (in solar masses)
        - 'radius_mpc': float (in Mpc)
        - 'mass_position_mpc': np.array([x, y, z]) in Mpc
        - 'observer_position_mpc': np.array([x, y, z]) in Mpc
        - 'mass_kg': float (in kg)
        - 'radius_m': float (in m)
        - 'mass_position_m': np.array([x, y, z]) in m
        - 'observer_position_m': np.array([x, y, z]) in m
        - 'distance_mpc': float (distance between observer and mass in Mpc)
    
    Returns None if filename doesn't match expected pattern.
    
    Examples
    --------
    >>> info = parse_trajectory_filename(
    ...     "backward_raytracing_perturbed_flrw_OPTIMAL_M1.0e15_R5.0_mass500_500_500_obs0_0_0_Mpc.h5"
    ... )
    >>> print(info['mass_msun'])
    1.0e+15
    >>> print(info['distance_mpc'])
    866.03
    """
    # Extract basename if full path provided
    basename = os.path.basename(filename)
    
    # Pattern to match the standardized filename format
    # backward_raytracing_{metric}_{version}_M{mass}_R{radius}_mass{x}_{y}_{z}_obs{x}_{y}_{z}_Mpc.h5
    pattern = (
        r'backward_raytracing_'
        r'(?P<metric_type>\w+)_'
        r'(?P<version>\w+)_'
        r'M(?P<mass>[0-9.e+-]+)_'
        r'R(?P<radius>[0-9.]+)_'
        r'mass(?P<mass_x>-?[0-9]+)_(?P<mass_y>-?[0-9]+)_(?P<mass_z>-?[0-9]+)_'
        r'obs(?P<obs_x>-?[0-9]+)_(?P<obs_y>-?[0-9]+)_(?P<obs_z>-?[0-9]+)_'
        r'Mpc\.h5'
    )
    
    match = re.match(pattern, basename)
    
    if not match:
        # Try legacy format (just mass position, for backward compatibility)
        legacy_pattern = (
            r'backward_raytracing.*?'
            r'mass_(?P<mass_x>[0-9]+)_(?P<mass_y>[0-9]+)_(?P<mass_z>[0-9]+)_Mpc\.h5'
        )
        legacy_match = re.match(legacy_pattern, basename)
        
        if legacy_match:
            # Return minimal info for legacy files
            mass_pos = np.array([
                float(legacy_match.group('mass_x')),
                float(legacy_match.group('mass_y')),
                float(legacy_match.group('mass_z'))
            ])
            
            return {
                'metric_type': 'unknown',
                'version': 'unknown',
                'mass_msun': None,
                'radius_mpc': None,
                'mass_position_mpc': mass_pos,
                'observer_position_mpc': None,
                'mass_kg': None,
                'radius_m': None,
                'mass_position_m': mass_pos * one_Mpc,
                'observer_position_m': None,
                'distance_mpc': None
            }
        
        return None
    
    # Extract values
    metric_type = match.group('metric_type')
    version = match.group('version')
    mass_msun = float(match.group('mass'))
    radius_mpc = float(match.group('radius'))
    
    mass_position_mpc = np.array([
        float(match.group('mass_x')),
        float(match.group('mass_y')),
        float(match.group('mass_z'))
    ])
    
    observer_position_mpc = np.array([
        float(match.group('obs_x')),
        float(match.group('obs_y')),
        float(match.group('obs_z'))
    ])
    
    # Convert to SI units
    mass_kg = mass_msun * one_Msun
    radius_m = radius_mpc * one_Mpc
    mass_position_m = mass_position_mpc * one_Mpc
    observer_position_m = observer_position_mpc * one_Mpc
    
    # Compute distance
    distance_mpc = np.linalg.norm(mass_position_mpc - observer_position_mpc)
    
    return {
        'metric_type': metric_type,
        'version': version,
        'mass_msun': mass_msun,
        'radius_mpc': radius_mpc,
        'mass_position_mpc': mass_position_mpc,
        'observer_position_mpc': observer_position_mpc,
        'mass_kg': mass_kg,
        'radius_m': radius_m,
        'mass_position_m': mass_position_m,
        'observer_position_m': observer_position_m,
        'distance_mpc': distance_mpc
    }


def format_simulation_info(filename):
    """
    Format simulation info from filename into human-readable string.
    
    Parameters
    ----------
    filename : str
        Trajectory filename
    
    Returns
    -------
    str
        Formatted information string, or error message if parsing fails
    """
    info = parse_trajectory_filename(filename)
    
    if info is None:
        return f"Could not parse filename: {filename}"
    
    lines = []
    lines.append("Simulation Parameters:")
    lines.append(f"  Metric: {info['metric_type']}")
    lines.append(f"  Version: {info['version']}")
    
    if info['mass_msun'] is not None:
        lines.append(f"  Mass: {info['mass_msun']:.2e} M_sun")
    if info['radius_mpc'] is not None:
        lines.append(f"  Radius: {info['radius_mpc']:.1f} Mpc")
    
    if info['mass_position_mpc'] is not None:
        mp = info['mass_position_mpc']
        lines.append(f"  Mass position: [{mp[0]:.0f}, {mp[1]:.0f}, {mp[2]:.0f}] Mpc")
    
    if info['observer_position_mpc'] is not None:
        op = info['observer_position_mpc']
        lines.append(f"  Observer position: [{op[0]:.0f}, {op[1]:.0f}, {op[2]:.0f}] Mpc")
    
    if info['distance_mpc'] is not None:
        lines.append(f"  Observer-Mass distance: {info['distance_mpc']:.2f} Mpc")
    
    return "\n".join(lines)
