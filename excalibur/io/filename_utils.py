"""
Utilities for standardized trajectory filename generation and parsing.

This module provides functions to create descriptive filenames for trajectory
data that encode key simulation parameters (mass, radius, positions, number of photons,
integrator type) and parse this information back from filenames.
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
    n_photons=None,
    integrator=None,
    version=None,
    stop_mode=None,
    stop_value=None,
    output_dir="_data",
    extra_tags=None,
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
    n_photons : int, optional
        Number of photons simulated
    integrator : str, optional
        Type of integrator used ("optimal", "persistent", "sequential", "parallel", etc.)
    version : str, optional
        Backward-compatible alias for ``integrator`` used in older scripts/tests.
    stop_mode : str, optional
        Integration stopping condition ("steps", "redshift", "a", "chi")
    stop_value : float or int, optional
        Value for the stopping condition (e.g., 1000 for steps, 10.0 for redshift)
    output_dir : str, optional
        Output directory path (default: "_data")
    extra_tags : dict or None, optional
        Extra short tags appended to the filename for precise run provenance.
        Values are formatted compactly and joined as "keyvalue" (e.g.
        {"static": 1, "cone": "5deg"} -> "static1_cone5deg").
    
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
    ...     observer_position_m=[0, 0, 0],
    ...     n_photons=500,
    ...     integrator="optimal",
    ...     stop_mode="redshift",
    ...     stop_value=10.0
    ... )
    >>> print(filename)
    _data/backward_raytracing_perturbed_flrw_M1.0e+15_R5.0_mass500_500_500_obs0_0_0_N500_optimal_z10.0_Mpc.h5
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
    
    # Backward compatibility: allow passing version=<...> as alias for integrator.
    if integrator is None and version is not None:
        integrator = version

    # Format photon count and integrator (optional)
    photon_str = f"N{n_photons}" if n_photons is not None else ""
    integrator_str = integrator if integrator is not None else ""
    
    # Format stop condition (optional)
    stop_str = ""
    if stop_mode is not None and stop_value is not None:
        if stop_mode == "steps":
            stop_str = f"S{int(stop_value)}"
        elif stop_mode == "redshift":
            stop_str = f"z{stop_value:.1f}"
        elif stop_mode == "a":
            stop_str = f"a{stop_value:.3f}"
        elif stop_mode == "chi":
            stop_str = f"chi{stop_value:.1f}"
        else:
            # Generic format for unknown stop modes
            stop_str = f"{stop_mode}{stop_value}"
    
    # Build filename with optional components
    filename_parts = [
        "excalibur_run",
        metric_type,
        mass_str,
        radius_str,
        mass_pos_str,
        obs_pos_str
    ]
    
    # Add optional components
    if photon_str:
        filename_parts.append(photon_str)
    if integrator_str:
        filename_parts.append(integrator_str)
    if stop_str:
        filename_parts.append(stop_str)
    
    # Add user-provided tags (optional)
    if extra_tags:
        for k, v in extra_tags.items():
            if v is None:
                continue
            key = str(k)
            if isinstance(v, bool):
                val = "1" if v else "0"
            elif isinstance(v, (int, np.integer)):
                val = str(int(v))
            elif isinstance(v, (float, np.floating)):
                val = f"{float(v):.6g}".replace("+", "")
            else:
                val = str(v)
            filename_parts.append(f"{key}{val}")

    # Join with underscores and add extension
    filename = "_".join(filename_parts) + "_Mpc.h5"
    
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
        - 'mass_msun': float (in solar masses)
        - 'radius_mpc': float (in Mpc)
        - 'mass_position_mpc': np.array([x, y, z]) in Mpc
        - 'observer_position_mpc': np.array([x, y, z]) in Mpc
        - 'n_photons': int (number of photons, if available)
        - 'integrator': str (integrator type, if available)
        - 'stop_mode': str (stopping condition mode, if available)
        - 'stop_value': float/int (stopping condition value, if available)
        - 'mass_kg': float (in kg)
        - 'radius_m': float (in m)
        - 'mass_position_m': np.array([x, y, z]) in m
        - 'observer_position_m': np.array([x, y, z]) in m
        - 'distance_mpc': float (distance between observer and mass in Mpc)
    
    Returns None if filename doesn't match expected pattern.
    
    Examples
    --------
    >>> info = parse_trajectory_filename(
    ...     "backward_raytracing_perturbed_flrw_M1.0e15_R5.0_mass500_500_500_obs0_0_0_N500_optimal_z10.0_Mpc.h5"
    ... )
    >>> print(info['mass_msun'])
    1.0e+15
    >>> print(info['n_photons'])
    500
    >>> print(info['integrator'])
    'optimal'
    >>> print(info['stop_mode'])
    'redshift'
    >>> print(info['stop_value'])
    10.0
    """
    # Extract basename if full path provided
    basename = os.path.basename(filename)
    
    # Enhanced pattern to match the new filename format with optional components
    # excalibur_run_{metric}_M{mass}_R{radius}_mass{x}_{y}_{z}_obs{x}_{y}_{z}
    #   [_N{photons}][_{integrator}][_{stop_condition}]_Mpc.h5
    # Also supports legacy "version" tokens appended as the last component
    # (e.g. ..._OPTIMAL_Mpc.h5).
    pattern = (
        r'excalibur_run_'
        r'(?P<metric_type>\w+)_'
        r'M(?P<mass>[0-9.e+-]+)_'
        r'R(?P<radius>[0-9.]+)_'
        r'mass(?P<mass_x>-?[0-9]+)_(?P<mass_y>-?[0-9]+)_(?P<mass_z>-?[0-9]+)_'
        r'obs(?P<obs_x>-?[0-9]+)_(?P<obs_y>-?[0-9]+)_(?P<obs_z>-?[0-9]+)'
        r'(?:_N(?P<n_photons>[0-9]+))?'                    # Optional photon count
        r'(?:_(?P<integrator>sequential|parallel|optimal|persistent|rk45|rk4|leapfrog4|OPTIMAL|OPTIMIZED|standard))?'  # Optional integrator/version
        r'(?:_(?P<stop_condition>S[0-9]+|z[0-9.]+|a[0-9.]+|chi[0-9.]+))?'  # Optional stop condition
        r'(?:_[A-Za-z0-9.]+)*'                             # Optional extra tags
        r'_Mpc\.h5'
    )
    
    match = re.match(pattern, basename)
    
    if not match:
        # Try legacy format (just mass position, for backward compatibility)
        legacy_pattern = (
            r'(backward_raytracing|excalibur_run).*?'
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
                'mass_msun': None,
                'radius_mpc': None,
                'mass_position_mpc': mass_pos,
                'observer_position_mpc': None,
                'n_photons': None,
                'integrator': None,
                'version': None,
                'stop_mode': None,
                'stop_value': None,
                'mass_kg': None,
                'radius_m': None,
                'mass_position_m': mass_pos * one_Mpc,
                'observer_position_m': None,
                'distance_mpc': None
            }
        
        return None
    
    # Extract values
    metric_type = match.group('metric_type')
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
    
    # Extract optional parameters
    n_photons = int(match.group('n_photons')) if match.group('n_photons') else None
    integrator = match.group('integrator') if match.group('integrator') else None
    version = integrator  # backward-compatible alias
    
    # Parse stop condition
    stop_mode = None
    stop_value = None
    stop_condition_str = match.group('stop_condition')
    if stop_condition_str:
        if stop_condition_str.startswith('S'):
            stop_mode = 'steps'
            stop_value = int(stop_condition_str[1:])
        elif stop_condition_str.startswith('z'):
            stop_mode = 'redshift'
            stop_value = float(stop_condition_str[1:])
        elif stop_condition_str.startswith('a'):
            stop_mode = 'a'
            stop_value = float(stop_condition_str[1:])
        elif stop_condition_str.startswith('chi'):
            stop_mode = 'chi'
            stop_value = float(stop_condition_str[3:])
        else:
            # Try to parse generic format
            match_generic = re.match(r'([a-zA-Z]+)([0-9.]+)', stop_condition_str)
            if match_generic:
                stop_mode = match_generic.group(1)
                try:
                    stop_value = float(match_generic.group(2))
                except ValueError:
                    stop_value = int(match_generic.group(2))
    
    # Convert to SI units
    mass_kg = mass_msun * one_Msun
    radius_m = radius_mpc * one_Mpc
    mass_position_m = mass_position_mpc * one_Mpc
    observer_position_m = observer_position_mpc * one_Mpc
    
    # Compute distance
    distance_mpc = np.linalg.norm(mass_position_mpc - observer_position_mpc)
    
    return {
        'metric_type': metric_type,
        'mass_msun': mass_msun,
        'radius_mpc': radius_mpc,
        'mass_position_mpc': mass_position_mpc,
        'observer_position_mpc': observer_position_mpc,
        'n_photons': n_photons,
        'integrator': integrator,
        'version': version,
        'stop_mode': stop_mode,
        'stop_value': stop_value,
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
    
    if info['n_photons'] is not None:
        lines.append(f"  Number of photons: {info['n_photons']}")
    
    if info['integrator'] is not None:
        lines.append(f"  Integrator: {info['integrator']}")
    
    if info['stop_mode'] is not None and info['stop_value'] is not None:
        stop_mode = info['stop_mode']
        stop_value = info['stop_value']
        if stop_mode == 'steps':
            lines.append(f"  Stop condition: {int(stop_value)} steps")
        elif stop_mode == 'redshift':
            lines.append(f"  Stop condition: redshift z = {stop_value:.1f}")
        elif stop_mode == 'a':
            lines.append(f"  Stop condition: scale factor a = {stop_value:.3f}")
        elif stop_mode == 'chi':
            lines.append(f"  Stop condition: comoving distance χ = {stop_value:.1f}")
        else:
            lines.append(f"  Stop condition: {stop_mode} = {stop_value}")
    
    if info['distance_mpc'] is not None:
        lines.append(f"  Observer-Mass distance: {info['distance_mpc']:.2f} Mpc")
    
    return "\n".join(lines)