"""
Helpers for coordinate transformations.
"""

import numpy as np
from excalibur.core.constants import *

def spherical_to_cartesian(r, theta, phi):
    """
    Convert spherical coordinates (r, theta, phi) to Cartesian (x, y, z).
    
    Parameters:
    -----------
    r : float or np.ndarray
        Radial distance
    theta : float or np.ndarray
        Polar angle (0 <= theta <= pi)
    phi : float or np.ndarray
        Azimuthal angle (0 <= phi < 2pi)
        
    Returns:
    --------
    x, y, z : float or np.ndarray
        Cartesian coordinates
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def cartesian_to_spherical(x, y, z):
    """
    Convert Cartesian coordinates (x, y, z) to spherical (r, theta, phi).
    
    Parameters:
    -----------
    x : float or np.ndarray
        X coordinate
    y : float or np.ndarray
        Y coordinate
    z : float or np.ndarray
        Z coordinate
        
    Returns:
    --------
    r, theta, phi : float or np.ndarray
        Spherical coordinates
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi

def cartesian_velocity_to_spherical(x, y, z, vx, vy, vz):
    """
    Convert Cartesian velocity components (vx, vy, vz) to spherical (vr, vtheta, vphi).
    
    Parameters:
    -----------
    x : float or np.ndarray
        X coordinate
    y : float or np.ndarray
        Y coordinate
    z : float or np.ndarray
        Z coordinate
    vx : float or np.ndarray
        Velocity in X direction
    vy : float or np.ndarray
        Velocity in Y direction
    vz : float or np.ndarray
        Velocity in Z direction
        
    Returns:
    --------
    vr, vtheta, vphi : float or np.ndarray
        Spherical velocity components
    """
    r, theta, phi = cartesian_to_spherical(x, y, z)
    
    vr = (x * vx + y * vy + z * vz) / r
    vtheta = ( (z * (x * vx + y * vy) - (x**2 + y**2) * vz) ) / (r**2 * np.sqrt(x**2 + y**2))
    vphi = (x * vy - y * vx) / (x**2 + y**2)
    
    return vr, vtheta, vphi

def spherical_to_isotropic_radius(r_spherical, r_s):
    """
    Convert spherical radial coordinate to isotropic radial coordinate.
    
    Parameters:
    -----------
    r_spherical : float or np.ndarray
        Spherical radial coordinate
    r_s : float
        Schwarzschild radius
        
    Returns:
    --------
    r_isotropic : float or np.ndarray
        Isotropic radial coordinate
    """
    # Solve quadratic equation: r_spherical = r * (1 + r_s/(4r))^2
    a = 1
    b = r_s / 2
    c = (r_s**2) / 16 - r_spherical
    
    discriminant = b**2 - 4*a*c
    if np.any(discriminant < 0):
        raise ValueError("No real solution for isotropic radius.")
    
    sqrt_discriminant = np.sqrt(discriminant)
    r_isotropic = (-b + sqrt_discriminant) / (2*a)
    
    return r_isotropic

def isotropic_to_spherical_radius(r_isotropic, r_s):
    """
    Convert isotropic radial coordinate to spherical radial coordinate.
    
    Parameters:
    -----------
    r_isotropic : float or np.ndarray
        Isotropic radial coordinate
    r_s : float
        Schwarzschild radius
        
    Returns:
    --------
    r_spherical : float or np.ndarray
        Spherical radial coordinate
    """
    r_spherical = r_isotropic * (1 + r_s / (4 * r_isotropic))**2
    return r_spherical

def cartesian_to_isotropic_radius(x, y, z, center, r_s):
    """
    Convert Cartesian coordinates to isotropic radial coordinate.
    
    Parameters:
    -----------
    x : float or np.ndarray
        X coordinate
    y : float or np.ndarray
        Y coordinate
    z : float or np.ndarray
        Z coordinate
    center : array-like
        Center position [x0, y0, z0]
    r_s : float
        Schwarzschild radius
        
    Returns:
    --------
    r_isotropic : float or np.ndarray
        Isotropic radial coordinate
    """
    shifted_x = x - center[0]
    shifted_y = y - center[1]
    shifted_z = z - center[2]
    
    r = np.sqrt(shifted_x**2 + shifted_y**2 + shifted_z**2)
    r_isotropic = spherical_to_isotropic_radius(r, r_s)
    
    return r_isotropic
