# grid/interpolator_fast.py
import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def interpolate_3d_numba(x, y, z, field, origin_x, origin_y, origin_z, 
                         spacing_x, spacing_y, spacing_z, nx, ny, nz):
    """
    Fast trilinear interpolation compiled with Numba.
    
    Parameters:
    -----------
    x, y, z : float
        Position to interpolate at
    field : 3D array
        Field values on grid
    origin_x, origin_y, origin_z : float
        Grid origin coordinates
    spacing_x, spacing_y, spacing_z : float
        Grid spacing
    nx, ny, nz : int
        Grid dimensions
    """
    # Compute relative position
    rel_x = (x - origin_x) / spacing_x
    rel_y = (y - origin_y) / spacing_y
    rel_z = (z - origin_z) / spacing_z
    
    # Check bounds
    if rel_x < 0 or rel_x >= nx - 1 or rel_y < 0 or rel_y >= ny - 1 or rel_z < 0 or rel_z >= nz - 1:
        return np.nan  # Out of bounds
    
    # Get integer indices and fractional parts
    i0 = int(np.floor(rel_x))
    j0 = int(np.floor(rel_y))
    k0 = int(np.floor(rel_z))
    
    # Clip to valid range
    i0 = min(max(i0, 0), nx - 2)
    j0 = min(max(j0, 0), ny - 2)
    k0 = min(max(k0, 0), nz - 2)
    
    # Fractional parts
    dx = rel_x - i0
    dy = rel_y - j0
    dz = rel_z - k0
    
    # Trilinear interpolation
    c000 = field[i0, j0, k0]
    c100 = field[i0+1, j0, k0]
    c010 = field[i0, j0+1, k0]
    c001 = field[i0, j0, k0+1]
    c110 = field[i0+1, j0+1, k0]
    c101 = field[i0+1, j0, k0+1]
    c011 = field[i0, j0+1, k0+1]
    c111 = field[i0+1, j0+1, k0+1]
    
    interp = (c000 * (1-dx) * (1-dy) * (1-dz) +
              c100 * dx * (1-dy) * (1-dz) +
              c010 * (1-dx) * dy * (1-dz) +
              c001 * (1-dx) * (1-dy) * dz +
              c110 * dx * dy * (1-dz) +
              c101 * dx * (1-dy) * dz +
              c011 * (1-dx) * dy * dz +
              c111 * dx * dy * dz)
    
    return interp


@njit(cache=True, fastmath=True)
def gradient_3d_numba(x, y, z, field, origin_x, origin_y, origin_z,
                     spacing_x, spacing_y, spacing_z, nx, ny, nz):
    """
    Fast gradient calculation using central differences.
    Returns (grad_x, grad_y, grad_z).
    """
    # Compute relative position
    rel_x = (x - origin_x) / spacing_x
    rel_y = (y - origin_y) / spacing_y
    rel_z = (z - origin_z) / spacing_z
    
    # Get integer index
    i0 = int(np.floor(rel_x))
    j0 = int(np.floor(rel_y))
    k0 = int(np.floor(rel_z))
    
    # Clip to valid range (need ±1 for gradient)
    i0 = min(max(i0, 1), nx - 2)
    j0 = min(max(j0, 1), ny - 2)
    k0 = min(max(k0, 1), nz - 2)
    
    # Central differences
    grad_x = (field[i0+1, j0, k0] - field[i0-1, j0, k0]) / (2 * spacing_x)
    grad_y = (field[i0, j0+1, k0] - field[i0, j0-1, k0]) / (2 * spacing_y)
    grad_z = (field[i0, j0, k0+1] - field[i0, j0, k0-1]) / (2 * spacing_z)
    
    return grad_x, grad_y, grad_z


class InterpolatorFast:
    """
    Fast interpolator using Numba-compiled functions.
    Drop-in replacement for Interpolator with better performance.
    """
    def __init__(self, grid):
        self.grid = grid
        self.is_4d = len(grid.shape) == 4
        
        # Pre-extract grid parameters for fast access
        self.origin = grid.origin
        self.spacing = grid.spacing
        self.shape = grid.shape
        
    def interpolate(self, x, field, t=None):
        """Fast 3D interpolation."""
        if self.is_4d and t is not None:
            raise NotImplementedError("4D interpolation not yet optimized")
        
        f = self.grid.fields[field]
        result = interpolate_3d_numba(
            x[0], x[1], x[2], f,
            self.origin[0], self.origin[1], self.origin[2],
            self.spacing[0], self.spacing[1], self.spacing[2],
            self.shape[0], self.shape[1], self.shape[2]
        )
        
        if np.isnan(result):
            raise ValueError(f"Position {x} outside grid bounds")
        
        return result
    
    def gradient(self, x, field, t=None):
        """Fast gradient calculation."""
        f = self.grid.fields[field]
        grad_x, grad_y, grad_z = gradient_3d_numba(
            x[0], x[1], x[2], f,
            self.origin[0], self.origin[1], self.origin[2],
            self.spacing[0], self.spacing[1], self.spacing[2],
            self.shape[0], self.shape[1], self.shape[2]
        )
        return np.array([grad_x, grad_y, grad_z])
    
    def value_and_gradient(self, x, field, t=None):
        """Combined value and gradient calculation."""
        val = self.interpolate(x, field, t)
        grad = self.gradient(x, field, t)
        return val, grad
    
    def value_gradient_and_time_derivative(self, x, field, t):
        """Retourne valeur, gradient spatial et dérivée temporelle."""
        # Time derivative is zero for static field
        val = self.interpolate(x, field, t)
        grad = self.gradient(x, field, t)
        return val, grad, 0.0
