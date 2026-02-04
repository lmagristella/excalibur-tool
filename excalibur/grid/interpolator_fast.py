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

@njit(cache=True, fastmath=True)
def interpolate_wrapper_numba(x, field, origin, spacing, shape):
    return interpolate_3d_numba(
        x[0], x[1], x[2],
        field,
        origin[0], origin[1], origin[2],
        spacing[0], spacing[1], spacing[2],
        shape[0], shape[1], shape[2]
    )

@njit(cache=True, fastmath=True)
def gradient_wrapper_numba(x, field, origin, spacing, shape):
    return gradient_3d_numba(
        x[0], x[1], x[2],
        field,
        origin[0], origin[1], origin[2],
        spacing[0], spacing[1], spacing[2],
        shape[0], shape[1], shape[2]
    )

@njit(cache=True, fastmath=True)
def fused_interpolation_gradient(
    x, y, z,
    field,
    origin_x, origin_y, origin_z,
    spacing_x, spacing_y, spacing_z,
    nx, ny, nz
):
    """
    Retourne :
        val : float
        grad_x, grad_y, grad_z : floats
        dt : float (0 ici)
    """
    # -----------------------
    # Relative position
    # -----------------------
    rel_x = (x - origin_x) / spacing_x
    rel_y = (y - origin_y) / spacing_y
    rel_z = (z - origin_z) / spacing_z

    # -----------------------
    # Check bounds
    # -----------------------
    if rel_x < 1 or rel_x >= nx - 2 or \
       rel_y < 1 or rel_y >= ny - 2 or \
       rel_z < 1 or rel_z >= nz - 2:
        return np.nan, 0.0, 0.0, 0.0, 0.0

    # -----------------------
    # Base index + fractions
    # -----------------------
    i0 = int(np.floor(rel_x))
    j0 = int(np.floor(rel_y))
    k0 = int(np.floor(rel_z))

    dx = rel_x - i0
    dy = rel_y - j0
    dz = rel_z - k0

    # -----------------------
    # Trilinear INTERPOLATION
    # -----------------------
    c000 = field[i0,   j0,   k0]
    c100 = field[i0+1, j0,   k0]
    c010 = field[i0,   j0+1, k0]
    c001 = field[i0,   j0,   k0+1]
    c110 = field[i0+1, j0+1, k0]
    c101 = field[i0+1, j0,   k0+1]
    c011 = field[i0,   j0+1, k0+1]
    c111 = field[i0+1, j0+1, k0+1]

    val = (
        c000 * (1-dx)*(1-dy)*(1-dz) +
        c100 * dx*(1-dy)*(1-dz) +
        c010 * (1-dx)*dy*(1-dz) +
        c001 * (1-dx)*(1-dy)*dz +
        c110 * dx*dy*(1-dz) +
        c101 * dx*(1-dy)*dz +
        c011 * (1-dx)*dy*dz +
        c111 * dx*dy*dz
    )

    # -----------------------
    # GRADIENT (central diff)
    # -----------------------
    grad_x = (field[i0+1, j0,   k0] - field[i0-1, j0,   k0]) / (2 * spacing_x)
    grad_y = (field[i0,   j0+1, k0] - field[i0,   j0-1, k0]) / (2 * spacing_y)
    grad_z = (field[i0,   j0,   k0+1] - field[i0,   j0,   k0-1]) / (2 * spacing_z)

    # Champ statique → d/dt = 0
    dt = 0.0

    return val, grad_x, grad_y, grad_z, dt

@njit(cache=True, fastmath=True)
def _wrap_index(i, n):
    # works for i possibly negative
    return i % n

@njit(cache=True, fastmath=True)
def fused_interpolation_gradient_periodic(
    x, y, z,
    field,
    origin_x, origin_y, origin_z,
    spacing_x, spacing_y, spacing_z,
    nx, ny, nz
):
    # Map position into [0, L) periodic domain
    Lx = nx * spacing_x
    Ly = ny * spacing_y
    Lz = nz * spacing_z

    # bring into periodic range
    xx = (x - origin_x) % Lx
    yy = (y - origin_y) % Ly
    zz = (z - origin_z) % Lz

    rel_x = xx / spacing_x
    rel_y = yy / spacing_y
    rel_z = zz / spacing_z

    i0 = int(np.floor(rel_x))
    j0 = int(np.floor(rel_y))
    k0 = int(np.floor(rel_z))

    dx = rel_x - i0
    dy = rel_y - j0
    dz = rel_z - k0

    i1 = _wrap_index(i0 + 1, nx)
    j1 = _wrap_index(j0 + 1, ny)
    k1 = _wrap_index(k0 + 1, nz)

    i0 = _wrap_index(i0, nx)
    j0 = _wrap_index(j0, ny)
    k0 = _wrap_index(k0, nz)

    # Trilinear interpolation with wrapped neighbors
    c000 = field[i0, j0, k0]
    c100 = field[i1, j0, k0]
    c010 = field[i0, j1, k0]
    c001 = field[i0, j0, k1]
    c110 = field[i1, j1, k0]
    c101 = field[i1, j0, k1]
    c011 = field[i0, j1, k1]
    c111 = field[i1, j1, k1]

    val = (
        c000 * (1-dx)*(1-dy)*(1-dz) +
        c100 * dx*(1-dy)*(1-dz) +
        c010 * (1-dx)*dy*(1-dz) +
        c001 * (1-dx)*(1-dy)*dz +
        c110 * dx*dy*(1-dz) +
        c101 * dx*(1-dy)*dz +
        c011 * (1-dx)*dy*dz +
        c111 * dx*dy*dz
    )

    # Central differences with periodic wrap (use nearest cell index)
    ic = i0
    jc = j0
    kc = k0

    ip = _wrap_index(ic + 1, nx)
    im = _wrap_index(ic - 1, nx)
    jp = _wrap_index(jc + 1, ny)
    jm = _wrap_index(jc - 1, ny)
    kp = _wrap_index(kc + 1, nz)
    km = _wrap_index(kc - 1, nz)

    grad_x = (field[ip, jc, kc] - field[im, jc, kc]) / (2.0 * spacing_x)
    grad_y = (field[ic, jp, kc] - field[ic, jm, kc]) / (2.0 * spacing_y)
    grad_z = (field[ic, jc, kp] - field[ic, jc, km]) / (2.0 * spacing_z)

    return val, grad_x, grad_y, grad_z, 0.0


class InterpolatorFast:
    """
    Fast interpolator using Numba-compiled functions.
    Drop-in replacement for Interpolator with better performance.
    Parameters:
    -----------
    boundary : str
        Boundary handling mode ("error", "clamp", "periodic")
    """
    def __init__(self, grid, boundary = "clamp"):
        self.grid = grid
        self.is_4d = len(grid.shape) == 4
        self.boundary = boundary
        
        # Pre-extract grid parameters for fast access
        self.origin = np.array(grid.origin, dtype=np.float64)
        self.spacing = np.array(grid.spacing, dtype=np.float64)
        self.shape = np.array(grid.shape, dtype=np.int64)

    def value_gradient_and_time_derivative(self, x, field, t=None):
        f = self.grid.fields[field]
        val, gx, gy, gz, dt = fused_interpolation_gradient(
            x[0], x[1], x[2],
            f,
            self.origin[0], self.origin[1], self.origin[2],
            self.spacing[0], self.spacing[1], self.spacing[2],
            self.shape[0], self.shape[1], self.shape[2]
        )
        if np.isnan(val):
            raise ValueError(f"Point {x} outside grid bounds")
        return val, (gx, gy, gz), dt
        
    def interpolate_old(self, x, field, t=None):
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
    
    def interpolate(self, x, field, t=None):
        f = self.grid.fields[field]
        result = interpolate_wrapper_numba(x, f, self.origin, self.spacing, self.shape)
        
        if np.isnan(result):
            raise ValueError(f"Position {x} outside grid bounds")
        
        return result
    
    def gradient_old(self, x, field, t=None):
        """Fast gradient calculation — returns tuple (gx, gy, gz)."""
        f = self.grid.fields[field]
        grad_x, grad_y, grad_z = gradient_3d_numba(
            x[0], x[1], x[2], f,
            self.origin[0], self.origin[1], self.origin[2],
            self.spacing[0], self.spacing[1], self.spacing[2],
            self.shape[0], self.shape[1], self.shape[2]
        )
        # Return as plain Python tuple to avoid creating a numpy array
        return grad_x, grad_y, grad_z
    
    def gradient(self, x, field, t=None):
        f = self.grid.fields[field]
        grad_x, grad_y, grad_z = gradient_wrapper_numba(x, f, self.origin, self.spacing, self.shape)
        return grad_x, grad_y, grad_z

    def value_and_gradient(self, x, field, t=None):
        """Combined value and gradient calculation — returns (val, (gx,gy,gz))."""
        val = self.interpolate(x, field, t)
        grad = self.gradient(x, field, t)  # tuple
        return val, grad

    def value_gradient_and_time_derivative_old(self, x, field, t):
        """Return value, gradient tuple and time derivative (float)."""
        val = self.interpolate(x, field, t)
        grad = self.gradient(x, field, t)
        # If field static, time derivative is zero — keep float
        return val, grad, 0.0
