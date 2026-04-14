# grid/interpolator_fast.py
import numpy as np
from numba import njit

# ============================================================
#  NUMBA HELPERS
# ============================================================

@njit(cache=True, fastmath=True)
def _wrap_index(i: int, n: int) -> int:
    # modulo that works for negative indices
    return i % n

@njit(cache=True, fastmath=True)
def _wrap_pos(p: float, origin: float, L: float) -> float:
    # Map p into [origin, origin+L)
    return origin + ((p - origin) % L)

@njit(cache=True, fastmath=True)
def _clamp_pos(p: float, lo: float, hi: float) -> float:
    # Clamp to [lo, hi]
    if p < lo:
        return lo
    if p > hi:
        return hi
    return p

@njit(cache=True, fastmath=True)
def _interp_trilinear_from_corners(
    c000, c100, c010, c001,
    c110, c101, c011, c111,
    dx, dy, dz
) -> float:
    return (
        c000 * (1 - dx) * (1 - dy) * (1 - dz) +
        c100 * dx       * (1 - dy) * (1 - dz) +
        c010 * (1 - dx) * dy       * (1 - dz) +
        c001 * (1 - dx) * (1 - dy) * dz       +
        c110 * dx       * dy       * (1 - dz) +
        c101 * dx       * (1 - dy) * dz       +
        c011 * (1 - dx) * dy       * dz       +
        c111 * dx       * dy       * dz
    )

# ============================================================
#  3D: VALUE + GRADIENT (ERROR mode)
#    - strict bounds, requires central differences => i in [1..n-2)
# ============================================================

@njit(cache=True, fastmath=True)
def fused_interpolation_gradient_3d_error(
    x, y, z,
    field,
    origin_x, origin_y, origin_z,
    spacing_x, spacing_y, spacing_z,
    nx, ny, nz
):
    rel_x = (x - origin_x) / spacing_x
    rel_y = (y - origin_y) / spacing_y
    rel_z = (z - origin_z) / spacing_z

    # Need i0-1 and i0+1 for central diffs, and i0+1 for interpolation corners
    if rel_x < 1.0 or rel_x >= (nx - 2.0) or rel_y < 1.0 or rel_y >= (ny - 2.0) or rel_z < 1.0 or rel_z >= (nz - 2.0):
        return np.nan, 0.0, 0.0, 0.0, 0.0

    i0 = int(np.floor(rel_x))
    j0 = int(np.floor(rel_y))
    k0 = int(np.floor(rel_z))

    dx = rel_x - i0
    dy = rel_y - j0
    dz = rel_z - k0

    # interpolation corners
    c000 = field[i0,   j0,   k0]
    c100 = field[i0+1, j0,   k0]
    c010 = field[i0,   j0+1, k0]
    c001 = field[i0,   j0,   k0+1]
    c110 = field[i0+1, j0+1, k0]
    c101 = field[i0+1, j0,   k0+1]
    c011 = field[i0,   j0+1, k0+1]
    c111 = field[i0+1, j0+1, k0+1]

    val = _interp_trilinear_from_corners(c000, c100, c010, c001, c110, c101, c011, c111, dx, dy, dz)

    # central differences at base cell index
    gx = (field[i0+1, j0,   k0] - field[i0-1, j0,   k0]) / (2.0 * spacing_x)
    gy = (field[i0,   j0+1, k0] - field[i0,   j0-1, k0]) / (2.0 * spacing_y)
    gz = (field[i0,   j0,   k0+1] - field[i0,   j0,   k0-1]) / (2.0 * spacing_z)

    return val, gx, gy, gz, 0.0


# ============================================================
#  3D: VALUE + GRADIENT (CLAMP mode)
#    - position clamped into box
#    - interpolation uses clamped indices
#    - gradient uses one-sided at edges, central otherwise
# ============================================================

@njit(cache=True, fastmath=True)
def fused_interpolation_gradient_3d_clamp(
    x, y, z,
    field,
    origin_x, origin_y, origin_z,
    spacing_x, spacing_y, spacing_z,
    nx, ny, nz
):
    # Clamp position to the grid domain in physical units.
    # For interpolation, we need i0 in [0..n-2]. That corresponds to rel in [0..n-1).
    # We clamp to [origin, origin + (n-1)*dx - eps] to avoid rel == n-1 exactly.
    epsx = 1e-12 * spacing_x
    epsy = 1e-12 * spacing_y
    epsz = 1e-12 * spacing_z

    x = _clamp_pos(x, origin_x, origin_x + (nx - 1) * spacing_x - epsx)
    y = _clamp_pos(y, origin_y, origin_y + (ny - 1) * spacing_y - epsy)
    z = _clamp_pos(z, origin_z, origin_z + (nz - 1) * spacing_z - epsz)

    rel_x = (x - origin_x) / spacing_x
    rel_y = (y - origin_y) / spacing_y
    rel_z = (z - origin_z) / spacing_z

    i0 = int(np.floor(rel_x))
    j0 = int(np.floor(rel_y))
    k0 = int(np.floor(rel_z))

    # Clamp indices for interpolation corner access
    if i0 < 0: i0 = 0
    if j0 < 0: j0 = 0
    if k0 < 0: k0 = 0
    if i0 > nx - 2: i0 = nx - 2
    if j0 > ny - 2: j0 = ny - 2
    if k0 > nz - 2: k0 = nz - 2

    dx = rel_x - i0
    dy = rel_y - j0
    dz = rel_z - k0

    c000 = field[i0,   j0,   k0]
    c100 = field[i0+1, j0,   k0]
    c010 = field[i0,   j0+1, k0]
    c001 = field[i0,   j0,   k0+1]
    c110 = field[i0+1, j0+1, k0]
    c101 = field[i0+1, j0,   k0+1]
    c011 = field[i0,   j0+1, k0+1]
    c111 = field[i0+1, j0+1, k0+1]

    val = _interp_trilinear_from_corners(c000, c100, c010, c001, c110, c101, c011, c111, dx, dy, dz)

    # Gradient: one-sided at edges, central otherwise.
    # Choose a center index ic in [0..n-1].
    ic = i0
    jc = j0
    kc = k0

    # X
    if ic <= 0:
        gx = (field[1, jc, kc] - field[0, jc, kc]) / spacing_x
    elif ic >= nx - 1:
        gx = (field[nx - 1, jc, kc] - field[nx - 2, jc, kc]) / spacing_x
    else:
        # if ic == nx-1 can't happen here (ic<=nx-2), but keep robust:
        ip = ic + 1 if ic + 1 < nx else nx - 1
        im = ic - 1 if ic - 1 >= 0 else 0
        gx = (field[ip, jc, kc] - field[im, jc, kc]) / (2.0 * spacing_x) if (ip != ic and im != ic) else (field[ip, jc, kc] - field[im, jc, kc]) / spacing_x

    # Y
    if jc <= 0:
        gy = (field[ic, 1, kc] - field[ic, 0, kc]) / spacing_y
    elif jc >= ny - 1:
        gy = (field[ic, ny - 1, kc] - field[ic, ny - 2, kc]) / spacing_y
    else:
        jp = jc + 1 if jc + 1 < ny else ny - 1
        jm = jc - 1 if jc - 1 >= 0 else 0
        gy = (field[ic, jp, kc] - field[ic, jm, kc]) / (2.0 * spacing_y) if (jp != jc and jm != jc) else (field[ic, jp, kc] - field[ic, jm, kc]) / spacing_y

    # Z
    if kc <= 0:
        gz = (field[ic, jc, 1] - field[ic, jc, 0]) / spacing_z
    elif kc >= nz - 1:
        gz = (field[ic, jc, nz - 1] - field[ic, jc, nz - 2]) / spacing_z
    else:
        kp = kc + 1 if kc + 1 < nz else nz - 1
        km = kc - 1 if kc - 1 >= 0 else 0
        gz = (field[ic, jc, kp] - field[ic, jc, km]) / (2.0 * spacing_z) if (kp != kc and km != kc) else (field[ic, jc, kp] - field[ic, jc, km]) / spacing_z

    return val, gx, gy, gz, 0.0


# ============================================================
#  3D: VALUE + GRADIENT (PERIODIC mode) - spatial periodicity
# ============================================================

@njit(cache=True, fastmath=True)
def fused_interpolation_gradient_3d_periodic(
    x, y, z,
    field,
    origin_x, origin_y, origin_z,
    spacing_x, spacing_y, spacing_z,
    nx, ny, nz
):
    Lx = nx * spacing_x
    Ly = ny * spacing_y
    Lz = nz * spacing_z

    x = _wrap_pos(x, origin_x, Lx)
    y = _wrap_pos(y, origin_y, Ly)
    z = _wrap_pos(z, origin_z, Lz)

    rel_x = (x - origin_x) / spacing_x
    rel_y = (y - origin_y) / spacing_y
    rel_z = (z - origin_z) / spacing_z

    i0 = int(np.floor(rel_x))
    j0 = int(np.floor(rel_y))
    k0 = int(np.floor(rel_z))

    dx = rel_x - i0
    dy = rel_y - j0
    dz = rel_z - k0

    i0 = _wrap_index(i0, nx)
    j0 = _wrap_index(j0, ny)
    k0 = _wrap_index(k0, nz)

    i1 = _wrap_index(i0 + 1, nx)
    j1 = _wrap_index(j0 + 1, ny)
    k1 = _wrap_index(k0 + 1, nz)

    # interpolation corners with wrap
    c000 = field[i0, j0, k0]
    c100 = field[i1, j0, k0]
    c010 = field[i0, j1, k0]
    c001 = field[i0, j0, k1]
    c110 = field[i1, j1, k0]
    c101 = field[i1, j0, k1]
    c011 = field[i0, j1, k1]
    c111 = field[i1, j1, k1]

    val = _interp_trilinear_from_corners(c000, c100, c010, c001, c110, c101, c011, c111, dx, dy, dz)

    # central diffs with wrap at cell index
    ip = _wrap_index(i0 + 1, nx)
    im = _wrap_index(i0 - 1, nx)
    jp = _wrap_index(j0 + 1, ny)
    jm = _wrap_index(j0 - 1, ny)
    kp = _wrap_index(k0 + 1, nz)
    km = _wrap_index(k0 - 1, nz)

    gx = (field[ip, j0, k0] - field[im, j0, k0]) / (2.0 * spacing_x)
    gy = (field[i0, jp, k0] - field[i0, jm, k0]) / (2.0 * spacing_y)
    gz = (field[i0, j0, kp] - field[i0, j0, km]) / (2.0 * spacing_z)

    return val, gx, gy, gz, 0.0


# ============================================================
#  4D: VALUE + GRADIENT + TIME DERIVATIVE
#  - time interpolation is linear between snapshots it0/it1
#  - time derivative is (val1 - val0)/dt (snapshot spacing)
#  - spatial boundary handling uses chosen mode per snapshot
# ============================================================

@njit(cache=True, fastmath=True)
def fused_interpolation_gradient_4d_error(
    x, y, z, t,
    field4d,
    origin_x, origin_y, origin_z, origin_t,
    spacing_x, spacing_y, spacing_z, spacing_t,
    nx, ny, nz, nt
):
    rel_t = (t - origin_t) / spacing_t
    if rel_t < 0.0 or rel_t >= (nt - 1.0):
        return np.nan, 0.0, 0.0, 0.0, 0.0

    it0 = int(np.floor(rel_t))
    if it0 < 0:
        it0 = 0
    if it0 > nt - 2:
        it0 = nt - 2
    it1 = it0 + 1
    wt = rel_t - it0

    # snapshot 0
    v0, gx0, gy0, gz0, _ = fused_interpolation_gradient_3d_error(
        x, y, z, field4d[:, :, :, it0],
        origin_x, origin_y, origin_z,
        spacing_x, spacing_y, spacing_z,
        nx, ny, nz
    )
    # snapshot 1
    v1, gx1, gy1, gz1, _ = fused_interpolation_gradient_3d_error(
        x, y, z, field4d[:, :, :, it1],
        origin_x, origin_y, origin_z,
        spacing_x, spacing_y, spacing_z,
        nx, ny, nz
    )

    if np.isnan(v0) or np.isnan(v1):
        return np.nan, 0.0, 0.0, 0.0, 0.0

    val = (1.0 - wt) * v0 + wt * v1
    gx = (1.0 - wt) * gx0 + wt * gx1
    gy = (1.0 - wt) * gy0 + wt * gy1
    gz = (1.0 - wt) * gz0 + wt * gz1

    dtd = (v1 - v0) / spacing_t
    return val, gx, gy, gz, dtd


@njit(cache=True, fastmath=True)
def fused_interpolation_gradient_4d_clamp(
    x, y, z, t,
    field4d,
    origin_x, origin_y, origin_z, origin_t,
    spacing_x, spacing_y, spacing_z, spacing_t,
    nx, ny, nz, nt
):
    # Clamp time into [t0, tN-1) for safe it0/it1
    eps_t = 1e-12 * spacing_t
    t = _clamp_pos(t, origin_t, origin_t + (nt - 1) * spacing_t - eps_t)

    rel_t = (t - origin_t) / spacing_t
    it0 = int(np.floor(rel_t))
    if it0 < 0:
        it0 = 0
    if it0 > nt - 2:
        it0 = nt - 2
    it1 = it0 + 1
    wt = rel_t - it0

    v0, gx0, gy0, gz0, _ = fused_interpolation_gradient_3d_clamp(
        x, y, z, field4d[:, :, :, it0],
        origin_x, origin_y, origin_z,
        spacing_x, spacing_y, spacing_z,
        nx, ny, nz
    )
    v1, gx1, gy1, gz1, _ = fused_interpolation_gradient_3d_clamp(
        x, y, z, field4d[:, :, :, it1],
        origin_x, origin_y, origin_z,
        spacing_x, spacing_y, spacing_z,
        nx, ny, nz
    )

    # clamp mode never returns nan for spatial; keep safe anyway
    if np.isnan(v0) or np.isnan(v1):
        return np.nan, 0.0, 0.0, 0.0, 0.0

    val = (1.0 - wt) * v0 + wt * v1
    gx = (1.0 - wt) * gx0 + wt * gx1
    gy = (1.0 - wt) * gy0 + wt * gy1
    gz = (1.0 - wt) * gz0 + wt * gz1

    dtd = (v1 - v0) / spacing_t
    return val, gx, gy, gz, dtd


@njit(cache=True, fastmath=True)
def fused_interpolation_gradient_4d_periodic(
    x, y, z, t,
    field4d,
    origin_x, origin_y, origin_z, origin_t,
    spacing_x, spacing_y, spacing_z, spacing_t,
    nx, ny, nz, nt
):
    # Time: clamp (NOT periodic) unless you really have periodic time snapshots.
    rel_t = (t - origin_t) / spacing_t
    if rel_t < 0.0 or rel_t >= (nt - 1.0):
        return np.nan, 0.0, 0.0, 0.0, 0.0

    it0 = int(np.floor(rel_t))
    if it0 < 0:
        it0 = 0
    if it0 > nt - 2:
        it0 = nt - 2
    it1 = it0 + 1
    wt = rel_t - it0

    v0, gx0, gy0, gz0, _ = fused_interpolation_gradient_3d_periodic(
        x, y, z, field4d[:, :, :, it0],
        origin_x, origin_y, origin_z,
        spacing_x, spacing_y, spacing_z,
        nx, ny, nz
    )
    v1, gx1, gy1, gz1, _ = fused_interpolation_gradient_3d_periodic(
        x, y, z, field4d[:, :, :, it1],
        origin_x, origin_y, origin_z,
        spacing_x, spacing_y, spacing_z,
        nx, ny, nz
    )

    if np.isnan(v0) or np.isnan(v1):
        return np.nan, 0.0, 0.0, 0.0, 0.0

    val = (1.0 - wt) * v0 + wt * v1
    gx = (1.0 - wt) * gx0 + wt * gx1
    gy = (1.0 - wt) * gy0 + wt * gy1
    gz = (1.0 - wt) * gz0 + wt * gz1

    dtd = (v1 - v0) / spacing_t
    return val, gx, gy, gz, dtd


# ============================================================
#  PUBLIC CLASS
# ============================================================

class InterpolatorFast:
    """
    Fast interpolator using Numba-compiled functions.
    Drop-in replacement for Interpolator.

    Supports:
      - 3D and 4D fields
      - boundary handling: "error" (strict), "clamp" (safe), "periodic" (spatial periodic)
      - value + gradient + time derivative (for 4D)
    """

    def __init__(self, grid, boundary: str = "periodic"):
        self.grid = grid
        self.is_4d = len(grid.shape) == 4

        if boundary not in ("error", "clamp", "periodic"):
            raise ValueError("boundary must be one of: 'error', 'clamp', 'periodic'")
        self.boundary = boundary

        # Pre-extract grid parameters for fast access
        self.origin = np.array(grid.origin, dtype=np.float64)
        self.spacing = np.array(grid.spacing, dtype=np.float64)
        self.shape = np.array(grid.shape, dtype=np.int64)

        if self.is_4d:
            if self.origin.shape[0] != 4 or self.spacing.shape[0] != 4 or self.shape.shape[0] != 4:
                raise ValueError("4D grid requires origin/spacing/shape of length 4")
        else:
            if self.origin.shape[0] < 3 or self.spacing.shape[0] < 3 or self.shape.shape[0] < 3:
                raise ValueError("3D grid requires origin/spacing/shape of length >= 3")

    # -------------------------
    # Core combined call
    # -------------------------
    def value_gradient_and_time_derivative(self, x, field, t=None):
        f = self.grid.fields[field]

        if self.is_4d:
            if t is None:
                raise ValueError("4D grid: you must pass t for time interpolation/derivative")

            if self.boundary == "periodic":
                val, gx, gy, gz, dtd = fused_interpolation_gradient_4d_periodic(
                    x[0], x[1], x[2], t,
                    f,
                    self.origin[0], self.origin[1], self.origin[2], self.origin[3],
                    self.spacing[0], self.spacing[1], self.spacing[2], self.spacing[3],
                    int(self.shape[0]), int(self.shape[1]), int(self.shape[2]), int(self.shape[3]),
                )
            elif self.boundary == "clamp":
                val, gx, gy, gz, dtd = fused_interpolation_gradient_4d_clamp(
                    x[0], x[1], x[2], t,
                    f,
                    self.origin[0], self.origin[1], self.origin[2], self.origin[3],
                    self.spacing[0], self.spacing[1], self.spacing[2], self.spacing[3],
                    int(self.shape[0]), int(self.shape[1]), int(self.shape[2]), int(self.shape[3]),
                )
            else:
                val, gx, gy, gz, dtd = fused_interpolation_gradient_4d_error(
                    x[0], x[1], x[2], t,
                    f,
                    self.origin[0], self.origin[1], self.origin[2], self.origin[3],
                    self.spacing[0], self.spacing[1], self.spacing[2], self.spacing[3],
                    int(self.shape[0]), int(self.shape[1]), int(self.shape[2]), int(self.shape[3]),
                )
        else:
            if self.boundary == "periodic":
                val, gx, gy, gz, dtd = fused_interpolation_gradient_3d_periodic(
                    x[0], x[1], x[2],
                    f,
                    self.origin[0], self.origin[1], self.origin[2],
                    self.spacing[0], self.spacing[1], self.spacing[2],
                    int(self.shape[0]), int(self.shape[1]), int(self.shape[2]),
                )
            elif self.boundary == "clamp":
                val, gx, gy, gz, dtd = fused_interpolation_gradient_3d_clamp(
                    x[0], x[1], x[2],
                    f,
                    self.origin[0], self.origin[1], self.origin[2],
                    self.spacing[0], self.spacing[1], self.spacing[2],
                    int(self.shape[0]), int(self.shape[1]), int(self.shape[2]),
                )
            else:
                val, gx, gy, gz, dtd = fused_interpolation_gradient_3d_error(
                    x[0], x[1], x[2],
                    f,
                    self.origin[0], self.origin[1], self.origin[2],
                    self.spacing[0], self.spacing[1], self.spacing[2],
                    int(self.shape[0]), int(self.shape[1]), int(self.shape[2]),
                )

        if np.isnan(val):
            raise ValueError(f"Point {x} outside grid bounds (boundary='{self.boundary}')")
        return val, (gx, gy, gz), dtd

    # -------------------------
    # Compatibility methods
    # -------------------------
    def interpolate(self, x, field, t=None):
        # Use fused call and discard gradient, d/dt
        if self.is_4d:
            if t is None:
                raise ValueError("4D grid: you must pass t to interpolate")
            val, _, _ = self.value_gradient_and_time_derivative(x, field, t)
            return val
        val, _, _ = self.value_gradient_and_time_derivative(x, field, 0.0)
        return val

    def gradient(self, x, field, t=None):
        if self.is_4d:
            if t is None:
                raise ValueError("4D grid: you must pass t to gradient")
            _, grad, _ = self.value_gradient_and_time_derivative(x, field, t)
            return grad
        _, grad, _ = self.value_gradient_and_time_derivative(x, field, 0.0)
        return grad

    def value_and_gradient(self, x, field, t=None):
        if self.is_4d:
            if t is None:
                raise ValueError("4D grid: you must pass t to value_and_gradient")
            val, grad, _ = self.value_gradient_and_time_derivative(x, field, t)
            return val, grad
        val, grad, _ = self.value_gradient_and_time_derivative(x, field, 0.0)
        return val, grad
