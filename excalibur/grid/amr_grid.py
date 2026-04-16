"""
Patch-based AMR (Adaptive Mesh Refinement) for excalibur.

Architecture
------------
  AMRPatch   - one rectangular refined region at a given level.
               Wraps a normal ``Grid`` + ``InterpolatorFast`` instance.
  AMRGrid    - root (coarse) grid plus an ordered list of patches.
  AMRInterpolator - drop-in replacement for ``InterpolatorFast``;
               queries are transparently dispatched to the finest
               patch covering the requested point.

The design is inspired by Berger-Colella structured AMR but
simplified: patches are axis-aligned boxes at integral refinement
ratios, and they carry their own copy of the field data (no
pointer-based sharing).

Usage
-----
>>> grid = AMRGrid.from_field(base_grid, "Phi", refine_fn, max_level=3)
>>> interp = AMRInterpolator(grid, boundary="clamp", scheme="tricubic")
>>> val, grad, hess, dtd = interp.value_gradient_hessian_and_time_derivative(pos, "Phi")
"""
from __future__ import annotations

import numpy as np
from typing import Callable, List, Optional, Tuple

from excalibur.grid.grid import Grid
from excalibur.grid.interpolator_4d_fast import InterpolatorFast


# ===================================================================
#  AMR Patch
# ===================================================================

class AMRPatch:
    """
    One rectangular refined region.

    Parameters
    ----------
    level : int
        Refinement level (0 = coarse root).
    origin : ndarray (3,)
        Physical origin of the patch [m].
    extent : ndarray (3,)
        Physical size of the patch in each dimension [m].
    shape : tuple (Nx, Ny, Nz)
        Number of cells in each direction.
    parent_level : int
        Level of the parent grid that was refined.
    """

    def __init__(
        self,
        level: int,
        origin: np.ndarray,
        extent: np.ndarray,
        shape: Tuple[int, int, int],
        parent_level: int = 0,
    ):
        self.level = level
        self.origin = np.asarray(origin, dtype=np.float64)
        self.extent = np.asarray(extent, dtype=np.float64)
        self.shape = tuple(shape)
        self.parent_level = parent_level

        self.spacing = self.extent / np.array(self.shape, dtype=np.float64)
        self.upper = self.origin + self.extent

        # Will be populated by AMRGrid
        self.grid: Optional[Grid] = None
        self.interp: Optional[InterpolatorFast] = None

    def contains(self, pos: np.ndarray) -> bool:
        """Return True if spatial position *pos* (3,) lies inside this patch."""
        return bool(
            np.all(pos >= self.origin) and np.all(pos < self.upper)
        )

    def __repr__(self):
        dx_kpc = self.spacing[0] / 3.0857e19  # rough kpc
        return (
            f"AMRPatch(level={self.level}, "
            f"shape={self.shape}, "
            f"dx~{dx_kpc:.0f} kpc, "
            f"origin_Mpc=[{self.origin[0]/3.0857e22:.2f}, "
            f"{self.origin[1]/3.0857e22:.2f}, "
            f"{self.origin[2]/3.0857e22:.2f}])"
        )


# ===================================================================
#  AMR Grid
# ===================================================================

class AMRGrid:
    """
    Root grid plus a list of refined patches.

    The root grid (level 0) covers the entire domain.
    Each patch at level *L* has spacing = root_spacing / ratio^L.

    Attributes
    ----------
    root : Grid
        The coarse base grid.
    patches : list of AMRPatch
        Sorted from finest (highest level) to coarsest so that
        look-ups find the finest covering patch first.
    field_names : list of str
        Names of the fields present on all levels.
    """

    def __init__(self, root: Grid):
        self.root = root
        self.patches: List[AMRPatch] = []
        self.field_names: List[str] = list(root.fields.keys())
        self.max_level = 0

        # Expose root fields for backward compatibility with code that
        # accesses ``grid.fields["Phi"]`` directly.
        self.fields = root.fields
        self.shape = root.shape
        self.spacing = root.spacing
        self.origin = root.origin

    # -----------------------------------------------------------------
    #  Factory: automatic refinement from a scalar field
    # -----------------------------------------------------------------
    @classmethod
    def from_field(
        cls,
        base_grid: Grid,
        field_name: str,
        potential_fn: Callable,
        max_level: int = 3,
        ratio: int = 2,
        refine_threshold: float = 0.05,
        refine_mode: str = "gradient",
        min_patch_cells: int = 32,
        boundary: str = "clamp",
        scheme: str = "tricubic",
        verbose: bool = True,
    ) -> "AMRGrid":
        """
        Build an AMR hierarchy by recursively refining regions where
        the field has large gradients (or curvature).

        Parameters
        ----------
        base_grid : Grid
            Already-populated coarse grid with ``field_name`` in ``.fields``.
        field_name : str
            Name of the field to analyse for refinement (e.g. "Phi").
        potential_fn : callable(x, y, z) -> float or ndarray
            Function that evaluates the *exact* field value at arbitrary
            positions.  Used to populate refined patches.
        max_level : int
            Maximum number of refinement levels beyond root.
        ratio : int
            Refinement ratio per level (typically 2 or 4).
        refine_threshold : float
            Relative threshold for the refinement criterion.
            For ``refine_mode="gradient"``: refine where
            |grad Phi| * dx / |Phi_max| > threshold.
        refine_mode : str
            "gradient" - refine where spatial gradient is large.
            "curvature" - refine where Laplacian is large.
        min_patch_cells : int
            Minimum number of cells per dimension in any refined patch.
        boundary, scheme : str
            Interpolation settings for each patch's InterpolatorFast.
        verbose : bool
            Print refinement statistics.

        Returns
        -------
        amr : AMRGrid
            The complete AMR hierarchy.
        """
        amr = cls(base_grid)

        if verbose:
            print(f"  AMR: root grid {tuple(base_grid.shape)}, "
                  f"dx = {base_grid.spacing[0]/3.0857e19:.1f} kpc")

        # Build interpolator for root
        # (used internally only for gradient estimation)
        root_interp = InterpolatorFast(base_grid, boundary=boundary, scheme=scheme)

        current_field = base_grid.fields[field_name]
        current_origin = np.array(base_grid.origin, dtype=np.float64)
        current_spacing = np.array(base_grid.spacing, dtype=np.float64)
        current_shape = np.array(base_grid.shape, dtype=np.int64)

        field_max = np.max(np.abs(current_field))
        if field_max == 0:
            field_max = 1.0

        for level in range(1, max_level + 1):
            # Identify cells that need refinement on the current finest level
            flagged = _flag_cells(
                current_field, current_spacing, field_max,
                refine_threshold, refine_mode,
            )

            if not np.any(flagged):
                if verbose:
                    print(f"  AMR level {level}: no cells flagged  --  stopping.")
                break

            n_flagged = np.sum(flagged)

            # Compute bounding box of flagged region (with buffer)
            bbox_lo, bbox_hi = _flagged_bbox(
                flagged, current_origin, current_spacing, buffer_cells=4,
            )

            # Ensure minimum patch size
            patch_extent = bbox_hi - bbox_lo
            new_spacing = current_spacing / ratio
            patch_shape_raw = np.round(patch_extent / new_spacing).astype(int)
            patch_shape = np.maximum(patch_shape_raw, min_patch_cells)
            # Recompute extent to be consistent
            patch_extent = patch_shape * new_spacing

            # Re-centre: keep the patch centred on the flagged region
            flagged_center = 0.5 * (bbox_lo + bbox_hi)
            patch_origin = flagged_center - 0.5 * patch_extent

            # Clamp to root domain
            root_upper = current_origin + current_shape * current_spacing
            patch_origin = np.maximum(patch_origin, np.array(base_grid.origin, dtype=np.float64))
            patch_upper = patch_origin + patch_extent
            patch_upper = np.minimum(patch_upper, np.array(base_grid.origin, dtype=np.float64) + np.array(base_grid.shape) * np.array(base_grid.spacing))
            patch_extent = patch_upper - patch_origin
            patch_shape = np.round(patch_extent / new_spacing).astype(int)
            patch_shape = np.maximum(patch_shape, min_patch_cells)
            patch_extent = patch_shape * new_spacing

            patch = AMRPatch(
                level=level,
                origin=patch_origin,
                extent=patch_extent,
                shape=tuple(patch_shape),
                parent_level=level - 1,
            )

            # Populate the patch field by evaluating potential_fn on the
            # refined grid.
            _populate_patch(patch, field_name, potential_fn, verbose=verbose)

            # Create Grid + InterpolatorFast for this patch
            patch.grid = Grid(
                shape=patch.shape,
                spacing=tuple(patch.spacing),
                origin=tuple(patch.origin),
            )
            patch.grid.add_field(field_name, patch._field_data)
            patch.interp = InterpolatorFast(
                patch.grid, boundary=boundary, scheme=scheme,
            )

            amr.patches.append(patch)
            amr.max_level = level

            if verbose:
                mem_MB = np.prod(patch_shape) * 8 / 1e6
                print(f"  AMR level {level}: {tuple(patch_shape)}, "
                      f"dx = {new_spacing[0]/3.0857e19:.1f} kpc, "
                      f"flagged = {n_flagged}, "
                      f"mem = {mem_MB:.1f} MB")

            # For the next iteration, analyse *this* patch's field
            current_field = patch._field_data
            current_origin = patch.origin.copy()
            current_spacing = patch.spacing.copy()
            current_shape = np.array(patch.shape, dtype=np.int64)

        # Sort patches finest-first for fast lookup
        amr.patches.sort(key=lambda p: -p.level)

        # Also build root interp (for queries outside all patches)
        amr._root_interp = root_interp

        if verbose:
            total_cells = int(np.prod(base_grid.shape))
            for p in amr.patches:
                total_cells += int(np.prod(p.shape))
            total_mem_GB = total_cells * 8 / 1e9
            print(f"  AMR total: {len(amr.patches)} patches + root, "
                  f"{total_cells/1e6:.1f} M cells, "
                  f"{total_mem_GB:.2f} GB")

        return amr

    def find_patch(self, pos: np.ndarray) -> Optional[AMRPatch]:
        """Return the finest patch containing *pos*, or None (use root)."""
        for patch in self.patches:   # sorted finest-first
            if patch.contains(pos):
                return patch
        return None

    def __repr__(self):
        return (
            f"AMRGrid(root={tuple(self.root.shape)}, "
            f"patches={len(self.patches)}, "
            f"max_level={self.max_level})"
        )


# ===================================================================
#  AMR Interpolator  (drop-in for InterpolatorFast)
# ===================================================================

class AMRInterpolator:
    """
    Drop-in replacement for ``InterpolatorFast`` that transparently
    queries the finest AMR patch covering the requested point.

    Implements the same public API:
      - value_gradient_hessian_and_time_derivative(x, field, t=None)
      - value_gradient_and_time_derivative(x, field, t=None)
      - interpolate(x, field, t=None)
      - gradient(x, field, t=None)
      - value_and_gradient(x, field, t=None)
      - hessian(x, field, t=None)
      - laplacian(x, field, t=None)
    """

    def __init__(self, amr_grid: AMRGrid, boundary: str = "clamp",
                 scheme: str = "tricubic"):
        self.amr = amr_grid
        self.boundary = boundary
        self.scheme = scheme

        # Ensure root interpolator exists
        if not hasattr(amr_grid, '_root_interp') or amr_grid._root_interp is None:
            amr_grid._root_interp = InterpolatorFast(
                amr_grid.root, boundary=boundary, scheme=scheme,
            )

        # Also expose grid for compatibility (metric checks self.interp.grid)
        self.grid = amr_grid.root

    def _get_interp(self, pos: np.ndarray) -> InterpolatorFast:
        """Find the finest interpolator covering *pos*."""
        patch = self.amr.find_patch(pos)
        if patch is not None and patch.interp is not None:
            return patch.interp
        return self.amr._root_interp

    # ----------------------------------------------------------
    #  Core
    # ----------------------------------------------------------
    def value_gradient_hessian_and_time_derivative(self, x, field, t=None):
        interp = self._get_interp(x)
        return interp.value_gradient_hessian_and_time_derivative(x, field, t)

    # ----------------------------------------------------------
    #  Backward-compatible API
    # ----------------------------------------------------------
    def value_gradient_and_time_derivative(self, x, field, t=None):
        interp = self._get_interp(x)
        return interp.value_gradient_and_time_derivative(x, field, t)

    def interpolate(self, x, field, t=None):
        interp = self._get_interp(x)
        return interp.interpolate(x, field, t)

    def gradient(self, x, field, t=None):
        interp = self._get_interp(x)
        return interp.gradient(x, field, t)

    def value_and_gradient(self, x, field, t=None):
        interp = self._get_interp(x)
        return interp.value_and_gradient(x, field, t)

    def hessian(self, x, field, t=None):
        interp = self._get_interp(x)
        return interp.hessian(x, field, t)

    def laplacian(self, x, field, t=None):
        interp = self._get_interp(x)
        return interp.laplacian(x, field, t)


# ===================================================================
#  Internal helpers
# ===================================================================

def _flag_cells(
    field: np.ndarray,
    spacing: np.ndarray,
    field_max: float,
    threshold: float,
    mode: str,
) -> np.ndarray:
    """
    Return a boolean mask of cells that need refinement.
    """
    if mode == "gradient":
        # Central-difference gradient magnitude (interior cells only)
        gx = np.zeros_like(field)
        gy = np.zeros_like(field)
        gz = np.zeros_like(field)
        gx[1:-1, :, :] = (field[2:, :, :] - field[:-2, :, :]) / (2 * spacing[0])
        gy[:, 1:-1, :] = (field[:, 2:, :] - field[:, :-2, :]) / (2 * spacing[1])
        gz[:, :, 1:-1] = (field[:, :, 2:] - field[:, :, :-2]) / (2 * spacing[2])
        grad_mag = np.sqrt(gx**2 + gy**2 + gz**2)
        # Normalised criterion: |grad Phi| * dx / |Phi_max|
        criterion = grad_mag * spacing[0] / field_max
    elif mode == "curvature":
        # Laplacian (second derivative)
        lap = np.zeros_like(field)
        lap[1:-1, :, :] += (field[2:, :, :] - 2*field[1:-1, :, :] + field[:-2, :, :]) / spacing[0]**2
        lap[:, 1:-1, :] += (field[:, 2:, :] - 2*field[:, 1:-1, :] + field[:, :-2, :]) / spacing[1]**2
        lap[:, :, 1:-1] += (field[:, :, 2:] - 2*field[:, :, 1:-1] + field[:, :, :-2]) / spacing[2]**2
        criterion = np.abs(lap) * spacing[0]**2 / field_max
    else:
        raise ValueError(f"Unknown refine_mode: {mode!r}")

    return criterion > threshold


def _flagged_bbox(
    flagged: np.ndarray,
    origin: np.ndarray,
    spacing: np.ndarray,
    buffer_cells: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the bounding box (physical coordinates) of flagged cells,
    padded by *buffer_cells* in each direction.
    """
    idx = np.argwhere(flagged)
    lo_idx = idx.min(axis=0) - buffer_cells
    hi_idx = idx.max(axis=0) + buffer_cells + 1  # +1 for upper edge

    lo_idx = np.maximum(lo_idx, 0)
    hi_idx = np.minimum(hi_idx, np.array(flagged.shape))

    bbox_lo = origin + lo_idx * spacing
    bbox_hi = origin + hi_idx * spacing
    return bbox_lo, bbox_hi


def _populate_patch(
    patch: AMRPatch,
    field_name: str,
    potential_fn: Callable,
    verbose: bool = False,
):
    """
    Evaluate *potential_fn* on the patch's refined grid to fill
    the field data.
    """
    nx, ny, nz = patch.shape
    ox, oy, oz = patch.origin
    dx, dy, dz = patch.spacing

    # Build coordinate arrays for the patch
    x1d = ox + (np.arange(nx) + 0.0) * dx  # cell-vertex convention
    y1d = oy + (np.arange(ny) + 0.0) * dy
    z1d = oz + (np.arange(nz) + 0.0) * dz

    # Evaluate in slices to keep memory bounded
    data = np.empty((nx, ny, nz), dtype=np.float64)
    Y2d, Z2d = np.meshgrid(y1d, z1d, indexing="ij")

    for ix in range(nx):
        X2d = np.full_like(Y2d, x1d[ix])
        data[ix, :, :] = potential_fn(X2d, Y2d, Z2d)

    patch._field_data = data
