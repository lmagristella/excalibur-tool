#!/usr/bin/env python3
"""Post-processing for equivalent-mass lens-shape comparison runs.

Reads ``lensing_mass_shape_*_results.npz`` and produces:
    1. background-subtracted kappa/gamma profiles for the selected shapes
    2. Delta-kappa and Delta-gamma radial profiles relative to the reference
    3. Delta-kappa and Delta-gamma 2D maps for each non-reference shape
    4. Raw kappa and |gamma| 2D maps with critical-curve overlays
    5. Relative-percent kappa and |gamma| profiles versus the reference
    6. Common-screen tangential/cross-shear maps in the circular polar basis
    7. Common-screen tangential/cross-shear maps in a projected-ellipse-aligned basis

Usage::

    ./.venv/bin/python _postprocessing/analyze_equivalent_mass_profiles.py
    ./.venv/bin/python _postprocessing/analyze_equivalent_mass_profiles.py path/to/results.npz
    ./.venv/bin/python _postprocessing/analyze_equivalent_mass_profiles.py path/to/results.npz --profiles los_cigar disturbed_cluster
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from _postprocessing.profile_plot_styles import equivalent_mass_profile_style
from excalibur.io.filename_utils import RunNamer, latest_run


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze equivalent-mass lens-shape comparison outputs."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Path to a lensing_mass_shape result file. Defaults to the latest one in _data/output.",
    )
    parser.add_argument(
        "--profiles",
        nargs="*",
        default=None,
        help="Subset of profile names to display. The reference profile is always included automatically.",
    )
    parser.add_argument(
        "--skip-maps",
        action="store_true",
        help="Skip the 2D Delta-kappa / Delta-gamma map figure.",
    )
    parser.add_argument(
        "--skip-raw-maps",
        action="store_true",
        help="Skip the raw 2D kappa / |gamma| maps with critical-curve overlays.",
    )
    return parser.parse_args()


def _load_string_scalar(value):
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(arr)


def _load_string_list(value):
    arr = np.asarray(value)
    if arr.shape == ():
        return [str(arr.item())]
    return [str(item) for item in arr.tolist()]


def load_data(path=None):
    if path is None:
        path = latest_run("lensing_mass_shape")
        if path is None:
            raise FileNotFoundError("No lensing_mass_shape result file found in _data/output")
    data = np.load(path, allow_pickle=True)
    return data, path


def _selected_profile_indices(profile_names, reference_index, requested_names):
    ref_idx = int(reference_index)
    if not requested_names:
        ordered = list(range(len(profile_names)))
    else:
        name_to_index = {name: idx for idx, name in enumerate(profile_names)}
        invalid = [name for name in requested_names if name not in name_to_index]
        if invalid:
            valid = ", ".join(profile_names)
            raise ValueError(f"Unknown profile name(s): {', '.join(invalid)}. Available: {valid}")
        ordered = [name_to_index[name] for name in requested_names]
        if ref_idx not in ordered:
            ordered.insert(0, ref_idx)

    deduped = []
    seen = set()
    for idx in ordered:
        if idx not in seen:
            deduped.append(idx)
            seen.add(idx)
    return deduped


def _project_component_centers(d, profile_index):
    centers_mpc = np.asarray(d["component_centers_Mpc"])[profile_index]
    valid = np.all(np.isfinite(centers_mpc), axis=1)
    if not np.any(valid):
        return np.empty(0), np.empty(0)

    centers_mpc = centers_mpc[valid]
    lens_center_mpc = np.asarray(d["lens_center_Mpc"], dtype=float)
    screen_e1 = np.asarray(d["screen_e1"], dtype=float)
    screen_e2 = np.asarray(d["screen_e2"], dtype=float)

    rel = centers_mpc - lens_center_mpc[None, :]
    b1 = rel @ screen_e1
    b2 = rel @ screen_e2
    return b1, b2


def _map_coordinates_1d(d, n_map_1d):
    if "b1_map_Mpc" in d and "b2_map_Mpc" in d:
        b1 = np.unique(np.asarray(d["b1_map_Mpc"], dtype=float))
        b2 = np.unique(np.asarray(d["b2_map_Mpc"], dtype=float))
        if b1.size == n_map_1d and b2.size == n_map_1d:
            return b1, b2

    half = float(d["map_half_Mpc"])
    coords = np.linspace(-half, half, n_map_1d)
    return coords, coords


def _overlay_reference_geometry(ax, comp_b1, comp_b2, theta, R200, rs):
    ax.plot(R200 * np.cos(theta), R200 * np.sin(theta), "w--", lw=0.8, alpha=0.9)
    ax.plot(rs * np.cos(theta), rs * np.sin(theta), color="cyan", ls=":", lw=0.8, alpha=0.9)
    ax.scatter([0.0], [0.0], marker="+", s=80, color="white", linewidths=1.0)
    if comp_b1.size > 0:
        ax.scatter(comp_b1, comp_b2, marker="x", s=40, color="white", linewidths=1.0)


def _overlay_critical_curves(ax, b1_coords, b2_coords, kappa_panel, gamma_panel):
    tangential = 1.0 - kappa_panel - gamma_panel
    radial = 1.0 - kappa_panel + gamma_panel

    if np.nanmin(tangential) <= 0.0 <= np.nanmax(tangential):
        ax.contour(
            b1_coords,
            b2_coords,
            tangential.T,
            levels=[0.0],
            colors=["white"],
            linewidths=1.2,
            linestyles="-",
        )
    if np.nanmin(radial) <= 0.0 <= np.nanmax(radial):
        ax.contour(
            b1_coords,
            b2_coords,
            radial.T,
            levels=[0.0],
            colors=["cyan"],
            linewidths=1.2,
            linestyles="--",
        )


def _safe_relative_percent(values, reference, min_fraction=1e-6):
    values = np.asarray(values, dtype=float)
    reference = np.asarray(reference, dtype=float)
    scale = max(np.nanmax(np.abs(reference)), 1.0)
    floor = max(scale * min_fraction, 1e-12)

    relative = np.full_like(values, np.nan, dtype=float)
    mask = np.abs(reference) > floor
    relative[..., mask] = 100.0 * (values[..., mask] - reference[mask]) / reference[mask]
    return relative, floor


def _finite_abs_max(values, floor=1e-12):
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return float(floor)
    return max(float(np.nanmax(np.abs(arr[finite]))), float(floor))


def _finite_positive_min(values, fallback):
    arr = np.asarray(values, dtype=float)
    finite_positive = arr[np.isfinite(arr) & (arr > 0.0)]
    if finite_positive.size == 0:
        return float(fallback)
    return max(float(np.nanmin(finite_positive)), float(fallback))


def _finite_panel_or_zeros(panel):
    arr = np.asarray(panel, dtype=float)
    if np.any(np.isfinite(arr)):
        return arr
    return np.zeros_like(arr, dtype=float)


def _shear_components_from_d_flat(d_flat):
    arr = np.asarray(d_flat, dtype=float)
    if arr.shape[-1] != 4:
        raise ValueError("Expected last D_flat axis to have length 4")

    gamma1 = 0.5 * (arr[..., 0] - arr[..., 3])
    gamma2 = 0.5 * (arr[..., 1] + arr[..., 2])
    omega = 0.5 * (arr[..., 2] - arr[..., 1])
    invalid = ~np.all(np.isfinite(arr), axis=-1)
    gamma1 = np.where(invalid, np.nan, gamma1)
    gamma2 = np.where(invalid, np.nan, gamma2)
    omega = np.where(invalid, np.nan, omega)
    return gamma1, gamma2, omega


def _load_shear_component_maps(d, n_map_1d):
    if "D_flat_map_by_shape" not in d:
        return None, None, None

    d_flat_map = np.asarray(d["D_flat_map_by_shape"], dtype=float).reshape(-1, n_map_1d, n_map_1d, 4)
    return _shear_components_from_d_flat(d_flat_map)


def _orthonormal_screen_basis_from_seeds(direction, seed1, seed2):
    direction = np.asarray(direction, dtype=float)
    seed1 = np.asarray(seed1, dtype=float)
    seed2 = np.asarray(seed2, dtype=float)

    if not np.all(np.isfinite(direction)):
        return None, None
    direction_norm = np.linalg.norm(direction)
    if direction_norm < 1e-14:
        return None, None
    n_hat = direction / direction_norm

    basis1 = seed1 - np.dot(seed1, n_hat) * n_hat
    basis1_norm = np.linalg.norm(basis1)
    if basis1_norm < 1e-14:
        basis1 = seed2 - np.dot(seed2, n_hat) * n_hat
        basis1_norm = np.linalg.norm(basis1)
        if basis1_norm < 1e-14:
            return None, None
    basis1 /= basis1_norm

    basis2 = seed2 - np.dot(seed2, n_hat) * n_hat - np.dot(seed2, basis1) * basis1
    basis2_norm = np.linalg.norm(basis2)
    if basis2_norm < 1e-14:
        basis2 = np.cross(n_hat, basis1)
        basis2_norm = np.linalg.norm(basis2)
        if basis2_norm < 1e-14:
            return None, None
    basis2 /= basis2_norm

    if np.dot(np.cross(basis1, basis2), n_hat) < 0.0:
        basis2 = -basis2

    return basis1, basis2


def _common_screen_rotation(common_basis, local_basis):
    common_b1, common_b2 = common_basis
    local_e1, local_e2 = local_basis
    return np.array(
        [
            [np.dot(common_b1, local_e1), np.dot(common_b1, local_e2)],
            [np.dot(common_b2, local_e1), np.dot(common_b2, local_e2)],
        ],
        dtype=float,
    )


def _load_common_screen_projected_shear_maps(d, n_map_1d):
    required = [
        "D_flat_map_by_shape",
        "initial_sachs_e1_map_by_shape",
        "initial_sachs_e2_map_by_shape",
        "final_sachs_e1_map_by_shape",
        "final_sachs_e2_map_by_shape",
        "final_k_mu_map_by_shape",
        "target_map_Mpc",
        "obs_pos_Mpc",
        "screen_e1",
        "screen_e2",
        "source_reached_map_by_shape",
    ]
    if any(key not in d for key in required):
        return None, None

    d_flat_map = np.asarray(d["D_flat_map_by_shape"], dtype=float)
    initial_e1_map = np.asarray(d["initial_sachs_e1_map_by_shape"], dtype=float)
    initial_e2_map = np.asarray(d["initial_sachs_e2_map_by_shape"], dtype=float)
    final_e1_map = np.asarray(d["final_sachs_e1_map_by_shape"], dtype=float)
    final_e2_map = np.asarray(d["final_sachs_e2_map_by_shape"], dtype=float)
    final_k_mu_map = np.asarray(d["final_k_mu_map_by_shape"], dtype=float)
    source_reached = np.asarray(d["source_reached_map_by_shape"], dtype=bool)

    n_profiles = d_flat_map.shape[0]
    n_pixels = n_map_1d * n_map_1d

    d_flat_map = d_flat_map.reshape(n_profiles, n_pixels, 4)
    initial_e1_map = initial_e1_map.reshape(n_profiles, n_pixels, 4)
    initial_e2_map = initial_e2_map.reshape(n_profiles, n_pixels, 4)
    final_e1_map = final_e1_map.reshape(n_profiles, n_pixels, 4)
    final_e2_map = final_e2_map.reshape(n_profiles, n_pixels, 4)
    final_k_mu_map = final_k_mu_map.reshape(n_profiles, n_pixels, 4)
    source_reached = source_reached.reshape(n_profiles, n_pixels)

    targets = np.asarray(d["target_map_Mpc"], dtype=float).reshape(n_pixels, 3)
    obs_pos = np.asarray(d["obs_pos_Mpc"], dtype=float)
    screen_e1 = np.asarray(d["screen_e1"], dtype=float)
    screen_e2 = np.asarray(d["screen_e2"], dtype=float)

    gamma1 = np.full((n_profiles, n_pixels), np.nan, dtype=float)
    gamma2 = np.full((n_profiles, n_pixels), np.nan, dtype=float)
    omega = np.full((n_profiles, n_pixels), np.nan, dtype=float)

    observer_common_bases = []
    for pixel_index in range(n_pixels):
        observer_direction = targets[pixel_index] - obs_pos
        observer_common_bases.append(
            _orthonormal_screen_basis_from_seeds(observer_direction, screen_e1, screen_e2)
        )

    for profile_index in range(n_profiles):
        for pixel_index in range(n_pixels):
            if not source_reached[profile_index, pixel_index]:
                continue

            d_flat = d_flat_map[profile_index, pixel_index]
            if not np.all(np.isfinite(d_flat)):
                continue

            observer_common_basis = observer_common_bases[pixel_index]
            if observer_common_basis[0] is None:
                continue

            observer_local_basis = _orthonormal_screen_basis_from_seeds(
                targets[pixel_index] - obs_pos,
                initial_e1_map[profile_index, pixel_index, 1:4],
                initial_e2_map[profile_index, pixel_index, 1:4],
            )
            source_common_basis = _orthonormal_screen_basis_from_seeds(
                final_k_mu_map[profile_index, pixel_index, 1:4],
                screen_e1,
                screen_e2,
            )
            source_local_basis = _orthonormal_screen_basis_from_seeds(
                final_k_mu_map[profile_index, pixel_index, 1:4],
                final_e1_map[profile_index, pixel_index, 1:4],
                final_e2_map[profile_index, pixel_index, 1:4],
            )
            if (
                observer_local_basis[0] is None
                or source_common_basis[0] is None
                or source_local_basis[0] is None
            ):
                continue

            rotation_obs = _common_screen_rotation(observer_common_basis, observer_local_basis)
            rotation_src = _common_screen_rotation(source_common_basis, source_local_basis)
            d_matrix = np.asarray(
                [[d_flat[0], d_flat[1]], [d_flat[2], d_flat[3]]],
                dtype=float,
            )
            d_common = rotation_src @ d_matrix @ rotation_obs.T
            gamma1[profile_index, pixel_index] = 0.5 * (d_common[0, 0] - d_common[1, 1])
            gamma2[profile_index, pixel_index] = 0.5 * (d_common[0, 1] + d_common[1, 0])
            omega[profile_index, pixel_index] = 0.5 * (d_common[1, 0] - d_common[0, 1])

    return (
        gamma1.reshape(n_profiles, n_map_1d, n_map_1d),
        gamma2.reshape(n_profiles, n_map_1d, n_map_1d),
        omega.reshape(n_profiles, n_map_1d, n_map_1d),
    )


def _map_coordinate_grids(d, n_map_1d):
    if "b1_map_Mpc" in d and "b2_map_Mpc" in d:
        try:
            b1_grid = np.asarray(d["b1_map_Mpc"], dtype=float).reshape(n_map_1d, n_map_1d)
            b2_grid = np.asarray(d["b2_map_Mpc"], dtype=float).reshape(n_map_1d, n_map_1d)
            return b1_grid, b2_grid
        except ValueError:
            pass

    b1_coords, b2_coords = _map_coordinates_1d(d, n_map_1d)
    return np.meshgrid(b1_coords, b2_coords, indexing="ij")


def _shear_tangential_cross_from_components(gamma1_map, gamma2_map, polar_angle_rad):
    """Convert common-screen gamma1/gamma2 into tangential/cross shear.

    This script works directly with the Jacobi-map convention used in
    ``_shear_components_from_d_flat``. In that convention the spherical
    tangential branch is positive for

        gamma_t = gamma1 cos(2 phi) + gamma2 sin(2 phi)
        gamma_x = -gamma1 sin(2 phi) + gamma2 cos(2 phi)
    """
    gamma1_map = np.asarray(gamma1_map, dtype=float)
    gamma2_map = np.asarray(gamma2_map, dtype=float)
    polar_angle_rad = np.asarray(polar_angle_rad, dtype=float)

    cos2 = np.cos(2.0 * polar_angle_rad)
    sin2 = np.sin(2.0 * polar_angle_rad)
    gamma_t = gamma1_map * cos2 + gamma2_map * sin2
    gamma_x = -gamma1_map * sin2 + gamma2_map * cos2

    invalid = (~np.isfinite(gamma1_map)) | (~np.isfinite(gamma2_map)) | (~np.isfinite(polar_angle_rad))
    gamma_t = np.where(invalid, np.nan, gamma_t)
    gamma_x = np.where(invalid, np.nan, gamma_x)
    return gamma_t, gamma_x


def _estimate_projected_shape_bases(d, n_map_1d):
    if "kappa_map_by_shape" not in d:
        return None

    kappa_map = np.asarray(d["kappa_map_by_shape"], dtype=float).reshape(-1, n_map_1d, n_map_1d)
    b1_grid, b2_grid = _map_coordinate_grids(d, n_map_1d)
    radius = np.hypot(b1_grid, b2_grid)
    finite_radius = radius[np.isfinite(radius)]
    if finite_radius.size == 0:
        return None

    outer_mask_base = radius >= 0.85 * float(np.nanmax(finite_radius))
    basis_info = []
    for panel in kappa_map:
        finite = np.isfinite(panel)
        if not np.any(finite):
            basis_info.append(
                {
                    "theta_major_rad": 0.0,
                    "theta_major_deg": 0.0,
                    "axis_ratio": 1.0,
                    "moment_ratio": 1.0,
                    "is_nearly_circular": True,
                    "tangential_sign": 1.0,
                }
            )
            continue

        outer_mask = outer_mask_base & finite
        if np.any(outer_mask):
            background = float(np.nanmedian(panel[outer_mask]))
        else:
            background = float(np.nanmedian(panel[finite]))

        weights = np.where(finite, np.clip(panel - background, 0.0, None), 0.0)
        weight_sum = float(np.sum(weights))
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            basis_info.append(
                {
                    "theta_major_rad": 0.0,
                    "theta_major_deg": 0.0,
                    "axis_ratio": 1.0,
                    "moment_ratio": 1.0,
                    "is_nearly_circular": True,
                    "tangential_sign": 1.0,
                }
            )
            continue

        i_xx = float(np.sum(weights * b1_grid * b1_grid))
        i_yy = float(np.sum(weights * b2_grid * b2_grid))
        i_xy = float(np.sum(weights * b1_grid * b2_grid))
        inertia = np.array([[i_xx, i_xy], [i_xy, i_yy]], dtype=float)
        eigvals, eigvecs = np.linalg.eigh(inertia)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        lambda_major = float(max(eigvals[0], 0.0))
        lambda_minor = float(max(eigvals[1], 0.0))
        if lambda_major <= 0.0:
            basis_info.append(
                {
                    "theta_major_rad": 0.0,
                    "theta_major_deg": 0.0,
                    "axis_ratio": 1.0,
                    "moment_ratio": 1.0,
                    "is_nearly_circular": True,
                    "tangential_sign": 1.0,
                }
            )
            continue

        moment_ratio = lambda_major / max(lambda_minor, 1e-30)
        projected_q = np.sqrt(lambda_minor / lambda_major) if lambda_minor > 0.0 else 0.0
        projected_q = float(np.clip(projected_q, 0.0, 1.0))

        if moment_ratio < 1.03:
            theta_major_rad = 0.0
            theta_major_deg = 0.0
            projected_q = 1.0
            is_nearly_circular = True
        else:
            major_axis = eigvecs[:, 0]
            theta_major_rad = float(np.arctan2(major_axis[1], major_axis[0]))
            theta_major_rad = ((theta_major_rad + 0.5 * np.pi) % np.pi) - 0.5 * np.pi
            theta_major_deg = float(np.rad2deg(theta_major_rad))
            is_nearly_circular = False

        basis_info.append(
            {
                "theta_major_rad": theta_major_rad,
                "theta_major_deg": theta_major_deg,
                "axis_ratio": projected_q,
                "moment_ratio": float(moment_ratio),
                "is_nearly_circular": is_nearly_circular,
                "tangential_sign": 1.0,
            }
        )

    return basis_info


def _ellipse_aligned_polar_angle(b1_grid, b2_grid, basis_info):
    theta = float(basis_info["theta_major_rad"])
    axis_ratio = max(float(basis_info["axis_ratio"]), 1e-6)

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    b_major = b1_grid * cos_t + b2_grid * sin_t
    b_minor = -b1_grid * sin_t + b2_grid * cos_t
    return np.arctan2(b_minor / axis_ratio, b_major)


def _projected_basis_label(basis_info):
    if basis_info is None:
        return ""
    if basis_info["is_nearly_circular"]:
        return "q_proj≈1.000 (circular fallback)"
    return (
        f"phi_major={basis_info['theta_major_deg']:+.1f} deg, "
        f"q_proj={basis_info['axis_ratio']:.3f}, "
        f"t_sign={basis_info['tangential_sign']:+.0f}"
    )


def _projected_ellipse_coordinates(b1_grid, b2_grid, basis_info):
    theta = float(basis_info["theta_major_rad"])
    axis_ratio = max(float(basis_info["axis_ratio"]), 1e-6)

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    b_major = b1_grid * cos_t + b2_grid * sin_t
    b_minor = -b1_grid * sin_t + b2_grid * cos_t
    semi_major = np.hypot(b_major, b_minor / axis_ratio)
    ellipse_angle = np.arctan2(b_minor / axis_ratio, b_major)
    return b_major, b_minor, semi_major, ellipse_angle


def _bilinear_interpolate_regular_grid(panel, x_coords, y_coords, x_query, y_query):
    panel = np.asarray(panel, dtype=float)
    x_coords = np.asarray(x_coords, dtype=float)
    y_coords = np.asarray(y_coords, dtype=float)
    x_query = np.asarray(x_query, dtype=float)
    y_query = np.asarray(y_query, dtype=float)

    result = np.full(x_query.shape, np.nan, dtype=float)
    if panel.ndim != 2 or x_coords.ndim != 1 or y_coords.ndim != 1:
        return result
    if panel.shape != (x_coords.size, y_coords.size):
        return result
    if x_coords.size < 2 or y_coords.size < 2:
        return result

    valid = (
        np.isfinite(x_query)
        & np.isfinite(y_query)
        & (x_query >= x_coords[0])
        & (x_query <= x_coords[-1])
        & (y_query >= y_coords[0])
        & (y_query <= y_coords[-1])
    )
    if not np.any(valid):
        return result

    xv = x_query[valid]
    yv = y_query[valid]
    ix1 = np.searchsorted(x_coords, xv, side="right")
    iy1 = np.searchsorted(y_coords, yv, side="right")
    ix1 = np.clip(ix1, 1, x_coords.size - 1)
    iy1 = np.clip(iy1, 1, y_coords.size - 1)
    ix0 = ix1 - 1
    iy0 = iy1 - 1

    x0 = x_coords[ix0]
    x1 = x_coords[ix1]
    y0 = y_coords[iy0]
    y1 = y_coords[iy1]
    with np.errstate(invalid="ignore", divide="ignore"):
        fx = (xv - x0) / np.maximum(x1 - x0, 1e-30)
        fy = (yv - y0) / np.maximum(y1 - y0, 1e-30)

    v00 = panel[ix0, iy0]
    v10 = panel[ix1, iy0]
    v01 = panel[ix0, iy1]
    v11 = panel[ix1, iy1]
    result_valid = (
        (1.0 - fx) * (1.0 - fy) * v00
        + fx * (1.0 - fy) * v10
        + (1.0 - fx) * fy * v01
        + fx * fy * v11
    )
    finite_valid = np.isfinite(v00) & np.isfinite(v10) & np.isfinite(v01) & np.isfinite(v11)
    result_subset = result[valid]
    result_subset[finite_valid] = result_valid[finite_valid]
    result[valid] = result_subset
    return result


def _sample_common_screen_shear_on_projected_ellipses(d, semi_major_values_mpc, n_theta=181):
    n_map_1d = int(d["n_map_1d"])
    gamma1_map, gamma2_map, _ = _load_common_screen_projected_shear_maps(d, n_map_1d)
    if gamma1_map is None or gamma2_map is None:
        return None

    gamma_abs_map = np.hypot(gamma1_map, gamma2_map)
    basis_info = _estimate_projected_shape_bases(d, n_map_1d)
    if basis_info is None:
        return None

    b1_coords, b2_coords = _map_coordinates_1d(d, n_map_1d)
    theta_quarter = np.linspace(0.0, 0.5 * np.pi, n_theta)
    theta_full = np.linspace(0.0, 2.0 * np.pi, 4 * n_theta, endpoint=False)

    profiles = []
    for profile_index in range(gamma_abs_map.shape[0]):
        basis = basis_info[profile_index]
        theta_major = float(basis["theta_major_rad"])
        q_proj = max(float(basis["axis_ratio"]), 1e-6)
        cos_t = np.cos(theta_major)
        sin_t = np.sin(theta_major)

        profile_curves = []
        for semi_major in semi_major_values_mpc:
            a_mpc = float(semi_major)
            if not np.isfinite(a_mpc) or a_mpc <= 0.0:
                continue

            x_major_quarter = a_mpc * np.cos(theta_quarter)
            y_minor_quarter = q_proj * a_mpc * np.sin(theta_quarter)
            b1_quarter = x_major_quarter * cos_t - y_minor_quarter * sin_t
            b2_quarter = x_major_quarter * sin_t + y_minor_quarter * cos_t
            gamma_quarter = _bilinear_interpolate_regular_grid(
                gamma_abs_map[profile_index],
                b1_coords,
                b2_coords,
                b1_quarter,
                b2_quarter,
            )

            x_major_full = a_mpc * np.cos(theta_full)
            y_minor_full = q_proj * a_mpc * np.sin(theta_full)
            b1_full = x_major_full * cos_t - y_minor_full * sin_t
            b2_full = x_major_full * sin_t + y_minor_full * cos_t
            gamma_full = _bilinear_interpolate_regular_grid(
                gamma_abs_map[profile_index],
                b1_coords,
                b2_coords,
                b1_full,
                b2_full,
            )

            harmonic = _fit_spin2_angular_harmonics(gamma_full, theta_full)
            if not np.any(np.isfinite(gamma_quarter)):
                continue

            finite_quarter = np.isfinite(gamma_quarter)
            gamma_major = float(gamma_quarter[finite_quarter][0]) if np.any(finite_quarter) else np.nan
            gamma_minor = float(gamma_quarter[finite_quarter][-1]) if np.any(finite_quarter) else np.nan
            profile_curves.append(
                {
                    "a_mpc": a_mpc,
                    "theta_quarter_rad": theta_quarter,
                    "gamma_quarter": gamma_quarter,
                    "gamma_major": gamma_major,
                    "gamma_minor": gamma_minor,
                    "gamma_min": float(np.nanmin(gamma_quarter)),
                    "gamma_max": float(np.nanmax(gamma_quarter)),
                    "harmonic": harmonic,
                }
            )

        profiles.append(
            {
                "basis": basis,
                "curves": profile_curves,
            }
        )
    return profiles


def _common_screen_projected_principal_frame_maps(d, n_map_1d):
    gamma1_map, gamma2_map, _omega_map = _load_common_screen_projected_shear_maps(d, n_map_1d)
    if gamma1_map is None or gamma2_map is None:
        return None

    b1_grid, b2_grid = _map_coordinate_grids(d, n_map_1d)
    basis_info = _estimate_projected_shape_bases(d, n_map_1d)
    if basis_info is None:
        return None

    n_profiles = gamma1_map.shape[0]
    x_rot = np.full_like(gamma1_map, np.nan, dtype=float)
    y_rot = np.full_like(gamma1_map, np.nan, dtype=float)
    gamma1_rot = np.full_like(gamma1_map, np.nan, dtype=float)
    gamma2_rot = np.full_like(gamma1_map, np.nan, dtype=float)
    gamma_abs = np.full_like(gamma1_map, np.nan, dtype=float)
    phase_map = np.full_like(gamma1_map, np.nan, dtype=float)

    for profile_index in range(n_profiles):
        basis = basis_info[profile_index]
        theta = float(basis["theta_major_rad"])
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        cos_2t = np.cos(2.0 * theta)
        sin_2t = np.sin(2.0 * theta)

        x_rot[profile_index] = cos_t * b1_grid + sin_t * b2_grid
        y_rot[profile_index] = -sin_t * b1_grid + cos_t * b2_grid
        gamma1_rot[profile_index] = gamma1_map[profile_index] * cos_2t + gamma2_map[profile_index] * sin_2t
        gamma2_rot[profile_index] = -gamma1_map[profile_index] * sin_2t + gamma2_map[profile_index] * cos_2t
        gamma_abs[profile_index] = np.hypot(gamma1_rot[profile_index], gamma2_rot[profile_index])
        valid = np.isfinite(gamma1_rot[profile_index]) & np.isfinite(gamma2_rot[profile_index])
        phase_map[profile_index, valid] = 0.5 * np.arctan2(
            gamma2_rot[profile_index, valid],
            gamma1_rot[profile_index, valid],
        )

    return {
        "basis": basis_info,
        "x_rot": x_rot,
        "y_rot": y_rot,
        "gamma1_rot": gamma1_rot,
        "gamma2_rot": gamma2_rot,
        "gamma_abs": gamma_abs,
        "phase": phase_map,
    }


def _load_common_screen_tangential_cross_maps(d, n_map_1d, *, ellipse_aligned=False):
    gamma1_map, gamma2_map, _omega_map = _load_common_screen_projected_shear_maps(d, n_map_1d)
    if gamma1_map is None or gamma2_map is None:
        return None, None, None

    b1_grid, b2_grid = _map_coordinate_grids(d, n_map_1d)
    if not ellipse_aligned:
        polar_angle = np.arctan2(b2_grid, b1_grid)
        gamma_t_map, gamma_x_map = _shear_tangential_cross_from_components(
            gamma1_map,
            gamma2_map,
            polar_angle[None, :, :],
        )
        return gamma_t_map, gamma_x_map, None

    basis_info = _estimate_projected_shape_bases(d, n_map_1d)
    if basis_info is None:
        return None, None, None

    gamma_t_map = np.full_like(gamma1_map, np.nan, dtype=float)
    gamma_x_map = np.full_like(gamma2_map, np.nan, dtype=float)
    radius = np.hypot(b1_grid, b2_grid)
    finite_radius = radius[np.isfinite(radius)]
    if finite_radius.size == 0:
        return None, None, None
    annulus_mask = (radius > 0.15 * float(np.nanmax(finite_radius))) & (radius < 0.55 * float(np.nanmax(finite_radius)))

    for profile_index, info in enumerate(basis_info):
        polar_angle = _ellipse_aligned_polar_angle(b1_grid, b2_grid, info)
        gamma_t_map[profile_index], gamma_x_map[profile_index] = _shear_tangential_cross_from_components(
            gamma1_map[profile_index],
            gamma2_map[profile_index],
            polar_angle,
        )
        valid_annulus = annulus_mask & np.isfinite(gamma_t_map[profile_index])
        if np.any(valid_annulus):
            median_t = float(np.nanmedian(gamma_t_map[profile_index][valid_annulus]))
            if median_t < 0.0:
                gamma_t_map[profile_index] *= -1.0
                gamma_x_map[profile_index] *= -1.0
                info["tangential_sign"] = -1.0

    return gamma_t_map, gamma_x_map, basis_info


def _plot_common_screen_tangential_cross_maps(d, namer, selected_indices, *, ellipse_aligned, delta):
    profile_names = _load_string_list(d["profile_names"])
    reference_index = int(d["reference_profile_index"])
    reference_name = _load_string_scalar(d["reference_profile_name"])

    plot_indices = selected_indices if not delta else [idx for idx in selected_indices if idx != reference_index]
    if not plot_indices:
        kind = "Delta" if delta else "raw"
        print(f"   [skip] no profiles selected for {kind} tangential/cross-shear plotting")
        return

    n_map_1d = int(d["n_map_1d"])
    gamma_t_map, gamma_x_map, basis_info = _load_common_screen_tangential_cross_maps(
        d,
        n_map_1d,
        ellipse_aligned=ellipse_aligned,
    )
    if gamma_t_map is None or gamma_x_map is None:
        print("   [skip] common-screen shear projection inputs missing in NPZ")
        return

    if ellipse_aligned:
        print(
            "   [note] ellipse-aligned gamma_t/gamma_x use a circularized polar angle "
            "estimated from each profile's projected kappa quadrupole."
        )
    else:
        print(
            "   [note] common-screen gamma_t/gamma_x use the circular polar angle about the lens centre; "
            "with this D-map convention the spherical tangential branch is positive."
        )

    half = float(d["map_half_Mpc"])
    extent = [-half, half, -half, half]
    b1_coords, b2_coords = _map_coordinates_1d(d, n_map_1d)
    theta = np.linspace(0.0, 2.0 * np.pi, 200)
    R200 = float(d["R_200_Mpc"])
    rs = float(d["r_s_Mpc"])
    kappa_map = np.asarray(d["kappa_map_by_shape"], dtype=float).reshape(-1, n_map_1d, n_map_1d)
    gamma_abs_map = np.asarray(d["gamma_map_by_shape"], dtype=float).reshape(-1, n_map_1d, n_map_1d)

    if delta:
        ref_t = gamma_t_map[reference_index]
        ref_x = gamma_x_map[reference_index]
        component_vmax = None
    else:
        component_vmax = _finite_abs_max(np.stack([gamma_t_map[plot_indices], gamma_x_map[plot_indices]], axis=0))

    fig, axes = plt.subplots(len(plot_indices), 2, figsize=(13, 4.8 * len(plot_indices)), squeeze=False)
    for row, profile_index in enumerate(plot_indices):
        name = profile_names[profile_index]
        comp_b1, comp_b2 = _project_component_centers(d, profile_index)
        basis_suffix = ""
        if ellipse_aligned and basis_info is not None:
            basis_suffix = f" ({_projected_basis_label(basis_info[profile_index])})"

        if delta:
            panels = [
                (
                    gamma_t_map[profile_index] - ref_t,
                    f"Delta common-screen gamma_t: {name} - {reference_name}{basis_suffix}",
                ),
                (
                    gamma_x_map[profile_index] - ref_x,
                    f"Delta common-screen gamma_x: {name} - {reference_name}{basis_suffix}",
                ),
            ]
        else:
            panels = [
                (gamma_t_map[profile_index], f"Raw common-screen gamma_t: {name}{basis_suffix}"),
                (gamma_x_map[profile_index], f"Raw common-screen gamma_x: {name}{basis_suffix}"),
            ]

        for col, (panel, title) in enumerate(panels):
            ax = axes[row, col]
            vmax = _finite_abs_max(panel) if delta else component_vmax
            image_panel = _finite_panel_or_zeros(panel)
            im = ax.imshow(
                image_panel.T,
                origin="lower",
                extent=extent,
                cmap="RdBu_r",
                aspect="equal",
                norm=SymLogNorm(linthresh=max(vmax * 1e-3, 1e-12), vmin=-vmax, vmax=vmax),
            )
            _overlay_reference_geometry(ax, comp_b1, comp_b2, theta, R200, rs)
            if not delta:
                _overlay_critical_curves(ax, b1_coords, b2_coords, kappa_map[profile_index], gamma_abs_map[profile_index])
            ax.set_title(title)
            ax.set_xlabel("b1 [Mpc]")
            ax.set_ylabel("b2 [Mpc]")
            div = make_axes_locatable(ax)
            cbar = plt.colorbar(im, cax=div.append_axes("right", size="5%", pad=0.05))
            cbar.set_label("shear")

    basis_label = "projected-ellipse circularized basis" if ellipse_aligned else "circular polar basis"
    if delta:
        title = (
            "Equivalent-mass Delta common-screen tangential/cross-shear maps  --  "
            f"{namer.title_line()}  --  basis: {basis_label}  --  ref: {reference_name}"
        )
        output_tag = (
            "shape_delta_common_screen_ellipse_aligned_tangential_cross_maps"
            if ellipse_aligned
            else "shape_delta_common_screen_tangential_cross_maps"
        )
    else:
        title = (
            "Equivalent-mass raw common-screen tangential/cross-shear maps with critical curves  --  "
            f"{namer.title_line()}  --  basis: {basis_label}  --  tangential=white solid, radial=cyan dashed"
        )
        output_tag = (
            "shape_raw_common_screen_ellipse_aligned_tangential_cross_maps"
            if ellipse_aligned
            else "shape_raw_common_screen_tangential_cross_maps"
        )

    fig.suptitle(title, fontsize=12, y=1.01)
    fig.tight_layout()
    outfile = namer.plot(output_tag)
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"   [ok] {outfile}")
    plt.close(fig)


def plot_delta_common_screen_tangential_cross_maps(d, namer, selected_indices):
    _plot_common_screen_tangential_cross_maps(
        d,
        namer,
        selected_indices,
        ellipse_aligned=False,
        delta=True,
    )


def plot_raw_common_screen_tangential_cross_maps(d, namer, selected_indices):
    _plot_common_screen_tangential_cross_maps(
        d,
        namer,
        selected_indices,
        ellipse_aligned=False,
        delta=False,
    )


def plot_delta_common_screen_ellipse_aligned_tangential_cross_maps(d, namer, selected_indices):
    _plot_common_screen_tangential_cross_maps(
        d,
        namer,
        selected_indices,
        ellipse_aligned=True,
        delta=True,
    )


def plot_raw_common_screen_ellipse_aligned_tangential_cross_maps(d, namer, selected_indices):
    _plot_common_screen_tangential_cross_maps(
        d,
        namer,
        selected_indices,
        ellipse_aligned=True,
        delta=False,
    )


def _plot_valid_line(ax, x, y, *plot_args, **plot_kwargs):
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if np.count_nonzero(mask) == 0:
        return False
    ax.plot(x_arr[mask], y_arr[mask], *plot_args, **plot_kwargs)
    return True


def _fill_between_valid(ax, x, y1, y2, **kwargs):
    x_arr = np.asarray(x, dtype=float)
    y1_arr = np.asarray(y1, dtype=float)
    y2_arr = np.asarray(y2, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y1_arr) & np.isfinite(y2_arr)
    if np.count_nonzero(mask) == 0:
        return False
    ax.fill_between(x_arr[mask], y1_arr[mask], y2_arr[mask], **kwargs)
    return True


def _annulus_edges_from_centers(radius_centers):
    radius_centers = np.asarray(radius_centers, dtype=float)
    if radius_centers.ndim != 1:
        raise ValueError("radius_centers must be 1D")
    if radius_centers.size == 0:
        return np.empty(0, dtype=float)

    edges = np.empty(radius_centers.size + 1, dtype=float)
    edges[0] = 0.0
    if radius_centers.size == 1:
        edges[1] = max(radius_centers[0], 1e-12)
        return edges

    edges[1:-1] = 0.5 * (radius_centers[:-1] + radius_centers[1:])
    edges[-1] = radius_centers[-1] + 0.5 * (radius_centers[-1] - radius_centers[-2])
    return edges


def _annular_map_stats(map_by_shape, radius_grid, radius_centers):
    map_by_shape = np.asarray(map_by_shape, dtype=float)
    radius_grid = np.asarray(radius_grid, dtype=float)
    radius_centers = np.asarray(radius_centers, dtype=float)
    edges = _annulus_edges_from_centers(radius_centers)

    n_profiles = map_by_shape.shape[0]
    n_bins = radius_centers.size
    mean = np.full((n_profiles, n_bins), np.nan, dtype=float)
    std = np.full((n_profiles, n_bins), np.nan, dtype=float)
    counts = np.zeros((n_profiles, n_bins), dtype=np.int32)

    flat_radius = radius_grid.reshape(-1)
    flat_maps = map_by_shape.reshape(n_profiles, -1)

    for bin_index in range(n_bins):
        if bin_index == n_bins - 1:
            mask = (flat_radius >= edges[bin_index]) & (flat_radius <= edges[bin_index + 1])
        else:
            mask = (flat_radius >= edges[bin_index]) & (flat_radius < edges[bin_index + 1])
        if not np.any(mask):
            continue

        annulus_values = flat_maps[:, mask]
        finite = np.isfinite(annulus_values)
        counts[:, bin_index] = np.sum(finite, axis=1, dtype=np.int32)
        sample_sum = np.sum(np.where(finite, annulus_values, 0.0), axis=1)
        np.divide(sample_sum, counts[:, bin_index], out=mean[:, bin_index], where=counts[:, bin_index] > 0)

        centered = np.where(finite, annulus_values - mean[:, bin_index][:, None], 0.0)
        variance = np.full(n_profiles, np.nan, dtype=float)
        np.divide(
            np.sum(centered * centered, axis=1),
            counts[:, bin_index],
            out=variance,
            where=counts[:, bin_index] > 0,
        )
        std[:, bin_index] = np.sqrt(variance)

    return mean, std, counts


def _load_annular_map_profiles(d):
    n_map_1d = int(d["n_map_1d"])
    b1_coords, b2_coords = _map_coordinates_1d(d, n_map_1d)
    b1_grid, b2_grid = np.meshgrid(b1_coords, b2_coords, indexing="ij")
    radius_grid = np.sqrt(b1_grid**2 + b2_grid**2)

    max_complete_radius = float(d["map_half_Mpc"])
    if "b_profile_Mpc" in d:
        radius_centers = np.asarray(d["b_profile_Mpc"], dtype=float)
        radius_centers = radius_centers[radius_centers <= max_complete_radius + 1e-12]
    else:
        radius_centers = np.linspace(0.0, max_complete_radius, max(n_map_1d // 2, 2))
    if radius_centers.size == 0:
        return None

    kappa_map = np.asarray(d["kappa_map_by_shape"], dtype=float).reshape(-1, n_map_1d, n_map_1d)
    gamma_map = np.asarray(d["gamma_map_by_shape"], dtype=float).reshape(-1, n_map_1d, n_map_1d)
    kappa_mean, kappa_std, kappa_count = _annular_map_stats(kappa_map, radius_grid, radius_centers)
    gamma_mean, gamma_std, gamma_count = _annular_map_stats(gamma_map, radius_grid, radius_centers)

    return {
        "radius_Mpc": radius_centers,
        "kappa_mean": kappa_mean,
        "kappa_std": kappa_std,
        "kappa_count": kappa_count,
        "gamma_mean": gamma_mean,
        "gamma_std": gamma_std,
        "gamma_count": gamma_count,
        "max_complete_radius_Mpc": max_complete_radius,
    }


def plot_profile_comparison(d, namer, selected_indices):
    b_profile = np.asarray(d["b_profile_Mpc"], dtype=float)
    profile_names = _load_string_list(d["profile_names"])
    reference_index = int(d["reference_profile_index"])
    reference_name = _load_string_scalar(d["reference_profile_name"])
    component_counts = np.asarray(d["component_counts"], dtype=int)
    axis_rms = np.asarray(d["profile_axis_rms_Mpc"], dtype=float)

    kappa_profile = np.asarray(d["kappa_profile_by_shape"], dtype=float)
    gamma_profile = np.asarray(d["gamma_profile_by_shape"], dtype=float)
    kappa_profile_std = np.asarray(d["kappa_profile_std_by_shape"], dtype=float) if "kappa_profile_std_by_shape" in d else None
    gamma_profile_std = np.asarray(d["gamma_profile_std_by_shape"], dtype=float) if "gamma_profile_std_by_shape" in d else None
    profile_definition = _load_string_scalar(d["profile_definition"]) if "profile_definition" in d else "line_cut"
    profile_nphi = int(np.asarray(d["profile_nphi"]).item()) if "profile_nphi" in d else 1
    kappa_signal = kappa_profile - kappa_profile[:, -1][:, None]
    ref_signal = kappa_signal[reference_index]
    delta_kappa_signal = kappa_signal - ref_signal[None, :]
    delta_gamma = gamma_profile - gamma_profile[reference_index][None, :]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for color_idx, profile_index in enumerate(selected_indices):
        name = profile_names[profile_index]
        label = (
            f"{name}  (N={component_counts[profile_index]}, "
            f"rms=({axis_rms[profile_index, 0]:.2f}, {axis_rms[profile_index, 1]:.2f}, {axis_rms[profile_index, 2]:.2f}) Mpc)"
        )
        style = equivalent_mass_profile_style(
            name,
            is_reference=profile_index == reference_index,
            fallback_index=color_idx,
        )

        _plot_valid_line(
            axes[0, 0],
            b_profile,
            kappa_signal[profile_index],
            "-",
            marker=style["marker"],
            ms=style["ms"],
            lw=style["lw"],
            color=style["color"],
            alpha=style["alpha"],
            markeredgecolor=style["mec"],
            markeredgewidth=style["mew"],
            zorder=style["zorder"],
            label=label,
        )
        if kappa_profile_std is not None:
            bg_std = kappa_profile_std[profile_index, -1]
            kappa_signal_std = np.sqrt(kappa_profile_std[profile_index] ** 2 + bg_std ** 2)
            _fill_between_valid(
                axes[0, 0],
                b_profile,
                kappa_signal[profile_index] - kappa_signal_std,
                kappa_signal[profile_index] + kappa_signal_std,
                color=style["color"],
                alpha=0.12 if profile_index == reference_index else 0.08,
                linewidth=0,
            )

        mask_gamma = (
            (b_profile > 0.0)
            & np.isfinite(gamma_profile[profile_index])
            & (gamma_profile[profile_index] > 0.0)
        )
        axes[0, 1].loglog(
            b_profile[mask_gamma],
            gamma_profile[profile_index, mask_gamma],
            "-",
            marker=style["marker"],
            ms=style["ms"],
            lw=style["lw"],
            color=style["color"],
            alpha=style["alpha"],
            markeredgecolor=style["mec"],
            markeredgewidth=style["mew"],
            zorder=style["zorder"],
            label=label,
        )
        if gamma_profile_std is not None and np.any(mask_gamma):
            lower = np.clip(
                gamma_profile[profile_index, mask_gamma] - gamma_profile_std[profile_index, mask_gamma],
                1e-30,
                None,
            )
            upper = gamma_profile[profile_index, mask_gamma] + gamma_profile_std[profile_index, mask_gamma]
            axes[0, 1].fill_between(
                b_profile[mask_gamma],
                lower,
                upper,
                color=style["color"],
                alpha=0.12 if profile_index == reference_index else 0.08,
                linewidth=0,
            )

        if profile_index != reference_index:
            _plot_valid_line(
                axes[1, 0],
                b_profile,
                delta_kappa_signal[profile_index],
                "-",
                marker=style["marker"],
                ms=style["ms"],
                lw=1.3,
                color=style["color"],
                alpha=0.9,
                markeredgecolor=style["mec"],
                markeredgewidth=style["mew"],
                label=name,
            )
            _plot_valid_line(
                axes[1, 1],
                b_profile,
                delta_gamma[profile_index],
                "-",
                marker=style["marker"],
                ms=style["ms"],
                lw=1.3,
                color=style["color"],
                alpha=0.9,
                markeredgecolor=style["mec"],
                markeredgewidth=style["mew"],
                label=name,
            )

    if profile_definition == "geometric_center_radial_mean":
        title_suffix = f" (radial mean, Nphi={profile_nphi})"
        x_label = "Radius b [Mpc]"
    else:
        title_suffix = ""
        x_label = "Impact parameter b [Mpc]"

    axes[0, 0].set_title(f"Background-subtracted kappa profiles{title_suffix}")
    axes[0, 0].set_xlabel(x_label)
    axes[0, 0].set_ylabel("kappa - kappa_bg")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].set_title(f"Shear profiles{title_suffix}")
    axes[0, 1].set_xlabel(x_label)
    axes[0, 1].set_ylabel("|gamma|")
    axes[0, 1].grid(True, alpha=0.3, which="both")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].axhline(0.0, color="k", lw=0.8, alpha=0.5)
    axes[1, 0].set_xscale("log")
    kappa_vmax = _finite_abs_max(delta_kappa_signal[selected_indices])
    axes[1, 0].set_yscale("symlog", linthresh=max(kappa_vmax * 1e-3, 1e-12))
    axes[1, 0].set_title(f"Delta-kappa relative to {reference_name}")
    axes[1, 0].set_xlabel(x_label)
    axes[1, 0].set_ylabel("Delta (kappa - kappa_bg)")
    axes[1, 0].grid(True, alpha=0.3, which="both")
    if len(selected_indices) > 1:
        axes[1, 0].legend(fontsize=8)

    axes[1, 1].axhline(0.0, color="k", lw=0.8, alpha=0.5)
    axes[1, 1].set_xscale("log")
    gamma_vmax = _finite_abs_max(delta_gamma[selected_indices])
    axes[1, 1].set_yscale("symlog", linthresh=max(gamma_vmax * 1e-3, 1e-12))
    axes[1, 1].set_title(f"Delta-gamma relative to {reference_name}")
    axes[1, 1].set_xlabel(x_label)
    axes[1, 1].set_ylabel("Delta |gamma|")
    axes[1, 1].grid(True, alpha=0.3, which="both")
    if len(selected_indices) > 1:
        axes[1, 1].legend(fontsize=8)

    fig.suptitle(
        f"Equivalent-mass profile comparison  --  {namer.title_line()}  --  ref: {reference_name}",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    outfile = namer.plot("shape_profiles")
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"   [ok] {outfile}")
    plt.close(fig)


def plot_annular_map_profiles(d, namer, selected_indices):
    profile_names = _load_string_list(d["profile_names"])
    reference_index = int(d["reference_profile_index"])
    reference_name = _load_string_scalar(d["reference_profile_name"])
    component_counts = np.asarray(d["component_counts"], dtype=int)
    axis_rms = np.asarray(d["profile_axis_rms_Mpc"], dtype=float)

    annular = _load_annular_map_profiles(d)
    if annular is None:
        print("   [skip] could not derive annular map profiles")
        return

    radius = annular["radius_Mpc"]
    kappa_mean = annular["kappa_mean"]
    kappa_std = annular["kappa_std"]
    gamma_mean = annular["gamma_mean"]
    gamma_std = annular["gamma_std"]
    max_complete_radius = annular["max_complete_radius_Mpc"]

    kappa_signal = kappa_mean - kappa_mean[:, -1][:, None]
    kappa_signal_std = np.sqrt(kappa_std**2 + kappa_std[:, -1][:, None] ** 2)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))

    for color_idx, profile_index in enumerate(selected_indices):
        name = profile_names[profile_index]
        label = (
            f"{name}  (N={component_counts[profile_index]}, "
            f"rms=({axis_rms[profile_index, 0]:.2f}, {axis_rms[profile_index, 1]:.2f}, {axis_rms[profile_index, 2]:.2f}) Mpc)"
        )
        style = equivalent_mass_profile_style(
            name,
            is_reference=profile_index == reference_index,
            fallback_index=color_idx,
        )

        _plot_valid_line(
            axes[0],
            radius,
            kappa_signal[profile_index],
            "-",
            marker=style["marker"],
            ms=style["ms"],
            lw=style["lw"],
            color=style["color"],
            alpha=style["alpha"],
            markeredgecolor=style["mec"],
            markeredgewidth=style["mew"],
            label=label,
        )
        _fill_between_valid(
            axes[0],
            radius,
            kappa_signal[profile_index] - kappa_signal_std[profile_index],
            kappa_signal[profile_index] + kappa_signal_std[profile_index],
            color=style["color"],
            alpha=0.12 if profile_index == reference_index else 0.08,
            linewidth=0,
        )

        mask_gamma = (
            (radius > 0.0)
            & np.isfinite(gamma_mean[profile_index])
            & (gamma_mean[profile_index] > 0.0)
        )
        if np.any(mask_gamma):
            axes[1].loglog(
                radius[mask_gamma],
                gamma_mean[profile_index, mask_gamma],
                "-",
                marker=style["marker"],
                ms=style["ms"],
                lw=style["lw"],
                color=style["color"],
                alpha=style["alpha"],
                markeredgecolor=style["mec"],
                markeredgewidth=style["mew"],
                label=label,
            )
            lower = np.clip(
                gamma_mean[profile_index, mask_gamma] - gamma_std[profile_index, mask_gamma],
                1e-30,
                None,
            )
            upper = gamma_mean[profile_index, mask_gamma] + gamma_std[profile_index, mask_gamma]
            axes[1].fill_between(
                radius[mask_gamma],
                lower,
                upper,
                color=style["color"],
                alpha=0.12 if profile_index == reference_index else 0.08,
                linewidth=0,
            )

    axes[0].set_title("Annular-mean kappa profiles from 2D map")
    axes[0].set_xlabel("Radius b [Mpc]")
    axes[0].set_ylabel("kappa - kappa_bg")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_title("Annular-mean shear profiles from 2D map")
    axes[1].set_xlabel("Radius b [Mpc]")
    axes[1].set_ylabel("|gamma|")
    axes[1].grid(True, alpha=0.3, which="both")
    axes[1].legend(fontsize=8)

    fig.suptitle(
        "Equivalent-mass annular map profiles  --  "
        f"{namer.title_line()}  --  ref: {reference_name}  "
        f"(shaded = 1 sigma annular scatter, r <= {max_complete_radius:.2f} Mpc)",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    outfile = namer.plot("shape_annular_map_profiles")
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"   [ok] {outfile}")
    plt.close(fig)


def plot_relative_profiles(d, namer, selected_indices):
    profile_names = _load_string_list(d["profile_names"])
    reference_index = int(d["reference_profile_index"])
    reference_name = _load_string_scalar(d["reference_profile_name"])
    b_profile = np.asarray(d["b_profile_Mpc"], dtype=float)
    profile_definition = _load_string_scalar(d["profile_definition"]) if "profile_definition" in d else "line_cut"
    profile_nphi = int(np.asarray(d["profile_nphi"]).item()) if "profile_nphi" in d else 1

    non_reference_indices = [idx for idx in selected_indices if idx != reference_index]
    if not non_reference_indices:
        print("   [skip] no non-reference profile selected for relative-profile plotting")
        return

    kappa_profile = np.asarray(d["kappa_profile_by_shape"], dtype=float)
    gamma_profile = np.asarray(d["gamma_profile_by_shape"], dtype=float)
    kappa_rel, kappa_floor = _safe_relative_percent(kappa_profile, kappa_profile[reference_index])
    gamma_rel, gamma_floor = _safe_relative_percent(gamma_profile, gamma_profile[reference_index])

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))

    for color_idx, profile_index in enumerate(non_reference_indices):
        name = profile_names[profile_index]
        style = equivalent_mass_profile_style(name, is_reference=False, fallback_index=color_idx)

        mask_kappa = np.isfinite(kappa_rel[profile_index])
        if np.any(mask_kappa):
            axes[0].plot(
                b_profile[mask_kappa],
                kappa_rel[profile_index, mask_kappa],
                "-",
                marker=style["marker"],
                ms=style["ms"],
                lw=1.4,
                color=style["color"],
                alpha=0.9,
                markeredgecolor=style["mec"],
                markeredgewidth=style["mew"],
                label=name,
            )

        mask_gamma = np.isfinite(gamma_rel[profile_index])
        if np.any(mask_gamma):
            axes[1].plot(
                b_profile[mask_gamma],
                gamma_rel[profile_index, mask_gamma],
                "-",
                marker=style["marker"],
                ms=style["ms"],
                lw=1.4,
                color=style["color"],
                alpha=0.9,
                markeredgecolor=style["mec"],
                markeredgewidth=style["mew"],
                label=name,
            )

    if profile_definition == "geometric_center_radial_mean":
        title_suffix = f" (radial mean, Nphi={profile_nphi})"
        x_label = "Radius b [Mpc]"
    else:
        title_suffix = ""
        x_label = "Impact parameter b [Mpc]"

    for ax in axes:
        ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)
        ax.set_xlabel(x_label)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8)

    kappa_vals = np.abs(kappa_rel[np.isfinite(kappa_rel)])
    gamma_vals = np.abs(gamma_rel[np.isfinite(gamma_rel)])
    kappa_linthresh = max((np.nanmax(kappa_vals) if kappa_vals.size else 1.0) * 1e-3, 1e-6)
    gamma_linthresh = max((np.nanmax(gamma_vals) if gamma_vals.size else 1.0) * 1e-3, 1e-6)

    axes[0].set_yscale("symlog", linthresh=kappa_linthresh)
    axes[0].set_title(f"Relative kappa vs {reference_name}{title_suffix}")
    axes[0].set_ylabel("100 * (kappa / kappa_ref - 1) [%]")

    axes[1].set_yscale("symlog", linthresh=gamma_linthresh)
    axes[1].set_title(f"Relative |gamma| vs {reference_name}{title_suffix}")
    axes[1].set_ylabel("100 * (|gamma| / |gamma|_ref - 1) [%]")

    fig.suptitle(
        "Equivalent-mass relative profiles  --  "
        f"{namer.title_line()}  --  ref: {reference_name}  "
        f"(masked where ref < {kappa_floor:.1e} for kappa, {gamma_floor:.1e} for gamma)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    outfile = namer.plot("shape_relative_profiles")
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"   [ok] {outfile}")
    plt.close(fig)


def plot_delta_maps(d, namer, selected_indices):
    profile_names = _load_string_list(d["profile_names"])
    reference_index = int(d["reference_profile_index"])
    reference_name = _load_string_scalar(d["reference_profile_name"])

    non_reference_indices = [idx for idx in selected_indices if idx != reference_index]
    if not non_reference_indices:
        print("   [skip] no non-reference profile selected for Delta-map plotting")
        return

    n_map_1d = int(d["n_map_1d"])
    half = float(d["map_half_Mpc"])
    extent = [-half, half, -half, half]
    theta = np.linspace(0.0, 2.0 * np.pi, 200)
    R200 = float(d["R_200_Mpc"])
    rs = float(d["r_s_Mpc"])

    kappa_map = np.asarray(d["kappa_map_by_shape"], dtype=float).reshape(-1, n_map_1d, n_map_1d)
    gamma_map = np.asarray(d["gamma_map_by_shape"], dtype=float).reshape(-1, n_map_1d, n_map_1d)

    kappa_signal_map = kappa_map - kappa_map[:, :1, :1]
    ref_kappa_signal_map = kappa_signal_map[reference_index]
    ref_gamma_map = gamma_map[reference_index]

    nrows = len(non_reference_indices)
    fig, axes = plt.subplots(nrows, 2, figsize=(13, 4.8 * nrows), squeeze=False)

    for row, profile_index in enumerate(non_reference_indices):
        name = profile_names[profile_index]
        delta_kappa_map = kappa_signal_map[profile_index] - ref_kappa_signal_map
        delta_gamma_map = gamma_map[profile_index] - ref_gamma_map
        comp_b1, comp_b2 = _project_component_centers(d, profile_index)

        for col, (panel, title, cmap_name) in enumerate(
            [
                (delta_kappa_map, f"Delta-kappa map: {name} - {reference_name}", "RdBu_r"),
                (delta_gamma_map, f"Delta-gamma map: {name} - {reference_name}", "RdBu_r"),
            ]
        ):
            ax = axes[row, col]
            vmax = _finite_abs_max(panel)
            image_panel = _finite_panel_or_zeros(panel)
            im = ax.imshow(
                image_panel.T,
                origin="lower",
                extent=extent,
                cmap=cmap_name,
                aspect="equal",
                norm=SymLogNorm(linthresh=max(vmax * 1e-3, 1e-12), vmin=-vmax, vmax=vmax),
            )
            _overlay_reference_geometry(ax, comp_b1, comp_b2, theta, R200, rs)
            ax.set_title(title)
            ax.set_xlabel("b1 [Mpc]")
            ax.set_ylabel("b2 [Mpc]")
            div = make_axes_locatable(ax)
            plt.colorbar(im, cax=div.append_axes("right", size="5%", pad=0.05))

    fig.suptitle(
        f"Equivalent-mass Delta maps  --  {namer.title_line()}  --  ref: {reference_name}",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    outfile = namer.plot("shape_delta_maps")
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"   [ok] {outfile}")
    plt.close(fig)


def plot_raw_maps(d, namer, selected_indices):
    profile_names = _load_string_list(d["profile_names"])

    n_map_1d = int(d["n_map_1d"])
    half = float(d["map_half_Mpc"])
    extent = [-half, half, -half, half]
    b1_coords, b2_coords = _map_coordinates_1d(d, n_map_1d)
    theta = np.linspace(0.0, 2.0 * np.pi, 200)
    R200 = float(d["R_200_Mpc"])
    rs = float(d["r_s_Mpc"])

    kappa_map = np.asarray(d["kappa_map_by_shape"], dtype=float).reshape(-1, n_map_1d, n_map_1d)
    gamma_map = np.asarray(d["gamma_map_by_shape"], dtype=float).reshape(-1, n_map_1d, n_map_1d)
    if "mu_map_by_shape" in d:
        mu_map = np.asarray(d["mu_map_by_shape"], dtype=float).reshape(-1, n_map_1d, n_map_1d)
    else:
        inv_mu_map = (1.0 - kappa_map) ** 2 - gamma_map ** 2
        mu_map = np.zeros_like(inv_mu_map)
        valid = np.abs(inv_mu_map) > 1e-30
        mu_map[valid] = 1.0 / inv_mu_map[valid]
    inv_mu_map = (1.0 - kappa_map) ** 2 - gamma_map ** 2

    selected_kappa = kappa_map[selected_indices]
    selected_gamma = gamma_map[selected_indices]
    selected_mu = mu_map[selected_indices]
    selected_inv_mu = inv_mu_map[selected_indices]
    kappa_vmax = _finite_abs_max(selected_kappa)
    gamma_vmax = _finite_abs_max(selected_gamma)
    mu_vmax = _finite_abs_max(selected_mu)
    inv_mu_vmax = _finite_abs_max(selected_inv_mu)
    kappa_vmin = _finite_positive_min(selected_kappa, kappa_vmax * 1e-4)
    gamma_vmin = _finite_positive_min(selected_gamma, gamma_vmax * 1e-4)

    fig, axes = plt.subplots(len(selected_indices), 4, figsize=(24, 4.8 * len(selected_indices)), squeeze=False)

    for row, profile_index in enumerate(selected_indices):
        name = profile_names[profile_index]
        comp_b1, comp_b2 = _project_component_centers(d, profile_index)
        panels = [
            (kappa_map[profile_index], "Raw kappa map", "magma", LogNorm(vmin=kappa_vmin, vmax=kappa_vmax), "kappa", kappa_vmin),
            (gamma_map[profile_index], "Raw |gamma| map", "viridis", LogNorm(vmin=gamma_vmin, vmax=gamma_vmax), "|gamma|", gamma_vmin),
            (
                mu_map[profile_index],
                "Raw mu map",
                "RdBu_r",
                SymLogNorm(linthresh=max(mu_vmax * 1e-3, 1e-6), vmin=-mu_vmax, vmax=mu_vmax),
                "mu",
                None,
            ),
            (
                inv_mu_map[profile_index],
                "Raw 1/mu map",
                "RdBu_r",
                SymLogNorm(linthresh=max(inv_mu_vmax * 1e-3, 1e-6), vmin=-inv_mu_vmax, vmax=inv_mu_vmax),
                "1/mu",
                None,
            ),
        ]

        for col, (panel, title, cmap_name, norm, cbar_label, floor) in enumerate(panels):
            ax = axes[row, col]
            finite_panel = _finite_panel_or_zeros(panel)
            image_panel = np.clip(finite_panel, floor, None) if floor is not None else finite_panel
            im = ax.imshow(
                image_panel.T,
                origin="lower",
                extent=extent,
                cmap=cmap_name,
                aspect="equal",
                norm=norm,
            )
            _overlay_reference_geometry(ax, comp_b1, comp_b2, theta, R200, rs)
            _overlay_critical_curves(ax, b1_coords, b2_coords, kappa_map[profile_index], gamma_map[profile_index])
            ax.set_title(f"{title}: {name}")
            ax.set_xlabel("b1 [Mpc]")
            ax.set_ylabel("b2 [Mpc]")
            div = make_axes_locatable(ax)
            cax = div.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label(cbar_label)

    fig.suptitle(
        "Equivalent-mass raw maps with critical curves  --  "
        f"{namer.title_line()}  --  tangential=white solid, radial=cyan dashed",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    outfile = namer.plot("shape_raw_maps")
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"   [ok] {outfile}")
    plt.close(fig)


def plot_delta_shear_component_maps(d, namer, selected_indices):
    profile_names = _load_string_list(d["profile_names"])
    reference_index = int(d["reference_profile_index"])
    reference_name = _load_string_scalar(d["reference_profile_name"])

    non_reference_indices = [idx for idx in selected_indices if idx != reference_index]
    if not non_reference_indices:
        print("   [skip] no non-reference profile selected for Delta shear-component plotting")
        return

    n_map_1d = int(d["n_map_1d"])
    gamma1_map, gamma2_map, _omega_map = _load_shear_component_maps(d, n_map_1d)
    if gamma1_map is None or gamma2_map is None:
        print("   [skip] D_flat_map_by_shape missing; cannot derive Delta gamma1/gamma2 maps")
        return

    print(
        "   [note] gamma1/gamma2 are derived in each ray's transported local Sachs basis; "
        "they are diagnostics, not a global common-frame shear decomposition."
    )

    half = float(d["map_half_Mpc"])
    extent = [-half, half, -half, half]
    theta = np.linspace(0.0, 2.0 * np.pi, 200)
    R200 = float(d["R_200_Mpc"])
    rs = float(d["r_s_Mpc"])

    ref_gamma1_map = gamma1_map[reference_index]
    ref_gamma2_map = gamma2_map[reference_index]

    fig, axes = plt.subplots(len(non_reference_indices), 2, figsize=(13, 4.8 * len(non_reference_indices)), squeeze=False)

    for row, profile_index in enumerate(non_reference_indices):
        name = profile_names[profile_index]
        comp_b1, comp_b2 = _project_component_centers(d, profile_index)
        panels = [
            (
                gamma1_map[profile_index] - ref_gamma1_map,
                f"Delta local-basis gamma1 map: {name} - {reference_name}",
            ),
            (
                gamma2_map[profile_index] - ref_gamma2_map,
                f"Delta local-basis gamma2 map: {name} - {reference_name}",
            ),
        ]

        for col, (panel, title) in enumerate(panels):
            ax = axes[row, col]
            vmax = _finite_abs_max(panel)
            image_panel = _finite_panel_or_zeros(panel)
            im = ax.imshow(
                image_panel.T,
                origin="lower",
                extent=extent,
                cmap="RdBu_r",
                aspect="equal",
                norm=SymLogNorm(linthresh=max(vmax * 1e-3, 1e-12), vmin=-vmax, vmax=vmax),
            )
            _overlay_reference_geometry(ax, comp_b1, comp_b2, theta, R200, rs)
            ax.set_title(title)
            ax.set_xlabel("b1 [Mpc]")
            ax.set_ylabel("b2 [Mpc]")
            div = make_axes_locatable(ax)
            cbar = plt.colorbar(im, cax=div.append_axes("right", size="5%", pad=0.05))
            cbar.set_label("shear component")

    fig.suptitle(
        "Equivalent-mass Delta local-Sachs-basis shear-component maps  --  "
        f"{namer.title_line()}  --  ref: {reference_name}",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    outfile = namer.plot("shape_delta_shear_component_maps")
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"   [ok] {outfile}")
    plt.close(fig)


def plot_raw_shear_component_maps(d, namer, selected_indices):
    profile_names = _load_string_list(d["profile_names"])

    n_map_1d = int(d["n_map_1d"])
    gamma1_map, gamma2_map, _omega_map = _load_shear_component_maps(d, n_map_1d)
    if gamma1_map is None or gamma2_map is None:
        print("   [skip] D_flat_map_by_shape missing; cannot derive raw gamma1/gamma2 maps")
        return

    print(
        "   [note] raw gamma1/gamma2 are shown in each ray's transported local Sachs basis; "
        "sign patterns are not those of a common observer-frame shear map."
    )

    half = float(d["map_half_Mpc"])
    extent = [-half, half, -half, half]
    b1_coords, b2_coords = _map_coordinates_1d(d, n_map_1d)
    theta = np.linspace(0.0, 2.0 * np.pi, 200)
    R200 = float(d["R_200_Mpc"])
    rs = float(d["r_s_Mpc"])

    kappa_map = np.asarray(d["kappa_map_by_shape"], dtype=float).reshape(-1, n_map_1d, n_map_1d)
    gamma_abs_map = np.asarray(d["gamma_map_by_shape"], dtype=float).reshape(-1, n_map_1d, n_map_1d)
    component_vmax = _finite_abs_max(np.stack([gamma1_map[selected_indices], gamma2_map[selected_indices]], axis=0))

    fig, axes = plt.subplots(len(selected_indices), 2, figsize=(13, 4.8 * len(selected_indices)), squeeze=False)

    for row, profile_index in enumerate(selected_indices):
        name = profile_names[profile_index]
        comp_b1, comp_b2 = _project_component_centers(d, profile_index)
        panels = [
            (gamma1_map[profile_index], f"Raw local-basis gamma1 map: {name}"),
            (gamma2_map[profile_index], f"Raw local-basis gamma2 map: {name}"),
        ]

        for col, (panel, title) in enumerate(panels):
            ax = axes[row, col]
            image_panel = _finite_panel_or_zeros(panel)
            im = ax.imshow(
                image_panel.T,
                origin="lower",
                extent=extent,
                cmap="RdBu_r",
                aspect="equal",
                norm=SymLogNorm(
                    linthresh=max(component_vmax * 1e-3, 1e-12),
                    vmin=-component_vmax,
                    vmax=component_vmax,
                ),
            )
            _overlay_reference_geometry(ax, comp_b1, comp_b2, theta, R200, rs)
            _overlay_critical_curves(ax, b1_coords, b2_coords, kappa_map[profile_index], gamma_abs_map[profile_index])
            ax.set_title(title)
            ax.set_xlabel("b1 [Mpc]")
            ax.set_ylabel("b2 [Mpc]")
            div = make_axes_locatable(ax)
            cbar = plt.colorbar(im, cax=div.append_axes("right", size="5%", pad=0.05))
            cbar.set_label("shear component")

    fig.suptitle(
        "Equivalent-mass raw local-Sachs-basis shear-component maps with critical curves  --  "
        f"{namer.title_line()}  --  tangential=white solid, radial=cyan dashed",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    outfile = namer.plot("shape_raw_shear_component_maps")
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"   [ok] {outfile}")
    plt.close(fig)


def plot_delta_common_screen_shear_component_maps(d, namer, selected_indices):
    profile_names = _load_string_list(d["profile_names"])
    reference_index = int(d["reference_profile_index"])
    reference_name = _load_string_scalar(d["reference_profile_name"])

    non_reference_indices = [idx for idx in selected_indices if idx != reference_index]
    if not non_reference_indices:
        print("   [skip] no non-reference profile selected for common-screen Delta shear-component plotting")
        return

    n_map_1d = int(d["n_map_1d"])
    gamma1_map, gamma2_map, _omega_map = _load_common_screen_projected_shear_maps(d, n_map_1d)
    if gamma1_map is None or gamma2_map is None:
        print("   [skip] common-screen shear projection inputs missing in NPZ")
        return

    half = float(d["map_half_Mpc"])
    extent = [-half, half, -half, half]
    theta = np.linspace(0.0, 2.0 * np.pi, 200)
    R200 = float(d["R_200_Mpc"])
    rs = float(d["r_s_Mpc"])
    ref_gamma1_map = gamma1_map[reference_index]
    ref_gamma2_map = gamma2_map[reference_index]

    fig, axes = plt.subplots(len(non_reference_indices), 2, figsize=(13, 4.8 * len(non_reference_indices)), squeeze=False)

    for row, profile_index in enumerate(non_reference_indices):
        name = profile_names[profile_index]
        comp_b1, comp_b2 = _project_component_centers(d, profile_index)
        panels = [
            (gamma1_map[profile_index] - ref_gamma1_map, f"Delta common-screen gamma1 map: {name} - {reference_name}"),
            (gamma2_map[profile_index] - ref_gamma2_map, f"Delta common-screen gamma2 map: {name} - {reference_name}"),
        ]

        for col, (panel, title) in enumerate(panels):
            ax = axes[row, col]
            vmax = _finite_abs_max(panel)
            image_panel = _finite_panel_or_zeros(panel)
            im = ax.imshow(
                image_panel.T,
                origin="lower",
                extent=extent,
                cmap="RdBu_r",
                aspect="equal",
                norm=SymLogNorm(linthresh=max(vmax * 1e-3, 1e-12), vmin=-vmax, vmax=vmax),
            )
            _overlay_reference_geometry(ax, comp_b1, comp_b2, theta, R200, rs)
            ax.set_title(title)
            ax.set_xlabel("b1 [Mpc]")
            ax.set_ylabel("b2 [Mpc]")
            div = make_axes_locatable(ax)
            cbar = plt.colorbar(im, cax=div.append_axes("right", size="5%", pad=0.05))
            cbar.set_label("shear component")

    fig.suptitle(
        "Equivalent-mass Delta common-screen shear-component maps  --  "
        f"{namer.title_line()}  --  ref: {reference_name}",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    outfile = namer.plot("shape_delta_common_screen_shear_component_maps")
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"   [ok] {outfile}")
    plt.close(fig)


def plot_raw_common_screen_shear_component_maps(d, namer, selected_indices):
    profile_names = _load_string_list(d["profile_names"])

    n_map_1d = int(d["n_map_1d"])
    gamma1_map, gamma2_map, _omega_map = _load_common_screen_projected_shear_maps(d, n_map_1d)
    if gamma1_map is None or gamma2_map is None:
        print("   [skip] common-screen shear projection inputs missing in NPZ")
        return

    half = float(d["map_half_Mpc"])
    extent = [-half, half, -half, half]
    b1_coords, b2_coords = _map_coordinates_1d(d, n_map_1d)
    theta = np.linspace(0.0, 2.0 * np.pi, 200)
    R200 = float(d["R_200_Mpc"])
    rs = float(d["r_s_Mpc"])
    kappa_map = np.asarray(d["kappa_map_by_shape"], dtype=float).reshape(-1, n_map_1d, n_map_1d)
    gamma_abs_map = np.asarray(d["gamma_map_by_shape"], dtype=float).reshape(-1, n_map_1d, n_map_1d)
    component_vmax = _finite_abs_max(np.stack([gamma1_map[selected_indices], gamma2_map[selected_indices]], axis=0))

    fig, axes = plt.subplots(len(selected_indices), 2, figsize=(13, 4.8 * len(selected_indices)), squeeze=False)

    for row, profile_index in enumerate(selected_indices):
        name = profile_names[profile_index]
        comp_b1, comp_b2 = _project_component_centers(d, profile_index)
        panels = [
            (gamma1_map[profile_index], f"Raw common-screen gamma1 map: {name}"),
            (gamma2_map[profile_index], f"Raw common-screen gamma2 map: {name}"),
        ]

        for col, (panel, title) in enumerate(panels):
            ax = axes[row, col]
            image_panel = _finite_panel_or_zeros(panel)
            im = ax.imshow(
                image_panel.T,
                origin="lower",
                extent=extent,
                cmap="RdBu_r",
                aspect="equal",
                norm=SymLogNorm(
                    linthresh=max(component_vmax * 1e-3, 1e-12),
                    vmin=-component_vmax,
                    vmax=component_vmax,
                ),
            )
            _overlay_reference_geometry(ax, comp_b1, comp_b2, theta, R200, rs)
            _overlay_critical_curves(ax, b1_coords, b2_coords, kappa_map[profile_index], gamma_abs_map[profile_index])
            ax.set_title(title)
            ax.set_xlabel("b1 [Mpc]")
            ax.set_ylabel("b2 [Mpc]")
            div = make_axes_locatable(ax)
            cbar = plt.colorbar(im, cax=div.append_axes("right", size="5%", pad=0.05))
            cbar.set_label("shear component")

    fig.suptitle(
        "Equivalent-mass raw common-screen shear-component maps with critical curves  --  "
        f"{namer.title_line()}  --  tangential=white solid, radial=cyan dashed",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    outfile = namer.plot("shape_raw_common_screen_shear_component_maps")
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"   [ok] {outfile}")
    plt.close(fig)


def plot_delta_omega_maps(d, namer, selected_indices):
    profile_names = _load_string_list(d["profile_names"])
    reference_index = int(d["reference_profile_index"])
    reference_name = _load_string_scalar(d["reference_profile_name"])

    non_reference_indices = [idx for idx in selected_indices if idx != reference_index]
    if not non_reference_indices:
        print("   [skip] no non-reference profile selected for Delta omega plotting")
        return

    n_map_1d = int(d["n_map_1d"])
    _gamma1_map, _gamma2_map, omega_map = _load_shear_component_maps(d, n_map_1d)
    if omega_map is None:
        print("   [skip] D_flat_map_by_shape missing; cannot derive Delta omega maps")
        return

    print(
        "   [note] omega is derived in each ray's transported local Sachs basis; "
        "it is a local rotation diagnostic, not a global common-frame scalar."
    )

    half = float(d["map_half_Mpc"])
    extent = [-half, half, -half, half]
    theta = np.linspace(0.0, 2.0 * np.pi, 200)
    R200 = float(d["R_200_Mpc"])
    rs = float(d["r_s_Mpc"])
    ref_omega_map = omega_map[reference_index]

    fig, axes = plt.subplots(len(non_reference_indices), 1, figsize=(7.4, 4.8 * len(non_reference_indices)), squeeze=False)

    for row, profile_index in enumerate(non_reference_indices):
        name = profile_names[profile_index]
        comp_b1, comp_b2 = _project_component_centers(d, profile_index)
        panel = omega_map[profile_index] - ref_omega_map
        ax = axes[row, 0]
        vmax = _finite_abs_max(panel)
        image_panel = _finite_panel_or_zeros(panel)
        im = ax.imshow(
            image_panel.T,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            aspect="equal",
            norm=SymLogNorm(linthresh=max(vmax * 1e-3, 1e-12), vmin=-vmax, vmax=vmax),
        )
        _overlay_reference_geometry(ax, comp_b1, comp_b2, theta, R200, rs)
        ax.set_title(f"Delta local-basis omega map: {name} - {reference_name}")
        ax.set_xlabel("b1 [Mpc]")
        ax.set_ylabel("b2 [Mpc]")
        div = make_axes_locatable(ax)
        cbar = plt.colorbar(im, cax=div.append_axes("right", size="5%", pad=0.05))
        cbar.set_label("rotation component")

    fig.suptitle(
        "Equivalent-mass Delta local-Sachs-basis omega maps  --  "
        f"{namer.title_line()}  --  ref: {reference_name}",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    outfile = namer.plot("shape_delta_omega_maps")
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"   [ok] {outfile}")
    plt.close(fig)


def plot_raw_omega_maps(d, namer, selected_indices):
    profile_names = _load_string_list(d["profile_names"])

    n_map_1d = int(d["n_map_1d"])
    _gamma1_map, _gamma2_map, omega_map = _load_shear_component_maps(d, n_map_1d)
    if omega_map is None:
        print("   [skip] D_flat_map_by_shape missing; cannot derive raw omega maps")
        return

    print(
        "   [note] raw omega is shown in each ray's transported local Sachs basis; "
        "it is a local rotation diagnostic, not a common observer-frame scalar."
    )

    half = float(d["map_half_Mpc"])
    extent = [-half, half, -half, half]
    b1_coords, b2_coords = _map_coordinates_1d(d, n_map_1d)
    theta = np.linspace(0.0, 2.0 * np.pi, 200)
    R200 = float(d["R_200_Mpc"])
    rs = float(d["r_s_Mpc"])
    kappa_map = np.asarray(d["kappa_map_by_shape"], dtype=float).reshape(-1, n_map_1d, n_map_1d)
    gamma_abs_map = np.asarray(d["gamma_map_by_shape"], dtype=float).reshape(-1, n_map_1d, n_map_1d)
    omega_vmax = _finite_abs_max(omega_map[selected_indices])

    fig, axes = plt.subplots(len(selected_indices), 1, figsize=(7.4, 4.8 * len(selected_indices)), squeeze=False)

    for row, profile_index in enumerate(selected_indices):
        name = profile_names[profile_index]
        comp_b1, comp_b2 = _project_component_centers(d, profile_index)
        panel = omega_map[profile_index]
        ax = axes[row, 0]
        image_panel = _finite_panel_or_zeros(panel)
        im = ax.imshow(
            image_panel.T,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            aspect="equal",
            norm=SymLogNorm(linthresh=max(omega_vmax * 1e-3, 1e-12), vmin=-omega_vmax, vmax=omega_vmax),
        )
        _overlay_reference_geometry(ax, comp_b1, comp_b2, theta, R200, rs)
        _overlay_critical_curves(ax, b1_coords, b2_coords, kappa_map[profile_index], gamma_abs_map[profile_index])
        ax.set_title(f"Raw local-basis omega map: {name}")
        ax.set_xlabel("b1 [Mpc]")
        ax.set_ylabel("b2 [Mpc]")
        div = make_axes_locatable(ax)
        cbar = plt.colorbar(im, cax=div.append_axes("right", size="5%", pad=0.05))
        cbar.set_label("rotation component")

    fig.suptitle(
        "Equivalent-mass raw local-Sachs-basis omega maps with critical curves  --  "
        f"{namer.title_line()}  --  tangential=white solid, radial=cyan dashed",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    outfile = namer.plot("shape_raw_omega_maps")
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"   [ok] {outfile}")
    plt.close(fig)


def plot_delta_common_screen_omega_maps(d, namer, selected_indices):
    profile_names = _load_string_list(d["profile_names"])
    reference_index = int(d["reference_profile_index"])
    reference_name = _load_string_scalar(d["reference_profile_name"])

    non_reference_indices = [idx for idx in selected_indices if idx != reference_index]
    if not non_reference_indices:
        print("   [skip] no non-reference profile selected for common-screen Delta omega plotting")
        return

    n_map_1d = int(d["n_map_1d"])
    _gamma1_map, _gamma2_map, omega_map = _load_common_screen_projected_shear_maps(d, n_map_1d)
    if omega_map is None:
        print("   [skip] common-screen shear projection inputs missing in NPZ")
        return

    half = float(d["map_half_Mpc"])
    extent = [-half, half, -half, half]
    theta = np.linspace(0.0, 2.0 * np.pi, 200)
    R200 = float(d["R_200_Mpc"])
    rs = float(d["r_s_Mpc"])
    ref_omega_map = omega_map[reference_index]

    fig, axes = plt.subplots(len(non_reference_indices), 1, figsize=(7.4, 4.8 * len(non_reference_indices)), squeeze=False)

    for row, profile_index in enumerate(non_reference_indices):
        name = profile_names[profile_index]
        comp_b1, comp_b2 = _project_component_centers(d, profile_index)
        panel = omega_map[profile_index] - ref_omega_map
        ax = axes[row, 0]
        vmax = _finite_abs_max(panel)
        image_panel = _finite_panel_or_zeros(panel)
        im = ax.imshow(
            image_panel.T,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            aspect="equal",
            norm=SymLogNorm(linthresh=max(vmax * 1e-3, 1e-12), vmin=-vmax, vmax=vmax),
        )
        _overlay_reference_geometry(ax, comp_b1, comp_b2, theta, R200, rs)
        ax.set_title(f"Delta common-screen omega map: {name} - {reference_name}")
        ax.set_xlabel("b1 [Mpc]")
        ax.set_ylabel("b2 [Mpc]")
        div = make_axes_locatable(ax)
        cbar = plt.colorbar(im, cax=div.append_axes("right", size="5%", pad=0.05))
        cbar.set_label("rotation component")

    fig.suptitle(
        "Equivalent-mass Delta common-screen omega maps  --  "
        f"{namer.title_line()}  --  ref: {reference_name}",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    outfile = namer.plot("shape_delta_common_screen_omega_maps")
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"   [ok] {outfile}")
    plt.close(fig)


def plot_raw_common_screen_omega_maps(d, namer, selected_indices):
    profile_names = _load_string_list(d["profile_names"])

    n_map_1d = int(d["n_map_1d"])
    _gamma1_map, _gamma2_map, omega_map = _load_common_screen_projected_shear_maps(d, n_map_1d)
    if omega_map is None:
        print("   [skip] common-screen shear projection inputs missing in NPZ")
        return

    half = float(d["map_half_Mpc"])
    extent = [-half, half, -half, half]
    b1_coords, b2_coords = _map_coordinates_1d(d, n_map_1d)
    theta = np.linspace(0.0, 2.0 * np.pi, 200)
    R200 = float(d["R_200_Mpc"])
    rs = float(d["r_s_Mpc"])
    kappa_map = np.asarray(d["kappa_map_by_shape"], dtype=float).reshape(-1, n_map_1d, n_map_1d)
    gamma_abs_map = np.asarray(d["gamma_map_by_shape"], dtype=float).reshape(-1, n_map_1d, n_map_1d)
    omega_vmax = _finite_abs_max(omega_map[selected_indices])

    fig, axes = plt.subplots(len(selected_indices), 1, figsize=(7.4, 4.8 * len(selected_indices)), squeeze=False)

    for row, profile_index in enumerate(selected_indices):
        name = profile_names[profile_index]
        comp_b1, comp_b2 = _project_component_centers(d, profile_index)
        panel = omega_map[profile_index]
        ax = axes[row, 0]
        image_panel = _finite_panel_or_zeros(panel)
        im = ax.imshow(
            image_panel.T,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            aspect="equal",
            norm=SymLogNorm(linthresh=max(omega_vmax * 1e-3, 1e-12), vmin=-omega_vmax, vmax=omega_vmax),
        )
        _overlay_reference_geometry(ax, comp_b1, comp_b2, theta, R200, rs)
        _overlay_critical_curves(ax, b1_coords, b2_coords, kappa_map[profile_index], gamma_abs_map[profile_index])
        ax.set_title(f"Raw common-screen omega map: {name}")
        ax.set_xlabel("b1 [Mpc]")
        ax.set_ylabel("b2 [Mpc]")
        div = make_axes_locatable(ax)
        cbar = plt.colorbar(im, cax=div.append_axes("right", size="5%", pad=0.05))
        cbar.set_label("rotation component")

    fig.suptitle(
        "Equivalent-mass raw common-screen omega maps with critical curves  --  "
        f"{namer.title_line()}  --  tangential=white solid, radial=cyan dashed",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    outfile = namer.plot("shape_raw_common_screen_omega_maps")
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"   [ok] {outfile}")
    plt.close(fig)


def plot_raw_common_screen_projected_ellipse_shear_profiles(d, namer, selected_indices):
    n_map_1d = int(d["n_map_1d"])
    b1_grid, b2_grid = _map_coordinate_grids(d, n_map_1d)
    radius_grid = np.hypot(b1_grid, b2_grid)
    annuli = _diagnostic_annuli_mpc(d, radius_grid)
    semi_major_values = [0.5 * (rmin + rmax) for rmin, rmax in annuli]
    sampled_profiles = _sample_common_screen_shear_on_projected_ellipses(d, semi_major_values)
    if sampled_profiles is None:
        return

    profile_names = _load_string_list(d["profile_names"])
    fig, axes = plt.subplots(
        len(selected_indices),
        2,
        figsize=(12.6, 4.4 * len(selected_indices)),
        squeeze=False,
    )
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(semi_major_values), 1)))

    for row, profile_index in enumerate(selected_indices):
        ax_theta = axes[row, 0]
        ax_axes = axes[row, 1]
        profile = sampled_profiles[profile_index]
        basis = profile["basis"]
        curves = profile["curves"]
        if not curves:
            ax_theta.set_visible(False)
            ax_axes.set_visible(False)
            continue

        axis_a = []
        axis_major = []
        axis_minor = []

        for curve_index, curve in enumerate(curves):
            theta_deg = np.rad2deg(curve["theta_quarter_rad"])
            color = colors[min(curve_index, len(colors) - 1)]
            ax_theta.plot(theta_deg, curve["gamma_quarter"], color=color, lw=1.8, label=f"a={curve['a_mpc']:.3f} Mpc")
            ax_theta.scatter([0.0, 90.0], [curve["gamma_major"], curve["gamma_minor"]], color=color, s=18, zorder=3)
            axis_a.append(curve["a_mpc"])
            axis_major.append(curve["gamma_major"])
            axis_minor.append(curve["gamma_minor"])

        ax_theta.set_xlim(0.0, 90.0)
        ax_theta.set_xlabel("projected ellipse angle theta [deg]")
        ax_theta.set_ylabel("|gamma|")
        ax_theta.grid(alpha=0.25)
        ax_theta.legend(loc="best", fontsize=8)
        ax_theta.set_title(
            f"{profile_names[profile_index]}  --  phi_major={basis['theta_major_deg']:+.1f} deg, "
            f"q_proj={basis['axis_ratio']:.3f}"
        )

        axis_a = np.asarray(axis_a, dtype=float)
        axis_major = np.asarray(axis_major, dtype=float)
        axis_minor = np.asarray(axis_minor, dtype=float)
        ax_axes.plot(axis_a, axis_major, "o-", color="tab:red", lw=1.8, label="gamma along major axis")
        ax_axes.plot(axis_a, axis_minor, "s--", color="tab:blue", lw=1.8, label="gamma along minor axis")
        ax_axes.set_xlabel("projected semi-major axis a [Mpc]")
        ax_axes.set_ylabel("|gamma|")
        ax_axes.grid(alpha=0.25)
        ax_axes.legend(loc="best", fontsize=8)
        ax_axes.set_title("Projected-axis shear amplitudes")

    fig.suptitle(
        "Equivalent-mass raw common-screen shear amplitude along projected ellipses  --  "
        f"{namer.title_line()}  --  left: gamma(a, theta), right: axis cuts",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    outfile = namer.plot("shape_raw_common_screen_projected_ellipse_shear_profiles")
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"   [ok] {outfile}")
    plt.close(fig)


def plot_raw_common_screen_projected_principal_frame_maps(d, namer, selected_indices):
    profile_names = _load_string_list(d["profile_names"])
    n_map_1d = int(d["n_map_1d"])
    frame_maps = _common_screen_projected_principal_frame_maps(d, n_map_1d)
    if frame_maps is None:
        print("   [skip] common-screen shear projection inputs missing in NPZ")
        return

    gamma1_rot = frame_maps["gamma1_rot"]
    gamma2_rot = frame_maps["gamma2_rot"]
    gamma_abs = frame_maps["gamma_abs"]
    phase_map = frame_maps["phase"]
    x_rot = frame_maps["x_rot"]
    y_rot = frame_maps["y_rot"]
    basis_info = frame_maps["basis"]

    component_vmax = _finite_abs_max(np.stack([gamma1_rot[selected_indices], gamma2_rot[selected_indices]], axis=0))
    gamma_vmax = _finite_abs_max(gamma_abs[selected_indices])
    phase_limit = 0.5 * np.pi
    fig, axes = plt.subplots(len(selected_indices), 4, figsize=(22, 4.8 * len(selected_indices)), squeeze=False)

    for row, profile_index in enumerate(selected_indices):
        basis = basis_info[profile_index]
        comp_b1, comp_b2 = _project_component_centers(d, profile_index)
        theta = float(basis["theta_major_rad"])
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        comp_x = cos_t * comp_b1 + sin_t * comp_b2
        comp_y = -sin_t * comp_b1 + cos_t * comp_b2

        panels = [
            (
                gamma1_rot[profile_index],
                "RdBu_r",
                SymLogNorm(linthresh=max(component_vmax * 1e-3, 1e-12), vmin=-component_vmax, vmax=component_vmax),
                "gamma1 in projected frame",
                "shear component",
            ),
            (
                gamma2_rot[profile_index],
                "RdBu_r",
                SymLogNorm(linthresh=max(component_vmax * 1e-3, 1e-12), vmin=-component_vmax, vmax=component_vmax),
                "gamma2 in projected frame",
                "shear component",
            ),
            (
                gamma_abs[profile_index],
                "viridis",
                Normalize(vmin=0.0, vmax=gamma_vmax),
                "|gamma| in projected frame",
                "|gamma|",
            ),
            (
                phase_map[profile_index],
                "twilight_shifted",
                Normalize(vmin=-phase_limit, vmax=phase_limit),
                "phase in projected frame",
                "phase [rad]",
            ),
        ]

        for col, (panel, cmap, norm, title, cbar_label) in enumerate(panels):
            ax = axes[row, col]
            image_panel = _finite_panel_or_zeros(panel)
            im = ax.pcolormesh(
                x_rot[profile_index],
                y_rot[profile_index],
                image_panel,
                shading="auto",
                cmap=cmap,
                norm=norm,
            )
            ax.scatter([0.0], [0.0], marker="+", s=80, color="white", linewidths=1.0)
            if comp_x.size > 0:
                ax.scatter(comp_x, comp_y, marker="x", s=40, color="white", linewidths=1.0)
            ax.set_aspect("equal")
            ax.set_xlabel("projected major axis [Mpc]")
            ax.set_ylabel("projected minor axis [Mpc]")
            if col == 0:
                ax.set_title(
                    f"{profile_names[profile_index]}\n{title}"
                )
            else:
                ax.set_title(title)
            div = make_axes_locatable(ax)
            cbar = plt.colorbar(im, cax=div.append_axes("right", size="5%", pad=0.05))
            cbar.set_label(cbar_label)

    fig.suptitle(
        "Equivalent-mass raw common-screen shear maps in each profile's projected principal frame  --  "
        f"{namer.title_line()}",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    outfile = namer.plot("shape_raw_common_screen_projected_principal_frame_maps")
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"   [ok] {outfile}")
    plt.close(fig)


def _diagnostic_annuli_mpc(d, radius_grid):
    radius_grid = np.asarray(radius_grid, dtype=float)
    finite_radius = radius_grid[np.isfinite(radius_grid)]
    if finite_radius.size == 0:
        return []

    radius_max = float(np.nanmax(finite_radius))
    if radius_max <= 0.0:
        return []

    rs = float(d["r_s_Mpc"]) if "r_s_Mpc" in d else np.nan
    if np.isfinite(rs) and rs > 0.0:
        candidate_annuli = [(1.0 * rs, 2.0 * rs), (2.0 * rs, 3.0 * rs)]
    else:
        candidate_annuli = [(0.15 * radius_max, 0.30 * radius_max), (0.30 * radius_max, 0.45 * radius_max)]

    min_width = max(0.03 * radius_max, 1e-6)
    outer_cap = 0.90 * radius_max
    annuli = []
    for rmin, rmax in candidate_annuli:
        lo = max(float(rmin), 0.0)
        hi = min(float(rmax), outer_cap)
        if hi - lo >= min_width:
            annuli.append((lo, hi))

    if annuli:
        return annuli

    fallback_annuli = [(0.15 * radius_max, 0.35 * radius_max), (0.35 * radius_max, 0.55 * radius_max)]
    for rmin, rmax in fallback_annuli:
        lo = max(float(rmin), 0.0)
        hi = min(float(rmax), outer_cap)
        if hi - lo >= min_width:
            annuli.append((lo, hi))
    return annuli


def _fit_spin2_angular_harmonics(values, polar_angle_rad):
    values = np.asarray(values, dtype=float)
    polar_angle_rad = np.asarray(polar_angle_rad, dtype=float)
    mask = np.isfinite(values) & np.isfinite(polar_angle_rad)
    if np.count_nonzero(mask) < 8:
        return None

    phi = polar_angle_rad[mask]
    y = values[mask]
    design = np.column_stack(
        [
            np.ones_like(phi),
            np.cos(2.0 * phi),
            np.sin(2.0 * phi),
            np.cos(4.0 * phi),
            np.sin(4.0 * phi),
        ]
    )
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    return {
        "monopole": float(coeffs[0]),
        "amp_m2": float(np.hypot(coeffs[1], coeffs[2])),
        "amp_m4": float(np.hypot(coeffs[3], coeffs[4])),
    }


def _common_screen_shear_diagnostics(d):
    n_map_1d = int(d["n_map_1d"])
    gamma_t_circular, gamma_x_circular, _ = _load_common_screen_tangential_cross_maps(
        d,
        n_map_1d,
        ellipse_aligned=False,
    )
    gamma_t_ellipse, gamma_x_ellipse, basis_info = _load_common_screen_tangential_cross_maps(
        d,
        n_map_1d,
        ellipse_aligned=True,
    )
    if gamma_t_circular is None or gamma_x_circular is None:
        return None

    b1_grid, b2_grid = _map_coordinate_grids(d, n_map_1d)
    radius_grid = np.hypot(b1_grid, b2_grid)
    polar_angle = np.arctan2(b2_grid, b1_grid)
    annuli = _diagnostic_annuli_mpc(d, radius_grid)

    diagnostics = []
    for profile_index in range(gamma_t_circular.shape[0]):
        profile_info = {
            "basis": None if basis_info is None else basis_info[profile_index],
            "annuli": [],
        }
        for rmin, rmax in annuli:
            mask = (radius_grid > rmin) & (radius_grid < rmax)
            mask &= np.isfinite(gamma_t_circular[profile_index]) & np.isfinite(gamma_x_circular[profile_index])
            if gamma_t_ellipse is not None and gamma_x_ellipse is not None:
                mask &= np.isfinite(gamma_t_ellipse[profile_index]) & np.isfinite(gamma_x_ellipse[profile_index])
            if np.count_nonzero(mask) < 8:
                continue

            gt_c = gamma_t_circular[profile_index][mask]
            gx_c = gamma_x_circular[profile_index][mask]
            gt_e = gamma_t_ellipse[profile_index][mask] if gamma_t_ellipse is not None else None
            gx_e = gamma_x_ellipse[profile_index][mask] if gamma_x_ellipse is not None else None

            harmonic_gt = _fit_spin2_angular_harmonics(gt_c, polar_angle[mask])
            harmonic_gx = _fit_spin2_angular_harmonics(gx_c, polar_angle[mask])
            monopole_scale = 1e-30
            if harmonic_gt is not None:
                monopole_scale = max(abs(harmonic_gt["monopole"]), monopole_scale)

            annulus_info = {
                "rmin_mpc": float(rmin),
                "rmax_mpc": float(rmax),
                "mean_gt": float(np.nanmean(gt_c)),
                "mean_gx": float(np.nanmean(gx_c)),
                "mean_abs_gx": float(np.nanmean(np.abs(gx_c))),
                "frac_gt_pos_circular": float(np.mean(gt_c > 0.0)),
                "median_abs_ratio_circular": float(
                    np.nanmedian(np.abs(gx_c) / np.maximum(np.abs(gt_c), 1e-30))
                ),
                "median_abs_ratio_ellipse": np.nan,
                "frac_gt_pos_ellipse": np.nan,
                "gt_m2_rel": np.nan if harmonic_gt is None else harmonic_gt["amp_m2"] / monopole_scale,
                "gt_m4_rel": np.nan if harmonic_gt is None else harmonic_gt["amp_m4"] / monopole_scale,
                "gx_m2_rel": np.nan if harmonic_gx is None else harmonic_gx["amp_m2"] / monopole_scale,
                "gx_m4_rel": np.nan if harmonic_gx is None else harmonic_gx["amp_m4"] / monopole_scale,
            }
            if gt_e is not None and gx_e is not None:
                annulus_info["median_abs_ratio_ellipse"] = float(
                    np.nanmedian(np.abs(gx_e) / np.maximum(np.abs(gt_e), 1e-30))
                )
                annulus_info["frac_gt_pos_ellipse"] = float(np.mean(gt_e > 0.0))

            profile_info["annuli"].append(annulus_info)

        diagnostics.append(profile_info)
    return diagnostics


def _common_screen_projected_ellipse_gamma_diagnostics(d):
    n_map_1d = int(d["n_map_1d"])
    b1_grid, b2_grid = _map_coordinate_grids(d, n_map_1d)
    radius_grid = np.hypot(b1_grid, b2_grid)
    annuli = _diagnostic_annuli_mpc(d, radius_grid)
    semi_major_values = [0.5 * (rmin + rmax) for rmin, rmax in annuli]
    sampled_profiles = _sample_common_screen_shear_on_projected_ellipses(d, semi_major_values)
    if sampled_profiles is None:
        return None

    diagnostics = []
    for profile in sampled_profiles:
        profile_info = {
            "basis": profile["basis"],
            "ellipses": [],
        }
        for curve in profile["curves"]:
            harmonic = curve["harmonic"]
            monopole_scale = np.nan
            gamma_m2_rel = np.nan
            gamma_m4_rel = np.nan
            if harmonic is not None:
                monopole_scale = max(abs(harmonic["monopole"]), 1e-30)
                gamma_m2_rel = harmonic["amp_m2"] / monopole_scale
                gamma_m4_rel = harmonic["amp_m4"] / monopole_scale

            profile_info["ellipses"].append(
                {
                    "a_mpc": float(curve["a_mpc"]),
                    "gamma_major": float(curve["gamma_major"]),
                    "gamma_minor": float(curve["gamma_minor"]),
                    "gamma_min": float(curve["gamma_min"]),
                    "gamma_max": float(curve["gamma_max"]),
                    "gamma_m2_rel": float(gamma_m2_rel),
                    "gamma_m4_rel": float(gamma_m4_rel),
                }
            )

        diagnostics.append(profile_info)
    return diagnostics


def print_summary(d, selected_indices):
    profile_names = _load_string_list(d["profile_names"])
    reference_index = int(d["reference_profile_index"])
    reference_name = _load_string_scalar(d["reference_profile_name"])

    kappa_profile = np.asarray(d["kappa_profile_by_shape"], dtype=float)
    gamma_profile = np.asarray(d["gamma_profile_by_shape"], dtype=float)
    n_map_1d = int(d["n_map_1d"])
    kappa_map = np.asarray(d["kappa_map_by_shape"], dtype=float).reshape(-1, n_map_1d, n_map_1d)
    gamma_map = np.asarray(d["gamma_map_by_shape"], dtype=float).reshape(-1, n_map_1d, n_map_1d)

    kappa_signal = kappa_profile - kappa_profile[:, -1][:, None]
    delta_kappa_signal = kappa_signal - kappa_signal[reference_index][None, :]
    delta_gamma = gamma_profile - gamma_profile[reference_index][None, :]
    delta_kappa_map = (kappa_map - kappa_map[:, :1, :1]) - (kappa_map[reference_index] - kappa_map[reference_index, :1, :1])
    delta_gamma_map = gamma_map - gamma_map[reference_index]
    profile_definition = _load_string_scalar(d["profile_definition"]) if "profile_definition" in d else "line_cut"
    shear_diagnostics = _common_screen_shear_diagnostics(d)
    ellipse_gamma_diagnostics = _common_screen_projected_ellipse_gamma_diagnostics(d)

    print("\nSummary relative to reference profile:")
    print(f"  reference = {reference_name}")
    if profile_definition == "geometric_center_radial_mean":
        print("  profile_definition = geometric_center_radial_mean")
    for profile_index in selected_indices:
        if profile_index == reference_index:
            continue
        print(
            f"  {profile_names[profile_index]}: "
            f"max |Delta-kappa_profile| = {_finite_abs_max(delta_kappa_signal[profile_index]):.4e}, "
            f"max |Delta-gamma_profile| = {_finite_abs_max(delta_gamma[profile_index]):.4e}, "
            f"max |Delta-kappa_map| = {_finite_abs_max(delta_kappa_map[profile_index]):.4e}, "
            f"max |Delta-gamma_map| = {_finite_abs_max(delta_gamma_map[profile_index]):.4e}"
        )

    if shear_diagnostics is None:
        return

    print("\nCommon-screen tangential/cross-shear diagnostics:")
    print("  circular basis = polar angle around lens centre")
    print("  ellipse basis  = circularized polar angle from projected kappa quadrupole")
    for profile_index in selected_indices:
        profile_diag = shear_diagnostics[profile_index]
        basis = profile_diag["basis"]
        if basis is None:
            print(f"  {profile_names[profile_index]}: diagnostics unavailable")
            continue

        print(
            f"  {profile_names[profile_index]}: "
            f"phi_major={basis['theta_major_deg']:+.2f} deg, "
            f"q_proj={basis['axis_ratio']:.3f}, "
            f"t_sign={basis['tangential_sign']:+.0f}, "
            f"circular_fallback={basis['is_nearly_circular']}"
        )
        for annulus_info in profile_diag["annuli"]:
            print(
                f"    ring {annulus_info['rmin_mpc']:.3f}-{annulus_info['rmax_mpc']:.3f} Mpc: "
                f"<gt>={annulus_info['mean_gt']:+.3e}, "
                f"<gx>={annulus_info['mean_gx']:+.3e}, "
                f"<|gx|>={annulus_info['mean_abs_gx']:.3e}, "
                f"frac(gt>0)_circ={annulus_info['frac_gt_pos_circular']:.3f}, "
                f"med|gx/gt|_circ={annulus_info['median_abs_ratio_circular']:.3e}, "
                f"med|gx/gt|_ell={annulus_info['median_abs_ratio_ellipse']:.3e}, "
                f"frac(gt>0)_ell={annulus_info['frac_gt_pos_ellipse']:.3f}, "
                f"gt m2/m4={annulus_info['gt_m2_rel']:.3f}/{annulus_info['gt_m4_rel']:.3f}, "
                f"gx m2/m4={annulus_info['gx_m2_rel']:.3f}/{annulus_info['gx_m4_rel']:.3f}"
            )

    if ellipse_gamma_diagnostics is None:
        return

    print("\nProjected-ellipse common-screen shear-amplitude diagnostics (paper-style):")
    print("  gamma(a, theta) sampled along projected kappa ellipses at fixed semi-major axis a")
    for profile_index in selected_indices:
        profile_diag = ellipse_gamma_diagnostics[profile_index]
        basis = profile_diag["basis"]
        if basis is None:
            print(f"  {profile_names[profile_index]}: diagnostics unavailable")
            continue

        print(
            f"  {profile_names[profile_index]}: "
            f"phi_major={basis['theta_major_deg']:+.2f} deg, "
            f"q_proj={basis['axis_ratio']:.3f}"
        )
        for ellipse_info in profile_diag["ellipses"]:
            print(
                f"    a={ellipse_info['a_mpc']:.3f} Mpc: "
                f"gamma_major={ellipse_info['gamma_major']:.3e}, "
                f"gamma_minor={ellipse_info['gamma_minor']:.3e}, "
                f"gamma_min/max={ellipse_info['gamma_min']:.3e}/{ellipse_info['gamma_max']:.3e}, "
                f"gamma m2/m4={ellipse_info['gamma_m2_rel']:.3f}/{ellipse_info['gamma_m4_rel']:.3f}"
            )


def main():
    args = parse_args()
    data, path = load_data(args.path)
    namer = RunNamer.from_path(path)
    profile_names = _load_string_list(data["profile_names"])
    selected_indices = _selected_profile_indices(
        profile_names,
        data["reference_profile_index"],
        args.profiles,
    )

    print(f"Loaded {path}")
    print(f"Selected profiles: {', '.join(profile_names[idx] for idx in selected_indices)}")

    plot_profile_comparison(data, namer, selected_indices)
    plot_annular_map_profiles(data, namer, selected_indices)
    plot_relative_profiles(data, namer, selected_indices)
    if not args.skip_maps:
        plot_delta_maps(data, namer, selected_indices)
        plot_delta_shear_component_maps(data, namer, selected_indices)
        plot_delta_omega_maps(data, namer, selected_indices)
        plot_delta_common_screen_shear_component_maps(data, namer, selected_indices)
        plot_delta_common_screen_tangential_cross_maps(data, namer, selected_indices)
        plot_delta_common_screen_ellipse_aligned_tangential_cross_maps(data, namer, selected_indices)
        plot_delta_common_screen_omega_maps(data, namer, selected_indices)
    if not args.skip_raw_maps:
        plot_raw_maps(data, namer, selected_indices)
        plot_raw_shear_component_maps(data, namer, selected_indices)
        plot_raw_omega_maps(data, namer, selected_indices)
        plot_raw_common_screen_shear_component_maps(data, namer, selected_indices)
        plot_raw_common_screen_projected_principal_frame_maps(data, namer, selected_indices)
        plot_raw_common_screen_tangential_cross_maps(data, namer, selected_indices)
        plot_raw_common_screen_ellipse_aligned_tangential_cross_maps(data, namer, selected_indices)
        plot_raw_common_screen_omega_maps(data, namer, selected_indices)
        plot_raw_common_screen_projected_ellipse_shear_profiles(data, namer, selected_indices)
    print_summary(data, selected_indices)


if __name__ == "__main__":
    main()