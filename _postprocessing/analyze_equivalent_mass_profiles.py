#!/usr/bin/env python3
"""Post-processing for equivalent-mass lens-shape comparison runs.

Reads ``lensing_mass_shape_*_results.npz`` and produces:
    1. background-subtracted kappa/gamma profiles for the selected shapes
    2. Delta-kappa and Delta-gamma radial profiles relative to the reference
    3. Delta-kappa and Delta-gamma 2D maps for each non-reference shape
    4. Raw kappa and |gamma| 2D maps with critical-curve overlays
    5. Relative-percent kappa and |gamma| profiles versus the reference

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
from matplotlib.colors import LogNorm, SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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
    cmap = plt.get_cmap("tab10", max(len(selected_indices), 1))

    for color_idx, profile_index in enumerate(selected_indices):
        name = profile_names[profile_index]
        label = (
            f"{name}  (N={component_counts[profile_index]}, "
            f"rms=({axis_rms[profile_index, 0]:.2f}, {axis_rms[profile_index, 1]:.2f}, {axis_rms[profile_index, 2]:.2f}) Mpc)"
        )
        color = "k" if profile_index == reference_index else cmap(color_idx)
        lw = 2.2 if profile_index == reference_index else 1.4
        alpha = 0.95 if profile_index == reference_index else 0.9

        _plot_valid_line(
            axes[0, 0],
            b_profile,
            kappa_signal[profile_index],
            "o-",
            ms=3,
            lw=lw,
            color=color,
            alpha=alpha,
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
                color=color,
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
            "o-",
            ms=3,
            lw=lw,
            color=color,
            alpha=alpha,
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
                color=color,
                alpha=0.12 if profile_index == reference_index else 0.08,
                linewidth=0,
            )

        if profile_index != reference_index:
            _plot_valid_line(
                axes[1, 0],
                b_profile,
                delta_kappa_signal[profile_index],
                "o-",
                ms=3,
                lw=1.3,
                color=color,
                alpha=0.9,
                label=name,
            )
            _plot_valid_line(
                axes[1, 1],
                b_profile,
                delta_gamma[profile_index],
                "o-",
                ms=3,
                lw=1.3,
                color=color,
                alpha=0.9,
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
    cmap = plt.get_cmap("tab10", max(len(selected_indices), 1))

    for color_idx, profile_index in enumerate(selected_indices):
        name = profile_names[profile_index]
        label = (
            f"{name}  (N={component_counts[profile_index]}, "
            f"rms=({axis_rms[profile_index, 0]:.2f}, {axis_rms[profile_index, 1]:.2f}, {axis_rms[profile_index, 2]:.2f}) Mpc)"
        )
        color = "k" if profile_index == reference_index else cmap(color_idx)
        lw = 2.2 if profile_index == reference_index else 1.4
        alpha = 0.95 if profile_index == reference_index else 0.9

        _plot_valid_line(
            axes[0],
            radius,
            kappa_signal[profile_index],
            "o-",
            ms=3,
            lw=lw,
            color=color,
            alpha=alpha,
            label=label,
        )
        _fill_between_valid(
            axes[0],
            radius,
            kappa_signal[profile_index] - kappa_signal_std[profile_index],
            kappa_signal[profile_index] + kappa_signal_std[profile_index],
            color=color,
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
                "o-",
                ms=3,
                lw=lw,
                color=color,
                alpha=alpha,
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
                color=color,
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
    cmap = plt.get_cmap("tab10", max(len(non_reference_indices), 1))

    for color_idx, profile_index in enumerate(non_reference_indices):
        color = cmap(color_idx)
        name = profile_names[profile_index]

        mask_kappa = np.isfinite(kappa_rel[profile_index])
        if np.any(mask_kappa):
            axes[0].plot(
                b_profile[mask_kappa],
                kappa_rel[profile_index, mask_kappa],
                "o-",
                ms=3,
                lw=1.4,
                color=color,
                alpha=0.9,
                label=name,
            )

        mask_gamma = np.isfinite(gamma_rel[profile_index])
        if np.any(mask_gamma):
            axes[1].plot(
                b_profile[mask_gamma],
                gamma_rel[profile_index, mask_gamma],
                "o-",
                ms=3,
                lw=1.4,
                color=color,
                alpha=0.9,
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
        plot_delta_common_screen_omega_maps(data, namer, selected_indices)
    if not args.skip_raw_maps:
        plot_raw_maps(data, namer, selected_indices)
        plot_raw_shear_component_maps(data, namer, selected_indices)
        plot_raw_omega_maps(data, namer, selected_indices)
        plot_raw_common_screen_shear_component_maps(data, namer, selected_indices)
        plot_raw_common_screen_omega_maps(data, namer, selected_indices)
    print_summary(data, selected_indices)


if __name__ == "__main__":
    main()