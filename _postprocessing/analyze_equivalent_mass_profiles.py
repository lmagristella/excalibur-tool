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

        axes[0, 0].plot(b_profile, kappa_signal[profile_index], "o-", ms=3, lw=lw, color=color, alpha=alpha, label=label)
        if kappa_profile_std is not None:
            bg_std = kappa_profile_std[profile_index, -1]
            kappa_signal_std = np.sqrt(kappa_profile_std[profile_index] ** 2 + bg_std ** 2)
            axes[0, 0].fill_between(
                b_profile,
                kappa_signal[profile_index] - kappa_signal_std,
                kappa_signal[profile_index] + kappa_signal_std,
                color=color,
                alpha=0.12 if profile_index == reference_index else 0.08,
                linewidth=0,
            )

        mask_gamma = (b_profile > 0.0) & (gamma_profile[profile_index] > 0.0)
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
            axes[1, 0].plot(b_profile, delta_kappa_signal[profile_index], "o-", ms=3, lw=1.3, color=color, alpha=0.9, label=name)
            axes[1, 1].plot(b_profile, delta_gamma[profile_index], "o-", ms=3, lw=1.3, color=color, alpha=0.9, label=name)

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
    kappa_vmax = max(np.max(np.abs(delta_kappa_signal[selected_indices])), 1e-12)
    axes[1, 0].set_yscale("symlog", linthresh=max(kappa_vmax * 1e-3, 1e-12))
    axes[1, 0].set_title(f"Delta-kappa relative to {reference_name}")
    axes[1, 0].set_xlabel(x_label)
    axes[1, 0].set_ylabel("Delta (kappa - kappa_bg)")
    axes[1, 0].grid(True, alpha=0.3, which="both")
    if len(selected_indices) > 1:
        axes[1, 0].legend(fontsize=8)

    axes[1, 1].axhline(0.0, color="k", lw=0.8, alpha=0.5)
    axes[1, 1].set_xscale("log")
    gamma_vmax = max(np.max(np.abs(delta_gamma[selected_indices])), 1e-12)
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
            vmax = max(np.max(np.abs(panel)), 1e-12)
            im = ax.imshow(
                panel.T,
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
    kappa_vmax = max(np.nanmax(selected_kappa), 1e-12)
    gamma_vmax = max(np.nanmax(selected_gamma), 1e-12)
    mu_vmax = max(np.nanmax(np.abs(selected_mu)), 1e-12)
    inv_mu_vmax = max(np.nanmax(np.abs(selected_inv_mu)), 1e-12)
    kappa_positive = selected_kappa[selected_kappa > 0.0]
    gamma_positive = selected_gamma[selected_gamma > 0.0]
    kappa_vmin = max(np.nanmin(kappa_positive), kappa_vmax * 1e-4) if kappa_positive.size else kappa_vmax * 1e-4
    gamma_vmin = max(np.nanmin(gamma_positive), gamma_vmax * 1e-4) if gamma_positive.size else gamma_vmax * 1e-4

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
            image_panel = np.clip(panel, floor, None) if floor is not None else panel
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
            f"max |Delta-kappa_profile| = {np.max(np.abs(delta_kappa_signal[profile_index])):.4e}, "
            f"max |Delta-gamma_profile| = {np.max(np.abs(delta_gamma[profile_index])):.4e}, "
            f"max |Delta-kappa_map| = {np.max(np.abs(delta_kappa_map[profile_index])):.4e}, "
            f"max |Delta-gamma_map| = {np.max(np.abs(delta_gamma_map[profile_index])):.4e}"
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
    plot_relative_profiles(data, namer, selected_indices)
    if not args.skip_maps:
        plot_delta_maps(data, namer, selected_indices)
    if not args.skip_raw_maps:
        plot_raw_maps(data, namer, selected_indices)
    print_summary(data, selected_indices)


if __name__ == "__main__":
    main()