#!/usr/bin/env python3
"""Fit a spherical NFW lens model to each equivalent-mass profile.

Reads a ``lensing_mass_shape_*_results.npz`` file, fits a spherical NFW model to
the stored radial ``kappa`` and/or ``gamma`` profiles for every compared shape,
and reports the inferred mass/concentration bias caused by forcing the wrong
model family.

Usage::

    ./.venv/bin/python _postprocessing/fit_equivalent_mass_spherical_nfw_bias.py
    ./.venv/bin/python _postprocessing/fit_equivalent_mass_spherical_nfw_bias.py path/to/results.npz
    ./.venv/bin/python _postprocessing/fit_equivalent_mass_spherical_nfw_bias.py path/to/results.npz --observable gamma
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import least_squares

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from excalibur.core.constants import one_Mpc, one_Msun
from excalibur.io.filename_utils import RunNamer, latest_run
from excalibur.objects.nfw_halo import NFWHalo
from _postprocessing.profile_plot_styles import (
    equivalent_mass_fit_style,
    equivalent_mass_profile_display_name,
    equivalent_mass_profile_style,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit a spherical NFW model to each profile in a lensing_mass_shape result file."
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
        help="Subset of profile names to fit. The reference profile is always included automatically.",
    )
    parser.add_argument(
        "--observable",
        choices=("both", "kappa", "gamma"),
        default="both",
        help="Which radial profile(s) to fit. Default: both.",
    )
    parser.add_argument(
        "--rmin-mpc",
        type=float,
        default=None,
        help="Optional minimum impact parameter for the fit [Mpc]. Defaults to the first positive stored bin.",
    )
    parser.add_argument(
        "--rmax-mpc",
        type=float,
        default=None,
        help="Optional maximum impact parameter for the fit [Mpc]. Defaults to the largest stored finite bin.",
    )
    parser.add_argument(
        "--mass-dex-window",
        type=float,
        default=1.0,
        help="Half-width of the allowed log10(M/Msun) fit interval around the injected mass. Default: 1.0 dex.",
    )
    parser.add_argument(
        "--c-min",
        type=float,
        default=0.5,
        help="Lower concentration bound. Default: 0.5.",
    )
    parser.add_argument(
        "--c-max",
        type=float,
        default=50.0,
        help="Upper concentration bound. Default: 50.0.",
    )
    return parser.parse_args()


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


def _infer_rho_cr_from_results(d):
    if "M_200_Msun" not in d or "R_200_Mpc" not in d:
        return None
    mass = float(d["M_200_Msun"]) * one_Msun
    radius = float(d["R_200_Mpc"]) * one_Mpc
    if not np.isfinite(mass) or not np.isfinite(radius) or mass <= 0.0 or radius <= 0.0:
        return None
    return 3.0 * mass / (4.0 * np.pi * 200.0 * radius**3)


def _nfw_model_profiles(radius_mpc, log10_mass_msun, log10_concentration, sigma_cr, rho_cr):
    mass_msun = 10.0 ** float(log10_mass_msun)
    concentration = 10.0 ** float(log10_concentration)
    halo = NFWHalo(
        M_200=mass_msun * one_Msun,
        c_NFW=concentration,
        center=np.zeros(3, dtype=float),
        rho_cr=rho_cr,
    )
    impact_parameter = np.asarray(radius_mpc, dtype=float) * one_Mpc
    kappa_model = halo.kappa_analytic(impact_parameter, sigma_cr)
    gamma_model = halo.gamma_analytic(impact_parameter, sigma_cr)
    return halo, kappa_model, gamma_model


def _fit_mask(radius_mpc, data_kappa, data_gamma, args):
    radius_mpc = np.asarray(radius_mpc, dtype=float)
    mask = np.isfinite(radius_mpc) & (radius_mpc > 0.0)
    if args.rmin_mpc is not None:
        mask &= radius_mpc >= float(args.rmin_mpc)
    if args.rmax_mpc is not None:
        mask &= radius_mpc <= float(args.rmax_mpc)

    if args.observable in ("both", "kappa"):
        mask &= np.isfinite(data_kappa) & (data_kappa > 0.0)
    if args.observable in ("both", "gamma"):
        mask &= np.isfinite(data_gamma) & (data_gamma > 0.0)
    return mask


def _stack_log_residuals(params, radius_mpc, data_kappa, data_gamma, sigma_cr, rho_cr, observable):
    _halo, model_kappa, model_gamma = _nfw_model_profiles(radius_mpc, params[0], params[1], sigma_cr, rho_cr)

    residuals = []
    if observable in ("both", "kappa"):
        valid_k = np.isfinite(data_kappa) & np.isfinite(model_kappa) & (data_kappa > 0.0) & (model_kappa > 0.0)
        if np.any(valid_k):
            residuals.append(np.log(model_kappa[valid_k]) - np.log(data_kappa[valid_k]))
    if observable in ("both", "gamma"):
        valid_g = np.isfinite(data_gamma) & np.isfinite(model_gamma) & (data_gamma > 0.0) & (model_gamma > 0.0)
        if np.any(valid_g):
            residuals.append(np.log(model_gamma[valid_g]) - np.log(data_gamma[valid_g]))

    if not residuals:
        return np.array([1e6], dtype=float)
    return np.concatenate(residuals)


def _fit_one_profile(radius_mpc, data_kappa, data_gamma, sigma_cr, rho_cr, injected_mass_msun, injected_c, args):
    mask = _fit_mask(radius_mpc, data_kappa, data_gamma, args)
    if np.count_nonzero(mask) < 4:
        raise ValueError("Not enough valid radial bins for fitting")

    radius_fit = np.asarray(radius_mpc[mask], dtype=float)
    kappa_fit = np.asarray(data_kappa[mask], dtype=float)
    gamma_fit = np.asarray(data_gamma[mask], dtype=float)

    x0 = np.array([np.log10(injected_mass_msun), np.log10(injected_c)], dtype=float)
    lower = np.array([np.log10(injected_mass_msun) - float(args.mass_dex_window), np.log10(float(args.c_min))], dtype=float)
    upper = np.array([np.log10(injected_mass_msun) + float(args.mass_dex_window), np.log10(float(args.c_max))], dtype=float)

    result = least_squares(
        _stack_log_residuals,
        x0=x0,
        bounds=(lower, upper),
        args=(radius_fit, kappa_fit, gamma_fit, sigma_cr, rho_cr, args.observable),
        method="trf",
    )

    halo_fit, kappa_model, gamma_model = _nfw_model_profiles(radius_fit, result.x[0], result.x[1], sigma_cr, rho_cr)
    rms_log_kappa = np.nan
    rms_log_gamma = np.nan
    if np.any(kappa_fit > 0.0):
        rms_log_kappa = float(np.sqrt(np.mean((np.log(kappa_model) - np.log(kappa_fit)) ** 2)))
    if np.any(gamma_fit > 0.0):
        rms_log_gamma = float(np.sqrt(np.mean((np.log(gamma_model) - np.log(gamma_fit)) ** 2)))

    return {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "radius_fit_Mpc": radius_fit,
        "kappa_fit": kappa_fit,
        "gamma_fit": gamma_fit,
        "kappa_model": kappa_model,
        "gamma_model": gamma_model,
        "M_fit_Msun": float(10.0 ** result.x[0]),
        "c_fit": float(10.0 ** result.x[1]),
        "R_200_fit_Mpc": float(halo_fit.R_200 / one_Mpc),
        "r_s_fit_Mpc": float(halo_fit.r_s / one_Mpc),
        "cost": float(result.cost),
        "nfev": int(result.nfev),
        "rms_log_kappa": float(rms_log_kappa),
        "rms_log_gamma": float(rms_log_gamma),
    }


def _plot_profile_overlays(
    radius_mpc,
    kappa_profiles,
    gamma_profiles,
    fit_results,
    profile_names,
    outfile,
    observable,
    injected_mass_msun=None,
    injected_c=None,
):
    radius_mpc = np.asarray(radius_mpc, dtype=float)
    panels = []
    if observable in ("both", "kappa"):
        panels.append((r"$\kappa(b)$", kappa_profiles, "kappa_model", r"Convergence profile  $\kappa(b)$"))
    if observable in ("both", "gamma"):
        panels.append((r"$\gamma(b)$", gamma_profiles, "gamma_model", r"Shear profile  $\gamma(b)$"))

    n_panels = len(panels)
    fig_height = 5.2 * n_panels + 1.4
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(13.5, fig_height), squeeze=False, sharex=True,
        gridspec_kw={"hspace": 0.22},
    )
    axes = axes[:, 0]
    fig.patch.set_facecolor("#ffffff")

    positive_radius = radius_mpc[np.isfinite(radius_mpc) & (radius_mpc > 0.0)]
    xlim = _positive_plot_limits(positive_radius, pad_decades=0.06)

    profile_handles = []
    for idx, name in enumerate(profile_names):
        style = equivalent_mass_profile_style(name, is_reference=name == "single_nfw", fallback_index=idx)
        profile_handles.append(
            Line2D(
                [0], [0],
                color=style["color"],
                lw=2.0,
                marker=style["marker"],
                markersize=7.0,
                markerfacecolor="#ffffff",
                markeredgecolor=style["color"],
                markeredgewidth=1.1,
                label=equivalent_mass_profile_display_name(name),
            )
        )

    style_handles = [
        Line2D(
            [0], [0],
            color="#36404a", lw=1.6, marker="o", markersize=6.5,
            markerfacecolor="#ffffff", markeredgecolor="#36404a", markeredgewidth=1.0,
            label="stored profile",
        ),
        Line2D(
            [0], [0],
            color="#36404a", lw=2.0, ls=(0, (4.0, 2.2)),
            label="spherical-NFW fit",
        ),
    ]

    for ax, (ylabel, profiles, model_key, title) in zip(axes, panels):
        panel_values = []
        for idx, name in enumerate(profile_names):
            fit = fit_results[idx]
            data_style = equivalent_mass_profile_style(name, is_reference=name == "single_nfw", fallback_index=idx)
            fit_style = equivalent_mass_fit_style(name, is_reference=name == "single_nfw", fallback_index=idx)

            profile_values = np.asarray(profiles[idx], dtype=float)
            fit_radius = np.asarray(fit["radius_fit_Mpc"], dtype=float)
            fit_values = np.asarray(fit[model_key], dtype=float)
            mask_data = np.isfinite(radius_mpc) & (radius_mpc > 0.0) & np.isfinite(profile_values) & (profile_values > 0.0)
            mask_fit = np.isfinite(fit_radius) & (fit_radius > 0.0) & np.isfinite(fit_values) & (fit_values > 0.0)
            if np.any(mask_data):
                ax.plot(
                    radius_mpc[mask_data],
                    profile_values[mask_data],
                    color=data_style["color"],
                    lw=data_style["lw"],
                    marker=data_style["marker"],
                    markersize=data_style["ms"] + 1.2,
                    markerfacecolor="#ffffff",
                    markeredgecolor=data_style["color"],
                    markeredgewidth=1.05,
                    alpha=data_style["alpha"],
                    zorder=data_style["zorder"],
                )
                panel_values.append(profile_values[mask_data])
            if np.any(mask_fit):
                ax.plot(
                    fit_radius[mask_fit],
                    fit_values[mask_fit],
                    color=fit_style["color"],
                    lw=fit_style["lw"] + 0.3,
                    ls=fit_style["linestyle"],
                    alpha=fit_style["alpha"],
                    zorder=fit_style["zorder"],
                )
                panel_values.append(fit_values[mask_fit])

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_facecolor("#fafbfc")
        ax.grid(which="major", alpha=0.32, color="#c6ccd3", linewidth=0.8)
        ax.grid(which="minor", alpha=0.14, color="#d9dee4", linewidth=0.55)
        ax.set_title(title, fontsize=13, pad=8)
        ax.tick_params(axis="both", which="major", labelsize=10)
        for spine in ax.spines.values():
            spine.set_color("#bfc6ce")
            spine.set_linewidth(0.9)
        if xlim is not None:
            ax.set_xlim(*xlim)
        ylim = _positive_plot_limits(*panel_values, pad_decades=0.10)
        if ylim is not None:
            ax.set_ylim(*ylim)

    style_legend = axes[0].legend(
        handles=style_handles,
        loc="lower left",
        frameon=True,
        framealpha=0.96,
        facecolor="#ffffff",
        edgecolor="#d8dde3",
        fontsize=9.5,
        handlelength=2.4,
        borderpad=0.55,
    )
    axes[0].add_artist(style_legend)

    axes[-1].set_xlabel(r"impact parameter $b$  [Mpc]", fontsize=13)
    fig.legend(
        handles=profile_handles,
        loc="center right",
        bbox_to_anchor=(0.99, 0.5),
        frameon=True,
        framealpha=0.96,
        facecolor="#ffffff",
        edgecolor="#d8dde3",
        fontsize=10,
        title="Mass profile",
        title_fontsize=11,
        handlelength=2.4,
        labelspacing=0.85,
        borderpad=0.7,
    )

    if injected_mass_msun is not None and injected_c is not None:
        suptitle = (
            r"Spherical-NFW fits to equivalent-mass profiles  —  "
            rf"injected $M_{{200}} = {injected_mass_msun:.2e}\,M_\odot$,  $c = {injected_c:.2f}$"
        )
    else:
        suptitle = "Spherical-NFW fits to equivalent-mass profiles"
    top_for_suptitle = 0.92 if n_panels >= 2 else 0.88
    fig.suptitle(suptitle, fontsize=14, y=0.985)

    fig.subplots_adjust(left=0.07, right=0.83, top=top_for_suptitle, bottom=0.08)
    fig.savefig(outfile, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _plot_bias_summary(profile_names, fit_results, injected_mass_msun, injected_c, outfile):
    y = np.arange(len(profile_names), dtype=float)
    display_names = [equivalent_mass_profile_display_name(name) for name in profile_names]
    mass_ratio = np.array([fit["M_fit_Msun"] / injected_mass_msun for fit in fit_results], dtype=float)
    concentration_ratio = np.array([fit["c_fit"] / injected_c for fit in fit_results], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 0.95 * len(profile_names) + 2.6), squeeze=False, sharey=True)
    fig.patch.set_facecolor("#ffffff")
    ax_mass, ax_c = axes[0]
    panels = (
        (ax_mass, mass_ratio, r"$M_{\rm fit} \, / \, M_{\rm true}$", r"Mass bias  —  $M_{\rm true}=$" + f"{injected_mass_msun:.2e}" + r"$\,M_\odot$"),
        (ax_c, concentration_ratio, r"$c_{\rm fit} \, / \, c_{\rm true}$", r"Concentration bias  —  $c_{\rm true}=$" + f"{injected_c:.2f}"),
    )

    for ax, ratios, xlabel, title in panels:
        ratio_min = min(float(np.min(ratios)), 1.0)
        ratio_max = max(float(np.max(ratios)), 1.0)
        span = max(ratio_max - ratio_min, 1e-3)
        pad = max(0.10, 0.28 * span)
        ax.set_xlim(ratio_min - pad, ratio_max + pad)

        ax.axvspan(1.0 - 0.0001, 1.0 + 0.0001, color="#dfe5ec", alpha=0.0)
        ax.axvline(1.0, color="#36404a", ls=(0, (2.5, 2.0)), lw=1.3, alpha=0.85, zorder=1)
        ax.text(
            1.0, len(profile_names) - 0.35,
            "  truth",
            color="#36404a", fontsize=9.0, alpha=0.85,
            ha="left", va="bottom", style="italic",
        )

        ax.set_facecolor("#fafbfc")
        ax.grid(axis="x", which="major", alpha=0.32, color="#c6ccd3", linewidth=0.8)
        ax.grid(axis="x", which="minor", alpha=0.14, color="#d9dee4", linewidth=0.55)
        ax.minorticks_on()
        ax.tick_params(axis="y", which="minor", left=False)
        ax.set_xlabel(xlabel, fontsize=13, labelpad=8)
        ax.set_title(title, fontsize=12.5, pad=10)
        ax.tick_params(axis="both", which="major", labelsize=10)
        for spine in ax.spines.values():
            spine.set_color("#bfc6ce")
            spine.set_linewidth(0.9)

        for idx, (yy, name, ratio) in enumerate(zip(y, profile_names, ratios)):
            style = equivalent_mass_profile_style(name, is_reference=name == "single_nfw", fallback_index=idx)
            ax.hlines(
                yy, 1.0, ratio,
                color=style["color"],
                lw=2.8 if name == "single_nfw" else 2.3,
                alpha=0.95,
                zorder=2,
            )
            ax.scatter(
                [ratio], [yy],
                s=120 if name == "single_nfw" else 95,
                marker=style["marker"],
                color=style["color"],
                edgecolors="#ffffff",
                linewidths=1.3,
                zorder=4,
            )
            dx = 0.020 * (ax.get_xlim()[1] - ax.get_xlim()[0])
            x_text = ratio + dx if ratio >= 1.0 else ratio - dx
            ha = "left" if ratio >= 1.0 else "right"
            pct = (ratio - 1.0) * 100.0
            ax.text(
                x_text, yy,
                f"{pct:+.1f}%",
                va="center", ha=ha,
                fontsize=10, fontweight="semibold", color="#2b333c",
                bbox={
                    "facecolor": "#ffffff",
                    "edgecolor": style["color"],
                    "boxstyle": "round,pad=0.22",
                    "alpha": 0.92,
                    "linewidth": 0.9,
                },
                zorder=5,
            )

    ax_mass.set_yticks(y)
    ax_mass.set_yticklabels(display_names, fontsize=11)
    ax_c.tick_params(axis="y", labelleft=False)
    ax_mass.set_ylim(len(profile_names) - 0.4, -0.6)
    fig.suptitle("Bias on inferred parameters when fitting a spherical NFW to non-spherical profiles", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(outfile, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _positive_plot_limits(*arrays, pad_decades=0.08):
    positive_values = []
    for arr in arrays:
        values = np.asarray(arr, dtype=float)
        mask = np.isfinite(values) & (values > 0.0)
        if np.any(mask):
            positive_values.append(values[mask].ravel())
    if not positive_values:
        return None
    values = np.concatenate(positive_values)
    log_min = float(np.log10(np.min(values)))
    log_max = float(np.log10(np.max(values)))
    if not np.isfinite(log_min) or not np.isfinite(log_max):
        return None
    if abs(log_max - log_min) < 1e-9:
        log_min -= 0.18
        log_max += 0.18
    return 10.0 ** (log_min - pad_decades), 10.0 ** (log_max + pad_decades)


def _observable_tag(observable):
    return f"observable_{str(observable).lower()}"


def _save_summary_npz(namer, profile_names, fit_results, injected_mass_msun, injected_c, observable):
    observable_tag = _observable_tag(observable)
    outfile = namer.path(f"{namer.stem}_shape_spherical_nfw_fit_bias_{observable_tag}_summary.npz")
    np.savez(
        outfile,
        profile_names=np.asarray(profile_names),
        observable=str(observable),
        M_true_Msun=float(injected_mass_msun),
        c_true=float(injected_c),
        M_fit_Msun=np.asarray([fit["M_fit_Msun"] for fit in fit_results], dtype=float),
        c_fit=np.asarray([fit["c_fit"] for fit in fit_results], dtype=float),
        R_200_fit_Mpc=np.asarray([fit["R_200_fit_Mpc"] for fit in fit_results], dtype=float),
        r_s_fit_Mpc=np.asarray([fit["r_s_fit_Mpc"] for fit in fit_results], dtype=float),
        success=np.asarray([fit["success"] for fit in fit_results], dtype=bool),
        rms_log_kappa=np.asarray([fit["rms_log_kappa"] for fit in fit_results], dtype=float),
        rms_log_gamma=np.asarray([fit["rms_log_gamma"] for fit in fit_results], dtype=float),
    )
    return outfile


def main():
    args = parse_args()
    data, path = load_data(args.path)
    namer = RunNamer.from_path(path)

    profile_names = _load_string_list(data["profile_names"])
    selected_indices = _selected_profile_indices(profile_names, data["reference_profile_index"], args.profiles)
    selected_names = [profile_names[idx] for idx in selected_indices]

    radius_mpc = np.asarray(data["b_profile_Mpc"], dtype=float)
    kappa_profiles = np.asarray(data["kappa_profile_by_shape"], dtype=float)[selected_indices]
    gamma_profiles = np.asarray(data["gamma_profile_by_shape"], dtype=float)[selected_indices]

    injected_mass_msun = float(data["M_200_Msun"])
    injected_c = float(data["c_NFW"])
    sigma_cr = float(data["Sigma_cr"])
    rho_cr = _infer_rho_cr_from_results(data)
    if rho_cr is None:
        raise ValueError("Could not infer rho_cr from M_200_Msun and R_200_Mpc in the results file")

    positive_radii = radius_mpc[np.isfinite(radius_mpc) & (radius_mpc > 0.0)]
    effective_rmin = float(args.rmin_mpc) if args.rmin_mpc is not None else float(np.min(positive_radii))
    effective_rmax = float(args.rmax_mpc) if args.rmax_mpc is not None else float(np.max(positive_radii))

    fit_results = []
    for idx, name in zip(selected_indices, selected_names):
        fit = _fit_one_profile(
            radius_mpc=radius_mpc,
            data_kappa=np.asarray(data["kappa_profile_by_shape"], dtype=float)[idx],
            data_gamma=np.asarray(data["gamma_profile_by_shape"], dtype=float)[idx],
            sigma_cr=sigma_cr,
            rho_cr=rho_cr,
            injected_mass_msun=injected_mass_msun,
            injected_c=injected_c,
            args=args,
        )
        fit_results.append(fit)

    print(f"Loaded {path}")
    print(f"Selected profiles: {', '.join(selected_names)}")
    print(
        f"Spherical NFW fit setup: observable={args.observable}, "
        f"fit range={effective_rmin:.4f}-{effective_rmax:.4f} Mpc, "
        f"Sigma_cr={sigma_cr * one_Mpc ** 2 / one_Msun:.3e} Msun/Mpc^2"
    )
    print(
        f"Injected spherical reference: M_true={injected_mass_msun:.6e} Msun, "
        f"c_true={injected_c:.4f}"
    )
    print("\nBest-fit spherical-NFW parameters:")
    for name, fit in zip(selected_names, fit_results):
        mass_bias = 100.0 * (fit["M_fit_Msun"] / injected_mass_msun - 1.0)
        concentration_bias = 100.0 * (fit["c_fit"] / injected_c - 1.0)
        print(
            f"  {name}: "
            f"M_fit={fit['M_fit_Msun']:.6e} Msun ({mass_bias:+.2f}%), "
            f"c_fit={fit['c_fit']:.4f} ({concentration_bias:+.2f}%), "
            f"rms_log(kappa)={fit['rms_log_kappa']:.3e}, "
            f"rms_log(gamma)={fit['rms_log_gamma']:.3e}, "
            f"success={fit['success']}"
        )

    observable_tag = _observable_tag(args.observable)
    profile_plot = namer.plot(f"shape_spherical_nfw_fit_bias_{observable_tag}_profiles")
    bias_plot = namer.plot(f"shape_spherical_nfw_fit_bias_{observable_tag}_summary")
    summary_npz = _save_summary_npz(namer, selected_names, fit_results, injected_mass_msun, injected_c, args.observable)

    _plot_profile_overlays(
        radius_mpc,
        kappa_profiles,
        gamma_profiles,
        fit_results,
        selected_names,
        profile_plot,
        args.observable,
        injected_mass_msun=injected_mass_msun,
        injected_c=injected_c,
    )
    _plot_bias_summary(selected_names, fit_results, injected_mass_msun, injected_c, bias_plot)

    print(f"\n   [ok] {profile_plot}")
    print(f"   [ok] {bias_plot}")
    print(f"   [ok] {summary_npz}")


if __name__ == "__main__":
    main()