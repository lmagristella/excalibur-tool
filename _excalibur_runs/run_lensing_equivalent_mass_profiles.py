#!/usr/bin/env python3
"""Compare lensing observables for mass-equivalent halo shapes.

This runner mirrors the analytical NFW workflow but replaces the single
spherical halo by a list of analytical sources with the same total mass.
The shapes are built as exact superpositions of spherical NFW halos, which
lets us probe shape-induced biases on kappa and gamma while keeping the
total mass fixed.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from scipy import interpolate
from scipy.optimize import brentq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from excalibur.core.constants import G, c, one_Mpc, one_Msun
from excalibur.core.cosmology import LCDM_Cosmology
from excalibur.grid.amr_grid import AMRGrid
from excalibur.grid.analytical_bypass import AnalyticalBypassInterpolator
from excalibur.grid.grid import Grid
from excalibur.grid.interpolator_fast import InterpolatorFast
from excalibur.integration.integrator import Integrator
from excalibur.integration.integrator_numba import (
    NumbaAMRBackend as StandardNumbaAMRBackend,
    integrate_photon_numba as integrate_photon_numba_standard,
)
from excalibur.integration.integrator_numba_schemes import (
    STOP_REASON_DT_INVALID,
    STOP_REASON_LAMBDA_STOP,
    STOP_REASON_PYTHON_BACKEND,
    STOP_REASON_STEP_BUDGET,
    STOP_REASON_UNKNOWN,
    integrator_stop_reason_labels,
)
from excalibur.io.filename_utils import RunNamer
from excalibur.metrics.perturbed_flrw_metric_fast import PerturbedFLRWMetricFast
from excalibur.objects.equivalent_mass_profiles import (
    available_equivalent_mass_presets,
    build_equivalent_mass_profile,
    parse_component_specs,
    science_ready_triaxial_preset_suite,
)
from excalibur.objects.nfw_halo import NFWHalo
from excalibur.observables.lensing_conventions import (
    DEFAULT_LENSING_REFERENCE_CONVENTION,
    PHYSICAL_LENSING_REFERENCE_CONVENTION,
    lensing_convention_label,
    sigma_cr_conventions,
)
from excalibur.observables.optical_tidal_matrix import lensing_from_jacobi
from excalibur.observables.sachs_basis import init_sachs_basis
from excalibur.photon.photon import Photon


def parse_args():
    presets = ", ".join(available_equivalent_mass_presets())
    parser = argparse.ArgumentParser(
        description="Analytical lensing comparison for equivalent-mass halo shapes."
    )
    parser.add_argument("--backend", choices=("python", "numba"), default="python")
    parser.add_argument(
        "--numba-kernel",
        choices=("standard", "lowalloc", "specialized"),
        default="standard",
        help="Numba backend variant. The specialized kernel is only available for conformal-screen lensing and supports rk4/dopri5.",
    )
    parser.add_argument(
        "--integrator",
        type=str,
        default="rk4",
        help="Integration scheme: rk4, rk45, dopri5, or dop853. dop853 is python-only; the specialized numba kernel supports rk4/dopri5 only.",
    )
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-13)
    parser.add_argument(
        "--dt-min-factor",
        type=float,
        default=None,
        help=(
            "Override the numba adaptive dt floor as a fraction of the initial dt_fine. "
            "Default backend behavior is dt_min = dt_fine / 1000 for adaptive schemes."
        ),
    )
    parser.add_argument(
        "--max-rejected",
        type=int,
        default=200,
        help="Maximum number of consecutive rejected adaptive steps in the numba backend.",
    )
    parser.add_argument(
        "--parallel-batch",
        action="store_true",
        help="Integrate all photons of one profile in a single numba.prange batch (requires --backend=numba --numba-kernel=specialized --integrator=dopri5).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print per-photon progress every N photons on the non-batch path. Use 0 to disable these progress lines.",
    )
    parser.add_argument("--n-root", type=int, default=16)
    parser.add_argument("--box-mpc", type=float, default=1950.0)
    parser.add_argument("--mass-200-msun", type=float, default=2e15)
    parser.add_argument("--c-nfw", type=float, default=7.0)
    parser.add_argument("--z-lens", type=float, default=None)
    parser.add_argument("--z-source", type=float, default=1.0)
    parser.add_argument("--obs-z-mpc", type=float, default=5.0)
    parser.add_argument("--source-margin-mpc", type=float, default=20.0)
    parser.add_argument("--map-n1d", type=int, default=21)
    parser.add_argument("--map-half-mpc", type=float, default=1.5)
    parser.add_argument("--n-fine-per-rs", type=int, default=8)
    parser.add_argument(
        "--profile-nphi",
        type=int,
        default=24,
        help="Number of azimuthal samples per non-zero radial bin for the geometric-center radial profile.",
    )
    parser.add_argument(
        "--profile-presets",
        nargs="+",
        default=["single_nfw", "los_cigar", "sky_cigar"],
        help=(
            "Profiles to compare. Use 'all' to expand every preset or 'science_ready_triaxial' for a curated triaxial suite. "
            f"Available: {presets}"
        ),
    )
    parser.add_argument(
        "--reference-preset",
        type=str,
        default=None,
        help="Reference profile for delta-kappa and delta-gamma outputs.",
    )
    parser.add_argument(
        "--shape-scale-rs",
        type=float,
        default=1.25,
        help="Global offset scale for non-spherical presets, in units of the reference r_s.",
    )
    parser.add_argument(
        "--profile-axis-scale",
        type=float,
        nargs=3,
        metavar=("S_PERP1", "S_PERP2", "S_LOS"),
        default=(1.0, 1.0, 1.0),
        help="Anisotropic scale applied to component offsets in the local (perp1, perp2, los) frame.",
    )
    parser.add_argument(
        "--profile-orientation-deg",
        type=float,
        nargs=3,
        metavar=("RX", "RY", "RZ"),
        default=(0.0, 0.0, 0.0),
        help="Euler rotations applied after local scaling, in degrees, around the local (perp1, perp2, los) axes.",
    )
    parser.add_argument(
        "--triaxial-axis-ratios",
        type=float,
        nargs=2,
        metavar=("B_OVER_A", "C_OVER_A"),
        default=None,
        help="Override the intrinsic axis ratios of the direct triaxial NFW presets.",
    )
    parser.add_argument(
        "--triaxial-quadrature-order",
        type=int,
        default=96,
        help="Quadrature order used by the direct triaxial NFW presets. Higher is slower but more accurate.",
    )
    parser.add_argument(
        "--cigar-axis",
        choices=("perp1", "perp2", "los"),
        default="los",
        help="Axis used by the cigar_parametric preset.",
    )
    parser.add_argument(
        "--cigar-core-fraction",
        type=float,
        default=0.50,
        help="Mass fraction assigned to the central component in cigar_parametric.",
    )
    parser.add_argument(
        "--cigar-satellite-concentration-scale",
        type=float,
        default=1.15,
        help="Concentration multiplier for the two outer cigar components.",
    )
    parser.add_argument(
        "--custom-components",
        nargs="*",
        default=None,
        help=(
            "Component specs for the custom_components preset. Format per component: "
            "mass_fraction:dx:dy:dz[:concentration_scale], with offsets in the local (perp1, perp2, los) frame."
        ),
    )
    parser.add_argument(
        "--save-trajectories",
        action="store_true",
        help="Store per-profile trajectories. Disabled by default because multi-profile runs get large.",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Print the available shape presets and exit.",
    )
    return parser.parse_args()


def _select_numba_backend(kernel_name):
    if kernel_name == "standard":
        return StandardNumbaAMRBackend, integrate_photon_numba_standard, None, None

    if kernel_name == "specialized":
        from excalibur.integration.integrator_numba_specialized import (
            NumbaAMRBackend as SpecializedNumbaAMRBackend,
            integrate_photon_numba_dopri5,
            integrate_photons_batch_dopri5,
        )

        return SpecializedNumbaAMRBackend, None, integrate_photon_numba_dopri5, integrate_photons_batch_dopri5

    from excalibur.integration.integrator_numba_lowalloc import (
        NumbaAMRBackend as LowAllocNumbaAMRBackend,
        integrate_photon_numba as integrate_photon_numba_lowalloc,
    )

    return LowAllocNumbaAMRBackend, integrate_photon_numba_lowalloc, None, None


def _string_array(values):
    values = [str(v) for v in values]
    max_len = max((len(v) for v in values), default=1)
    return np.asarray(values, dtype=f"U{max_len}")


def _expand_profile_presets(requested):
    valid = available_equivalent_mass_presets()
    valid_set = set(valid)
    suite_aliases = {
        "science_ready_triaxial": science_ready_triaxial_preset_suite(),
    }
    names = []
    for name in requested:
        if not name:
            continue
        lowered = name.lower()
        if lowered == "all":
            names.extend(valid)
            continue
        if lowered in suite_aliases:
            names.extend(suite_aliases[lowered])
            continue
        names.append(name)

    ordered = []
    seen = set()
    invalid = []
    for name in names:
        if name not in valid_set:
            invalid.append(name)
            continue
        if name not in seen:
            ordered.append(name)
            seen.add(name)

    if invalid:
        valid_str = ", ".join(valid)
        raise ValueError(f"Unknown profile preset(s): {', '.join(invalid)}. Valid presets: {valid_str}")
    if not ordered:
        raise ValueError("No valid profile preset selected")
    return ordered


def _resolve_reference_preset(profile_presets, requested_reference):
    if requested_reference is not None:
        if requested_reference not in profile_presets:
            selected = ", ".join(profile_presets)
            raise ValueError(
                f"--reference-preset={requested_reference} is not in --profile-presets ({selected})"
            )
        return requested_reference
    if "single_nfw" in profile_presets:
        return "single_nfw"
    return profile_presets[0]


def _source_components_for_metadata(source):
    if hasattr(source, "component_sources"):
        return list(source.component_sources())
    if hasattr(source, "halos"):
        return list(source.halos)
    return [source]


def _source_supports_numba_bypass(source):
    return bool(getattr(source, "supports_numba_nfw_bypass", hasattr(source, "halos")))


def _source_supports_numba_path(source, *, kernel_name, integrator_name):
    if _source_supports_numba_bypass(source):
        return True
    return bool(
        kernel_name == "specialized"
        and integrator_name == "dopri5"
        and getattr(source, "supports_numba_specialized_bypass", False)
    )


def _normalize_runner_integrator(name):
    key = name.lower().strip()
    alias_map = {
        "rkf45": "rk45",
        "dopri54": "dopri5",
        "dopri5(4)": "dopri5",
    }
    key = alias_map.get(key, key)
    valid = {"rk4", "rk45", "dopri5", "dop853"}
    if key not in valid:
        raise ValueError(
            f"Unsupported integrator '{name}'. Valid choices: {', '.join(sorted(valid))}"
        )
    return key


def _python_integrator_name(name):
    if name == "dopri5":
        return "rk45"
    return name


def _integrator_is_adaptive(name):
    return name in {"rk45", "dopri5", "dop853"}


def _effective_numba_dt_min(dt_fine, integrator_name, dt_min_factor):
    if not _integrator_is_adaptive(integrator_name):
        return abs(float(dt_fine))
    if dt_min_factor is not None:
        return abs(float(dt_fine)) * float(dt_min_factor)
    return abs(float(dt_fine)) / 1000.0


def _effective_numba_dt_max(dt_fine, integrator_name):
    abs_dt = abs(float(dt_fine))
    if not _integrator_is_adaptive(integrator_name):
        return abs_dt
    return abs_dt * 50.0


def _append_photon_history(photon, traj):
    if not hasattr(photon, "history"):
        return

    hist = photon.history
    for row in traj:
        if hasattr(hist, "append"):
            hist.append(row.copy())
        else:
            hist += [row.copy()]


def _pack_lensing_state(photon):
    x = np.asarray(photon.x, dtype=np.float64)
    u = np.asarray(photon.u, dtype=np.float64)
    e1 = np.asarray(getattr(photon, "e1"), dtype=np.float64)
    e2 = np.asarray(getattr(photon, "e2"), dtype=np.float64)
    D_flat = np.asarray(getattr(photon, "D_flat", [0, 0, 0, 0]), dtype=np.float64)
    P_flat = np.asarray(getattr(photon, "P_flat", [1, 0, 0, 1]), dtype=np.float64)
    return np.concatenate([x, u, e1, e2, D_flat, P_flat])


def _unpack_lensing_state(photon, final, lam):
    photon.x = final[0:4].copy()
    photon.u = final[4:8].copy()
    photon.e1 = final[8:12].copy()
    photon.e2 = final[12:16].copy()
    photon.D_flat = final[16:20].copy()
    photon.P_flat = final[20:24].copy()
    photon.lambda_affine = lam


def _lambda_stop_reached(lam, lambda_stop):
    tol = 1e-10 * max(1.0, abs(float(lambda_stop)))
    return lambda_stop <= 0.0 or float(lam) >= float(lambda_stop) - tol


def _specialized_stop_reason(lam, lambda_stop, n_accept, n_reject, max_steps):
    if _lambda_stop_reached(lam, lambda_stop):
        return STOP_REASON_LAMBDA_STOP
    if (int(n_accept) + int(n_reject)) >= int(max_steps):
        return STOP_REASON_STEP_BUDGET
    # The specialized DOPRI5 kernels now exit early when the step remains
    # rejected at the dt floor. If lambda_stop was not reached and the total
    # attempt budget was not exhausted, the only remaining termination mode is
    # effectively a dt floor / invalid-step condition.
    return STOP_REASON_DT_INVALID


def _photon_reached_source(lam, stop_reason, lambda_total):
    if int(stop_reason) == STOP_REASON_LAMBDA_STOP:
        return True
    return _lambda_stop_reached(lam, lambda_total)


def _clear_photon_history(photon):
    history = getattr(photon, "history", None)
    if history is not None and hasattr(history, "states"):
        history.states = []


def _restore_photon_initial_state(photon, initial_state):
    _unpack_lensing_state(photon, initial_state, 0.0)
    _clear_photon_history(photon)


def _integrate_specialized_dopri5_with_retry(
    photon,
    *,
    initial_state,
    numba_backend,
    integrate_photon_dopri5_impl,
    dt_fine,
    lambda_total,
    record_every,
    adaptive_kwargs,
):
    base_dt_min = float(adaptive_kwargs["dt_min"])
    base_max_steps = int(adaptive_kwargs["max_steps"])
    retry_schedule = (
        (1.0, 1),
        (0.1, 4),
        (0.01, 16),
        (0.001, 64),
        (0.0001, 256),
    )
    last_result = None

    for dt_min_scale, max_steps_scale in retry_schedule:
        _restore_photon_initial_state(photon, initial_state)
        dt_min = base_dt_min * dt_min_scale if base_dt_min > 0.0 else 0.0
        max_steps = max(base_max_steps, int(base_max_steps * max_steps_scale))
        _, lam, n_accept, n_reject = integrate_photon_dopri5_impl(
            photon,
            numba_backend,
            dt_init=dt_fine,
            lambda_stop=lambda_total,
            rtol=adaptive_kwargs["rtol"],
            atol=adaptive_kwargs["atol"],
            dt_min=dt_min,
            dt_max=adaptive_kwargs["dt_max"],
            max_steps=max_steps,
            record_every=record_every,
        )
        stop_reason = _specialized_stop_reason(lam, lambda_total, n_accept, n_reject, max_steps)
        last_result = (
            lam,
            int(n_accept),
            stop_reason,
            int(n_reject),
            -1,
            dt_min,
            np.nan,
            np.nan,
        )
        if _photon_reached_source(lam, stop_reason, lambda_total):
            return last_result
        if stop_reason not in (
            STOP_REASON_STEP_BUDGET,
            STOP_REASON_UNKNOWN,
            STOP_REASON_DT_INVALID,
        ):
            break

    return last_result


def _integrate_one_photon(
    photon,
    backend_name,
    *,
    numba_kernel,
    integrator_fine,
    numba_backend,
    integrate_photon_impl,
    integrate_photon_dopri5_impl,
    dt_fine,
    lambda_total,
    n_steps_budget,
    record_every,
    adaptive_kwargs=None,
):
    if backend_name == "numba":
        if numba_kernel == "specialized":
            if adaptive_kwargs is not None:
                initial_state = _pack_lensing_state(photon)
                return _integrate_specialized_dopri5_with_retry(
                    photon,
                    initial_state=initial_state,
                    numba_backend=numba_backend,
                    integrate_photon_dopri5_impl=integrate_photon_dopri5_impl,
                    dt_fine=dt_fine,
                    lambda_total=lambda_total,
                    record_every=record_every,
                    adaptive_kwargs=adaptive_kwargs,
                )

            state0 = _pack_lensing_state(photon)
            traj, final, lam, steps = numba_backend.integrate_rk4(
                state0,
                dt_fine,
                n_steps_budget,
                lambda_stop=lambda_total,
                record_every=record_every,
            )
            _unpack_lensing_state(photon, final, lam)
            _append_photon_history(photon, traj)
            return (
                lam,
                int(steps),
                _specialized_stop_reason(lam, lambda_total, steps, 0, n_steps_budget),
                0,
                0,
                abs(float(dt_fine)),
                float(dt_fine),
                float(dt_fine),
            )

        _, lam, _ = integrate_photon_impl(
            photon,
            numba_backend,
            dt=dt_fine,
            n_steps=n_steps_budget,
            lambda_stop=lambda_total,
            record_every=record_every,
        )
        steps_actual = int(getattr(photon, "integration_n_steps_actual", -1))
        stop_reason = int(getattr(photon, "integration_stop_reason", STOP_REASON_UNKNOWN))
        rejected_total = int(getattr(photon, "integration_rejected_total", 0))
        rejected_streak_peak = int(getattr(photon, "integration_rejected_streak_peak", 0))
        dt_min_attempted = float(getattr(photon, "integration_dt_min_attempted", abs(dt_fine)))
        dt_last_attempted = float(getattr(photon, "integration_dt_last_attempted", dt_fine))
        dt_last_suggested = float(getattr(photon, "integration_dt_last_suggested", dt_fine))
        return (
            lam,
            steps_actual,
            stop_reason,
            rejected_total,
            rejected_streak_peak,
            dt_min_attempted,
            dt_last_attempted,
            dt_last_suggested,
        )

    integrator_fine.integrate_single(
        photon,
        stop_mode="affine",
        stop_value=lambda_total,
        record_every=record_every,
    )
    photon.integration_n_steps_actual = -1
    photon.integration_stop_reason = STOP_REASON_PYTHON_BACKEND
    return (
        photon.lambda_affine,
        -1,
        STOP_REASON_PYTHON_BACKEND,
        -1,
        -1,
        np.nan,
        np.nan,
        np.nan,
    )


def make_photon(obs_pos, target, metric, eta_0, a_0, *, record_lensing):
    obs_4d = np.array([eta_0, *obs_pos])
    direction = target - obs_pos
    direction /= np.linalg.norm(direction)
    screen_convention = getattr(metric, "sachs_screen_convention", "metric")

    g = metric.metric_tensor(obs_4d)
    if screen_convention == "conformal_metric":
        g_init = g / (a_0 * a_0)
        basis_a = 1.0
    else:
        g_init = g
        basis_a = a_0

    k_spatial = direction * c
    spatial_sq = (
        g_init[1, 1] * k_spatial[0] ** 2
        + g_init[2, 2] * k_spatial[1] ** 2
        + g_init[3, 3] * k_spatial[2] ** 2
    )
    k0 = -np.sqrt(abs(-spatial_sq / g_init[0, 0]))
    k_mu = np.array([k0, *k_spatial])

    e1, e2 = init_sachs_basis(k_mu, g_init, basis_a, convention=screen_convention)

    photon = Photon(obs_4d.copy(), k_mu.copy(), record_lensing=record_lensing)
    photon.e1 = e1.copy()
    photon.e2 = e2.copy()
    return photon


def _build_common_targets(center, e_perp1, e_perp2, *, R200_Mpc, rs_Mpc, map_half_Mpc, map_n1d, profile_nphi):
    b_E_approx = 0.46
    b_values_Mpc = np.unique(
        np.sort(
            np.concatenate(
                [
                    np.array([0.0]),
                    np.linspace(0.005, 0.05, 12),
                    np.linspace(0.05, 0.10, 6),
                    np.linspace(0.10, b_E_approx * 1.5, 24),
                    np.linspace(b_E_approx * 1.5, rs_Mpc, 8),
                    np.linspace(rs_Mpc + 0.1, R200_Mpc, 12),
                    np.linspace(R200_Mpc + 0.2, 5.0, 8),
                    np.logspace(np.log10(0.005), np.log10(15.0), 30),
                ]
            )
        )
    )
    b_values = b_values_Mpc * one_Mpc

    target_profile_Mpc = []
    target_profile_samples_Mpc = []
    profile_sample_radius_index = []
    profile_n_samples_by_radius = []
    profile_sample_angles = []
    profile_sample_b1 = []
    profile_sample_b2 = []

    if profile_nphi <= 0:
        raise ValueError("--profile-nphi must be strictly positive")

    for radius_index, b in enumerate(b_values):
        b_mpc = b / one_Mpc
        target_profile_Mpc.append((center + b * e_perp1) / one_Mpc)
        if np.isclose(b_mpc, 0.0):
            angles = np.array([0.0], dtype=np.float64)
        else:
            angles = np.linspace(0.0, 2.0 * np.pi, profile_nphi, endpoint=False, dtype=np.float64)
        profile_n_samples_by_radius.append(int(angles.size))
        profile_sample_angles.append(angles)
        profile_sample_b1.append(b_mpc * np.cos(angles))
        profile_sample_b2.append(b_mpc * np.sin(angles))

        for angle in angles:
            offset_dir = np.cos(angle) * e_perp1 + np.sin(angle) * e_perp2
            target_profile_samples_Mpc.append((center + b * offset_dir) / one_Mpc)
            profile_sample_radius_index.append(radius_index)

    max_profile_samples = max(profile_n_samples_by_radius)
    profile_sample_angle_rad = np.full((len(b_values_Mpc), max_profile_samples), np.nan, dtype=np.float64)
    profile_sample_b1_Mpc = np.full((len(b_values_Mpc), max_profile_samples), np.nan, dtype=np.float64)
    profile_sample_b2_Mpc = np.full((len(b_values_Mpc), max_profile_samples), np.nan, dtype=np.float64)
    for radius_index, count in enumerate(profile_n_samples_by_radius):
        profile_sample_angle_rad[radius_index, :count] = profile_sample_angles[radius_index]
        profile_sample_b1_Mpc[radius_index, :count] = profile_sample_b1[radius_index]
        profile_sample_b2_Mpc[radius_index, :count] = profile_sample_b2[radius_index]

    b1_arr = np.linspace(-map_half_Mpc, map_half_Mpc, map_n1d) * one_Mpc
    b2_arr = np.linspace(-map_half_Mpc, map_half_Mpc, map_n1d) * one_Mpc

    target_map_Mpc = []
    map_b1_Mpc = []
    map_b2_Mpc = []
    for b1 in b1_arr:
        for b2 in b2_arr:
            target_map_Mpc.append((center + b1 * e_perp1 + b2 * e_perp2) / one_Mpc)
            map_b1_Mpc.append(b1 / one_Mpc)
            map_b2_Mpc.append(b2 / one_Mpc)

    return {
        "b_profile_Mpc": b_values_Mpc,
        "target_profile_Mpc": np.array(target_profile_Mpc, dtype=np.float64),
        "target_profile_samples_Mpc": np.array(target_profile_samples_Mpc, dtype=np.float64),
        "profile_sample_radius_index": np.array(profile_sample_radius_index, dtype=np.int32),
        "profile_n_samples_by_radius": np.array(profile_n_samples_by_radius, dtype=np.int32),
        "profile_sample_angle_rad": profile_sample_angle_rad,
        "profile_sample_b1_Mpc": profile_sample_b1_Mpc,
        "profile_sample_b2_Mpc": profile_sample_b2_Mpc,
        "b1_map_Mpc": np.array(map_b1_Mpc, dtype=np.float64),
        "b2_map_Mpc": np.array(map_b2_Mpc, dtype=np.float64),
        "target_map_Mpc": np.array(target_map_Mpc, dtype=np.float64),
        "map_n1d": map_n1d,
        "map_half_Mpc": map_half_Mpc,
    }


def _prepare_profile_sources(
    profile_presets,
    *,
    total_mass_msun,
    c_nfw,
    center,
    dir_hat,
    e_perp1,
    e_perp2,
    shape_scale_mpc,
    axis_scale_xyz,
    orientation_euler_deg,
    cigar_axis,
    cigar_core_fraction,
    cigar_satellite_concentration_scale,
    custom_component_specs,
    triaxial_axis_ratios,
    triaxial_quadrature_order,
):
    return [
        build_equivalent_mass_profile(
            preset_name,
            total_mass_msun=total_mass_msun,
            c_nfw=c_nfw,
            center=center,
            los_dir=dir_hat,
            e_perp1=e_perp1,
            e_perp2=e_perp2,
            shape_scale_mpc=shape_scale_mpc,
            axis_scale_xyz=axis_scale_xyz,
            orientation_euler_deg=orientation_euler_deg,
            cigar_axis=cigar_axis,
            cigar_core_fraction=cigar_core_fraction,
            cigar_satellite_concentration_scale=cigar_satellite_concentration_scale,
            custom_component_specs=custom_component_specs,
            triaxial_axis_ratios=triaxial_axis_ratios,
            triaxial_quadrature_order=triaxial_quadrature_order,
        )
        for preset_name in profile_presets
    ]


def _collect_trajectory_payload(results_per_profile, n_total):
    max_pts = 0
    for result in results_per_profile:
        if result["traj_n_pts"] is not None:
            max_pts = max(max_pts, int(result["traj_n_pts"].max()))

    traj_x4 = np.full((len(results_per_profile), n_total, max_pts, 4), np.nan, dtype=np.float32)
    traj_D = np.full((len(results_per_profile), n_total, max_pts, 4), np.nan, dtype=np.float32)
    traj_P = np.full((len(results_per_profile), n_total, max_pts, 4), np.nan, dtype=np.float32)
    traj_n_pts = np.zeros((len(results_per_profile), n_total), dtype=np.int32)

    for i, result in enumerate(results_per_profile):
        pts = result["traj_n_pts"]
        traj_n_pts[i] = pts
        for j in range(n_total):
            n = int(pts[j])
            traj_x4[i, j, :n] = result["traj_x4_Mpc"][j, :n]
            traj_D[i, j, :n] = result["traj_D_flat"][j, :n]
            traj_P[i, j, :n] = result["traj_P_flat"][j, :n]

    return traj_x4, traj_D, traj_P, traj_n_pts, max_pts


def _group_profile_samples(values, profile_sample_radius_index, profile_n_samples_by_radius):
    values = np.asarray(values, dtype=np.float64)
    radius_index = np.asarray(profile_sample_radius_index, dtype=np.int32)
    counts = np.asarray(profile_n_samples_by_radius, dtype=np.int32)
    n_radii = counts.shape[0]
    max_samples = int(counts.max())
    grouped = np.full((n_radii, max_samples) + values.shape[1:], np.nan, dtype=np.float64)
    fill_count = np.zeros(n_radii, dtype=np.int32)

    for sample_index, radius_idx in enumerate(radius_index):
        slot = int(fill_count[radius_idx])
        grouped[(radius_idx, slot)] = values[sample_index]
        fill_count[radius_idx] += 1

    return grouped


def _group_profile_samples_int(values, profile_sample_radius_index, profile_n_samples_by_radius, *, fill_value=-1):
    values = np.asarray(values, dtype=np.int32)
    radius_index = np.asarray(profile_sample_radius_index, dtype=np.int32)
    counts = np.asarray(profile_n_samples_by_radius, dtype=np.int32)
    n_radii = counts.shape[0]
    max_samples = int(counts.max())
    grouped = np.full((n_radii, max_samples) + values.shape[1:], fill_value, dtype=np.int32)
    fill_count = np.zeros(n_radii, dtype=np.int32)

    for sample_index, radius_idx in enumerate(radius_index):
        slot = int(fill_count[radius_idx])
        grouped[(radius_idx, slot)] = values[sample_index]
        fill_count[radius_idx] += 1

    return grouped


def _mean_and_std_from_grouped(grouped, profile_n_samples_by_radius):
    counts = np.asarray(profile_n_samples_by_radius, dtype=np.int32)
    n_radii = counts.shape[0]
    mean = np.empty((n_radii,) + grouped.shape[2:], dtype=np.float64)
    std = np.empty((n_radii,) + grouped.shape[2:], dtype=np.float64)

    for radius_index, count in enumerate(counts):
        samples = grouped[radius_index, :count]
        finite = np.isfinite(samples)
        finite_count = np.sum(finite, axis=0)
        sample_sum = np.sum(np.where(finite, samples, 0.0), axis=0)
        mean_row = np.full(samples.shape[1:], np.nan, dtype=np.float64)
        np.divide(sample_sum, finite_count, out=mean_row, where=finite_count > 0)

        centered = np.where(finite, samples - mean_row, 0.0)
        variance = np.full(samples.shape[1:], np.nan, dtype=np.float64)
        np.divide(
            np.sum(centered * centered, axis=0),
            finite_count,
            out=variance,
            where=finite_count > 0,
        )

        mean[radius_index] = mean_row
        std[radius_index] = np.sqrt(variance)

    return mean, std


def _nanabsmax(values):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.nan
    return float(np.nanmax(np.abs(values[finite])))


def _run_one_profile(
    source,
    *,
    args,
    root_grid,
    base_interp,
    a_of_eta,
    cosmo,
    eta_0,
    eta_min,
    a_0,
    obs_pos,
    targets_profile_samples_Mpc,
    profile_sample_radius_index,
    profile_n_samples_by_radius,
    targets_map_Mpc,
    dt_fine,
    lambda_total,
    traj_stride,
    n_steps_total,
    bardeen_a_lens=None,
):
    bypass_radius = 1e3 * args.box_mpc * one_Mpc
    interp = AnalyticalBypassInterpolator(
        base_interp=base_interp,
        analytical_source=source,
        bypass_radius=bypass_radius,
        bypass_fields=("Phi",),
        time_derivative=0.0,
    )
    metric = PerturbedFLRWMetricFast(
        a_of_eta=a_of_eta,
        grid=root_grid,
        interpolator=interp,
        adot_of_eta=cosmo.adot_of_eta,
        cosmology=cosmo,
        enable_lensing=True,
        slow_roll=True,
        analytical_geodesics=True,
        sachs_screen_convention="conformal_metric",
    )

    integrator_fine = None
    numba_backend = None
    integrate_photon_impl = None
    integrate_photon_dopri5_impl = None
    integrate_photons_batch_impl = None
    adaptive_kwargs = None
    if args.backend == "python":
        integrator_fine = Integrator(
            metric=metric,
            dt=dt_fine,
            mode="sequential",
            integrator=args.python_integrator,
            rtol=args.rtol,
            atol=args.atol,
        )
    else:
        (
            numba_backend_cls,
            integrate_photon_impl,
            integrate_photon_dopri5_impl,
            integrate_photons_batch_impl,
        ) = _select_numba_backend(args.numba_kernel)
        analytical_amr = AMRGrid(root_grid)
        dt_min_override = None
        if _integrator_is_adaptive(args.integrator):
            dt_min_override = _effective_numba_dt_min(dt_fine, args.integrator, args.dt_min_factor)
        numba_backend = numba_backend_cls(
            analytical_amr,
            cosmo,
            c_val=c,
            slow_roll=True,
            lensing=True,
            sachs_screen_convention=getattr(metric, "sachs_screen_convention", "metric"),
            analytical_source=source,
            bypass_radius=bypass_radius,
            integrator=args.integrator,
            rtol=args.rtol,
            atol=args.atol,
            dt_min=dt_min_override,
            max_rejected=args.max_rejected,
            eta_range=(eta_min, eta_0),
            bardeen_a_lens=bardeen_a_lens,
        )
        numba_backend.warmup()
        if args.numba_kernel == "specialized" and args.integrator == "dopri5":
            adaptive_kwargs = {
                "rtol": float(args.rtol),
                "atol": float(args.atol),
                "dt_min": _effective_numba_dt_min(dt_fine, args.integrator, args.dt_min_factor),
                "dt_max": _effective_numba_dt_max(dt_fine, args.integrator),
                "max_steps": max(4 * n_steps_total, 1024),
            }

    photons_profile = [
        make_photon(
            obs_pos,
            target_mpc * one_Mpc,
            metric,
            eta_0,
            a_0,
            record_lensing=args.save_trajectories,
        )
        for target_mpc in targets_profile_samples_Mpc
    ]
    photons_map = [
        make_photon(
            obs_pos,
            target_mpc * one_Mpc,
            metric,
            eta_0,
            a_0,
            record_lensing=args.save_trajectories,
        )
        for target_mpc in targets_map_Mpc
    ]
    all_photons = photons_profile + photons_map
    n_profile_samples = len(photons_profile)
    n_profile = len(profile_n_samples_by_radius)
    n_total = len(all_photons)
    initial_states = None
    if args.backend == "numba" and args.numba_kernel == "specialized" and args.integrator == "dopri5":
        initial_states = [_pack_lensing_state(photon) for photon in all_photons]

    kappas = np.empty(n_total)
    gammas = np.empty(n_total)
    mus = np.empty(n_total)
    D_flats = np.empty((n_total, 4))
    initial_sachs_e1 = np.full((n_total, 4), np.nan, dtype=np.float64)
    initial_sachs_e2 = np.full((n_total, 4), np.nan, dtype=np.float64)
    final_sachs_e1 = np.full((n_total, 4), np.nan, dtype=np.float64)
    final_sachs_e2 = np.full((n_total, 4), np.nan, dtype=np.float64)
    final_k_mu = np.full((n_total, 4), np.nan, dtype=np.float64)
    final_pos = np.empty((n_total, 3))
    lambda_actuals = np.empty(n_total)
    source_reached = np.zeros(n_total, dtype=bool)
    steps_actuals = np.full(n_total, -1, dtype=np.int32)
    stop_reasons = np.full(n_total, STOP_REASON_UNKNOWN, dtype=np.int32)
    rejected_totals = np.full(n_total, -1, dtype=np.int32)
    rejected_streak_peaks = np.full(n_total, -1, dtype=np.int32)
    dt_min_attempteds = np.full(n_total, np.nan)
    dt_last_attempteds = np.full(n_total, np.nan)
    dt_last_suggesteds = np.full(n_total, np.nan)

    for photon_index, photon in enumerate(all_photons):
        if hasattr(photon, "e1"):
            initial_sachs_e1[photon_index] = np.asarray(photon.e1, dtype=np.float64)
        if hasattr(photon, "e2"):
            initial_sachs_e2[photon_index] = np.asarray(photon.e2, dtype=np.float64)

    def _store_photon_result(
        i,
        lam,
        steps_actual,
        stop_reason,
        rejected_total,
        rejected_streak_peak,
        dt_min_attempted,
        dt_last_attempted,
        dt_last_suggested,
    ):
        photon = all_photons[i]
        final_pos[i] = photon.x[1:4]
        if hasattr(photon, "e1"):
            final_sachs_e1[i] = np.asarray(photon.e1, dtype=np.float64)
        if hasattr(photon, "e2"):
            final_sachs_e2[i] = np.asarray(photon.e2, dtype=np.float64)
        if hasattr(photon, "u"):
            final_k_mu[i] = np.asarray(photon.u, dtype=np.float64)
        lambda_actuals[i] = lam
        steps_actuals[i] = steps_actual
        stop_reasons[i] = stop_reason
        rejected_totals[i] = rejected_total
        rejected_streak_peaks[i] = rejected_streak_peak
        dt_min_attempteds[i] = dt_min_attempted
        dt_last_attempteds[i] = dt_last_attempted
        dt_last_suggesteds[i] = dt_last_suggested
        reached_source = _photon_reached_source(lam, stop_reason, lambda_total)
        source_reached[i] = reached_source

        if reached_source and lam > 0.0 and np.all(np.isfinite(photon.D_flat)):
            D_norm = photon.D_flat / lam
            kappa, mu, shear = lensing_from_jacobi(D_norm)
            kappas[i] = kappa
            mus[i] = mu
            gammas[i] = shear
            D_flats[i] = D_norm
            return kappa, shear

        kappas[i] = np.nan
        mus[i] = np.nan
        gammas[i] = np.nan
        D_flats[i] = np.nan
        return np.nan, np.nan

    t_int = time.time()
    print(f"\n   Profile '{source.label}' ...")
    if args.parallel_batch and adaptive_kwargs is not None and integrate_photons_batch_impl is not None:
        max_steps_batch = adaptive_kwargs["max_steps"]
        map_record_every = traj_stride if args.save_trajectories else 0
        traj_capacity_profile = 2
        traj_capacity_map = 2 if map_record_every <= 0 else 2 + max_steps_batch // max(map_record_every, 1) + 2

        profile_lams = np.empty(0, dtype=np.float64)
        profile_n_accepts = np.empty(0, dtype=np.int64)
        profile_n_rejects = np.empty(0, dtype=np.int64)
        if n_profile_samples > 0:
            _, _, profile_lams, profile_n_accepts, profile_n_rejects = integrate_photons_batch_impl(
                photons_profile,
                numba_backend,
                dt_init=dt_fine,
                lambda_stop=lambda_total,
                rtol=adaptive_kwargs["rtol"],
                atol=adaptive_kwargs["atol"],
                dt_min=adaptive_kwargs["dt_min"],
                dt_max=adaptive_kwargs["dt_max"],
                max_steps=max_steps_batch,
                record_every=0,
                traj_capacity=traj_capacity_profile,
            )

        map_lams = np.empty(0, dtype=np.float64)
        map_n_accepts = np.empty(0, dtype=np.int64)
        map_n_rejects = np.empty(0, dtype=np.int64)
        if photons_map:
            _, _, map_lams, map_n_accepts, map_n_rejects = integrate_photons_batch_impl(
                photons_map,
                numba_backend,
                dt_init=dt_fine,
                lambda_stop=lambda_total,
                rtol=adaptive_kwargs["rtol"],
                atol=adaptive_kwargs["atol"],
                dt_min=adaptive_kwargs["dt_min"],
                dt_max=adaptive_kwargs["dt_max"],
                max_steps=max_steps_batch,
                record_every=map_record_every,
                traj_capacity=traj_capacity_map,
            )

        for i in range(n_profile_samples):
            lam_i = float(profile_lams[i])
            steps_i = int(profile_n_accepts[i])
            rejected_i = int(profile_n_rejects[i])
            stop_reason_i = _specialized_stop_reason(
                lam_i,
                lambda_total,
                steps_i,
                rejected_i,
                max_steps_batch,
            )
            if not _photon_reached_source(lam_i, stop_reason_i, lambda_total):
                (
                    lam_i,
                    steps_i,
                    stop_reason_i,
                    rejected_i,
                    _rejected_streak_peak_i,
                    dt_min_attempted_i,
                    dt_last_attempted_i,
                    dt_last_suggested_i,
                ) = _integrate_specialized_dopri5_with_retry(
                    photons_profile[i],
                    initial_state=initial_states[i],
                    numba_backend=numba_backend,
                    integrate_photon_dopri5_impl=integrate_photon_dopri5_impl,
                    dt_fine=dt_fine,
                    lambda_total=lambda_total,
                    record_every=0,
                    adaptive_kwargs=adaptive_kwargs,
                )
            else:
                dt_min_attempted_i = np.nan
                dt_last_attempted_i = np.nan
                dt_last_suggested_i = np.nan
            kappa, shear = _store_photon_result(
                i,
                lam_i,
                steps_i,
                stop_reason_i,
                rejected_i,
                -1,
                dt_min_attempted_i,
                dt_last_attempted_i,
                dt_last_suggested_i,
            )

        for j in range(len(photons_map)):
            i = n_profile_samples + j
            lam_j = float(map_lams[j])
            steps_j = int(map_n_accepts[j])
            rejected_j = int(map_n_rejects[j])
            stop_reason_j = _specialized_stop_reason(
                lam_j,
                lambda_total,
                steps_j,
                rejected_j,
                max_steps_batch,
            )
            if not _photon_reached_source(lam_j, stop_reason_j, lambda_total):
                (
                    lam_j,
                    steps_j,
                    stop_reason_j,
                    rejected_j,
                    _rejected_streak_peak_j,
                    dt_min_attempted_j,
                    dt_last_attempted_j,
                    dt_last_suggested_j,
                ) = _integrate_specialized_dopri5_with_retry(
                    photons_map[j],
                    initial_state=initial_states[i],
                    numba_backend=numba_backend,
                    integrate_photon_dopri5_impl=integrate_photon_dopri5_impl,
                    dt_fine=dt_fine,
                    lambda_total=lambda_total,
                    record_every=map_record_every,
                    adaptive_kwargs=adaptive_kwargs,
                )
            else:
                dt_min_attempted_j = np.nan
                dt_last_attempted_j = np.nan
                dt_last_suggested_j = np.nan
            kappa, shear = _store_photon_result(
                i,
                lam_j,
                steps_j,
                stop_reason_j,
                rejected_j,
                -1,
                dt_min_attempted_j,
                dt_last_attempted_j,
                dt_last_suggested_j,
            )

        elapsed = time.time() - t_int
        print(
            f"      [{n_total:4d}/{n_total}] batch done in {elapsed:.2f}s "
            f"({elapsed / max(n_total, 1) * 1000:.2f} ms/photon)"
        )
    else:
        for i, photon in enumerate(all_photons):
            record_every = traj_stride if args.save_trajectories and i >= n_profile_samples else 0
            (
                lam,
                steps_actual,
                stop_reason,
                rejected_total,
                rejected_streak_peak,
                dt_min_attempted,
                dt_last_attempted,
                dt_last_suggested,
            ) = _integrate_one_photon(
                photon,
                args.backend,
                numba_kernel=args.numba_kernel,
                integrator_fine=integrator_fine,
                numba_backend=numba_backend,
                integrate_photon_impl=integrate_photon_impl,
                integrate_photon_dopri5_impl=integrate_photon_dopri5_impl,
                dt_fine=dt_fine,
                lambda_total=lambda_total,
                n_steps_budget=n_steps_total,
                record_every=record_every,
                adaptive_kwargs=adaptive_kwargs,
            )

            kappa, shear = _store_photon_result(
                i,
                lam,
                steps_actual,
                stop_reason,
                rejected_total,
                rejected_streak_peak,
                dt_min_attempted,
                dt_last_attempted,
                dt_last_suggested,
            )

            if args.progress_every > 0 and (i % args.progress_every == 0 or i == 0 or i == n_total - 1):
                elapsed = time.time() - t_int
                rate = (i + 1) / elapsed if elapsed > 0 else 0.0
                eta = (n_total - i - 1) / rate if rate > 0 else 0.0
                print(
                    f"      [{i+1:4d}/{n_total}]  kappa = {kappa:+.6e}  |gamma| = {shear:.3e}  "
                    f"({elapsed:.0f}s, ~{eta:.0f}s left)"
                )

    kappa_profile_samples = _group_profile_samples(
        kappas[:n_profile_samples],
        profile_sample_radius_index,
        profile_n_samples_by_radius,
    )
    gamma_profile_samples = _group_profile_samples(
        gammas[:n_profile_samples],
        profile_sample_radius_index,
        profile_n_samples_by_radius,
    )
    mu_profile_samples = _group_profile_samples(
        mus[:n_profile_samples],
        profile_sample_radius_index,
        profile_n_samples_by_radius,
    )
    D_flat_profile_samples = _group_profile_samples(
        D_flats[:n_profile_samples],
        profile_sample_radius_index,
        profile_n_samples_by_radius,
    )
    lambda_actual_profile_samples = _group_profile_samples(
        lambda_actuals[:n_profile_samples],
        profile_sample_radius_index,
        profile_n_samples_by_radius,
    )
    source_reached_profile_samples = _group_profile_samples_int(
        source_reached[:n_profile_samples].astype(np.int32),
        profile_sample_radius_index,
        profile_n_samples_by_radius,
        fill_value=1,
    ).astype(bool)
    rejected_total_profile_samples = _group_profile_samples_int(
        rejected_totals[:n_profile_samples],
        profile_sample_radius_index,
        profile_n_samples_by_radius,
    )
    rejected_streak_peak_profile_samples = _group_profile_samples_int(
        rejected_streak_peaks[:n_profile_samples],
        profile_sample_radius_index,
        profile_n_samples_by_radius,
    )
    dt_min_attempted_profile_samples = _group_profile_samples(
        dt_min_attempteds[:n_profile_samples],
        profile_sample_radius_index,
        profile_n_samples_by_radius,
    )
    dt_last_attempted_profile_samples = _group_profile_samples(
        dt_last_attempteds[:n_profile_samples],
        profile_sample_radius_index,
        profile_n_samples_by_radius,
    )
    dt_last_suggested_profile_samples = _group_profile_samples(
        dt_last_suggesteds[:n_profile_samples],
        profile_sample_radius_index,
        profile_n_samples_by_radius,
    )
    steps_actual_profile_samples = _group_profile_samples_int(
        steps_actuals[:n_profile_samples],
        profile_sample_radius_index,
        profile_n_samples_by_radius,
    )
    stop_reason_profile_samples = _group_profile_samples_int(
        stop_reasons[:n_profile_samples],
        profile_sample_radius_index,
        profile_n_samples_by_radius,
        fill_value=STOP_REASON_UNKNOWN,
    )
    final_pos_profile_samples_Mpc = _group_profile_samples(
        final_pos[:n_profile_samples] / one_Mpc,
        profile_sample_radius_index,
        profile_n_samples_by_radius,
    )

    kappa_profile, kappa_profile_std = _mean_and_std_from_grouped(
        kappa_profile_samples,
        profile_n_samples_by_radius,
    )
    gamma_profile, gamma_profile_std = _mean_and_std_from_grouped(
        gamma_profile_samples,
        profile_n_samples_by_radius,
    )
    mu_profile, mu_profile_std = _mean_and_std_from_grouped(
        mu_profile_samples,
        profile_n_samples_by_radius,
    )
    D_flat_profile, _ = _mean_and_std_from_grouped(
        D_flat_profile_samples,
        profile_n_samples_by_radius,
    )
    lambda_actual_profile, _ = _mean_and_std_from_grouped(
        lambda_actual_profile_samples,
        profile_n_samples_by_radius,
    )
    final_pos_profile_Mpc, _ = _mean_and_std_from_grouped(
        final_pos_profile_samples_Mpc,
        profile_n_samples_by_radius,
    )
    n_failed_profile = int(np.count_nonzero(~source_reached[:n_profile_samples]))
    n_failed_map = int(np.count_nonzero(~source_reached[n_profile_samples:]))
    if n_failed_profile or n_failed_map:
        print(
            f"      excluded {n_failed_profile} profile sample(s) and {n_failed_map} map photon(s) "
            "that did not reach the source plane"
        )

    result = {
        "kappa_profile": kappa_profile,
        "kappa_profile_std": kappa_profile_std,
        "gamma_profile": gamma_profile,
        "gamma_profile_std": gamma_profile_std,
        "mu_profile": mu_profile,
        "mu_profile_std": mu_profile_std,
        "kappa_profile_samples": kappa_profile_samples,
        "gamma_profile_samples": gamma_profile_samples,
        "mu_profile_samples": mu_profile_samples,
        "kappa_map": kappas[n_profile_samples:],
        "gamma_map": gammas[n_profile_samples:],
        "mu_map": mus[n_profile_samples:],
        "D_flat_profile": D_flat_profile,
        "D_flat_map": D_flats[n_profile_samples:],
        "initial_sachs_e1_map": initial_sachs_e1[n_profile_samples:],
        "initial_sachs_e2_map": initial_sachs_e2[n_profile_samples:],
        "final_sachs_e1_map": final_sachs_e1[n_profile_samples:],
        "final_sachs_e2_map": final_sachs_e2[n_profile_samples:],
        "final_k_mu_map": final_k_mu[n_profile_samples:],
        "lambda_actual_profile": lambda_actual_profile,
        "lambda_actual_profile_samples": lambda_actual_profile_samples,
        "lambda_actual_map": lambda_actuals[n_profile_samples:],
        "source_reached_profile_samples": source_reached_profile_samples,
        "source_reached_map": source_reached[n_profile_samples:],
        "steps_actual_profile_samples": steps_actual_profile_samples,
        "steps_actual_map": steps_actuals[n_profile_samples:],
        "stop_reason_profile_samples": stop_reason_profile_samples,
        "stop_reason_map": stop_reasons[n_profile_samples:],
        "rejected_total_profile_samples": rejected_total_profile_samples,
        "rejected_total_map": rejected_totals[n_profile_samples:],
        "rejected_streak_peak_profile_samples": rejected_streak_peak_profile_samples,
        "rejected_streak_peak_map": rejected_streak_peaks[n_profile_samples:],
        "dt_min_attempted_profile_samples": dt_min_attempted_profile_samples,
        "dt_min_attempted_map": dt_min_attempteds[n_profile_samples:],
        "dt_last_attempted_profile_samples": dt_last_attempted_profile_samples,
        "dt_last_attempted_map": dt_last_attempteds[n_profile_samples:],
        "dt_last_suggested_profile_samples": dt_last_suggested_profile_samples,
        "dt_last_suggested_map": dt_last_suggesteds[n_profile_samples:],
        "final_pos_profile_Mpc": final_pos_profile_Mpc,
        "final_pos_map_Mpc": final_pos[n_profile_samples:] / one_Mpc,
        "traj_x4_Mpc": None,
        "traj_D_flat": None,
        "traj_P_flat": None,
        "traj_n_pts": None,
    }

    if args.save_trajectories:
        raw_trajs = [
            np.array([[s[0], *s[1:4] / one_Mpc] for s in photon.history.states], dtype=np.float32)
            for photon in all_photons
        ]
        raw_D = [np.array([s[8:12] for s in photon.history.states], dtype=np.float32) for photon in all_photons]
        raw_P = [np.array([s[12:16] for s in photon.history.states], dtype=np.float32) for photon in all_photons]
        traj_n_pts = np.array([traj.shape[0] for traj in raw_trajs], dtype=np.int32)
        max_pts = int(traj_n_pts.max())

        traj_x4_Mpc = np.full((n_total, max_pts, 4), np.nan, dtype=np.float32)
        traj_D_flat = np.full((n_total, max_pts, 4), np.nan, dtype=np.float32)
        traj_P_flat = np.full((n_total, max_pts, 4), np.nan, dtype=np.float32)
        for i in range(n_total):
            n = traj_n_pts[i]
            traj_x4_Mpc[i, :n] = raw_trajs[i]
            traj_D_flat[i, :n] = raw_D[i]
            traj_P_flat[i, :n] = raw_P[i]

        result["traj_x4_Mpc"] = traj_x4_Mpc
        result["traj_D_flat"] = traj_D_flat
        result["traj_P_flat"] = traj_P_flat
        result["traj_n_pts"] = traj_n_pts

    print(
        f"      done in {time.time() - t_int:.1f}s  ({(time.time() - t_int) / max(n_total, 1):.2f}s/photon)"
    )
    return result


def main():
    args = parse_args()
    args.integrator = _normalize_runner_integrator(args.integrator)
    args.python_integrator = _python_integrator_name(args.integrator)
    if args.list_presets:
        print("Available equivalent-mass presets:")
        for name in available_equivalent_mass_presets():
            print(f"  - {name}")
        print("Preset suites:")
        print("  - science_ready_triaxial")
        return

    profile_presets = _expand_profile_presets(args.profile_presets)
    reference_preset = _resolve_reference_preset(profile_presets, args.reference_preset)
    custom_component_specs = parse_component_specs(args.custom_components or [])

    if args.backend == "numba" and args.integrator == "dop853":
        raise ValueError("The numba backend supports rk4, rk45, and dopri5, but not dop853.")
    if args.backend == "numba" and args.numba_kernel == "specialized" and args.integrator not in {"rk4", "dopri5"}:
        raise ValueError("--numba-kernel=specialized supports only --integrator=rk4 or --integrator=dopri5.")
    if args.parallel_batch and not (
        args.backend == "numba"
        and args.numba_kernel == "specialized"
        and args.integrator == "dopri5"
    ):
        raise ValueError(
            "--parallel-batch requires --backend=numba --numba-kernel=specialized --integrator=dopri5."
        )

    t_total = time.time()

    print("=" * 70)
    print("  EQUIVALENT-MASS LENSING  --  ANALYTICAL COMPARISON")
    print("=" * 70)
    print("\n1. Cosmology ...")
    print(f"   backend = {args.backend}")
    if args.backend == "numba":
        print(f"   numba kernel = {args.numba_kernel}")
    print(f"   integrator = {args.integrator}")
    if args.parallel_batch:
        print("   parallel-batch = on (numba.prange over photons)")
    print(f"   profiles = {', '.join(profile_presets)}")
    print(f"   reference = {reference_preset}")
    print(f"   profile_nphi = {args.profile_nphi}")
    if not np.allclose(args.profile_axis_scale, 1.0):
        print(
            "   local axis scale = "
            f"({args.profile_axis_scale[0]:.3f}, {args.profile_axis_scale[1]:.3f}, {args.profile_axis_scale[2]:.3f})"
        )
    if not np.allclose(args.profile_orientation_deg, 0.0):
        print(
            "   local orientation (Rx, Ry, Rz) [deg] = "
            f"({args.profile_orientation_deg[0]:.1f}, {args.profile_orientation_deg[1]:.1f}, {args.profile_orientation_deg[2]:.1f})"
        )
    if "cigar_parametric" in profile_presets:
        print(
            f"   cigar_parametric axis = {args.cigar_axis}, core fraction = {args.cigar_core_fraction:.3f}, "
            f"satellite c-scale = {args.cigar_satellite_concentration_scale:.3f}"
        )
    if "custom_components" in profile_presets:
        print(f"   custom components = {len(custom_component_specs)}")

    if args.z_lens is not None:
        print(f"   requested z_l = {args.z_lens:.4f}, requested z_s = {args.z_source:.4f}")

    H0 = 70.0
    Omega_m, Omega_lambda = 0.3, 0.7
    cosmo = LCDM_Cosmology(H0, Omega_m=Omega_m, Omega_r=0, Omega_lambda=Omega_lambda)

    explicit_redshift_geometry = args.z_lens is not None
    if explicit_redshift_geometry:
        if args.z_source <= args.z_lens:
            raise ValueError("--z-source must be larger than --z-lens for a lensing configuration")
        D_l_target = cosmo.comoving_distance(args.z_lens)
        D_s_target = cosmo.comoving_distance(args.z_source)
    else:
        D_l_target = None
        D_s_target = None

    _ = cosmo.a_of_eta(1e18)
    eta_0 = cosmo._eta_at_a1
    a_0 = cosmo.a_of_eta(eta_0)

    eta_min = 0.5 * eta_0
    eta_arr = np.linspace(eta_min, eta_0, 2000)
    a_arr = np.array([cosmo.a_of_eta(eta) for eta in eta_arr])
    a_of_eta = interpolate.interp1d(eta_arr, a_arr, kind="cubic", fill_value="extrapolate")
    print(f"   eta_0 = {eta_0:.4e} s,  a(eta_0) = {a_0:.6f}")

    print("2. Geometry + placeholder grid ...")
    box_Mpc = args.box_mpc
    if explicit_redshift_geometry:
        required_box_mpc = args.obs_z_mpc + D_s_target / one_Mpc + args.source_margin_mpc
        if box_Mpc < required_box_mpc:
            box_Mpc = required_box_mpc
            print(f"   auto-expanded box to {box_Mpc:.1f} Mpc for requested z_s")

    grid_size = box_Mpc * one_Mpc
    N_root = args.n_root
    root_grid = Grid(
        shape=(N_root, N_root, N_root),
        spacing=(grid_size / N_root,) * 3,
        origin=np.array([0.0, 0.0, 0.0]),
    )
    root_grid.add_field("Phi", np.zeros((N_root,) * 3))
    base_interp = InterpolatorFast(root_grid, boundary="clamp")

    M_200 = args.mass_200_msun * one_Msun
    c_NFW = args.c_nfw
    if explicit_redshift_geometry:
        center = np.array(
            [
                0.5 * box_Mpc,
                0.5 * box_Mpc,
                args.obs_z_mpc + D_l_target / one_Mpc,
            ]
        ) * one_Mpc
    else:
        center = np.array([0.5, 0.5, 0.5]) * grid_size

    reference_halo = NFWHalo(M_200, c_NFW, center)
    R200_Mpc = reference_halo.R_200 / one_Mpc
    rs_Mpc = reference_halo.r_s / one_Mpc
    print(f"   total target mass = {M_200 / one_Msun:.3e} Msun")
    print(f"   reference R_200 = {R200_Mpc:.3f} Mpc, r_s = {rs_Mpc:.4f} Mpc")

    print("3. Photon cone ...")
    obs_pos = np.array([box_Mpc / 2, box_Mpc / 2, args.obs_z_mpc]) * one_Mpc
    dir_to_center = center - obs_pos
    D_l = np.linalg.norm(dir_to_center)
    dir_hat = dir_to_center / D_l

    # Lens redshift (used to activate the Bardeen kernel rescaling, so that
    # the simulation produces kappa, gamma matching Bartelmann-Schneider 2001
    # analytical predictions at b_phys = a_l * b_co without post-processing).
    if explicit_redshift_geometry:
        z_l_for_bardeen = args.z_lens
    else:
        z_l_for_bardeen = brentq(
            lambda z: cosmo.comoving_distance(z) - D_l, 0.0, 5.0
        )
    bardeen_a_lens = 1.0 / (1.0 + z_l_for_bardeen)

    seed = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(seed, dir_hat)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    e_perp1 = seed - np.dot(seed, dir_hat) * dir_hat
    e_perp1 /= np.linalg.norm(e_perp1)
    e_perp2 = np.cross(dir_hat, e_perp1)
    e_perp2 /= np.linalg.norm(e_perp2)

    targets = _build_common_targets(
        center,
        e_perp1,
        e_perp2,
        R200_Mpc=R200_Mpc,
        rs_Mpc=rs_Mpc,
        map_half_Mpc=args.map_half_mpc,
        map_n1d=args.map_n1d,
        profile_nphi=args.profile_nphi,
    )
    n_profile = len(targets["b_profile_Mpc"])
    n_profile_samples = len(targets["target_profile_samples_Mpc"])
    n_map = len(targets["b1_map_Mpc"])
    n_total = n_profile_samples + n_map
    print(f"   D_l (obs -> barycenter) = {D_l / one_Mpc:.1f} Mpc")
    print(f"   Radial bins     : {n_profile}")
    print(f"   Profile photons : {n_profile_samples}")
    print(f"   Map photons     : {n_map}  ({args.map_n1d}x{args.map_n1d})")
    print(f"   Total / profile : {n_total}")

    print("4. Equivalent-mass profiles ...")
    shape_scale_mpc = args.shape_scale_rs * rs_Mpc
    sources = _prepare_profile_sources(
        profile_presets,
        total_mass_msun=M_200 / one_Msun,
        c_nfw=c_NFW,
        center=center,
        dir_hat=dir_hat,
        e_perp1=e_perp1,
        e_perp2=e_perp2,
        shape_scale_mpc=shape_scale_mpc,
        axis_scale_xyz=args.profile_axis_scale,
        orientation_euler_deg=args.profile_orientation_deg,
        cigar_axis=args.cigar_axis,
        cigar_core_fraction=args.cigar_core_fraction,
        cigar_satellite_concentration_scale=args.cigar_satellite_concentration_scale,
        custom_component_specs=custom_component_specs,
        triaxial_axis_ratios=args.triaxial_axis_ratios,
        triaxial_quadrature_order=args.triaxial_quadrature_order,
    )
    reference_index = profile_presets.index(reference_preset)
    if args.backend == "numba":
        unsupported_sources = [
            source.label
            for source in sources
            if not _source_supports_numba_path(
                source,
                kernel_name=args.numba_kernel,
                integrator_name=args.integrator,
            )
        ]
        if unsupported_sources:
            names = ", ".join(unsupported_sources)
            raise ValueError(
                "The current numba analytical bypass supports spherical/composite NFW components on all numba paths, "
                "and direct triaxial NFW profiles only on --numba-kernel specialized --integrator dopri5. "
                f"Unsupported profiles for the selected numba path: {names}"
            )
    for source in sources:
        axis_lengths = source.mass_weighted_axis_lengths() / one_Mpc
        components = _source_components_for_metadata(source)
        print(
            f"   {source.label:>16s}: {len(components)} component(s), "
            f"rms axes = ({axis_lengths[0]:.3f}, {axis_lengths[1]:.3f}, {axis_lengths[2]:.3f}) Mpc"
        )
        print(f"                    {source.description}")

    print("5. Integrator setup ...")
    if explicit_redshift_geometry:
        D_s = D_s_target
        z_source = args.z_source
    else:
        D_s = cosmo.comoving_distance(args.z_source)

    max_dist_in_box = grid_size - np.min(obs_pos)
    if explicit_redshift_geometry:
        if D_s > max_dist_in_box:
            raise ValueError(
                "Requested z_source lies outside the simulation box even after expansion; "
                "increase --source-margin-mpc or --box-mpc"
            )
    else:
        D_s = min(D_s, 0.95 * max_dist_in_box)

    D_ls = D_s - D_l
    if D_ls <= 0.0:
        raise ValueError(
            "The selected source plane lies in front of the lens or exactly on it. "
            "Increase --z-source, enlarge --box-mpc, or use explicit --z-lens/--z-source geometry."
        )
    if not explicit_redshift_geometry:
        try:
            z_source = brentq(lambda z: cosmo.comoving_distance(z) - D_s, 0.0, 5.0)
        except ValueError:
            z_source = 0.05

    characteristic_rs = min(source.r_s for source in sources)
    dt_fine = characteristic_rs / (args.n_fine_per_rs * c)
    step_fine = c * dt_fine
    lambda_total = D_s / c
    n_steps_uniform = int(np.ceil(D_s / step_fine))
    if _integrator_is_adaptive(args.integrator):
        n_steps_total = max(32, 8 * n_steps_uniform)
    else:
        n_steps_total = n_steps_uniform
    traj_stride = max(1, n_steps_total // 500)

    print(f"   D_s  = {D_s / one_Mpc:.1f} Mpc")
    print(f"   D_ls = {D_ls / one_Mpc:.1f} Mpc")
    print(f"   z_s  ~ {z_source:.4f}")
    print(f"   shape scale = {shape_scale_mpc:.3f} Mpc = {args.shape_scale_rs:.2f} r_s(ref)")
    print(f"   integrator = {args.integrator}")
    if _integrator_is_adaptive(args.integrator):
        dt_min_eff = _effective_numba_dt_min(dt_fine, args.integrator, args.dt_min_factor)
        print(
            f"   Initial fine step: {step_fine / one_Mpc * 1000:.0f} kpc, "
            f"accepted-step budget = {n_steps_total}, traj_stride = {traj_stride}"
        )
        print(f"   tolerances = (rtol={args.rtol:.1e}, atol={args.atol:.1e})")
        print(
            f"   dt_min floor = {dt_min_eff:.3e} s "
            f"({c * dt_min_eff / one_Mpc * 1000:.3f} kpc equivalent)"
        )
    else:
        print(
            f"   Uniform fine step: {step_fine / one_Mpc * 1000:.0f} kpc, "
            f"n_steps = {n_steps_total}, traj_stride = {traj_stride}"
        )

    if args.backend == "numba":
        print("   Using compiled analytical bypass for every selected profile.")

    print(f"\n6. Integrating {len(sources)} profile(s) ...")
    results_per_profile = []
    for source in sources:
        results_per_profile.append(
            _run_one_profile(
                source,
                args=args,
                root_grid=root_grid,
                base_interp=base_interp,
                a_of_eta=a_of_eta,
                cosmo=cosmo,
                eta_0=eta_0,
                eta_min=eta_min,
                a_0=a_0,
                obs_pos=obs_pos,
                targets_profile_samples_Mpc=targets["target_profile_samples_Mpc"],
                profile_sample_radius_index=targets["profile_sample_radius_index"],
                profile_n_samples_by_radius=targets["profile_n_samples_by_radius"],
                targets_map_Mpc=targets["target_map_Mpc"],
                dt_fine=dt_fine,
                lambda_total=lambda_total,
                traj_stride=traj_stride,
                n_steps_total=n_steps_total,
                bardeen_a_lens=bardeen_a_lens,
            )
        )

    print("7. Comparison metrics ...")
    kappa_profile_by_shape = np.stack([result["kappa_profile"] for result in results_per_profile], axis=0)
    kappa_profile_std_by_shape = np.stack([result["kappa_profile_std"] for result in results_per_profile], axis=0)
    gamma_profile_by_shape = np.stack([result["gamma_profile"] for result in results_per_profile], axis=0)
    gamma_profile_std_by_shape = np.stack([result["gamma_profile_std"] for result in results_per_profile], axis=0)
    mu_profile_by_shape = np.stack([result["mu_profile"] for result in results_per_profile], axis=0)
    mu_profile_std_by_shape = np.stack([result["mu_profile_std"] for result in results_per_profile], axis=0)
    kappa_profile_samples_by_shape = np.stack([result["kappa_profile_samples"] for result in results_per_profile], axis=0)
    gamma_profile_samples_by_shape = np.stack([result["gamma_profile_samples"] for result in results_per_profile], axis=0)
    mu_profile_samples_by_shape = np.stack([result["mu_profile_samples"] for result in results_per_profile], axis=0)
    kappa_map_by_shape = np.stack([result["kappa_map"] for result in results_per_profile], axis=0)
    gamma_map_by_shape = np.stack([result["gamma_map"] for result in results_per_profile], axis=0)
    mu_map_by_shape = np.stack([result["mu_map"] for result in results_per_profile], axis=0)
    D_flat_profile_by_shape = np.stack([result["D_flat_profile"] for result in results_per_profile], axis=0)
    D_flat_map_by_shape = np.stack([result["D_flat_map"] for result in results_per_profile], axis=0)
    initial_sachs_e1_map_by_shape = np.stack(
        [result["initial_sachs_e1_map"] for result in results_per_profile], axis=0
    )
    initial_sachs_e2_map_by_shape = np.stack(
        [result["initial_sachs_e2_map"] for result in results_per_profile], axis=0
    )
    final_sachs_e1_map_by_shape = np.stack(
        [result["final_sachs_e1_map"] for result in results_per_profile], axis=0
    )
    final_sachs_e2_map_by_shape = np.stack(
        [result["final_sachs_e2_map"] for result in results_per_profile], axis=0
    )
    final_k_mu_map_by_shape = np.stack(
        [result["final_k_mu_map"] for result in results_per_profile], axis=0
    )
    lambda_actual_profile_by_shape = np.stack(
        [result["lambda_actual_profile"] for result in results_per_profile], axis=0
    )
    lambda_actual_profile_samples_by_shape = np.stack(
        [result["lambda_actual_profile_samples"] for result in results_per_profile], axis=0
    )
    lambda_actual_map_by_shape = np.stack(
        [result["lambda_actual_map"] for result in results_per_profile], axis=0
    )
    source_reached_profile_samples_by_shape = np.stack(
        [result["source_reached_profile_samples"] for result in results_per_profile], axis=0
    )
    source_reached_map_by_shape = np.stack(
        [result["source_reached_map"] for result in results_per_profile], axis=0
    )
    steps_actual_profile_samples_by_shape = np.stack(
        [result["steps_actual_profile_samples"] for result in results_per_profile], axis=0
    )
    steps_actual_map_by_shape = np.stack(
        [result["steps_actual_map"] for result in results_per_profile], axis=0
    )
    stop_reason_profile_samples_by_shape = np.stack(
        [result["stop_reason_profile_samples"] for result in results_per_profile], axis=0
    )
    stop_reason_map_by_shape = np.stack(
        [result["stop_reason_map"] for result in results_per_profile], axis=0
    )
    rejected_total_profile_samples_by_shape = np.stack(
        [result["rejected_total_profile_samples"] for result in results_per_profile], axis=0
    )
    rejected_total_map_by_shape = np.stack(
        [result["rejected_total_map"] for result in results_per_profile], axis=0
    )
    rejected_streak_peak_profile_samples_by_shape = np.stack(
        [result["rejected_streak_peak_profile_samples"] for result in results_per_profile], axis=0
    )
    rejected_streak_peak_map_by_shape = np.stack(
        [result["rejected_streak_peak_map"] for result in results_per_profile], axis=0
    )
    dt_min_attempted_profile_samples_by_shape = np.stack(
        [result["dt_min_attempted_profile_samples"] for result in results_per_profile], axis=0
    )
    dt_min_attempted_map_by_shape = np.stack(
        [result["dt_min_attempted_map"] for result in results_per_profile], axis=0
    )
    dt_last_attempted_profile_samples_by_shape = np.stack(
        [result["dt_last_attempted_profile_samples"] for result in results_per_profile], axis=0
    )
    dt_last_attempted_map_by_shape = np.stack(
        [result["dt_last_attempted_map"] for result in results_per_profile], axis=0
    )
    dt_last_suggested_profile_samples_by_shape = np.stack(
        [result["dt_last_suggested_profile_samples"] for result in results_per_profile], axis=0
    )
    dt_last_suggested_map_by_shape = np.stack(
        [result["dt_last_suggested_map"] for result in results_per_profile], axis=0
    )
    final_pos_profile_by_shape_Mpc = np.stack(
        [result["final_pos_profile_Mpc"] for result in results_per_profile], axis=0
    )
    final_pos_map_by_shape_Mpc = np.stack(
        [result["final_pos_map_Mpc"] for result in results_per_profile], axis=0
    )

    ref_kappa_profile = kappa_profile_by_shape[reference_index]
    ref_gamma_profile = gamma_profile_by_shape[reference_index]
    ref_kappa_map = kappa_map_by_shape[reference_index]
    ref_gamma_map = gamma_map_by_shape[reference_index]

    delta_kappa_profile_by_shape = kappa_profile_by_shape - ref_kappa_profile[None, :]
    delta_gamma_profile_by_shape = gamma_profile_by_shape - ref_gamma_profile[None, :]
    delta_kappa_map_by_shape = kappa_map_by_shape - ref_kappa_map[None, :]
    delta_gamma_map_by_shape = gamma_map_by_shape - ref_gamma_map[None, :]

    # z_l was computed earlier for the Bardeen kernel rescaling; reuse it.
    z_l = z_l_for_bardeen

    Sigma_cr_comoving, Sigma_cr_physical = sigma_cr_conventions(D_l, D_s, D_ls, z_l)
    # With the Bardeen kernel fix active, the simulated kappa corresponds to
    # the physical impact parameter b_phys = a_l * b_co at the lens. The
    # correct analytical reference is therefore kappa_BS evaluated at b_phys
    # with the standard physical Sigma_cr (Bartelmann-Schneider 2001).
    bardeen_active = bardeen_a_lens is not None
    if bardeen_active:
        Sigma_cr = Sigma_cr_physical
        reference_label = lensing_convention_label(PHYSICAL_LENSING_REFERENCE_CONVENTION)
    else:
        Sigma_cr = Sigma_cr_comoving
        reference_label = lensing_convention_label(DEFAULT_LENSING_REFERENCE_CONVENTION)
    DA_l = cosmo.angular_diameter_distance(z_l)
    DA_s = cosmo.angular_diameter_distance(z_source)
    DA_ls = cosmo.angular_diameter_distance_z1z2(z_l, z_source)
    source_plane_center_Mpc = obs_pos / one_Mpc + (D_s / one_Mpc) * dir_hat

    reference_result = results_per_profile[reference_index]
    b_profile_abs_m = np.maximum(np.abs(targets["b_profile_Mpc"]) * one_Mpc, 1e-3 * one_Mpc)
    # Evaluate analytic NFW at the right b: physical when Bardeen is active
    # (so kappa_sim matches kappa_BS directly), legacy comoving otherwise.
    if bardeen_active:
        b_eval_m = bardeen_a_lens * b_profile_abs_m
    else:
        b_eval_m = b_profile_abs_m
    kappa_analytic_comoving = reference_halo.kappa_analytic(b_profile_abs_m, Sigma_cr_comoving)
    gamma_analytic_comoving = reference_halo.gamma_analytic(b_profile_abs_m, Sigma_cr_comoving)
    kappa_analytic_physical = reference_halo.kappa_analytic(b_eval_m, Sigma_cr_physical)
    gamma_analytic_physical = reference_halo.gamma_analytic(b_eval_m, Sigma_cr_physical)
    if bardeen_active:
        kappa_analytic = kappa_analytic_physical
        gamma_analytic = gamma_analytic_physical
    else:
        kappa_analytic = kappa_analytic_comoving
        gamma_analytic = gamma_analytic_comoving

    profile_axis_rms_Mpc = np.array([source.mass_weighted_axis_lengths() / one_Mpc for source in sources])
    source_components = [_source_components_for_metadata(source) for source in sources]
    component_counts = np.array([len(components) for components in source_components], dtype=np.int32)
    max_components = int(component_counts.max())
    component_mass_fraction = np.full((len(sources), max_components), np.nan, dtype=np.float64)
    component_centers_Mpc = np.full((len(sources), max_components, 3), np.nan, dtype=np.float64)
    component_c_nfw = np.full((len(sources), max_components), np.nan, dtype=np.float64)
    component_r_s_Mpc = np.full((len(sources), max_components), np.nan, dtype=np.float64)
    for i, components in enumerate(source_components):
        for j, halo in enumerate(components):
            component_mass_fraction[i, j] = halo.M_200 / M_200
            component_centers_Mpc[i, j] = halo.center / one_Mpc
            component_c_nfw[i, j] = halo.c_NFW
            component_r_s_Mpc[i, j] = halo.r_s / one_Mpc

    namer = RunNamer(
        "lensing_mass_shape",
        integrator=args.integrator,
        metric="FLRWP1",
        profile="simulation",
        M=M_200 / one_Msun,
        c_NFW=c_NFW,
        Rvir=round(R200_Mpc, 3),
        rs=round(rs_Mpc, 4),
        zl=round(z_l, 5),
        zs=round(z_source, 5),
        Dl=round(D_l / one_Mpc, 1),
        Ds=round(D_s / one_Mpc, 1),
        obs=(
            round(obs_pos[0] / one_Mpc, 1),
            round(obs_pos[1] / one_Mpc, 1),
            round(obs_pos[2] / one_Mpc, 1),
        ),
        box_Mpc=box_Mpc,
        Nph=n_total,
        Nshape=len(sources),
        ref=reference_preset,
    )
    outfile = namer.npz()

    save_payload = dict(
        b_profile_Mpc=targets["b_profile_Mpc"],
        target_profile_Mpc=targets["target_profile_Mpc"],
        target_profile_samples_Mpc=targets["target_profile_samples_Mpc"],
        profile_n_samples_by_radius=targets["profile_n_samples_by_radius"],
        profile_sample_angle_rad=targets["profile_sample_angle_rad"],
        profile_sample_b1_Mpc=targets["profile_sample_b1_Mpc"],
        profile_sample_b2_Mpc=targets["profile_sample_b2_Mpc"],
        profile_definition="geometric_center_radial_mean",
        profile_nphi=args.profile_nphi,
        b1_map_Mpc=targets["b1_map_Mpc"],
        b2_map_Mpc=targets["b2_map_Mpc"],
        target_map_Mpc=targets["target_map_Mpc"],
        n_map_1d=targets["map_n1d"],
        map_half_Mpc=targets["map_half_Mpc"],
        profile_names=_string_array([source.label for source in sources]),
        profile_descriptions=_string_array([source.description for source in sources]),
        reference_profile_name=reference_preset,
        reference_profile_index=reference_index,
        legacy_profile_name=reference_preset,
        legacy_profile_index=reference_index,
        kappa_profile=reference_result["kappa_profile"],
        kappa_profile_std=reference_result["kappa_profile_std"],
        gamma_profile=reference_result["gamma_profile"],
        gamma_profile_std=reference_result["gamma_profile_std"],
        mu_profile=reference_result["mu_profile"],
        mu_profile_std=reference_result["mu_profile_std"],
        kappa_map=reference_result["kappa_map"],
        gamma_map=reference_result["gamma_map"],
        mu_map=reference_result["mu_map"],
        D_flat_profile=reference_result["D_flat_profile"],
        D_flat_map=reference_result["D_flat_map"],
        lambda_actual_profile=reference_result["lambda_actual_profile"],
        lambda_actual_map=reference_result["lambda_actual_map"],
        source_reached_profile_samples=reference_result["source_reached_profile_samples"],
        source_reached_map=reference_result["source_reached_map"],
        final_pos_profile_Mpc=reference_result["final_pos_profile_Mpc"],
        final_pos_map_Mpc=reference_result["final_pos_map_Mpc"],
        initial_sachs_e1_map=reference_result["initial_sachs_e1_map"],
        initial_sachs_e2_map=reference_result["initial_sachs_e2_map"],
        final_sachs_e1_map=reference_result["final_sachs_e1_map"],
        final_sachs_e2_map=reference_result["final_sachs_e2_map"],
        final_k_mu_map=reference_result["final_k_mu_map"],
        kappa_analytic=kappa_analytic,
        gamma_analytic=gamma_analytic,
        kappa_analytic_comoving=kappa_analytic_comoving,
        gamma_analytic_comoving=gamma_analytic_comoving,
        kappa_analytic_physical=kappa_analytic_physical,
        gamma_analytic_physical=gamma_analytic_physical,
        kappa_profile_by_shape=kappa_profile_by_shape,
        kappa_profile_std_by_shape=kappa_profile_std_by_shape,
        kappa_profile_samples_by_shape=kappa_profile_samples_by_shape,
        gamma_profile_by_shape=gamma_profile_by_shape,
        gamma_profile_std_by_shape=gamma_profile_std_by_shape,
        gamma_profile_samples_by_shape=gamma_profile_samples_by_shape,
        mu_profile_by_shape=mu_profile_by_shape,
        mu_profile_std_by_shape=mu_profile_std_by_shape,
        mu_profile_samples_by_shape=mu_profile_samples_by_shape,
        kappa_map_by_shape=kappa_map_by_shape,
        gamma_map_by_shape=gamma_map_by_shape,
        mu_map_by_shape=mu_map_by_shape,
        delta_kappa_profile_by_shape=delta_kappa_profile_by_shape,
        delta_gamma_profile_by_shape=delta_gamma_profile_by_shape,
        delta_kappa_map_by_shape=delta_kappa_map_by_shape,
        delta_gamma_map_by_shape=delta_gamma_map_by_shape,
        D_flat_profile_by_shape=D_flat_profile_by_shape,
        D_flat_map_by_shape=D_flat_map_by_shape,
        initial_sachs_e1_map_by_shape=initial_sachs_e1_map_by_shape,
        initial_sachs_e2_map_by_shape=initial_sachs_e2_map_by_shape,
        final_sachs_e1_map_by_shape=final_sachs_e1_map_by_shape,
        final_sachs_e2_map_by_shape=final_sachs_e2_map_by_shape,
        final_k_mu_map_by_shape=final_k_mu_map_by_shape,
        lambda_actual_profile_by_shape=lambda_actual_profile_by_shape,
        lambda_actual_profile_samples_by_shape=lambda_actual_profile_samples_by_shape,
        lambda_actual_map_by_shape=lambda_actual_map_by_shape,
        source_reached_profile_samples_by_shape=source_reached_profile_samples_by_shape,
        source_reached_map_by_shape=source_reached_map_by_shape,
        steps_actual_profile_samples_by_shape=steps_actual_profile_samples_by_shape,
        steps_actual_map_by_shape=steps_actual_map_by_shape,
        stop_reason_profile_samples_by_shape=stop_reason_profile_samples_by_shape,
        stop_reason_map_by_shape=stop_reason_map_by_shape,
        rejected_total_profile_samples_by_shape=rejected_total_profile_samples_by_shape,
        rejected_total_map_by_shape=rejected_total_map_by_shape,
        rejected_streak_peak_profile_samples_by_shape=rejected_streak_peak_profile_samples_by_shape,
        rejected_streak_peak_map_by_shape=rejected_streak_peak_map_by_shape,
        dt_min_attempted_profile_samples_by_shape=dt_min_attempted_profile_samples_by_shape,
        dt_min_attempted_map_by_shape=dt_min_attempted_map_by_shape,
        dt_last_attempted_profile_samples_by_shape=dt_last_attempted_profile_samples_by_shape,
        dt_last_attempted_map_by_shape=dt_last_attempted_map_by_shape,
        dt_last_suggested_profile_samples_by_shape=dt_last_suggested_profile_samples_by_shape,
        dt_last_suggested_map_by_shape=dt_last_suggested_map_by_shape,
        stop_reason_labels=_string_array(integrator_stop_reason_labels()),
        final_pos_profile_by_shape_Mpc=final_pos_profile_by_shape_Mpc,
        final_pos_map_by_shape_Mpc=final_pos_map_by_shape_Mpc,
        profile_axis_rms_Mpc=profile_axis_rms_Mpc,
        component_counts=component_counts,
        component_mass_fraction=component_mass_fraction,
        component_centers_Mpc=component_centers_Mpc,
        component_c_NFW=component_c_nfw,
        component_r_s_Mpc=component_r_s_Mpc,
        shape_scale_rs=args.shape_scale_rs,
        shape_scale_Mpc=shape_scale_mpc,
        profile_axis_scale=np.asarray(args.profile_axis_scale, dtype=np.float64),
        profile_orientation_deg=np.asarray(args.profile_orientation_deg, dtype=np.float64),
        cigar_axis=args.cigar_axis,
        cigar_core_fraction=args.cigar_core_fraction,
        cigar_satellite_concentration_scale=args.cigar_satellite_concentration_scale,
        custom_components_raw=_string_array(args.custom_components or []),
        halo_type="equivalent_mass_composite_nfw",
        comparison_mode="profile_vs_reference",
        N_grid=N_root,
        box_Mpc=box_Mpc,
        M_Msun=M_200 / one_Msun,
        M_200_Msun=M_200 / one_Msun,
        c_NFW=c_NFW,
        R_200_Mpc=R200_Mpc,
        r_s_Mpc=rs_Mpc,
        R_vir_Mpc=R200_Mpc,
        n_steps=n_steps_total,
        dt_fine=dt_fine,
        dt_min_effective=_effective_numba_dt_min(dt_fine, args.integrator, args.dt_min_factor),
        dt_min_factor=-1.0 if args.dt_min_factor is None else args.dt_min_factor,
        integrator_name=args.integrator,
        python_integrator_name=args.python_integrator,
        max_rejected=args.max_rejected,
        rtol=args.rtol,
        atol=args.atol,
        Sigma_cr=Sigma_cr,
        Sigma_cr_comoving=Sigma_cr_comoving,
        Sigma_cr_physical=Sigma_cr_physical,
        bardeen_a_lens=(bardeen_a_lens if bardeen_a_lens is not None else -1.0),
        lensing_reference_convention=(
            PHYSICAL_LENSING_REFERENCE_CONVENTION if bardeen_active
            else DEFAULT_LENSING_REFERENCE_CONVENTION
        ),
        reference_label=reference_label,
        D_l_Mpc=D_l / one_Mpc,
        D_s_Mpc=D_s / one_Mpc,
        D_ls_Mpc=D_ls / one_Mpc,
        DA_l_Mpc=DA_l / one_Mpc,
        DA_s_Mpc=DA_s / one_Mpc,
        DA_ls_Mpc=DA_ls / one_Mpc,
        z_l=z_l,
        z_source=z_source,
        geometry_mode="explicit_redshift" if explicit_redshift_geometry else "box_centered",
        requested_z_l=args.z_lens if explicit_redshift_geometry else -1.0,
        requested_z_source=args.z_source,
        lambda_S=lambda_total,
        H0_kms_Mpc=H0,
        Omega_m=Omega_m,
        Omega_lambda=Omega_lambda,
        grid_type="ANALYTIC_EQUIVALENT_MASS",
        lens_center_Mpc=center / one_Mpc,
        optical_axis_hat=dir_hat,
        screen_e1=e_perp1,
        screen_e2=e_perp2,
        source_plane_center_Mpc=source_plane_center_Mpc,
        obs_pos_Mpc=obs_pos / one_Mpc,
        backend=args.backend,
        numba_kernel=args.numba_kernel,
        parallel_batch=bool(args.parallel_batch),
        save_trajectories=bool(args.save_trajectories),
    )

    if args.save_trajectories:
        traj_x4, traj_D, traj_P, traj_n_pts, max_pts = _collect_trajectory_payload(results_per_profile, n_total)
        save_payload.update(
            traj_profile_name=reference_preset,
            traj_profile_index=reference_index,
            traj_x4_Mpc=traj_x4[reference_index],
            traj_D_flat=traj_D[reference_index],
            traj_P_flat=traj_P[reference_index],
            traj_n_pts=traj_n_pts[reference_index],
            traj_x4_Mpc_by_shape=traj_x4,
            traj_D_flat_by_shape=traj_D,
            traj_P_flat_by_shape=traj_P,
            traj_n_pts_by_shape=traj_n_pts,
            traj_stride=traj_stride,
            traj_max_pts=max_pts,
        )

    np.savez(outfile, **save_payload)
    print(f"\n   [ok] Saved to {outfile}")

    total = time.time() - t_total
    print("\n" + "=" * 70)
    print("  EQUIVALENT-MASS LENSING  --  SUMMARY")
    print("=" * 70)
    print(f"  Total mass : {M_200 / one_Msun:.3e} Msun")
    print(f"  Reference  : {reference_preset}")
    print(f"  Geometry   : chi_l = {D_l / one_Mpc:.1f}, chi_s = {D_s / one_Mpc:.1f}, chi_ls = {D_ls / one_Mpc:.1f} Mpc")
    print(f"               z_l = {z_l:.4f}, z_s = {z_source:.4f}")
    print(f"  Sigma_cr ({reference_label}) = {Sigma_cr * one_Mpc ** 2 / one_Msun:.3e} Msun/Mpc^2")
    print(f"  Radial bins: {n_profile}  (azimuth samples per non-zero ring = {args.profile_nphi})")
    print(f"  Photons    : {n_total} per profile, {len(sources)} profiles")
    if _integrator_is_adaptive(args.integrator):
        print(f"  Step budget: {n_steps_total} accepted steps")
    else:
        print(f"  Steps      : {n_steps_total}")

    ref_bg = ref_kappa_profile[-1]
    ref_signal = _nanabsmax(ref_kappa_profile - ref_bg)
    print(f"  Reference background kappa = {ref_bg:.6e}")
    print(f"  Reference peak signal      = {ref_signal:.6e}")
    for i, source in enumerate(sources):
        axis_lengths = profile_axis_rms_Mpc[i]
        dk_prof = delta_kappa_profile_by_shape[i]
        dg_prof = delta_gamma_profile_by_shape[i]
        dk_map = delta_kappa_map_by_shape[i]
        dg_map = delta_gamma_map_by_shape[i]
        n_failed_profile = int(np.count_nonzero(~source_reached_profile_samples_by_shape[i]))
        n_failed_map = int(np.count_nonzero(~source_reached_map_by_shape[i]))
        print(
            f"  {source.label:>16s}: comps={component_counts[i]}, "
            f"rms_axes=({axis_lengths[0]:.3f}, {axis_lengths[1]:.3f}, {axis_lengths[2]:.3f}) Mpc, "
            f"failed profile/map = {n_failed_profile} / {n_failed_map}"
        )
        print(
            f"{'':18s}max |Δkappa| profile/map = {_nanabsmax(dk_prof):.4e} / {_nanabsmax(dk_map):.4e}, "
            f"max |Δgamma| profile/map = {_nanabsmax(dg_prof):.4e} / {_nanabsmax(dg_map):.4e}"
        )

    print(f"  Total time : {total / 60:.1f} min")
    print("=" * 70)


if __name__ == "__main__":
    main()