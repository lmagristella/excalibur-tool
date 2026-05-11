#!/usr/bin/env python3
"""Independent benchmark for the simple analytical NFW lensing case.

This script compares four quantities on the same representative geometry:

1. Closed-form NFW lensing curves from ``NFWHalo``.
2. Independent direct projection of the 3D NFW density,
   yielding ``Sigma``, ``kappa`` and ``gamma_t``.
3. Independent straight-ray integration of the scalar weak-lensing kernel
   ``nabla_perp^2 Phi / (4 pi G)``.
4. The full optical/Jacobi solver used by the production pipeline.

The goal is to determine whether the ~20% mismatch comes from the analytic
reference curves or from the full optical solver.
"""

import os
import sys

import numpy as np
from scipy import interpolate
from scipy.integrate import quad
from scipy.optimize import brentq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from excalibur.core.constants import G, c, one_Mpc, one_Msun
from excalibur.core.cosmology import LCDM_Cosmology
from excalibur.grid.analytical_bypass import AnalyticalBypassInterpolator
from excalibur.grid.grid import Grid
from excalibur.grid.interpolator_fast import InterpolatorFast
from excalibur.integration.integrator import Integrator
from excalibur.metrics.perturbed_flrw_metric_fast import PerturbedFLRWMetricFast
from excalibur.objects.nfw_halo import NFWHalo
from excalibur.observables.lensing_conventions import (
    DEFAULT_LENSING_REFERENCE_CONVENTION,
    lensing_convention_label,
    sigma_cr_conventions,
)
from excalibur.observables.optical_tidal_matrix import lensing_from_jacobi
from run_lensing_nfw_analytic import make_photon


def sigma_numeric(halo, radius):
    """Direct 3D projection of rho(r) to Sigma(R)."""
    zmax = 5e3 * halo.r_s
    integrand = lambda z: halo.density(halo.x0 + radius, halo.y0, halo.z0 + z)
    value, _ = quad(integrand, 0.0, zmax, epsabs=1e-10, epsrel=1e-10, limit=400)
    return 2.0 * value


def mean_sigma_numeric(halo, radius):
    """Direct projection-based mean Sigma(<R)."""
    integrand = lambda radius_p: sigma_numeric(halo, radius_p) * radius_p
    value, _ = quad(integrand, 0.0, radius, epsabs=1e-9, epsrel=1e-8, limit=200)
    return 2.0 * value / (radius * radius)


def scalar_kappa_straight_ray(halo, obs_pos, target, chi_source, sigma_cr):
    """Straight-ray scalar weak-lensing kernel using nabla_perp^2 Phi."""
    direction = target - obs_pos
    direction /= np.linalg.norm(direction)

    chi = np.linspace(0.0, chi_source, 12001)
    positions = obs_pos[None, :] + chi[:, None] * direction[None, :]
    kernel = np.empty_like(chi)
    for i, xyz in enumerate(positions):
        hess = halo.potential_hessian(xyz[0], xyz[1], xyz[2])
        lap = np.trace(hess)
        los = direction @ hess @ direction
        kernel[i] = (lap - los) / (4.0 * np.pi * G)

    sigma_perp = np.trapezoid(kernel, chi)
    return sigma_perp / sigma_cr


def scalar_kappa_on_real_path(halo, photon, sigma_cr):
    """Same scalar transverse kernel, but sampled along the reconstructed path."""
    history = photon.history.states if hasattr(photon.history, "states") else photon.history
    positions = np.vstack([
        np.asarray(state, dtype=float)[:4][1:4]
        for state in history
        if np.asarray(state).shape[0] >= 4
    ])
    diffs = np.diff(positions, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)

    valid = segment_lengths > 0.0
    if not np.any(valid):
        return np.nan

    tangents = diffs[valid] / segment_lengths[valid, None]
    midpoints = 0.5 * (positions[:-1][valid] + positions[1:][valid])

    kernel = np.empty(midpoints.shape[0])
    for i, (xyz, tangent) in enumerate(zip(midpoints, tangents)):
        hess = halo.potential_hessian(xyz[0], xyz[1], xyz[2])
        lap = np.trace(hess)
        los = tangent @ hess @ tangent
        kernel[i] = (lap - los) / (4.0 * np.pi * G)

    sigma_perp = np.sum(kernel * segment_lengths[valid])
    return sigma_perp / sigma_cr


def build_setup():
    """Return the shared representative NFW setup used in the bias diagnostics."""
    h0 = 70.0
    cosmo = LCDM_Cosmology(h0, Omega_m=0.3, Omega_r=0.0, Omega_lambda=0.7)
    _ = cosmo.a_of_eta(1e18)
    eta_0 = cosmo._eta_at_a1
    a_0 = cosmo.a_of_eta(eta_0)

    eta_arr = np.linspace(0.5 * eta_0, eta_0, 2000)
    a_arr = np.array([cosmo.a_of_eta(eta) for eta in eta_arr])
    a_of_eta = interpolate.interp1d(eta_arr, a_arr, kind="cubic", fill_value="extrapolate")

    box_mpc = 1950.0
    grid_size = box_mpc * one_Mpc
    n_root = 16
    root_grid = Grid(
        shape=(n_root, n_root, n_root),
        spacing=(grid_size / n_root,) * 3,
        origin=np.zeros(3),
    )
    root_grid.add_field("Phi", np.zeros((n_root,) * 3))

    halo = NFWHalo(2e15 * one_Msun, 7.0, np.array([0.5, 0.5, 0.5]) * grid_size)
    base_interp = InterpolatorFast(root_grid, boundary="clamp")
    interp = AnalyticalBypassInterpolator(
        base_interp=base_interp,
        analytical_source=halo,
        bypass_radius=1e3 * grid_size,
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
    )

    obs_pos = np.array([box_mpc / 2, box_mpc / 2, 5.0]) * one_Mpc
    center = halo.center
    d_l = float(np.linalg.norm(center - obs_pos))
    dir_hat = (center - obs_pos) / d_l
    seed = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(seed, dir_hat)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    e_perp1 = seed - np.dot(seed, dir_hat) * dir_hat
    e_perp1 /= np.linalg.norm(e_perp1)

    d_s = cosmo.comoving_distance(1.0)
    d_s = min(d_s, 0.95 * (grid_size - np.min(obs_pos)))
    d_ls = d_s - d_l
    z_l = brentq(lambda z: cosmo.comoving_distance(z) - d_l, 0.0, 5.0)
    z_s = brentq(lambda z: cosmo.comoving_distance(z) - d_s, 0.0, 5.0)
    lambda_total = d_s / c
    sigma_cr_comoving, sigma_cr_physical = sigma_cr_conventions(d_l, d_s, d_ls, z_l)

    return {
        "cosmo": cosmo,
        "eta_0": eta_0,
        "a_0": a_0,
        "halo": halo,
        "metric": metric,
        "obs_pos": obs_pos,
        "center": center,
        "e_perp1": e_perp1,
        "d_l": d_l,
        "d_s": d_s,
        "d_ls": d_ls,
        "z_l": z_l,
        "z_s": z_s,
        "lambda_total": lambda_total,
        "sigma_cr": sigma_cr_physical,
        "sigma_cr_physical": sigma_cr_physical,
        "sigma_cr_comoving": sigma_cr_comoving,
        "sigma_cr_reference": sigma_cr_comoving,
        "reference_convention": DEFAULT_LENSING_REFERENCE_CONVENTION,
    }


def main():
    setup = build_setup()
    halo = setup["halo"]
    metric = setup["metric"]

    integrator = Integrator(
        metric=metric,
        dt=halo.r_s / (8.0 * c),
        mode="sequential",
        integrator="rk4",
        rtol=1e-8,
        atol=1e-13,
    )

    b_values_mpc = np.array([0.8, 1.0, 1.5, 3.0])

    print("=" * 86)
    print("  Independent NFW Benchmark  --  analytic vs projected vs full solver")
    print("=" * 86)
    print(
        f"  Geometry: z_l={setup['z_l']:.4f}, z_s={setup['z_s']:.4f}, "
        f"D_l={setup['d_l']/one_Mpc:.1f} Mpc, D_s={setup['d_s']/one_Mpc:.1f} Mpc"
    )
    reference_label = lensing_convention_label(setup["reference_convention"])
    print(
        f"  Sigma_cr_ref ({reference_label}) = "
        f"{setup['sigma_cr_reference'] * one_Mpc**2 / one_Msun:.4e} Msun/Mpc^2"
    )
    print(
        f"  Sigma_cr_physical             = "
        f"{setup['sigma_cr_physical'] * one_Mpc**2 / one_Msun:.4e} Msun/Mpc^2"
    )
    print()
    print(
        "  b[Mpc]   kappa_an     kappa_proj   kappa_perp   kappa_path   "
        "kappa_full   alpha_full"
    )

    rows = []
    for b_mpc in b_values_mpc:
        radius = b_mpc * one_Mpc
        target = setup["center"] + radius * setup["e_perp1"]

        kappa_an = float(halo.kappa_analytic(np.array([radius]), setup["sigma_cr_reference"])[0])
        gamma_an = float(halo.gamma_analytic(np.array([radius]), setup["sigma_cr_reference"])[0])
        kappa_an_physical = float(halo.kappa_analytic(np.array([radius]), setup["sigma_cr_physical"])[0])

        sigma_proj = sigma_numeric(halo, radius)
        mean_sigma_proj = mean_sigma_numeric(halo, radius)
        kappa_proj = sigma_proj / setup["sigma_cr_reference"]
        gamma_proj = (mean_sigma_proj - sigma_proj) / setup["sigma_cr_reference"]

        kappa_perp = scalar_kappa_straight_ray(
            halo,
            setup["obs_pos"],
            target,
            setup["d_s"],
            setup["sigma_cr_reference"],
        )

        photon = make_photon(setup["obs_pos"], target, metric, setup["eta_0"], setup["a_0"])
        integrator.integrate_single(photon, stop_mode="affine", stop_value=setup["lambda_total"])
        d_norm = photon.D_flat / photon.lambda_affine
        kappa_full, _, _ = lensing_from_jacobi(d_norm)
        kappa_path = scalar_kappa_on_real_path(halo, photon, setup["sigma_cr_reference"])

        alpha_full = kappa_full / kappa_an if abs(kappa_an) > 0 else np.nan
        alpha_full_physical = kappa_full / kappa_an_physical if abs(kappa_an_physical) > 0 else np.nan
        rows.append(
            (
                b_mpc,
                kappa_an,
                kappa_proj,
                kappa_perp,
                kappa_path,
                kappa_full,
                alpha_full,
                alpha_full_physical,
                gamma_an,
                gamma_proj,
            )
        )

        print(
            f"  {b_mpc:5.2f}   {kappa_an:+.6e}  {kappa_proj:+.6e}  {kappa_perp:+.6e}  "
            f"{kappa_path:+.6e}  {kappa_full:+.6e}   {alpha_full:.6f}"
        )

    rows = np.array(rows, dtype=float)
    print()
    print("  Relative agreement of the independent references:")
    print(
        f"    max |kappa_proj / kappa_an - 1|   = {np.max(np.abs(rows[:, 2] / rows[:, 1] - 1.0)):.3e}"
    )
    print(
        f"    max |kappa_perp / kappa_an - 1|   = {np.max(np.abs(rows[:, 3] / rows[:, 1] - 1.0)):.3e}"
    )
    print(
        f"    max |kappa_path / kappa_an - 1|   = {np.max(np.abs(rows[:, 4] / rows[:, 1] - 1.0)):.3e}"
    )
    print(
        f"    max |gamma_proj / gamma_an - 1|   = {np.max(np.abs(rows[:, 9] / rows[:, 8] - 1.0)):.3e}"
    )
    print()
    print(
        f"  Full solver ratio vs {reference_label} reference: mean alpha_full = {np.mean(rows[:, 6]):.6f}, "
        f"range = [{np.min(rows[:, 6]):.6f}, {np.max(rows[:, 6]):.6f}]"
    )
    print(
        f"  Full solver ratio vs physical reference: mean alpha_full = {np.mean(rows[:, 7]):.6f}, "
        f"range = [{np.min(rows[:, 7]):.6f}, {np.max(rows[:, 7]):.6f}]"
    )
    print("=" * 86)


if __name__ == "__main__":
    main()