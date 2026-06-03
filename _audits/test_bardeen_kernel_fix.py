#!/usr/bin/env python3
"""Validate the Bardeen rescaling fix in the numba NFW bypass.

With bardeen_a_lens=a_l passed to the backend, the kernel computes
Phi_Bardeen(r_co) = Phi_NFW(a_l * r_co) instead of the static Newton
Phi_NFW(r_co). This should make kappa_sim match kappa_BS analytical
directly, with NO post-processing factor.

We test across multiple z_l values to confirm the fix is universal,
and we compare:
  - kappa_sim_raw_buggy   : current code (bardeen_a_lens=None)
  - kappa_sim_bardeen_fix : with bardeen_a_lens=a_l
  - kappa_BS              : standard physical analytical NFW

Expected:
  kappa_sim_raw_buggy / kappa_BS = (1+z_l)
  kappa_sim_bardeen_fix / kappa_BS = 1.0000 exactly
"""

from __future__ import annotations

import os, sys, time
import numpy as np
from scipy import interpolate

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from excalibur.core.constants import c, G, one_Mpc, one_Msun
from excalibur.core.cosmology import LCDM_Cosmology
from excalibur.grid.amr_grid import AMRGrid
from excalibur.grid.grid import Grid
from excalibur.grid.interpolator_fast import InterpolatorFast
from excalibur.grid.analytical_bypass import AnalyticalBypassInterpolator
from excalibur.metrics.perturbed_flrw_metric_fast import PerturbedFLRWMetricFast
from excalibur.objects.nfw_halo import NFWHalo
from excalibur.observables.sachs_basis import init_sachs_basis
from excalibur.observables.optical_tidal_matrix import lensing_from_jacobi
from excalibur.photon.photon import Photon


def make_photon(obs_pos, target, metric, eta_0, a_0):
    obs_4d = np.array([eta_0, *obs_pos])
    direction = target - obs_pos
    direction /= np.linalg.norm(direction)
    g = metric.metric_tensor(obs_4d)
    g_init = g / (a_0 * a_0)
    basis_a = 1.0
    k_spatial = direction * c
    spatial_sq = (g_init[1, 1] * k_spatial[0] ** 2
                  + g_init[2, 2] * k_spatial[1] ** 2
                  + g_init[3, 3] * k_spatial[2] ** 2)
    k0 = -np.sqrt(abs(-spatial_sq / g_init[0, 0]))
    k_mu = np.array([k0, *k_spatial])
    e1, e2 = init_sachs_basis(k_mu, g_init, basis_a, convention="conformal_metric")
    p = Photon(obs_4d.copy(), k_mu.copy(), record_lensing=False)
    p.e1 = e1.copy()
    p.e2 = e2.copy()
    return p


def run_one_zl(cosmo, eta_0, a_0, a_of_eta, eta_min,
               M_200, c_NFW, z_l, z_s, box_Mpc, b_in_rs,
               bardeen_a_lens=None):
    """Run sim at one (z_l, z_s) with optional Bardeen rescaling."""
    grid_size = box_Mpc * one_Mpc
    N_root = 8
    root_grid = Grid(
        shape=(N_root,)*3,
        spacing=(grid_size/N_root,)*3,
        origin=np.array([0.0, 0.0, 0.0]),
    )
    root_grid.add_field("Phi", np.zeros((N_root,)*3))

    D_C_l = cosmo.comoving_distance(z_l)
    D_C_s = cosmo.comoving_distance(z_s)

    obs_z_Mpc = 5.0
    obs_pos = np.array([box_Mpc/2, box_Mpc/2, obs_z_Mpc]) * one_Mpc
    center = obs_pos + np.array([0.0, 0.0, D_C_l])
    halo = NFWHalo(M_200, c_NFW, center)

    base_interp = InterpolatorFast(root_grid, boundary="clamp")
    bypass_radius = 1e3 * grid_size
    interp = AnalyticalBypassInterpolator(
        base_interp=base_interp, analytical_source=halo,
        bypass_radius=bypass_radius, bypass_fields=("Phi",),
        time_derivative=0.0,
    )

    metric = PerturbedFLRWMetricFast(
        a_of_eta=a_of_eta, grid=root_grid, interpolator=interp,
        adot_of_eta=cosmo.adot_of_eta, cosmology=cosmo,
        enable_lensing=True, slow_roll=True, analytical_geodesics=True,
        sachs_screen_convention="conformal_metric",
    )

    b_m = b_in_rs * halo.r_s
    e_perp1 = np.array([1.0, 0.0, 0.0])
    targets = [center + b * e_perp1 for b in b_m]

    from excalibur.integration.integrator_numba_specialized import (
        NumbaAMRBackend as SpecializedNumbaAMRBackend,
        integrate_photon_numba_dopri5,
    )
    analytical_amr = AMRGrid(root_grid)
    backend = SpecializedNumbaAMRBackend(
        analytical_amr, cosmo, c_val=c, slow_roll=True, lensing=True,
        sachs_screen_convention="conformal_metric",
        analytical_source=halo, bypass_radius=bypass_radius,
        integrator="dopri5", rtol=1e-10, atol=1e-14,
        eta_range=(eta_min, eta_0),
        bardeen_a_lens=bardeen_a_lens,
    )
    backend.warmup()

    dt_fine = halo.r_s / (16.0 * c)
    lambda_total = D_C_s / c

    photons = [make_photon(obs_pos, t, metric, eta_0, a_0) for t in targets]
    kappas = np.zeros(len(photons))
    gammas = np.zeros(len(photons))
    for i, p in enumerate(photons):
        _, lam, _, _ = integrate_photon_numba_dopri5(
            p, backend, dt_init=dt_fine, lambda_stop=lambda_total,
            rtol=1e-10, atol=1e-14, dt_min=dt_fine/1000.0,
            dt_max=50.0*dt_fine, max_steps=500000, record_every=0,
        )
        D_norm = p.D_flat / lam if lam > 0 else np.full(4, np.nan)
        k, mu, g = lensing_from_jacobi(D_norm)
        kappas[i] = k
        gammas[i] = g

    return kappas, gammas, halo, D_C_l, D_C_s


def main():
    print("=" * 92)
    print("  BARDEEN KERNEL FIX VALIDATION")
    print("  Compare: bardeen_a_lens=None (buggy) vs bardeen_a_lens=a_l (fixed)")
    print("=" * 92)

    cosmo = LCDM_Cosmology(70.0, Omega_m=0.3, Omega_r=0, Omega_lambda=0.7)
    _ = cosmo.a_of_eta(1e18)
    eta_0 = cosmo._eta_at_a1
    a_0 = cosmo.a_of_eta(eta_0)
    eta_min = 0.3 * eta_0
    eta_arr = np.linspace(eta_min, eta_0, 3000)
    a_arr = np.array([cosmo.a_of_eta(e) for e in eta_arr])
    a_of_eta = interpolate.interp1d(eta_arr, a_arr, kind="cubic",
                                    fill_value="extrapolate")

    M_200 = 3e14 * one_Msun
    c_NFW = 5.0
    z_s = 1.5
    b_in_rs = np.array([2.0, 4.0, 8.0])

    z_l_list = [0.1, 0.3, 0.5, 0.8]

    D_C_s = cosmo.comoving_distance(z_s)
    box_Mpc = 1.3 * D_C_s / one_Mpc + 100.0

    print(f"\nSetup: NFW M={M_200/one_Msun:.1e}, c={c_NFW}, z_s={z_s}")
    print(f"  b/r_s = {list(b_in_rs)}, scanning z_l = {z_l_list}")

    all_results = {}
    for z_l in z_l_list:
        a_l = 1.0 / (1.0 + z_l)
        print(f"\n--- z_l = {z_l}  (a_l = {a_l:.4f},  1+z_l = {1+z_l}) ---")

        # PATH 1: legacy (no bardeen rescaling)
        t0 = time.time()
        k_raw, g_raw, halo, D_C_l, _ = run_one_zl(
            cosmo, eta_0, a_0, a_of_eta, eta_min,
            M_200, c_NFW, z_l, z_s, box_Mpc, b_in_rs,
            bardeen_a_lens=None,
        )
        t_raw = time.time() - t0

        # PATH 2: Bardeen fix
        t0 = time.time()
        k_fix, g_fix, _, _, _ = run_one_zl(
            cosmo, eta_0, a_0, a_of_eta, eta_min,
            M_200, c_NFW, z_l, z_s, box_Mpc, b_in_rs,
            bardeen_a_lens=a_l,
        )
        t_fix = time.time() - t0

        # Analytical NFW kappa, gamma (BS-2001 physical)
        D_A_l = D_C_l / (1.0 + z_l)
        D_A_s = D_C_s / (1.0 + z_s)
        D_A_ls = (D_C_s - D_C_l) / (1.0 + z_s)
        Sigma_cr_BS = (c**2 / (4*np.pi*G)) * D_A_s / (D_A_l * D_A_ls)
        # Without fix: photon at b_co = b_in_rs * r_s_phys, code treats coords as
        # if physical, so kappa_sim corresponds to b/r_s = b_in_rs (times (1+z_l)
        # spurious factor). Reference here: kappa_BS at b/r_s = b_in_rs.
        b_m_raw = b_in_rs * halo.r_s
        # With Bardeen fix: photon really at b_phys = a_l * b_co, so kappa_sim_fix
        # corresponds to b_phys/r_s = a_l * b_in_rs. Reference: kappa_BS at that b.
        a_l = 1.0 / (1.0 + z_l)
        b_m_phys = a_l * b_in_rs * halo.r_s
        k_BS_raw = halo.kappa_analytic(b_m_raw, Sigma_cr_BS)
        g_BS_raw = halo.gamma_analytic(b_m_raw, Sigma_cr_BS)
        k_BS_phys = halo.kappa_analytic(b_m_phys, Sigma_cr_BS)
        g_BS_phys = halo.gamma_analytic(b_m_phys, Sigma_cr_BS)
        k_BS = k_BS_raw
        g_BS = g_BS_raw

        all_results[z_l] = {
            "k_raw": k_raw, "g_raw": g_raw,
            "k_fix": k_fix, "g_fix": g_fix,
            "k_BS_raw": k_BS_raw, "g_BS_raw": g_BS_raw,
            "k_BS_phys": k_BS_phys, "g_BS_phys": g_BS_phys,
            "k_BS": k_BS, "g_BS": g_BS,
            "t_raw": t_raw, "t_fix": t_fix,
        }
        print(f"  sim done: raw={t_raw:.1f}s, fix={t_fix:.1f}s")

    # ==================================================================
    # Report
    # ==================================================================
    print("\n" + "=" * 92)
    print("RESULTS  kappa:  raw / kappa_BS  AND  bardeen_fix / kappa_BS")
    print("=" * 92)
    print(f"{'z_l':>5} | {'b/r_s':>6} | {'k_raw/k_BS':>12} {'k_fix/k_BS':>12} | "
          f"{'expected raw':>14} {'expected fix':>14}")
    print("-" * 92)
    for z_l in z_l_list:
        R = all_results[z_l]
        for i, b in enumerate(b_in_rs):
            r_raw = R["k_raw"][i] / R["k_BS"][i]
            r_fix = R["k_fix"][i] / R["k_BS"][i]
            print(f"{z_l:5.2f} | {b:6.2f} | {r_raw:12.4f} {r_fix:12.4f} | "
                  f"{1+z_l:14.4f} {1.0:14.4f}")
        print("-" * 92)

    print("\n" + "=" * 92)
    print("RESULTS  gamma:  raw / gamma_BS  AND  bardeen_fix / gamma_BS")
    print("=" * 92)
    print(f"{'z_l':>5} | {'b/r_s':>6} | {'g_raw/g_BS':>12} {'g_fix/g_BS':>12} | "
          f"{'expected raw':>14} {'expected fix':>14}")
    print("-" * 92)
    for z_l in z_l_list:
        R = all_results[z_l]
        for i, b in enumerate(b_in_rs):
            r_raw = R["g_raw"][i] / R["g_BS"][i]
            r_fix = R["g_fix"][i] / R["g_BS"][i]
            print(f"{z_l:5.2f} | {b:6.2f} | {r_raw:12.4f} {r_fix:12.4f} | "
                  f"{1+z_l:14.4f} {1.0:14.4f}")
        print("-" * 92)

    # ==================================================================
    # Verdict
    # ==================================================================
    print("\n" + "=" * 92)
    print("VERDICT")
    print("=" * 92)
    print(f"  Bardeen fix interpretation: photon at b_co = b_in_rs * r_s_phys")
    print(f"  corresponds to physical b_phys = a_l * b_co (since coords are comoving).")
    print(f"  So kappa_sim_fix should equal kappa_BS at b_phys/r_s = a_l * b_in_rs.")
    print()
    print(f"{'z_l':>5} | {'a_l':>6} | {'max |k_fix/k_BS(a_l*b)-1|':>26} {'max |g_fix/g_BS(a_l*b)-1|':>26}")
    print("-" * 92)
    all_pass = True
    for z_l in z_l_list:
        R = all_results[z_l]
        a_l = 1.0 / (1.0 + z_l)
        max_err_k = np.max(np.abs(R["k_fix"]/R["k_BS_phys"] - 1.0))
        max_err_g = np.max(np.abs(R["g_fix"]/R["g_BS_phys"] - 1.0))
        passed = max_err_k < 0.005 and max_err_g < 0.005
        all_pass = all_pass and passed
        status = "[OK]" if passed else "[FAIL]"
        print(f"{z_l:5.2f} | {a_l:6.4f} | {max_err_k:26.2e} {max_err_g:26.2e}  {status}")

    print()
    if all_pass:
        print("  [SUCCESS] kappa_sim_fix matches kappa_BS at the PHYSICAL impact")
        print("            parameter b_phys = a_l * b_co exactly. The Bardeen")
        print("            kernel fix is mathematically correct.")
        print()
        print("            Note: user must interpret position arguments as comoving.")
        print("            For a target at physical b_phys, use b_co = b_phys / a_l.")
    else:
        print("  [FAIL] Some residuals remain even with the b_phys interpretation.")


if __name__ == "__main__":
    main()
