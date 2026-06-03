#!/usr/bin/env python3
"""Numerical test: do "metric" and "conformal_metric" screen conventions
give equivalent physical observables on the same NFW lens setup ?

We compare three integration paths on the SAME photons:

  (A) Python backend, sachs_screen_convention = "metric"
      - Riemann blocks use the full FLRW values (a, H_conf, H_prime)
      - Sachs basis defined in the physical metric g
      - This is the historical / textbook physical screen.

  (B) Python backend, sachs_screen_convention = "conformal_metric"
      - Riemann blocks use the full FLRW values (a, H_conf, H_prime)
      - Sachs basis defined in the CONFORMAL metric g/a^2
      - Tests Fleury's conformal dictionary at the projection step
        WITHOUT zeroing FLRW terms in Riemann.

  (C) Numba SPECIALIZED kernel, conformal_metric
      - Sets a = 1, adot = 0, H_conf = 0, H_prime = 0 in Riemann + Christoffel
      - Sachs basis re-init with a = 1
      - This is the optimized production path.

The conformal dictionary predicts:
    D_phys = a_source * D_conformal   (conformal -> physical angular distance)
    kappa, gamma (dimensionless lensing observables) should be IDENTICAL
    after the proper normalization is applied.

For the dimensionless ratio  D_norm = D / lambda  (used by lensing_from_jacobi),
all three paths should yield the SAME kappa, gamma at the source IF the
conformal trick is mathematically consistent and correctly implemented.

We print:
    - kappa, gamma from each path at a handful of impact parameters
    - Pairwise relative differences
    - Final photon position (geodesic path equivalence check)
"""

from __future__ import annotations

import os, sys, time
import numpy as np
from scipy import interpolate

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from excalibur.core.constants import c, one_Mpc, one_Msun
from excalibur.core.cosmology import LCDM_Cosmology
from excalibur.grid.amr_grid import AMRGrid
from excalibur.grid.grid import Grid
from excalibur.grid.interpolator_fast import InterpolatorFast
from excalibur.grid.analytical_bypass import AnalyticalBypassInterpolator
from excalibur.integration.integrator import Integrator
from excalibur.metrics.perturbed_flrw_metric_fast import PerturbedFLRWMetricFast
from excalibur.objects.nfw_halo import NFWHalo
from excalibur.observables.sachs_basis import init_sachs_basis
from excalibur.observables.optical_tidal_matrix import lensing_from_jacobi
from excalibur.photon.photon import Photon


def make_photon(obs_pos, target, metric, eta_0, a_0):
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
    spatial_sq = (g_init[1, 1] * k_spatial[0] ** 2
                  + g_init[2, 2] * k_spatial[1] ** 2
                  + g_init[3, 3] * k_spatial[2] ** 2)
    k0 = -np.sqrt(abs(-spatial_sq / g_init[0, 0]))
    k_mu = np.array([k0, *k_spatial])
    e1, e2 = init_sachs_basis(k_mu, g_init, basis_a, convention=screen_convention)
    p = Photon(obs_4d.copy(), k_mu.copy(), record_lensing=False)
    p.e1 = e1.copy()
    p.e2 = e2.copy()
    return p


def integrate_photons(photons, metric, dt_fine, lambda_total):
    integrator = Integrator(
        metric=metric, dt=dt_fine, mode="sequential",
        integrator="rk4", rtol=1e-8, atol=1e-13,
    )
    results = []
    for p in photons:
        integrator.integrate_single(p, stop_mode="affine",
                                    stop_value=lambda_total, record_every=0)
        lam = p.lambda_affine
        D_norm = p.D_flat / lam if lam > 0 else np.full(4, np.nan)
        kappa, mu, shear = lensing_from_jacobi(D_norm)
        results.append({
            "kappa": kappa, "mu": mu, "gamma": shear,
            "lam": lam,
            "final_pos_Mpc": np.asarray(p.x[1:4]) / one_Mpc,
            "D_norm": D_norm,
        })
    return results


def integrate_photons_numba_specialized(photons, root_grid, halo,
                                         bypass_radius, dt_fine,
                                         lambda_total, cosmo, eta_min, eta_0,
                                         screen_convention):
    from excalibur.integration.integrator_numba_specialized import (
        NumbaAMRBackend as SpecializedNumbaAMRBackend,
        integrate_photon_numba_dopri5,
    )
    analytical_amr = AMRGrid(root_grid)
    backend = SpecializedNumbaAMRBackend(
        analytical_amr, cosmo, c_val=c, slow_roll=True, lensing=True,
        sachs_screen_convention=screen_convention,
        analytical_source=halo, bypass_radius=bypass_radius,
        integrator="dopri5", rtol=1e-8, atol=1e-13,
        eta_range=(eta_min, eta_0),
    )
    backend.warmup()
    results = []
    for p in photons:
        _, lam, n_acc, n_rej = integrate_photon_numba_dopri5(
            p, backend, dt_init=dt_fine, lambda_stop=lambda_total,
            rtol=1e-8, atol=1e-13, dt_min=dt_fine/1000.0, dt_max=50.0*dt_fine,
            max_steps=200000, record_every=0,
        )
        D_norm = p.D_flat / lam if lam > 0 else np.full(4, np.nan)
        kappa, mu, shear = lensing_from_jacobi(D_norm)
        results.append({
            "kappa": kappa, "mu": mu, "gamma": shear,
            "lam": lam,
            "final_pos_Mpc": np.asarray(p.x[1:4]) / one_Mpc,
            "D_norm": D_norm,
        })
    return results


def main():
    print("=" * 78)
    print("  SCREEN CONVENTION EQUIVALENCE TEST")
    print("  Comparing metric vs conformal_metric (python + numba specialized)")
    print("=" * 78)

    # --- Cosmology ---
    H0 = 70.0
    cosmo = LCDM_Cosmology(H0, Omega_m=0.3, Omega_r=0, Omega_lambda=0.7)
    _ = cosmo.a_of_eta(1e18)
    eta_0 = cosmo._eta_at_a1
    a_0 = cosmo.a_of_eta(eta_0)
    eta_min = 0.5 * eta_0
    eta_arr = np.linspace(eta_min, eta_0, 2000)
    a_arr = np.array([cosmo.a_of_eta(e) for e in eta_arr])
    a_of_eta = interpolate.interp1d(eta_arr, a_arr, kind="cubic",
                                    fill_value="extrapolate")
    print(f"\nCosmology: H0={H0}, Om=0.3, OL=0.7")
    print(f"   eta_0 = {eta_0:.4e} s,  a_0 = {a_0:.6f}")

    # --- Box / Grid (placeholder, analytical bypass active) ---
    box_Mpc = 1950.0
    grid_size = box_Mpc * one_Mpc
    N_root = 8
    root_grid = Grid(
        shape=(N_root,)*3,
        spacing=(grid_size/N_root,)*3,
        origin=np.array([0.0, 0.0, 0.0]),
    )
    root_grid.add_field("Phi", np.zeros((N_root,)*3))

    # --- NFW halo ---
    M_200 = 2e15 * one_Msun
    c_NFW = 7.0
    obs_z_Mpc = 5.0
    obs_pos = np.array([box_Mpc/2, box_Mpc/2, obs_z_Mpc]) * one_Mpc
    z_source = 1.0
    D_s = cosmo.comoving_distance(z_source)
    D_s = min(D_s, 0.95 * (grid_size - obs_z_Mpc * one_Mpc))
    D_l_Mpc = 0.45 * D_s / one_Mpc  # lens at ~mid-line
    center = obs_pos + np.array([0, 0, D_l_Mpc * one_Mpc])
    halo = NFWHalo(M_200, c_NFW, center)
    print(f"\nNFW halo:  M200={M_200/one_Msun:.2e} Msun, c={c_NFW}")
    print(f"   center @ ~ z={D_l_Mpc:.1f} Mpc from observer")
    print(f"   r_s = {halo.r_s/one_Mpc*1000:.0f} kpc")

    # --- Interpolator (analytical bypass) ---
    base_interp = InterpolatorFast(root_grid, boundary="clamp")
    bypass_radius = 1e3 * grid_size
    interp = AnalyticalBypassInterpolator(
        base_interp=base_interp, analytical_source=halo,
        bypass_radius=bypass_radius, bypass_fields=("Phi",),
        time_derivative=0.0,
    )

    # --- Build two metrics with different screen conventions ---
    metric_phys = PerturbedFLRWMetricFast(
        a_of_eta=a_of_eta, grid=root_grid, interpolator=interp,
        adot_of_eta=cosmo.adot_of_eta, cosmology=cosmo,
        enable_lensing=True, slow_roll=True, analytical_geodesics=True,
        sachs_screen_convention="metric",
    )
    metric_conf = PerturbedFLRWMetricFast(
        a_of_eta=a_of_eta, grid=root_grid, interpolator=interp,
        adot_of_eta=cosmo.adot_of_eta, cosmology=cosmo,
        enable_lensing=True, slow_roll=True, analytical_geodesics=True,
        sachs_screen_convention="conformal_metric",
    )

    # --- Impact parameters: cusp -> outer profile ---
    b_test_Mpc = np.array([0.05, 0.20, 0.50, 1.00, 2.00, 5.00])
    print(f"\nTest impact parameters (Mpc): {list(b_test_Mpc)}")
    targets_Mpc = [center / one_Mpc + np.array([b, 0.0, 0.0])
                   for b in b_test_Mpc]

    # --- Integration parameters ---
    dt_fine = halo.r_s / (8.0 * c)
    lambda_total = D_s / c
    print(f"   D_s = {D_s/one_Mpc:.1f} Mpc,  lambda_total = c*Ds")
    print(f"   dt_fine = r_s / (8c) = {dt_fine:.3e} s")

    # --- Run path (A): python, metric ---
    print(f"\n[A] Python backend, screen = 'metric' (physical, full FLRW Riemann)...")
    t = time.time()
    photons_A = [make_photon(obs_pos, t_mpc * one_Mpc, metric_phys, eta_0, a_0)
                 for t_mpc in targets_Mpc]
    res_A = integrate_photons(photons_A, metric_phys, dt_fine, lambda_total)
    print(f"   done in {time.time()-t:.1f}s")

    # --- Run path (B): python, conformal_metric ---
    print(f"\n[B] Python backend, screen = 'conformal_metric' (full FLRW Riemann)...")
    t = time.time()
    photons_B = [make_photon(obs_pos, t_mpc * one_Mpc, metric_conf, eta_0, a_0)
                 for t_mpc in targets_Mpc]
    res_B = integrate_photons(photons_B, metric_conf, dt_fine, lambda_total)
    print(f"   done in {time.time()-t:.1f}s")

    # --- Run path (C): numba specialized ---
    print(f"\n[C] Numba SPECIALIZED kernel, screen = 'conformal_metric'")
    print(f"    (a=1, H=0, H'=0 simplification)...")
    t = time.time()
    photons_C = [make_photon(obs_pos, t_mpc * one_Mpc, metric_conf, eta_0, a_0)
                 for t_mpc in targets_Mpc]
    res_C = integrate_photons_numba_specialized(
        photons_C, root_grid, halo, bypass_radius, dt_fine, lambda_total,
        cosmo, eta_min, eta_0, "conformal_metric",
    )
    print(f"   done in {time.time()-t:.1f}s")

    # =================================================================
    #  REPORT
    # =================================================================
    print("\n" + "=" * 78)
    print("RESULTS:  kappa  and  |gamma|  at each impact parameter")
    print("=" * 78)
    print(f"{'b (Mpc)':>8} | {'A: metric (py)':>18} | "
          f"{'B: conf (py)':>18} | {'C: conf (numba)':>18}")
    print("-" * 78)
    for i, b in enumerate(b_test_Mpc):
        print(f"{b:8.3f} | "
              f"k={res_A[i]['kappa']:+.4e} g={res_A[i]['gamma']:.3e} | "
              f"k={res_B[i]['kappa']:+.4e} g={res_B[i]['gamma']:.3e} | "
              f"k={res_C[i]['kappa']:+.4e} g={res_C[i]['gamma']:.3e}")

    print("\n" + "=" * 78)
    print("PAIRWISE RELATIVE DIFFERENCES on kappa, |gamma|")
    print("=" * 78)
    print(f"{'b (Mpc)':>8} | {'|A-B|/|B| kappa':>18} {'|A-B|/|B| gamma':>18} | "
          f"{'|B-C|/|B| kappa':>18} {'|B-C|/|B| gamma':>18}")
    print("-" * 78)
    for i, b in enumerate(b_test_Mpc):
        def safe_rel(x, ref):
            return abs(x - ref) / abs(ref) if abs(ref) > 1e-30 else 0.0
        kAB = safe_rel(res_A[i]['kappa'], res_B[i]['kappa'])
        gAB = safe_rel(res_A[i]['gamma'], res_B[i]['gamma'])
        kBC = safe_rel(res_C[i]['kappa'], res_B[i]['kappa'])
        gBC = safe_rel(res_C[i]['gamma'], res_B[i]['gamma'])
        print(f"{b:8.3f} | "
              f"{kAB:18.2e} {gAB:18.2e} | "
              f"{kBC:18.2e} {gBC:18.2e}")

    print("\n" + "=" * 78)
    print("FINAL PHOTON POSITIONS  (geodesic path equivalence check)")
    print("=" * 78)
    print(f"{'b (Mpc)':>8} | {'A final (x,y,z) Mpc':>32} | "
          f"{'|B-A|':>10} {'|C-A|':>10}")
    print("-" * 78)
    for i, b in enumerate(b_test_Mpc):
        pA = res_A[i]['final_pos_Mpc']
        pB = res_B[i]['final_pos_Mpc']
        pC = res_C[i]['final_pos_Mpc']
        dBA = np.linalg.norm(pB - pA)
        dCA = np.linalg.norm(pC - pA)
        print(f"{b:8.3f} | ({pA[0]:9.3f},{pA[1]:8.3f},{pA[2]:8.3f}) | "
              f"{dBA:10.3e} {dCA:10.3e}")

    print("\n" + "=" * 78)
    print("INTERPRETATION:")
    print("=" * 78)
    print(" - If kappa,gamma agree across A/B/C (relative diff << 1):")
    print("     -> The screen convention switch is a true mathematical")
    print("        identity. The 'a=1, H=0' simplification in numba is OK.")
    print(" - If kappa differs between A and B by a CONSTANT factor:")
    print("     -> Convention mixing: D normalization changed but lensing is OK.")
    print(" - If A vs B differ in a NON-uniform way:")
    print("     -> Real physical inconsistency, needs investigation.")
    print(" - Final-position differences should be ~0 to numerical precision")
    print("   (null geodesics are conformally invariant).")
    print("=" * 78)


if __name__ == "__main__":
    main()
