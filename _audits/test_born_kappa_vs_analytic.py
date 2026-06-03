#!/usr/bin/env python3
"""Born approximation test: simulated kappa vs analytical NFW kappa.

The goal is to determine which Sigma_cr convention the simulated kappa
corresponds to, by comparing against the three candidates:

  (1) Bartelmann-Schneider 2001 PHYSICAL:
        Sigma_cr_BS = c^2 / (4 pi G) * D_A_s / (D_A_l * D_A_ls)

  (2) Excalibur "comoving":
        Sigma_cr_co = c^2 / (4 pi G) * D_C_s / (D_C_l * D_C_ls)

  (3) Excalibur "physical":
        Sigma_cr_phys_code = Sigma_cr_co / (1 + z_l)

The simulation uses the production path: numba SPECIALIZED kernel,
conformal_metric screen, slow_roll, dopri5.

Setup:
  - NFW lens at z_l = 0.3 (lower than usual to make 1+z_l effect visible)
  - Source at z_s = 1.0
  - Probe at intermediate b: weak-lensing regime, where Born is accurate

Expected behaviour if the implementation is correct per Fleury:
  - kappa_code in conformal convention should equal kappa_BS (physical)
    because kappa is a conformal INVARIANT (Fleury, section 5.1.1).

We test 6 impact parameters and report the ratios.
"""

from __future__ import annotations

import os, sys, time
import numpy as np
from scipy import interpolate
from scipy.optimize import brentq

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
from excalibur.observables.lensing_conventions import sigma_cr_conventions
from excalibur.photon.photon import Photon


def make_photon(obs_pos, target, metric, eta_0, a_0):
    """Build a photon, screen-conformal Sachs basis."""
    obs_4d = np.array([eta_0, *obs_pos])
    direction = target - obs_pos
    direction /= np.linalg.norm(direction)
    g = metric.metric_tensor(obs_4d)
    g_init = g / (a_0 * a_0)   # conformal screen
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


def main():
    print("=" * 72)
    print("  BORN TEST: simulated kappa vs analytical NFW (3 conventions)")
    print("  Production path: numba specialized + conformal_metric + dopri5")
    print("=" * 72)

    # -----------------------------------------------------------------
    # Cosmology
    # -----------------------------------------------------------------
    cosmo = LCDM_Cosmology(70.0, Omega_m=0.3, Omega_r=0, Omega_lambda=0.7)
    _ = cosmo.a_of_eta(1e18)
    eta_0 = cosmo._eta_at_a1
    a_0 = cosmo.a_of_eta(eta_0)
    eta_min = 0.5 * eta_0
    eta_arr = np.linspace(eta_min, eta_0, 2000)
    a_arr = np.array([cosmo.a_of_eta(e) for e in eta_arr])
    a_of_eta = interpolate.interp1d(eta_arr, a_arr, kind="cubic",
                                    fill_value="extrapolate")

    # -----------------------------------------------------------------
    # Geometry: z_l = 0.3, z_s = 1.0
    # -----------------------------------------------------------------
    z_l = 0.3
    z_s = 1.0
    D_C_l  = cosmo.comoving_distance(z_l)
    D_C_s  = cosmo.comoving_distance(z_s)
    D_C_ls = D_C_s - D_C_l
    D_A_l  = D_C_l  / (1.0 + z_l)
    D_A_s  = D_C_s  / (1.0 + z_s)
    D_A_ls = D_C_ls / (1.0 + z_s)   # flat K=0

    print(f"\nGeometry:")
    print(f"  z_l = {z_l}, z_s = {z_s}")
    print(f"  D_C_l  = {D_C_l/one_Mpc:.1f} Mpc")
    print(f"  D_C_s  = {D_C_s/one_Mpc:.1f} Mpc")
    print(f"  D_C_ls = {D_C_ls/one_Mpc:.1f} Mpc")
    print(f"  (1+z_l) = {1+z_l}")

    # Three Sigma_cr candidates
    Sigma_cr_BS = (c**2 / (4*np.pi*G)) * D_A_s / (D_A_l * D_A_ls)
    Sigma_cr_co, Sigma_cr_phys_code = sigma_cr_conventions(D_C_l, D_C_s, D_C_ls, z_l)

    print(f"\nThree Sigma_cr candidates (Msun/Mpc^2):")
    print(f"  (1) BS-2001 physical (D_A)        : {Sigma_cr_BS*one_Mpc**2/one_Msun:.3e}")
    print(f"  (2) excalibur 'comoving' (D_C)    : {Sigma_cr_co*one_Mpc**2/one_Msun:.3e}")
    print(f"  (3) excalibur 'physical' = (2)/(1+z_l): {Sigma_cr_phys_code*one_Mpc**2/one_Msun:.3e}")
    print(f"  ratio (2)/(1) = 1/(1+z_l) = {1.0/(1+z_l):.4f}  measured {Sigma_cr_co/Sigma_cr_BS:.4f}")
    print(f"  ratio (3)/(1) = 1/(1+z_l)^2 = {1.0/(1+z_l)**2:.4f}  measured {Sigma_cr_phys_code/Sigma_cr_BS:.4f}")

    # -----------------------------------------------------------------
    # Box + grid (placeholder for AnalyticalBypassInterpolator)
    # -----------------------------------------------------------------
    box_Mpc = 1.5 * D_C_s / one_Mpc + 50.0
    grid_size = box_Mpc * one_Mpc
    N_root = 8
    root_grid = Grid(
        shape=(N_root,)*3,
        spacing=(grid_size/N_root,)*3,
        origin=np.array([0.0, 0.0, 0.0]),
    )
    root_grid.add_field("Phi", np.zeros((N_root,)*3))

    # -----------------------------------------------------------------
    # NFW halo at z_l
    # -----------------------------------------------------------------
    M_200 = 5e14 * one_Msun  # lighter than test 1, keeps kappa < ~0.5 for Born
    c_NFW = 5.0
    obs_z_Mpc = 5.0
    obs_pos = np.array([box_Mpc/2, box_Mpc/2, obs_z_Mpc]) * one_Mpc
    center = obs_pos + np.array([0.0, 0.0, D_C_l])
    halo = NFWHalo(M_200, c_NFW, center)
    r_s_Mpc = halo.r_s / one_Mpc
    print(f"\nNFW halo:")
    print(f"  M200 = {M_200/one_Msun:.2e} Msun, c = {c_NFW}")
    print(f"  r_s = {r_s_Mpc*1000:.0f} kpc, R200 = {halo.R_200/one_Mpc:.2f} Mpc")
    print(f"  Sigma_s = 2 rho_s r_s = {halo.Sigma_s*one_Mpc**2/one_Msun:.3e} Msun/Mpc^2")

    # -----------------------------------------------------------------
    # Analytical bypass interpolator
    # -----------------------------------------------------------------
    base_interp = InterpolatorFast(root_grid, boundary="clamp")
    bypass_radius = 1e3 * grid_size
    interp = AnalyticalBypassInterpolator(
        base_interp=base_interp, analytical_source=halo,
        bypass_radius=bypass_radius, bypass_fields=("Phi",),
        time_derivative=0.0,
    )

    # Metric in conformal screen convention
    metric = PerturbedFLRWMetricFast(
        a_of_eta=a_of_eta, grid=root_grid, interpolator=interp,
        adot_of_eta=cosmo.adot_of_eta, cosmology=cosmo,
        enable_lensing=True, slow_roll=True, analytical_geodesics=True,
        sachs_screen_convention="conformal_metric",
    )

    # -----------------------------------------------------------------
    # Impact parameters: weak-lensing regime (b > r_s) to validate Born
    # -----------------------------------------------------------------
    b_test_Mpc = np.array([1.0, 2.0, 3.0, 5.0, 8.0, 15.0])  # > r_s
    print(f"\nImpact parameters (Mpc, weak-lensing regime b > r_s): {list(b_test_Mpc)}")

    # Build photon targets transverse to LOS
    e_perp1 = np.array([1.0, 0.0, 0.0])  # x direction
    targets = [center + b * one_Mpc * e_perp1 for b in b_test_Mpc]

    # -----------------------------------------------------------------
    # Integration: numba SPECIALIZED kernel + dopri5
    # -----------------------------------------------------------------
    from excalibur.integration.integrator_numba_specialized import (
        NumbaAMRBackend as SpecializedNumbaAMRBackend,
        integrate_photon_numba_dopri5,
    )
    analytical_amr = AMRGrid(root_grid)
    backend = SpecializedNumbaAMRBackend(
        analytical_amr, cosmo, c_val=c, slow_roll=True, lensing=True,
        sachs_screen_convention="conformal_metric",
        analytical_source=halo, bypass_radius=bypass_radius,
        integrator="dopri5", rtol=1e-9, atol=1e-13,
        eta_range=(eta_min, eta_0),
    )
    backend.warmup()

    dt_fine = halo.r_s / (16.0 * c)   # finer for accuracy
    lambda_total = D_C_s / c
    print(f"\nIntegration:")
    print(f"  dt_fine = r_s / (16c), lambda_total = D_s/c = {lambda_total:.3e} s")
    print(f"  rtol=1e-9, atol=1e-13")

    photons = [make_photon(obs_pos, t, metric, eta_0, a_0) for t in targets]
    print(f"  Integrating {len(photons)} photons...")
    t0 = time.time()
    kappas_sim = np.zeros(len(photons))
    gammas_sim = np.zeros(len(photons))
    for i, p in enumerate(photons):
        _, lam, _, _ = integrate_photon_numba_dopri5(
            p, backend, dt_init=dt_fine, lambda_stop=lambda_total,
            rtol=1e-9, atol=1e-13, dt_min=dt_fine/1000.0,
            dt_max=50.0*dt_fine, max_steps=200000, record_every=0,
        )
        D_norm = p.D_flat / lam if lam > 0 else np.full(4, np.nan)
        k, mu, g = lensing_from_jacobi(D_norm)
        kappas_sim[i] = k
        gammas_sim[i] = g
    print(f"  done in {time.time()-t0:.1f}s")

    # -----------------------------------------------------------------
    # Analytical NFW kappa with each Sigma_cr
    # -----------------------------------------------------------------
    b_m = b_test_Mpc * one_Mpc
    kappa_BS  = halo.kappa_analytic(b_m, Sigma_cr_BS)
    kappa_co  = halo.kappa_analytic(b_m, Sigma_cr_co)
    kappa_phc = halo.kappa_analytic(b_m, Sigma_cr_phys_code)

    # -----------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------
    print("\n" + "=" * 78)
    print("RESULTS: kappa_sim vs kappa_analytic (3 conventions)")
    print("=" * 78)
    print(f"{'b (Mpc)':>8} | {'kappa_sim':>12} | {'k_BS':>12} {'k_comoving':>12} {'k_physical_code':>16}")
    print("-" * 78)
    for i, b in enumerate(b_test_Mpc):
        print(f"{b:8.2f} | {kappas_sim[i]:12.4e} | "
              f"{kappa_BS[i]:12.4e} {kappa_co[i]:12.4e} {kappa_phc[i]:16.4e}")

    print("\n" + "=" * 78)
    print("RATIOS kappa_sim / kappa_analytic")
    print("=" * 78)
    print(f"{'b (Mpc)':>8} | {'k_sim/k_BS':>14} {'k_sim/k_comov':>14} {'k_sim/k_phys_code':>18}")
    print("-" * 78)
    for i, b in enumerate(b_test_Mpc):
        r1 = kappas_sim[i] / kappa_BS[i]
        r2 = kappas_sim[i] / kappa_co[i]
        r3 = kappas_sim[i] / kappa_phc[i]
        print(f"{b:8.2f} | {r1:14.4f} {r2:14.4f} {r3:18.4f}")

    # Stats: averaged ratios
    r_BS  = np.mean(kappas_sim / kappa_BS)
    r_co  = np.mean(kappas_sim / kappa_co)
    r_phc = np.mean(kappas_sim / kappa_phc)
    print(f"{'mean':>8} | {r_BS:14.4f} {r_co:14.4f} {r_phc:18.4f}")

    print("\n" + "=" * 78)
    print("INTERPRETATION:")
    print("=" * 78)
    print(f"  (1+z_l)    = {1+z_l}")
    print(f"  (1+z_l)^2  = {(1+z_l)**2}")
    print()
    if 0.97 < r_BS < 1.03:
        print(f"  [OK] kappa_sim matches BS-2001 physical convention to ~{(1-r_BS)*100:+.1f}%")
        print(f"       -> kappa_sim is the PHYSICAL convergence (Fleury invariant).")
        print(f"       -> kappa_analytic = Sigma / Sigma_cr_BS is the correct comparator.")
    elif 0.97 < r_co < 1.03:
        print(f"  [OK] kappa_sim matches excalibur 'comoving' convention to ~{(1-r_co)*100:+.1f}%")
        print(f"       -> kappa_sim is the 'comoving' kappa (= physical kappa / (1+z_l))")
    elif 0.97 < r_phc < 1.03:
        print(f"  [OK] kappa_sim matches excalibur 'physical' (the /(1+z_l) one)")
    else:
        print(f"  [UNCLEAR] kappa_sim doesn't match any of the 3 candidates within 3%")
        print(f"            r_BS    = {r_BS:.4f}  (need 1.00 if conformal Fleury)")
        print(f"            r_co    = {r_co:.4f}  (= r_BS / (1+z_l) expected if scale issue)")
        print(f"            r_phc   = {r_phc:.4f}  (= r_BS / (1+z_l)^2)")


if __name__ == "__main__":
    main()
