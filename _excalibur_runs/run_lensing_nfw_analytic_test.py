#!/usr/bin/env python3
"""Quick smoke-test for run_lensing_nfw_analytic — reduced photon count and step size."""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Patch key parameters before importing main
import run_lensing_nfw_analytic as _m

# ---- monkey-patch the constants used inside main() ----
_ORIG_MAIN = _m.main

def main():
    import numpy as np
    import time
    from scipy import interpolate
    from scipy.optimize import brentq

    from excalibur.core.constants import c, G, one_Mpc, one_Msun
    from excalibur.core.cosmology import LCDM_Cosmology
    from excalibur.grid.grid import Grid
    from excalibur.grid.interpolator_fast import InterpolatorFast
    from excalibur.grid.analytical_bypass import AnalyticalBypassInterpolator
    from excalibur.metrics.perturbed_flrw_metric_fast import PerturbedFLRWMetricFast
    from excalibur.photon.photon import Photon
    from excalibur.integration.integrator import Integrator
    from excalibur.observables.lensing_conventions import (
        DEFAULT_LENSING_REFERENCE_CONVENTION,
        lensing_convention_label,
        sigma_cr_conventions,
    )
    from excalibur.observables.sachs_basis import init_sachs_basis
    from excalibur.observables.optical_tidal_matrix import lensing_from_jacobi
    from excalibur.objects.nfw_halo import NFWHalo
    from excalibur.io.filename_utils import RunNamer
    from run_lensing_nfw_analytic import make_photon

    t_total = time.time()
    print("=" * 70)
    print("  NFW LENSING  --  FULLY ANALYTICAL  [SMOKE TEST]")
    print("=" * 70)

    # 1. Cosmology
    H0 = 70.0
    Omega_m, Omega_lambda = 0.3, 0.7
    cosmo = LCDM_Cosmology(H0, Omega_m=Omega_m, Omega_r=0, Omega_lambda=Omega_lambda)
    _ = cosmo.a_of_eta(1e18)
    eta_0 = cosmo._eta_at_a1
    a_0   = cosmo.a_of_eta(eta_0)
    eta_min = 0.5 * eta_0
    eta_arr = np.linspace(eta_min, eta_0, 2000)
    a_arr   = np.array([cosmo.a_of_eta(e) for e in eta_arr])
    a_of_eta = interpolate.interp1d(eta_arr, a_arr, kind="cubic", fill_value="extrapolate")
    print(f"   eta_0 = {eta_0:.4e} s")

    # 2. Halo
    box_Mpc   = 1950.0
    grid_size = box_Mpc * one_Mpc
    N_root    = 16
    root_grid = Grid(
        shape   = (N_root, N_root, N_root),
        spacing = (grid_size / N_root,) * 3,
        origin  = np.array([0.0, 0.0, 0.0]),
    )
    root_grid.add_field("Phi", np.zeros((N_root,) * 3))
    M_200  = 2e15 * one_Msun
    c_NFW  = 7.0
    center = np.array([0.5, 0.5, 0.5]) * grid_size
    halo   = NFWHalo(M_200, c_NFW, center)
    R200_Mpc = halo.R_200 / one_Mpc
    rs_Mpc   = halo.r_s   / one_Mpc
    print(f"   {halo}  R_200={R200_Mpc:.2f} Mpc  r_s={rs_Mpc*1e3:.0f} kpc")

    # 3. Analytical bypass
    base_interp   = InterpolatorFast(root_grid, boundary="clamp")
    bypass_radius = 1e3 * grid_size
    interp = AnalyticalBypassInterpolator(
        base_interp=base_interp, analytical_source=halo,
        bypass_radius=bypass_radius, bypass_fields=("Phi",), time_derivative=0.0,
    )

    # 4. Metric
    metric = PerturbedFLRWMetricFast(
        a_of_eta=a_of_eta, grid=root_grid, interpolator=interp,
        adot_of_eta=cosmo.adot_of_eta, cosmology=cosmo,
        enable_lensing=True, slow_roll=True,
        sachs_screen_convention="conformal_metric",
    )

    # 5. Photon cone  --  REDUCED
    obs_z_Mpc = 5.0
    obs_pos = np.array([box_Mpc / 2, box_Mpc / 2, obs_z_Mpc]) * one_Mpc
    dir_to_center = center - obs_pos
    D_l = np.linalg.norm(dir_to_center)
    dir_hat = dir_to_center / D_l
    seed = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(seed, dir_hat)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    e_perp1 = seed - np.dot(seed, dir_hat) * dir_hat
    e_perp1 /= np.linalg.norm(e_perp1)
    e_perp2 = np.cross(dir_hat, e_perp1)
    e_perp2 /= np.linalg.norm(e_perp2)

    # Profile: 10 points seulement
    b_values_Mpc = np.array([0.0, 0.05, 0.1, 0.3, 0.46, 0.8, rs_Mpc, R200_Mpc, 3.0, 10.0])
    b_values = b_values_Mpc * one_Mpc
    photons_profile, b_profile_Mpc = [], []
    for b in b_values:
        target = center + b * e_perp1
        p = make_photon(obs_pos, target, metric, eta_0, a_0)
        photons_profile.append(p)
        b_profile_Mpc.append(b / one_Mpc)
    n_profile = len(photons_profile)

    # Map: 5×5
    map_half_Mpc = 1.5
    n_map_1d = 5
    b1_arr = np.linspace(-map_half_Mpc, map_half_Mpc, n_map_1d) * one_Mpc
    b2_arr = np.linspace(-map_half_Mpc, map_half_Mpc, n_map_1d) * one_Mpc
    photons_map, map_b1_Mpc, map_b2_Mpc = [], [], []
    for b1 in b1_arr:
        for b2 in b2_arr:
            target = center + b1 * e_perp1 + b2 * e_perp2
            p = make_photon(obs_pos, target, metric, eta_0, a_0)
            photons_map.append(p)
            map_b1_Mpc.append(b1 / one_Mpc)
            map_b2_Mpc.append(b2 / one_Mpc)
    n_map   = len(photons_map)
    n_total = n_profile + n_map
    print(f"   Profile: {n_profile}  Map: {n_map} ({n_map_1d}x{n_map_1d})  Total: {n_total}")

    # 6. Integrator  --  coarser step + relaxed tol
    n_fine_per_rs = 2                          # 4× fewer steps than production
    dt_fine   = halo.r_s / (n_fine_per_rs * c)
    step_fine = c * dt_fine

    D_s = cosmo.comoving_distance(1.0)
    max_dist_in_box = grid_size - np.min(obs_pos)
    D_s  = min(D_s, 0.95 * max_dist_in_box)
    D_ls = D_s - D_l
    try:
        z_source = brentq(lambda z: cosmo.comoving_distance(z) - D_s, 0.0, 5.0)
    except ValueError:
        z_source = 0.05
    DA_FLRW  = cosmo.angular_diameter_distance(z_source)
    DA_l     = cosmo.angular_diameter_distance(
                   brentq(lambda z: cosmo.comoving_distance(z) - D_l, 0.0, 5.0))
    z_l      = brentq(lambda z: cosmo.comoving_distance(z) - D_l, 0.0, 5.0)

    lambda_total  = D_s / c
    n_steps_total = int(np.ceil(D_s / step_fine))
    traj_stride   = max(1, n_steps_total // 100)   # fewer traj pts in test
    print(f"   step={step_fine/one_Mpc*1e3:.0f} kpc  n_steps={n_steps_total}  stride={traj_stride}")

    integrator = Integrator(
        metric=metric, dt=dt_fine, mode="sequential", integrator="rk4",
        rtol=1e-6, atol=1e-10,
    )

    # 7. Integrate
    all_photons = photons_profile + photons_map
    print(f"\nIntegrating {n_total} photons ...")
    t_int = time.time()
    kappas, mus, gammas = np.empty(n_total), np.empty(n_total), np.empty(n_total)
    D_flats, final_pos, lambda_actuals = (
        np.empty((n_total, 4)), np.empty((n_total, 3)), np.empty(n_total))

    for i, photon in enumerate(all_photons):
        integrator.integrate_single(photon, stop_mode="affine", stop_value=lambda_total,
                                    record_every=traj_stride)
        lam = photon.lambda_affine
        D_norm = photon.D_flat / lam
        kappa, mu, shear = lensing_from_jacobi(D_norm)
        kappas[i], mus[i], gammas[i] = kappa, mu, shear
        D_flats[i], final_pos[i], lambda_actuals[i] = D_norm, photon.x[1:4], lam
        elapsed = time.time() - t_int
        rate    = (i + 1) / elapsed
        eta     = (n_total - i - 1) / rate if rate > 0 else 0
        print(f"   [{i+1:3d}/{n_total}]  kappa={kappa:+.4e}  |γ|={shear:.3e}"
              f"  ({elapsed:.0f}s, ~{eta:.0f}s left)")

    # 8. Trajectories
    raw_trajs = [np.array([s[0:4] for s in p.history.states]) / one_Mpc for p in all_photons]
    raw_D     = [np.array([s[8:12] for s in p.history.states]) for p in all_photons]
    raw_P     = [np.array([s[12:16] for s in p.history.states]) for p in all_photons]
    traj_n_pts = np.array([t.shape[0] for t in raw_trajs], dtype=np.int32)
    max_pts    = int(traj_n_pts.max())
    traj_x4_Mpc = np.full((n_total, max_pts, 4), np.nan, dtype=np.float32)
    traj_D_flat  = np.full((n_total, max_pts, 4), np.nan, dtype=np.float32)
    traj_P_flat  = np.full((n_total, max_pts, 4), np.nan, dtype=np.float32)
    for i in range(n_total):
        n = traj_n_pts[i]
        traj_x4_Mpc[i, :n] = raw_trajs[i]
        traj_D_flat[i,  :n] = raw_D[i]
        traj_P_flat[i,  :n] = raw_P[i]

    # 9. Analytic reference
    Sigma_cr_comoving, Sigma_cr_physical = sigma_cr_conventions(D_l, D_s, D_ls, z_l)
    Sigma_cr = Sigma_cr_comoving
    reference_label = lensing_convention_label(DEFAULT_LENSING_REFERENCE_CONVENTION)
    b_m = np.maximum(np.abs(np.array(b_profile_Mpc)) * one_Mpc, 1e-3 * one_Mpc)
    k_analytic_comoving = halo.kappa_analytic(b_m, Sigma_cr_comoving)
    g_analytic_comoving = halo.gamma_analytic(b_m, Sigma_cr_comoving)
    k_analytic_physical = halo.kappa_analytic(b_m, Sigma_cr_physical)
    g_analytic_physical = halo.gamma_analytic(b_m, Sigma_cr_physical)
    k_analytic = k_analytic_comoving
    g_analytic = g_analytic_comoving

    # 10. Save
    namer  = RunNamer(
        "lensing_nfw_analytic_test",
        integrator="rk4", metric="FLRWP1", profile="nfw",
        M=M_200/one_Msun, c_NFW=c_NFW, Rvir=round(R200_Mpc,3),
        rs=round(rs_Mpc,4), zl=round(z_l,5), zs=round(z_source,5),
        Dl=round(D_l/one_Mpc,1), Ds=round(D_s/one_Mpc,1), Nph=n_total,
    )
    outfile = namer.npz()
    k_prof = kappas[:n_profile];  g_prof = gammas[:n_profile];  m_prof = mus[:n_profile]
    np.savez(
        outfile,
        b_profile_Mpc=np.array(b_profile_Mpc), kappa_profile=k_prof,
        gamma_profile=g_prof, mu_profile=m_prof,
        kappa_analytic=k_analytic, gamma_analytic=g_analytic,
        kappa_analytic_comoving=k_analytic_comoving,
        gamma_analytic_comoving=g_analytic_comoving,
        kappa_analytic_physical=k_analytic_physical,
        gamma_analytic_physical=g_analytic_physical,
        D_flat_profile=D_flats[:n_profile], lambda_actual_profile=lambda_actuals[:n_profile],
        b1_map_Mpc=np.array(map_b1_Mpc), b2_map_Mpc=np.array(map_b2_Mpc),
        kappa_map=kappas[n_profile:], gamma_map=gammas[n_profile:], mu_map=mus[n_profile:],
        D_flat_map=D_flats[n_profile:], lambda_actual_map=lambda_actuals[n_profile:],
        n_map_1d=n_map_1d, map_half_Mpc=map_half_Mpc,
        traj_x4_Mpc=traj_x4_Mpc, traj_D_flat=traj_D_flat, traj_P_flat=traj_P_flat,
        traj_n_pts=traj_n_pts, traj_stride=traj_stride,
        halo_type="NFW", M_200_Msun=M_200/one_Msun, c_NFW=c_NFW,
        R_200_Mpc=R200_Mpc, r_s_Mpc=rs_Mpc, z_l=z_l, z_source=z_source,
        D_l_Mpc=D_l/one_Mpc, D_s_Mpc=D_s/one_Mpc, D_ls_Mpc=D_ls/one_Mpc,
        Sigma_cr=Sigma_cr,
        Sigma_cr_comoving=Sigma_cr_comoving,
        Sigma_cr_physical=Sigma_cr_physical,
        lensing_reference_convention=DEFAULT_LENSING_REFERENCE_CONVENTION,
        H0_kms_Mpc=H0, Omega_m=Omega_m, Omega_lambda=Omega_lambda,
        grid_type="ANALYTIC", obs_pos_Mpc=obs_pos/one_Mpc, box_Mpc=box_Mpc,
    )
    print(f"\n   [ok] Saved → {outfile}")

    total = time.time() - t_total
    print(f"\n  kappa_max  = {k_prof.max():.4e}  (analytic {reference_label}: {k_analytic.max():.4e})")
    print(f"  |gamma|_max= {g_prof.max():.4e}  (analytic {reference_label}: {g_analytic.max():.4e})")
    print(f"  mu_max     = {m_prof.max():+.2f}")
    print(f"  Total time : {total:.1f} s")


if __name__ == "__main__":
    main()