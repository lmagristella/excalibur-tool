#!/usr/bin/env python3
"""
Dense z_l scan + halo-mass cross-check to discriminate:
  (1) post-Born / path-coupling effect (alpha depends on halo mass)
  (2) Sigma_cr (1+z_l) convention effect (alpha is mass-independent)

For each setup, integrate one FLRW reference photon (tiny halo) and one
NFW photon at b = 1 Mpc, extract alpha = signal_mult / kappa_analytic_eff.
"""
import os, sys, time
import numpy as np
from scipy import interpolate
from scipy.optimize import brentq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from excalibur.core.constants import c, G, one_Mpc, one_Msun
from excalibur.core.cosmology import LCDM_Cosmology
from excalibur.grid.grid import Grid
from excalibur.grid.interpolator_fast import InterpolatorFast
from excalibur.grid.analytical_bypass import AnalyticalBypassInterpolator
from excalibur.metrics.perturbed_flrw_metric_fast import PerturbedFLRWMetricFast
from excalibur.integration.integrator import Integrator
from excalibur.observables.optical_tidal_matrix import lensing_from_jacobi
from excalibur.objects.nfw_halo import NFWHalo
from run_lensing_nfw_analytic import make_photon


def build(cosmo, eta_0, a_0, a_of_eta, root_grid, grid_size, halo):
    base = InterpolatorFast(root_grid, boundary="clamp")
    interp = AnalyticalBypassInterpolator(
        base_interp=base, analytical_source=halo,
        bypass_radius=1e3*grid_size, bypass_fields=("Phi",), time_derivative=0.0,
    )
    metric = PerturbedFLRWMetricFast(
        a_of_eta=a_of_eta, grid=root_grid, interpolator=interp,
        adot_of_eta=cosmo.adot_of_eta, cosmology=cosmo,
        enable_lensing=True, slow_roll=True,
    )
    return metric


def integrate(metric, integrator, obs_pos, target, eta_0, a_0, lambda_total):
    p = make_photon(obs_pos, target, metric, eta_0, a_0)
    integrator.integrate_single(p, stop_mode="affine", stop_value=lambda_total)
    D_norm = p.D_flat / p.lambda_affine
    kappa, _, _ = lensing_from_jacobi(D_norm)
    return kappa


def run_one(obs_z_Mpc, halo_NFW, halo_tiny, cosmo, eta_0, a_0, a_of_eta,
            root_grid, grid_size, box_Mpc, center, b_target_Mpc, n_fine_per_rs=8):
    metric_NFW  = build(cosmo, eta_0, a_0, a_of_eta, root_grid, grid_size, halo_NFW)
    metric_tiny = build(cosmo, eta_0, a_0, a_of_eta, root_grid, grid_size, halo_tiny)

    obs_pos = np.array([box_Mpc/2, box_Mpc/2, obs_z_Mpc]) * one_Mpc
    D_l = float(np.linalg.norm(center - obs_pos))
    dir_hat = (center - obs_pos) / D_l
    seed = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(seed, dir_hat)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    e_perp1 = seed - np.dot(seed, dir_hat) * dir_hat
    e_perp1 /= np.linalg.norm(e_perp1)

    D_s = cosmo.comoving_distance(1.0)
    D_s = min(D_s, 0.95 * (grid_size - np.min(obs_pos)))
    D_ls = D_s - D_l
    z_l = brentq(lambda z: cosmo.comoving_distance(z) - D_l, 0.0, 5.0)
    z_s = brentq(lambda z: cosmo.comoving_distance(z) - D_s, 0.0, 5.0)
    lambda_total = D_s / c

    dt_fine = halo_NFW.r_s / (n_fine_per_rs * c)
    integ_NFW  = Integrator(metric=metric_NFW,  dt=dt_fine, mode="sequential",
                             integrator="rk4", rtol=1e-8, atol=1e-13)
    integ_tiny = Integrator(metric=metric_tiny, dt=dt_fine, mode="sequential",
                             integrator="rk4", rtol=1e-8, atol=1e-13)

    target = center + b_target_Mpc * one_Mpc * e_perp1

    k_FLRW = integrate(metric_tiny, integ_tiny, obs_pos, target, eta_0, a_0, lambda_total)
    k_NFW  = integrate(metric_NFW,  integ_NFW,  obs_pos, target, eta_0, a_0, lambda_total)

    Sigma_cr = (c**2/(4*np.pi*G)) * D_s/(D_l*D_ls) / (1.0 + z_l)
    b_m = np.array([b_target_Mpc * one_Mpc])
    ka_eff = float(halo_NFW.kappa_analytic(b_m, Sigma_cr)[0])

    signal_mult = 1.0 - (1.0 - k_NFW)/(1.0 - k_FLRW)
    alpha = signal_mult / ka_eff
    return dict(obs_z=obs_z_Mpc, D_l_Mpc=D_l/one_Mpc, D_s_Mpc=D_s/one_Mpc,
                D_ls_Mpc=D_ls/one_Mpc, z_l=z_l, z_s=z_s,
                k_FLRW=k_FLRW, k_NFW=k_NFW, ka_eff=ka_eff,
                signal_mult=signal_mult, alpha=alpha)


def main():
    print("="*84, flush=True)
    print(" Dense z_l scan + halo mass cross-check", flush=True)
    print("="*84, flush=True)

    H0 = 70.0
    cosmo = LCDM_Cosmology(H0, Omega_m=0.3, Omega_r=0, Omega_lambda=0.7)
    _ = cosmo.a_of_eta(1e18)
    eta_0 = cosmo._eta_at_a1
    a_0   = cosmo.a_of_eta(eta_0)
    eta_arr = np.linspace(0.5*eta_0, eta_0, 2000)
    a_arr   = np.array([cosmo.a_of_eta(e) for e in eta_arr])
    a_of_eta = interpolate.interp1d(eta_arr, a_arr, kind="cubic", fill_value="extrapolate")

    box_Mpc = 1950.0
    grid_size = box_Mpc * one_Mpc
    N_root = 16
    root_grid = Grid(shape=(N_root,)*3, spacing=(grid_size/N_root,)*3, origin=np.zeros(3))
    root_grid.add_field("Phi", np.zeros((N_root,)*3))
    center = np.array([0.5, 0.5, 0.5]) * grid_size

    halo_heavy = NFWHalo(2e15 * one_Msun, 7.0, center)   # M = 2e15 (10×)
    halo_light = NFWHalo(2e14 * one_Msun, 7.0, center)   # M = 2e14
    halo_tiny  = NFWHalo(1e-30 * one_Msun, 7.0, center)  # for FLRW reference

    obs_zs_heavy = [5.0, 100.0, 200.0, 400.0, 600.0, 800.0]   # 6 dense points
    obs_zs_light_check = [5.0, 600.0]                         # 2 control points

    print("\n--- M_200 = 2e15 Msun (dense z_l scan) ---", flush=True)
    rows_heavy = []
    for obs_z in obs_zs_heavy:
        t0 = time.time()
        r = run_one(obs_z, halo_heavy, halo_tiny, cosmo, eta_0, a_0, a_of_eta,
                    root_grid, grid_size, box_Mpc, center, b_target_Mpc=1.0)
        rows_heavy.append(r)
        print(f"  obs_z={r['obs_z']:5.0f}  D_l={r['D_l_Mpc']:6.1f}  z_l={r['z_l']:.4f}  "
              f"z_s={r['z_s']:.4f}  k_FLRW={r['k_FLRW']:.3e}  k_NFW={r['k_NFW']:.3e}  "
              f"ka={r['ka_eff']:.3e}  alpha={r['alpha']:.5f}   ({time.time()-t0:.0f}s)", flush=True)

    print("\n--- M_200 = 2e14 Msun (control: 10x lighter) ---", flush=True)
    rows_light = []
    for obs_z in obs_zs_light_check:
        t0 = time.time()
        r = run_one(obs_z, halo_light, halo_tiny, cosmo, eta_0, a_0, a_of_eta,
                    root_grid, grid_size, box_Mpc, center, b_target_Mpc=1.0)
        rows_light.append(r)
        print(f"  obs_z={r['obs_z']:5.0f}  D_l={r['D_l_Mpc']:6.1f}  z_l={r['z_l']:.4f}  "
              f"z_s={r['z_s']:.4f}  k_FLRW={r['k_FLRW']:.3e}  k_NFW={r['k_NFW']:.3e}  "
              f"ka={r['ka_eff']:.3e}  alpha={r['alpha']:.5f}   ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== SUMMARY ===", flush=True)
    print("M=2e15:", flush=True)
    print(f"  {'z_l':>7} {'D_l/D_s':>8} {'D_ls/D_s':>9} {'1+z_l':>7} {'alpha':>9} {'1-alpha':>10}", flush=True)
    for r in rows_heavy:
        print(f"  {r['z_l']:7.4f} {r['D_l_Mpc']/r['D_s_Mpc']:8.4f} "
              f"{r['D_ls_Mpc']/r['D_s_Mpc']:9.4f} {1+r['z_l']:7.4f} "
              f"{r['alpha']:9.5f} {1-r['alpha']:10.5f}", flush=True)
    print("M=2e14 (control):", flush=True)
    for r in rows_light:
        print(f"  {r['z_l']:7.4f} {r['D_l_Mpc']/r['D_s_Mpc']:8.4f} "
              f"{r['D_ls_Mpc']/r['D_s_Mpc']:9.4f} {1+r['z_l']:7.4f} "
              f"{r['alpha']:9.5f} {1-r['alpha']:10.5f}", flush=True)

    # cross-check: at fixed z_l, does alpha change with halo mass?
    print("\nMass dependence at fixed z_l (post-Born vs Sigma_cr):", flush=True)
    for r_l in rows_light:
        # find matching heavy
        for r_h in rows_heavy:
            if abs(r_l['obs_z'] - r_h['obs_z']) < 0.1:
                d_alpha = r_l['alpha'] - r_h['alpha']
                print(f"  z_l={r_h['z_l']:.4f}  alpha(M=2e15)={r_h['alpha']:.5f}  "
                      f"alpha(M=2e14)={r_l['alpha']:.5f}  Delta={d_alpha:+.5f}", flush=True)


if __name__ == "__main__":
    main()
