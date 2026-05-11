#!/usr/bin/env python3
"""Test if the residual 2% bias scales as 1-1/c_NFW^2 or stays constant."""
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


def run_for_c(c_NFW, n_fine_per_rs=8):
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
    M_200 = 2e15 * one_Msun
    center = np.array([0.5, 0.5, 0.5]) * grid_size
    halo = NFWHalo(M_200, c_NFW, center)
    rs_Mpc = halo.r_s/one_Mpc
    R200_Mpc = halo.R_200/one_Mpc

    base_interp = InterpolatorFast(root_grid, boundary="clamp")
    interp = AnalyticalBypassInterpolator(
        base_interp=base_interp, analytical_source=halo,
        bypass_radius=1e3*grid_size, bypass_fields=("Phi",), time_derivative=0.0,
    )
    metric = PerturbedFLRWMetricFast(
        a_of_eta=a_of_eta, grid=root_grid, interpolator=interp,
        adot_of_eta=cosmo.adot_of_eta, cosmology=cosmo,
        enable_lensing=True, slow_roll=True,
    )

    obs_pos = np.array([box_Mpc/2, box_Mpc/2, 5.0]) * one_Mpc
    D_l = np.linalg.norm(center - obs_pos)
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
    lambda_total = D_s / c
    Sigma_cr = (c**2/(4*np.pi*G)) * D_s/(D_l*D_ls) / (1.0 + z_l)

    # Test photons:  several b inside halo + one far for bg
    b_test = [0.5, 1.0, 1.5, 3.0, 10.0]   # last is bg

    dt_fine = halo.r_s / (n_fine_per_rs * c)
    integ = Integrator(metric=metric, dt=dt_fine, mode="sequential",
                       integrator="rk4", rtol=1e-8, atol=1e-13)

    results = []
    for b_Mpc in b_test:
        target = center + b_Mpc * one_Mpc * e_perp1
        p = make_photon(obs_pos, target, metric, eta_0, a_0)
        integ.integrate_single(p, stop_mode="affine", stop_value=lambda_total)
        D_norm = p.D_flat / p.lambda_affine
        kappa, mu, shear = lensing_from_jacobi(D_norm)
        b_m = np.array([b_Mpc * one_Mpc])
        ka_eff  = float(halo.kappa_analytic(b_m, Sigma_cr)[0])
        ga_eff  = float(halo.gamma_analytic(b_m, Sigma_cr)[0])
        results.append((b_Mpc, kappa, shear, ka_eff, ga_eff))

    # bg = last photon
    k_bg = results[-1][1]
    g_bg = results[-1][2]
    ka_bg = results[-1][3]
    ga_bg = results[-1][4]

    print(f"\n  c_NFW = {c_NFW}  (1-1/c² = {1-1/c_NFW**2:.5f},  R_200={R200_Mpc:.3f} Mpc, r_s={rs_Mpc*1e3:.0f} kpc)")
    print(f"  k_bg(num) = {k_bg:.4e},  k_bg(analytic) = {ka_bg:.4e}")
    print(f"  Predictions for ratio after bg sub:")
    print(f"     1 - 1/c_NFW²  = {1-1/c_NFW**2:.5f}")
    print(f"  b[Mpc]   ratio_k_corrected   ratio_g_raw")
    for b_Mpc, k, g, ka, ga in results[:-1]:
        rk = (k - k_bg)/(ka - ka_bg) if (ka - ka_bg) != 0 else np.nan
        rg = g/ga if ga != 0 else np.nan
        print(f"  {b_Mpc:5.2f}     {rk:.5f}             {rg:.5f}")
    return results


def main():
    print("="*70)
    print(" Scan ratio_kappa vs c_NFW  --  if 1-1/c² is the cause, ratio scales")
    print("="*70)
    t0 = time.time()
    for c_NFW in (5, 7, 10):
        run_for_c(c_NFW, n_fine_per_rs=8)
    print(f"\n  Total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
