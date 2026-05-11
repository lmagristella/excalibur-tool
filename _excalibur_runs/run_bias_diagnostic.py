#!/usr/bin/env python3
"""
Diagnostic: is the 2% kappa/gamma bias an integration error or a convention issue?

Strategy: integrate 3 photons at large b (b > r_s, where Born approximation is
excellent and the bias is purely a global multiplicative factor) at TWO step
sizes. If the ratio num/analytic varies with step => integration error. If it
stays at 0.9796 => convention/physics issue.
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


def run_at_step(n_fine_per_rs, b_test_Mpc, halo, metric,
                obs_pos, center, e_perp1, eta_0, a_0,
                lambda_total):
    dt_fine = halo.r_s / (n_fine_per_rs * c)
    integ = Integrator(metric=metric, dt=dt_fine, mode="sequential",
                       integrator="rk4", rtol=1e-8, atol=1e-13)
    out = []
    for b_Mpc in b_test_Mpc:
        target = center + b_Mpc * one_Mpc * e_perp1
        p = make_photon(obs_pos, target, metric, eta_0, a_0)
        integ.integrate_single(p, stop_mode="affine", stop_value=lambda_total)
        D_norm = p.D_flat / p.lambda_affine
        kappa, mu, shear = lensing_from_jacobi(D_norm)
        out.append((b_Mpc, kappa, shear))
    return out


def main():
    # ---- Cosmo + halo ----
    H0 = 70.0
    cosmo = LCDM_Cosmology(H0, Omega_m=0.3, Omega_r=0, Omega_lambda=0.7)
    _ = cosmo.a_of_eta(1e18)
    eta_0 = cosmo._eta_at_a1
    a_0   = cosmo.a_of_eta(eta_0)
    eta_min = 0.5 * eta_0
    eta_arr = np.linspace(eta_min, eta_0, 2000)
    a_arr   = np.array([cosmo.a_of_eta(e) for e in eta_arr])
    a_of_eta = interpolate.interp1d(eta_arr, a_arr, kind="cubic", fill_value="extrapolate")

    box_Mpc = 1950.0
    grid_size = box_Mpc * one_Mpc
    N_root = 16
    root_grid = Grid(shape=(N_root,)*3, spacing=(grid_size/N_root,)*3,
                     origin=np.zeros(3))
    root_grid.add_field("Phi", np.zeros((N_root,)*3))
    M_200, c_NFW = 2e15 * one_Msun, 7.0
    center = np.array([0.5, 0.5, 0.5]) * grid_size
    halo = NFWHalo(M_200, c_NFW, center)
    R200_Mpc, rs_Mpc = halo.R_200/one_Mpc, halo.r_s/one_Mpc

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

    # ---- Geometry ----
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
    z_l      = brentq(lambda z: cosmo.comoving_distance(z) - D_l, 0.0, 5.0)
    z_source = brentq(lambda z: cosmo.comoving_distance(z) - D_s, 0.0, 5.0)
    lambda_total = D_s / c

    # ---- Sigma_cr (current convention) ----
    Sigma_cr_eff = (c**2/(4*np.pi*G)) * D_s/(D_l*D_ls) / (1.0 + z_l)
    Sigma_cr_phys = (c**2/(4*np.pi*G)) * D_s/(D_l*D_ls) * (1.0 + z_l)  # standard form
    print(f"  z_l={z_l:.4f}  z_s={z_source:.4f}  (1+z_l)^2 = {(1+z_l)**2:.4f}")
    print(f"  Sigma_cr_eff  (code, /(1+z_l)) = {Sigma_cr_eff:.4e}")
    print(f"  Sigma_cr_phys (textbook, *(1+z_l)) = {Sigma_cr_phys:.4e}")
    print(f"  ratio phys/eff = {Sigma_cr_phys/Sigma_cr_eff:.4f}\n")

    # ---- Test photons at large b ----
    b_test = [0.8, 1.0, 1.5, 3.0]   # Mpc

    for n_fine in (2, 8, 16):
        t0 = time.time()
        results = run_at_step(n_fine, b_test, halo, metric,
                              obs_pos, center, e_perp1, eta_0, a_0,
                              lambda_total)
        print(f"--- n_fine_per_rs={n_fine}  step={halo.r_s/n_fine/one_Mpc*1e3:.0f} kpc"
              f"  ({time.time()-t0:.0f}s) ---")
        print("  b[Mpc]   kappa_num     ka_eff      ka_phys     ratio_eff   ratio_phys")
        for b_Mpc, k_num, _g_num in results:
            b_m = b_Mpc * one_Mpc
            ka_eff  = halo.kappa_analytic(b_m, Sigma_cr_eff)
            ka_phys = halo.kappa_analytic(b_m, Sigma_cr_phys)
            r_eff   = k_num / ka_eff if ka_eff != 0 else float("nan")
            r_phys  = k_num / ka_phys if ka_phys != 0 else float("nan")
            print(f"  {b_Mpc:5.2f}    {k_num:+.4e}   {ka_eff:.4e}  "
                  f"{ka_phys:.4e}   {r_eff:.5f}    {r_phys:.5f}")
        print()


if __name__ == "__main__":
    main()
