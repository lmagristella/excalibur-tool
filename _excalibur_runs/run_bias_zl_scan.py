#!/usr/bin/env python3
"""alpha vs z_l scan  --  single setup, just move observer."""
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


def make_metric(halo, root_grid, grid_size, cosmo, a_of_eta):
    base = InterpolatorFast(root_grid, boundary="clamp")
    interp = AnalyticalBypassInterpolator(
        base_interp=base, analytical_source=halo,
        bypass_radius=1e3*grid_size, bypass_fields=("Phi",), time_derivative=0.0,
    )
    return PerturbedFLRWMetricFast(
        a_of_eta=a_of_eta, grid=root_grid, interpolator=interp,
        adot_of_eta=cosmo.adot_of_eta, cosmology=cosmo,
        enable_lensing=True, slow_roll=True,
    )


def integrate_photon(metric, integrator, obs_pos, target, eta_0, a_0, lambda_total):
    p = make_photon(obs_pos, target, metric, eta_0, a_0)
    integrator.integrate_single(p, stop_mode="affine", stop_value=lambda_total)
    D_norm = p.D_flat / p.lambda_affine
    kappa, _, _ = lensing_from_jacobi(D_norm)
    return kappa


def main():
    print("="*78, flush=True)
    print(" alpha vs z_l scan  (single setup, move observer only)", flush=True)
    print("="*78, flush=True)

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
    M_200, c_NFW = 2e15 * one_Msun, 7.0
    center = np.array([0.5, 0.5, 0.5]) * grid_size
    halo  = NFWHalo(M_200, c_NFW, center)
    tiny  = NFWHalo(1e-30 * one_Msun, c_NFW, center)
    rs_Mpc = halo.r_s / one_Mpc

    metric_NFW  = make_metric(halo, root_grid, grid_size, cosmo, a_of_eta)
    metric_FLRW = make_metric(tiny, root_grid, grid_size, cosmo, a_of_eta)

    n_fine_per_rs = 8
    dt_fine = halo.r_s / (n_fine_per_rs * c)
    integ_NFW  = Integrator(metric=metric_NFW,  dt=dt_fine, mode="sequential",
                            integrator="rk4", rtol=1e-8, atol=1e-13)
    integ_FLRW = Integrator(metric=metric_FLRW, dt=dt_fine, mode="sequential",
                            integrator="rk4", rtol=1e-8, atol=1e-13)
    print(f"  step={dt_fine*c/one_Mpc*1e3:.0f} kpc, halo M={M_200/one_Msun:.1e} Msun, "
          f"r_s={rs_Mpc*1e3:.0f} kpc", flush=True)

    obs_zs = [5.0, 300.0, 600.0]   # 3 setups for speed
    b_target_Mpc = 1.0
    rows = []

    for obs_z in obs_zs:
        t0 = time.time()
        obs_pos = np.array([box_Mpc/2, box_Mpc/2, obs_z]) * one_Mpc
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

        target = center + b_target_Mpc * one_Mpc * e_perp1

        print(f"\n  obs_z={obs_z:5.1f} Mpc  ->  D_l={D_l/one_Mpc:.1f} Mpc, "
              f"D_s={D_s/one_Mpc:.1f} Mpc, z_l={z_l:.4f}, z_s={z_s:.4f}", flush=True)
        print(f"    integrating FLRW reference ...", flush=True)
        k_FLRW = integrate_photon(metric_FLRW, integ_FLRW, obs_pos, target,
                                   eta_0, a_0, lambda_total)
        print(f"    integrating NFW photon ...", flush=True)
        k_NFW  = integrate_photon(metric_NFW, integ_NFW, obs_pos, target,
                                   eta_0, a_0, lambda_total)
        signal = k_NFW - k_FLRW

        Sigma_cr = (c**2/(4*np.pi*G)) * D_s/(D_l*D_ls) / (1.0 + z_l)
        b_m = np.array([b_target_Mpc * one_Mpc])
        ka_eff = float(halo.kappa_analytic(b_m, Sigma_cr)[0])
        alpha = signal / ka_eff
        rows.append((obs_z, D_l/one_Mpc, z_l, z_s, k_FLRW, k_NFW, ka_eff, signal, alpha))
        print(f"    k_FLRW={k_FLRW:+.5e}  k_NFW={k_NFW:+.5e}  signal={signal:+.4e}  "
              f"k_an={ka_eff:.4e}  alpha={alpha:.5f}  ({time.time()-t0:.0f}s)", flush=True)

    print()
    print(" SUMMARY  alpha vs z_l :", flush=True)
    print("  z_l       z_s       alpha       1+z_l       alpha*(1+z_l)    alpha/(1+z_l)", flush=True)
    for _, _, z_l, z_s, _, _, _, _, alpha in rows:
        print(f"  {z_l:.4f}    {z_s:.4f}    {alpha:.5f}    {1+z_l:.4f}      "
              f"{alpha*(1+z_l):.5f}        {alpha/(1+z_l):.5f}", flush=True)


if __name__ == "__main__":
    main()
