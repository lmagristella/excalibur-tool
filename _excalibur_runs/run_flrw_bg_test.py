#!/usr/bin/env python3
"""
Pure-FLRW Sachs test: photon with NO halo.
Convention says kappa_FLRW = 0  (D_norm = I in unlensed background).
We measure what the integrator actually returns.
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


def main():
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
    root_grid.add_field("Phi", np.zeros((N_root,)*3))    # Phi = 0 everywhere

    # Tiny-mass halo wrapped by analytical bypass: Phi -> 0 everywhere
    # (we keep the bypass to dodge a NaN bug in raw InterpolatorFast at certain points).
    center_box = np.array([0.5, 0.5, 0.5]) * grid_size
    tiny_halo  = NFWHalo(1e-30 * one_Msun, 7.0, center_box)
    base_interp = InterpolatorFast(root_grid, boundary="clamp")
    interp = AnalyticalBypassInterpolator(
        base_interp=base_interp, analytical_source=tiny_halo,
        bypass_radius=1e3*grid_size, bypass_fields=("Phi",), time_derivative=0.0,
    )

    use_analytical = bool(int(os.environ.get("ANALYTICAL_GEO", "0")))
    metric = PerturbedFLRWMetricFast(
        a_of_eta=a_of_eta, grid=root_grid, interpolator=interp,
        adot_of_eta=cosmo.adot_of_eta, cosmology=cosmo,
        enable_lensing=True, slow_roll=True,
        analytical_geodesics=use_analytical,
    )
    print(f"  analytical_geodesics = {use_analytical}")

    obs_pos = np.array([box_Mpc/2, box_Mpc/2, 5.0]) * one_Mpc
    center = np.array([0.5, 0.5, 0.5]) * grid_size
    D_l = np.linalg.norm(center - obs_pos)
    dir_hat = (center - obs_pos) / D_l
    seed = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(seed, dir_hat)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    e_perp1 = seed - np.dot(seed, dir_hat) * dir_hat
    e_perp1 /= np.linalg.norm(e_perp1)

    D_s = cosmo.comoving_distance(1.0)
    D_s = min(D_s, 0.95 * (grid_size - np.min(obs_pos)))
    z_l = brentq(lambda z: cosmo.comoving_distance(z) - D_l, 0.0, 5.0)
    z_s = brentq(lambda z: cosmo.comoving_distance(z) - D_s, 0.0, 5.0)
    lambda_total = D_s / c

    # Use the same step size as previous diagnostic (n_fine=8, with c_NFW=7 r_s)
    halo_for_step = NFWHalo(2e15*one_Msun, 7.0, center)
    dt_fine = halo_for_step.r_s / (8 * c)
    integ = Integrator(metric=metric, dt=dt_fine, mode="sequential",
                       integrator="rk4", rtol=1e-8, atol=1e-13)

    print("="*70)
    print(" PURE FLRW BACKGROUND  --  no halo, Phi = 0 everywhere")
    print("="*70)
    print(f"  z_l = {z_l:.4f}, z_s = {z_s:.4f}")
    print(f"  D_l = {D_l/one_Mpc:.1f} Mpc, D_s = {D_s/one_Mpc:.1f} Mpc")
    print(f"  Step = {dt_fine*c/one_Mpc*1e3:.0f} kpc, n_steps = {int(np.ceil(D_s/(dt_fine*c)))}")
    print()

    for b_Mpc, renorm in [(1.0, 0), (1.0, 1)]:
        target = center + b_Mpc * one_Mpc * e_perp1
        p = make_photon(obs_pos, target, metric, eta_0, a_0)
        t0 = time.time()
        integ.integrate_single(p, stop_mode="affine", stop_value=lambda_total,
                               renormalize_every=renorm, trace_norm=True)
        nh = p.norm_history
        max_null = max(nh) if nh else 0.0
        print(f"\n  renormalize_every={renorm}:  max relative null violation = {max_null:.4e}")
        D_norm = p.D_flat / p.lambda_affine
        kappa, mu, shear = lensing_from_jacobi(D_norm)
        print(f"  b={b_Mpc:5.2f} Mpc:  kappa = {kappa:+.6e}   "
              f"|gamma| = {shear:.4e}   mu = {mu:+.6f}   ({time.time()-t0:.0f}s)")
        print(f"            D_flat = {p.D_flat}")
        print(f"            lambda_actual = {p.lambda_affine:.6e}  "
              f"(expected {lambda_total:.6e})")
        p.norm_history = []  # reset for next iteration

    # Reference:  D_A_FLRW(z_s) / chi_s
    DA_s = cosmo.angular_diameter_distance(z_s)
    chi_s = D_s
    print()
    print(f"  D_A_FLRW(z_s) = {DA_s/one_Mpc:.2f} Mpc")
    print(f"  chi_s         = {chi_s/one_Mpc:.2f} Mpc")
    print(f"  D_A/chi       = {DA_s/chi_s:.6f} = 1/(1+z_s) = {1/(1+z_s):.6f}")


if __name__ == "__main__":
    main()
