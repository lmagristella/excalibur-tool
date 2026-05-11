#!/usr/bin/env python3
"""
Test if numerical kappa_FLRW matches the FLRW Ricci focusing prediction.

Plan: vary Omega_m at fixed observer/source geometry and check if
kappa_FLRW scales linearly with Omega_m (matter-only Ricci focusing).
If yes, the bug is the docstring (the code does include FLRW Ricci).
If no, something else is at play.
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


def run_flrw(Omega_m, Omega_lambda, obs_z_Mpc=5.0, n_fine_per_rs=8):
    H0 = 70.0
    cosmo = LCDM_Cosmology(H0, Omega_m=Omega_m, Omega_r=0,
                           Omega_lambda=Omega_lambda)
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
    tiny = NFWHalo(1e-30 * one_Msun, 7.0, center)

    base = InterpolatorFast(root_grid, boundary="clamp")
    interp = AnalyticalBypassInterpolator(
        base_interp=base, analytical_source=tiny,
        bypass_radius=1e3*grid_size, bypass_fields=("Phi",), time_derivative=0.0,
    )
    metric = PerturbedFLRWMetricFast(
        a_of_eta=a_of_eta, grid=root_grid, interpolator=interp,
        adot_of_eta=cosmo.adot_of_eta, cosmology=cosmo,
        enable_lensing=True, slow_roll=True,
    )

    obs_pos = np.array([box_Mpc/2, box_Mpc/2, obs_z_Mpc]) * one_Mpc
    D_l_marker = float(np.linalg.norm(center - obs_pos))   # for direction only
    dir_hat = (center - obs_pos) / D_l_marker
    seed = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(seed, dir_hat)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    e_perp1 = seed - np.dot(seed, dir_hat) * dir_hat
    e_perp1 /= np.linalg.norm(e_perp1)

    D_s = cosmo.comoving_distance(1.0)
    D_s = min(D_s, 0.95 * (grid_size - np.min(obs_pos)))
    z_s = brentq(lambda z: cosmo.comoving_distance(z) - D_s, 0.0, 5.0)
    lambda_total = D_s / c

    rs_for_step = 0.371 * one_Mpc
    dt_fine = rs_for_step / (n_fine_per_rs * c)
    integ = Integrator(metric=metric, dt=dt_fine, mode="sequential",
                       integrator="rk4", rtol=1e-8, atol=1e-13)

    target = center + 1.0 * one_Mpc * e_perp1
    p = make_photon(obs_pos, target, metric, eta_0, a_0)
    integ.integrate_single(p, stop_mode="affine", stop_value=lambda_total)
    D_norm = p.D_flat / p.lambda_affine
    kappa, _, _ = lensing_from_jacobi(D_norm)
    return dict(Omega_m=Omega_m, Omega_lambda=Omega_lambda,
                D_s_Mpc=D_s/one_Mpc, z_s=z_s, kappa_FLRW=kappa,
                D_norm=1.0 - kappa, lambda_S=p.lambda_affine)


def main():
    print("="*84, flush=True)
    print(" kappa_FLRW vs cosmology: probe Hubble drag in the Sachs equation", flush=True)
    print("="*84, flush=True)

    # Vary Omega_m, keep flat (Omega_Lambda = 1 - Omega_m)
    cosmos = [
        (0.0,  1.0),    # pure Lambda  -> Hubble drag from matter = 0
        (0.1,  0.9),
        (0.3,  0.7),    # default LCDM
        (0.5,  0.5),
        (1.0,  0.0),    # EdS, all matter
    ]

    rows = []
    for Om, Ol in cosmos:
        t0 = time.time()
        try:
            r = run_flrw(Om, Ol)
            rows.append(r)
            print(f"  Omega_m={Om:.2f} Omega_L={Ol:.2f}  D_s={r['D_s_Mpc']:.1f} Mpc  "
                  f"z_s={r['z_s']:.4f}  k_FLRW={r['kappa_FLRW']:+.4e}  "
                  f"D_norm={r['D_norm']:.5f}   ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"  Omega_m={Om:.2f} -> ERROR: {e}", flush=True)

    print("\n=== SCALING TEST ===", flush=True)
    if len(rows) >= 2:
        # If kappa_FLRW = A * Omega_m * (H0*D_s/c)^2 + B  with B ~ 0,
        # then kappa_FLRW / [Omega_m * (H0*D_s/c)^2]  should be ~constant
        H0 = 70.0 / (3e5)   # H0 in 1/Mpc
        for r in rows:
            Om = r['Omega_m']
            Ds = r['D_s_Mpc']
            HD2 = (H0 * Ds)**2
            denom = Om * HD2 if Om > 0 else 1.0
            print(f"  Om={Om:.2f} D_s={Ds:.0f}  (H0*Ds/c)^2={HD2:.4f}  "
                  f"k_FLRW={r['kappa_FLRW']:+.4e}  k/(Om*HD^2)={r['kappa_FLRW']/denom:.5e}", flush=True)


if __name__ == "__main__":
    main()
