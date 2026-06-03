#!/usr/bin/env python3
"""Diagnostic: why does z_end_py_conf differ from z_s by ~10% in
test_numba_nfw_screen_convention.py?

The failing assertion:
    assert abs(z_end_py_conf - setup["z_s"]) < 5e-3
    -> got 0.0086 (z_end = 0.098, z_s = 0.089)

Setup of that test:
    box = 400 Mpc
    halo at center, obs at z=5 Mpc (so d_l ~ 195 Mpc)
    d_s = min(comoving(z=0.12), 0.95 * (grid - 5 Mpc))
        = min(comoving(0.12), 0.95 * 395 Mpc)
        = min(~510 Mpc, ~375 Mpc) = 375 Mpc
        -> z_s computed by brentq to match 375 Mpc comoving
        -> z_s ~ 0.0893

Photon integrated with lambda_total = d_s / c.
At end of integration, x[0] = eta_end. z_end = 1/a(eta_end) - 1.

If the photon followed a pure FLRW null geodesic with k0 = -c/a^2 ish, then
eta_end - eta_0 = -lambda_total / a^2_avg, so:
    chi_traveled = c * lambda_total = d_s
    eta_change   = -d_s / c (for k0 ~ -c, conformal screen with a=1 effectively)

But the simulation uses a=1, adot=0 in the geodesic equations (specialized
kernel for the conformal screen). So in conformal coords, the photon travels
chi = c * lambda. Same here. So the eta arrival should match.

Then 1/a(eta_arrival) - 1 should equal z_s exactly. Unless...

Hypotheses to test:
 H1: the python backend (not numba spec) uses full FLRW a, adot in the
     conformal screen path. So lambda_conf != lambda_phys in the python case.
 H2: lambda_total = d_s / c is the CONFORMAL lambda? or physical?
 H3: precision issue in a_of_eta interpolator.

We'll reproduce the test setup and print:
 - z_s target
 - eta_0 and eta_end_python_conformal
 - integrated lambda
 - relations
"""

import os, sys
import numpy as np
from scipy import interpolate
from scipy.optimize import brentq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from excalibur.core.constants import c, one_Mpc, one_Msun
from excalibur.core.cosmology import LCDM_Cosmology
from excalibur.grid.amr_grid import AMRGrid, AMRInterpolator
from excalibur.grid.analytical_bypass import AnalyticalBypassInterpolator
from excalibur.grid.grid import Grid
from excalibur.integration.integrator import Integrator
from excalibur.metrics.perturbed_flrw_metric_fast import PerturbedFLRWMetricFast
from excalibur.objects.nfw_halo import NFWHalo
from excalibur.observables.sachs_basis import init_sachs_basis
from excalibur.photon.photon import Photon


def main():
    print("=" * 76)
    print(" DIAGNOSE z_end failure in test_numba_nfw_screen_convention.py")
    print("=" * 76)

    # Mirror the test setup exactly
    cosmo = LCDM_Cosmology(70.0, Omega_m=0.3, Omega_r=0.0, Omega_lambda=0.7)
    _ = cosmo.a_of_eta(1e18)
    eta_0 = cosmo._eta_at_a1
    a_0 = cosmo.a_of_eta(eta_0)

    eta_min = 0.5 * eta_0
    eta_arr = np.linspace(eta_min, eta_0, 2000)
    a_arr = np.array([cosmo.a_of_eta(eta) for eta in eta_arr])
    a_of_eta = interpolate.interp1d(eta_arr, a_arr, kind="cubic", fill_value="extrapolate")

    box_mpc = 400.0
    n_root = 64
    grid_size = box_mpc * one_Mpc
    root_grid = Grid(
        shape=(n_root, n_root, n_root),
        spacing=(grid_size / n_root,) * 3,
        origin=np.zeros(3),
    )

    halo = NFWHalo(2e15 * one_Msun, 7.0, np.array([0.5, 0.5, 0.5]) * grid_size)
    coords = np.linspace(0.0, grid_size, n_root)
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
    root_grid.add_field("Phi", halo.potential(x, y, z))

    amr_interp = AnalyticalBypassInterpolator(
        base_interp=AMRInterpolator(AMRGrid.from_field(
            root_grid, "Phi",
            lambda x_val, y_val, z_val: halo.potential(x_val, y_val, z_val),
            max_level=3, ratio=4, refine_threshold=0.005, refine_mode="gradient",
            min_patch_cells=32, boundary="clamp", scheme="tricubic", verbose=False,
        ), boundary="clamp", scheme="tricubic"),
        analytical_source=halo,
        bypass_radius=np.inf,
        bypass_fields=("Phi",),
        time_derivative=0.0,
    )

    obs_pos = np.array([box_mpc / 2.0, box_mpc / 2.0, 5.0]) * one_Mpc
    center = halo.center
    d_l = float(np.linalg.norm(center - obs_pos))
    d_s = min(cosmo.comoving_distance(0.12), 0.95 * (grid_size - np.min(obs_pos)))
    z_l = brentq(lambda redshift: cosmo.comoving_distance(redshift) - d_l, 0.0, 5.0)
    z_s = brentq(lambda redshift: cosmo.comoving_distance(redshift) - d_s, 0.0, 5.0)

    print(f"\nSetup:")
    print(f"  d_l = {d_l/one_Mpc:.2f} Mpc, z_l = {z_l:.6f}")
    print(f"  d_s = {d_s/one_Mpc:.2f} Mpc, z_s = {z_s:.6f}")

    dir_hat = (center - obs_pos) / d_l
    seed = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(seed, dir_hat)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    e_perp1 = seed - np.dot(seed, dir_hat) * dir_hat
    e_perp1 /= np.linalg.norm(e_perp1)
    impact_parameter = 1.0 * one_Mpc
    target = center + impact_parameter * e_perp1
    dt = halo.r_s / (8.0 * c)
    lambda_total = d_s / c

    # Build conformal-screen metric
    metric = PerturbedFLRWMetricFast(
        a_of_eta=a_of_eta, grid=root_grid, interpolator=amr_interp,
        adot_of_eta=cosmo.adot_of_eta, cosmology=cosmo,
        enable_lensing=True, slow_roll=True,
        sachs_screen_convention="conformal_metric",
    )

    # Make photon
    obs_4d = np.array([eta_0, *obs_pos])
    direction = (target - obs_pos) / np.linalg.norm(target - obs_pos)
    g_mu_nu = metric.metric_tensor(obs_4d)
    g_init = g_mu_nu / (a_0 * a_0)
    basis_a = 1.0
    k_spatial = direction * c
    spatial_sq = (g_init[1, 1] * k_spatial[0]**2 + g_init[2, 2] * k_spatial[1]**2
                  + g_init[3, 3] * k_spatial[2]**2)
    k0 = -np.sqrt(abs(-spatial_sq / g_init[0, 0]))
    k_mu = np.array([k0, *k_spatial])
    e1_mu, e2_mu = init_sachs_basis(k_mu, g_init, basis_a, convention="conformal_metric")

    photon = Photon(obs_4d.copy(), k_mu.copy())
    photon.e1 = e1_mu.copy()
    photon.e2 = e2_mu.copy()
    photon.D_flat = np.zeros(4)
    photon.P_flat = np.array([1.0, 0.0, 0.0, 1.0])

    print(f"\nInitial state:")
    print(f"  eta_0 = {eta_0:.6e} s,  a(eta_0) = {a_0:.6f}")
    print(f"  k^0 = {k0:.6e}")
    print(f"  k^z = {k_spatial[2]:.6e}")
    print(f"  k^0 / c = {k0/c:.6f}  (expected ~-1 for conformal screen)")

    # Integrate with python backend
    integrator = Integrator(
        metric=metric, dt=dt, mode="sequential",
        integrator="rk4", rtol=1e-8, atol=1e-13,
    )
    integrator.integrate_single(
        photon, stop_mode="affine", stop_value=lambda_total, record_every=0,
    )

    eta_end = photon.x[0]
    a_end = cosmo.a_of_eta(eta_end)
    z_end = 1.0 / a_end - 1.0

    print(f"\nFinal state after lambda = {photon.lambda_affine:.6e} s:")
    print(f"  eta_end = {eta_end:.6e} s")
    print(f"  a(eta_end) = {a_end:.6f}")
    print(f"  z_end = {z_end:.6f}   (target z_s = {z_s:.6f})")
    print(f"  z_end - z_s = {z_end - z_s:.4e}   (test tolerance: < 5e-3)")
    print(f"  delta_eta = eta_0 - eta_end = {(eta_0 - eta_end):.6e} s")
    print(f"  -> delta_eta * c / one_Mpc = {(eta_0 - eta_end)*c/one_Mpc:.4f} Mpc")
    print(f"  (compare to d_s = {d_s/one_Mpc:.4f} Mpc)")

    # ----------------------------------------------------------------
    # Compare with naive expectation
    # ----------------------------------------------------------------
    print("\n" + "-" * 76)
    print("ANALYSIS:")
    print("-" * 76)
    print(f"  The test sets lambda_total = d_s / c = {lambda_total:.6e} s.")
    print(f"  For the conformal screen, lambda_total = v_tilde_S (conformal affine).")
    print(f"  For a radial null geodesic in conformal coords: v_tilde = chi_S / c.")
    print(f"  So the photon SHOULD traverse chi = c * v_tilde = d_s in comoving distance.")
    print(f"  That puts it at z_s exactly, by construction.")
    print()
    print(f"  But python backend uses FULL FLRW geodesic equations with physical a,")
    print(f"  not the conformal-screen simplification (only the Sachs screen choice")
    print(f"  is conformal in the python path). So the integrated lambda is the")
    print(f"  PHYSICAL affine parameter, not the conformal one.")
    print()
    print(f"  Per Fleury Table 5.1: dv = a^2 * d(v_tilde).")
    print(f"  Hence lambda_phys integrates as int a^2 d(v_tilde).")
    print(f"  If we expect lambda_total = d_s/c (interpreted as PHYSICAL):")
    print(f"    chi_traveled / c = int (1/a^2) * dv_phys, evaluated from O to S.")
    print(f"    This is NOT equal to d_s / c unless a == 1 throughout.")
    print()
    print(f"  -> The test was assuming lambda_total = conformal affine, but the")
    print(f"     python solver integrates physical affine. Hence the photon arrives")
    print(f"     at a slightly different eta than z_s.")
    print(f"     This is consistent with z_end > z_s (photon went farther in eta).")
    print()

    # Compute what lambda would put us EXACTLY at z_s
    eta_s = brentq(lambda eta: cosmo.a_of_eta(eta) - 1.0/(1.0+z_s), eta_min, eta_0)
    print(f"  Target eta_s (for which a=1/(1+z_s)): {eta_s:.6e}")
    print(f"  delta_eta_needed = {eta_0 - eta_s:.6e} s")
    print(f"  -> corresponds to chi = {(eta_0-eta_s)*c/one_Mpc:.4f} Mpc (conformal)")
    print(f"     vs d_s = {d_s/one_Mpc:.4f} Mpc")
    print(f"     ratio = {(eta_0-eta_s)*c/d_s:.6f}")


if __name__ == "__main__":
    main()
