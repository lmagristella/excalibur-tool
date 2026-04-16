#!/usr/bin/env python3
r"""
First weak-lensing simulation with excalibur.

Backward ray tracing through a SIS-like halo with:
    - enable_lensing=True   -> 24-component state (geodesic + Sachs + Jacobi)
    - slow_roll=True        -> skip temporal derivatives of Phi (static potential)

At the end, the Jacobi map D_{AB} is read from the photon and the
convergence kappa, magnification mu, and shear |gamma| are printed.
"""

import numpy as np
from scipy import interpolate
import sys, os, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# -- excalibur imports ----------------------------------------------
from excalibur.grid.grid import Grid
from excalibur.grid.interpolator_4d_fast import InterpolatorFast as Interpolator4DFast
from excalibur.metrics.perturbed_flrw_metric_fast import PerturbedFLRWMetricFast
from excalibur.photon.photon import Photon
from excalibur.photon.photons import Photons
from excalibur.integration.integrator import Integrator
from excalibur.core.constants import c, one_Mpc, one_Msun
from excalibur.core.cosmology import LCDM_Cosmology
from excalibur.objects.spherical_mass import spherical_mass
from excalibur.observables.sachs_basis import init_sachs_basis
from excalibur.observables.optical_tidal_matrix import lensing_from_jacobi


def main():
    t0 = time.time()

    # =================================================================
    # 1.  COSMOLOGY
    # =================================================================
    print("1. Cosmology ...")
    H0 = 70                # km/s/Mpc
    Omega_m = 0.3
    Omega_lambda = 0.7
    cosmo = LCDM_Cosmology(H0, Omega_m=Omega_m, Omega_r=0, Omega_lambda=Omega_lambda)

    # Conformal time at a = 1
    _ = cosmo.a_of_eta(1e18)          # init
    eta_0 = cosmo._eta_at_a1          # eta today ~ 1.46e18 s
    a_0   = cosmo.a_of_eta(eta_0)

    # Build a(eta) interpolator over the backward range
    eta_min = 0.5 * eta_0
    eta_arr = np.linspace(eta_min, eta_0, 2000)
    a_arr   = np.array([cosmo.a_of_eta(e) for e in eta_arr])
    a_of_eta = interpolate.interp1d(eta_arr, a_arr, kind="cubic",
                                     fill_value="extrapolate")

    print(f"   eta0 = {eta_0:.4e} s,  a(eta0) = {a_0:.6f}")

    # =================================================================
    # 2.  GRID + MASS DISTRIBUTION
    # =================================================================
    print("2. Grid + mass ...")
    N         = 128
    box_Mpc   = 500.0
    grid_size = box_Mpc * one_Mpc
    dx = dy = dz = grid_size / N

    grid = Grid(
        shape   = (N, N, N),
        spacing = (dx, dy, dz),
        origin  = np.array([0.0, 0.0, 0.0]),
    )

    # Spherical halo at the box centre
    M      = 1e14 * one_Msun          # 10^14 Msun
    R_vir  = 5.0 * one_Mpc
    center = np.array([0.5, 0.5, 0.5]) * grid_size
    halo   = spherical_mass(M, R_vir, center)

    x1d = np.linspace(0, grid_size, N)
    X, Y, Z = np.meshgrid(x1d, x1d, x1d, indexing="ij")
    phi_field = halo.potential(X, Y, Z)
    grid.add_field("Phi", phi_field)

    phi_max = np.max(np.abs(phi_field))
    print(f"   Grid {N}^3,  box = {box_Mpc:.0f} Mpc")
    print(f"   M = {M/one_Msun:.1e} Msun,  R_vir = {R_vir/one_Mpc:.0f} Mpc")
    print(f"   |Phi|_max = {phi_max:.3e} m^2/s^2   ->  Phi/c^2 = {phi_max/c**2:.3e}")

    # =================================================================
    # 3.  INTERPOLATOR + METRIC  (with lensing & slow-roll)
    # =================================================================
    print("3. Metric (lensing ON, slow-roll ON) ...")
    interpolator = Interpolator4DFast(grid, boundary="clamp", scheme="trilinear")

    metric = PerturbedFLRWMetricFast(
        a_of_eta       = a_of_eta,
        grid           = grid,
        interpolator   = interpolator,
        adot_of_eta    = cosmo.adot_of_eta,
        cosmology      = cosmo,
        enable_lensing = True,
        slow_roll      = True,
    )
    print("   [ok] metric ready")

    # =================================================================
    # 4.  PHOTON SETUP  --  single photon aimed at the halo
    # =================================================================
    print("4. Photon setup ...")

    # Observer at one corner of the box, looking toward centre
    obs_pos = np.array([10.0, 10.0, 10.0]) * one_Mpc
    obs_4d  = np.array([eta_0, *obs_pos])

    direction = center - obs_pos
    direction /= np.linalg.norm(direction)

    # Build k^mu from spatial direction + null condition
    g = metric.metric_tensor(obs_4d)
    k_spatial = direction * c          # physical spatial components
    spatial_sq = (g[1,1]*k_spatial[0]**2
                + g[2,2]*k_spatial[1]**2
                + g[3,3]*k_spatial[2]**2)
    k0 = -np.sqrt(abs(-spatial_sq / g[0,0]))   # backward tracing  -> k^0 < 0
    k_mu = np.array([k0, *k_spatial])

    # Sachs basis at the observer
    e1, e2 = init_sachs_basis(-k_mu, g, a_0)   # pass forward-pointing k
    # For backward tracing we flip Sachs sign convention later if needed;
    # the transport equations are linear so the sign is irrelevant for |kappa|, |gamma|.

    photon = Photon(obs_4d.copy(), k_mu.copy())
    # Attach lensing IC
    photon.e1     = e1.copy()
    photon.e2     = e2.copy()
    photon.D_flat = np.array([0.0, 0.0, 0.0, 0.0])   # Jacobi IC: D(0) = 0
    photon.P_flat = np.array([1.0, 0.0, 0.0, 1.0])   # Jacobi IC: P(0) = I

    # Null-condition check
    rel_err = photon.null_condition_relative_error(metric=metric)
    print(f"   k^mu = [{k0:.4e}, {k_spatial[0]:.4e}, {k_spatial[1]:.4e}, {k_spatial[2]:.4e}]")
    print(f"   null condition relative error = {rel_err:.2e}")
    print(f"   e1^mu = {e1}")
    print(f"   e2^mu = {e2}")

    # =================================================================
    # 5.  INTEGRATOR
    # =================================================================
    print("5. Integrator ...")

    # Time-step: fraction of grid spacing / c
    # k^0 < 0 (backward tracing), so dlambda > 0 gives deta = k^0 dlambda < 0 (past)
    dt_init = grid.spacing[0] / (10.0 * c)      # positive affine step

    integrator = Integrator(
        metric     = metric,
        dt         = dt_init,
        mode       = "sequential",
        integrator = "rk4",         # RK4  --  simple, robust
        rtol       = 1e-8,
        atol       = 1e-12,
        dt_min     = 1e-20,
        dt_max     = abs(dt_init) * 50,
    )

    n_steps = 200
    print(f"   dt = {dt_init:.3e} s,  n_steps = {n_steps}")

    # =================================================================
    # 6.  INTEGRATION
    # =================================================================
    print("6. Integrating ...")
    t_int = time.time()

    photon.record()
    integrator.integrate_single(
        photon,
        stop_mode  = "steps",
        stop_value = n_steps,
    )

    dt_elapsed = time.time() - t_int
    print(f"   [ok] {n_steps} steps in {dt_elapsed:.2f} s")

    # =================================================================
    # 7.  READ LENSING OBSERVABLES
    # =================================================================
    print("7. Lensing observables ...")

    D_flat = photon.D_flat
    kappa, mu, shear = lensing_from_jacobi(D_flat)

    print(f"   D_flat = [{D_flat[0]:.6e}, {D_flat[1]:.6e}, {D_flat[2]:.6e}, {D_flat[3]:.6e}]")
    print(f"   kappa  (convergence)   = {kappa:.6e}")
    print(f"   mu  (magnification) = {mu:.6e}")
    print(f"   |gamma| (shear)        = {shear:.6e}")

    # Also print final position info
    eta_f = photon.x[0]
    pos_f = photon.x[1:4]
    a_f   = a_of_eta(eta_f)
    dist  = np.linalg.norm(pos_f - center) / one_Mpc

    print(f"\n   Final eta = {eta_f:.4e} s  (deta = {eta_f - eta_0:.3e} s)")
    print(f"   Final a = {a_f:.6f}")
    print(f"   Distance to halo = {dist:.2f} Mpc")

    # =================================================================
    # SUMMARY
    # =================================================================
    total = time.time() - t0
    print("\n" + "=" * 60)
    print("  LENSING SIMULATION SUMMARY")
    print("=" * 60)
    print(f"  Grid           : {N}^3, {box_Mpc:.0f} Mpc")
    print(f"  Halo           : {M/one_Msun:.1e} Msun, R = {R_vir/one_Mpc:.0f} Mpc")
    print(f"  Integrator     : RK4, {n_steps} steps, slow_roll=True")
    print(f"  Convergence kappa  : {kappa:.6e}")
    print(f"  Magnification mu: {mu:.6e}")
    print(f"  Shear |gamma|      : {shear:.6e}")
    print(f"  Time           : {total:.1f} s")
    print("=" * 60)


if __name__ == "__main__":
    main()
