#!/usr/bin/env python3
"""
Regression test: kappa_FLRW must vanish in pure FLRW (Phi = 0 everywhere).

Why this test exists
--------------------
The Sachs equation in `PerturbedFLRWMetricFast` integrates the *full*
Riemann tensor (FLRW background + perturbations). Analytically the FLRW
contributions cancel in the optical tidal matrix R_AB (Block 1's +H'
diagonal exactly cancels Block 3's -H' Kronecker piece), so
kappa_FLRW = 0 in pure FLRW.

A previous bug in `compute_tensorial_acceleration` violated the null
condition over Gpc-scale paths, contaminating R_AB and producing a
spurious kappa_FLRW ~ 6e-3 in pure FLRW. This test guards against
re-introduction of that or similar bugs.

We test both geodesic backends (analytical and tensorial) at a few
cosmologies. Pass criterion: |kappa_FLRW| < KAPPA_TOL.

Usage
-----
    python _tests/test_flrw_regression_kappa.py
"""
import os
import sys
import time

import numpy as np
from scipy import interpolate
from scipy.optimize import brentq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from excalibur.core.constants import c, one_Mpc
from excalibur.core.cosmology import LCDM_Cosmology
from excalibur.grid.grid import Grid
from excalibur.grid.interpolator_fast import InterpolatorFast
from excalibur.grid.analytical_bypass import AnalyticalBypassInterpolator
from excalibur.objects.nfw_halo import NFWHalo
from excalibur.core.constants import one_Msun
from excalibur.metrics.perturbed_flrw_metric_fast import PerturbedFLRWMetricFast
from excalibur.photon.photon import Photon
from excalibur.integration.integrator import Integrator
from excalibur.observables.sachs_basis import init_sachs_basis
from excalibur.observables.optical_tidal_matrix import lensing_from_jacobi


KAPPA_TOL = 1e-5    # the historical bug produced |kappa| ~ 6e-3
GAMMA_TOL = 1e-5


def _make_zero_phi_setup(H0, Omega_m, Omega_lambda):
    """Build a zero-Phi grid + interpolator + metric. Pure FLRW."""
    cosmo = LCDM_Cosmology(H0, Omega_m=Omega_m, Omega_r=0,
                           Omega_lambda=Omega_lambda)
    _ = cosmo.a_of_eta(1e18)
    eta_0 = cosmo._eta_at_a1
    a_0 = cosmo.a_of_eta(eta_0)

    eta_arr = np.linspace(0.5 * eta_0, eta_0, 1500)
    a_arr = np.array([cosmo.a_of_eta(e) for e in eta_arr])
    a_of_eta = interpolate.interp1d(eta_arr, a_arr, kind="cubic",
                                    fill_value="extrapolate")

    box_Mpc = 1500.0
    grid_size = box_Mpc * one_Mpc
    N = 8                                   # placeholder grid; never queried
    grid = Grid(shape=(N,) * 3, spacing=(grid_size / N,) * 3,
                origin=np.zeros(3))
    grid.add_field("Phi", np.zeros((N,) * 3))
    base = InterpolatorFast(grid, boundary="clamp")

    # Wrap the placeholder grid with an analytical bypass on a halo of
    # negligible mass and infinite radius. This routes every potential
    # query to the analytical NFW formulas, which return ~0 everywhere
    # (numerically clean Phi = 0) while exposing the Hessian needed by
    # the lensing extension.
    center = np.array([0.5, 0.5, 0.5]) * grid_size
    tiny_halo = NFWHalo(M_200=1e-30 * one_Msun, c_NFW=7.0, center=center)
    interp = AnalyticalBypassInterpolator(
        base_interp=base,
        analytical_source=tiny_halo,
        bypass_radius=1e3 * grid_size,
        bypass_fields=("Phi",),
        time_derivative=0.0,
    )
    return cosmo, eta_0, a_0, a_of_eta, grid, interp, grid_size


def _trace_one_photon(metric, eta_0, a_0, grid_size, cosmo, dt):
    """Trace a single photon through pure FLRW from observer to z~1.
    Returns (kappa, |gamma|, lambda_S)."""
    # Keep observer and target away from the grid boundary; the
    # interpolator needs >= 1 cell of margin (here ~47 Mpc with N=32).
    obs_pos = np.array([grid_size / 2, grid_size / 2, 0.1 * grid_size])
    target = obs_pos + np.array([1.0 * one_Mpc, 0.0, 0.7 * grid_size])

    obs_4d = np.array([eta_0, *obs_pos])
    direction = target - obs_pos
    direction /= np.linalg.norm(direction)

    g = metric.metric_tensor(obs_4d)
    k_spatial = direction * c
    spatial_sq = (g[1, 1] * k_spatial[0] ** 2
                  + g[2, 2] * k_spatial[1] ** 2
                  + g[3, 3] * k_spatial[2] ** 2)
    k0 = -np.sqrt(abs(-spatial_sq / g[0, 0]))
    k_mu = np.array([k0, *k_spatial])

    e1, e2 = init_sachs_basis(k_mu, g, a_0)
    p = Photon(obs_4d.copy(), k_mu.copy())
    p.e1, p.e2 = e1.copy(), e2.copy()
    p.D_flat = np.zeros(4)
    p.P_flat = np.array([1.0, 0.0, 0.0, 1.0])

    D_s = cosmo.comoving_distance(1.0)
    D_s = min(D_s, 0.45 * grid_size)
    lambda_total = D_s / c

    integ = Integrator(metric=metric, dt=dt, mode="sequential",
                       integrator="rk4", rtol=1e-8, atol=1e-13)
    integ.integrate_single(p, stop_mode="affine", stop_value=lambda_total)

    D_norm = p.D_flat / p.lambda_affine
    kappa, _, gamma = lensing_from_jacobi(D_norm)
    return kappa, gamma, p.lambda_affine


def _run_case(H0, Omega_m, Omega_lambda, analytical_geodesics):
    cosmo, eta_0, a_0, a_of_eta, grid, interp, grid_size = \
        _make_zero_phi_setup(H0, Omega_m, Omega_lambda)
    metric = PerturbedFLRWMetricFast(
        a_of_eta=a_of_eta, grid=grid, interpolator=interp,
        adot_of_eta=cosmo.adot_of_eta, cosmology=cosmo,
        enable_lensing=True, slow_roll=True,
        analytical_geodesics=analytical_geodesics,
    )
    dt = (1.0 * one_Mpc) / c     # ~1 Mpc step (safe for FLRW)
    return _trace_one_photon(metric, eta_0, a_0, grid_size, cosmo, dt)


def test_kappa_flrw_vanishes():
    """Pure FLRW must give |kappa| < KAPPA_TOL across cosmologies and backends."""
    cases = [
        # (label, H0, Om, Ol, analytical_geodesics)
        ("LCDM, analytical",  70.0, 0.3, 0.7, True),
        ("LCDM, tensorial",   70.0, 0.3, 0.7, False),
        ("EdS, analytical",   70.0, 1.0, 0.0, True),
        ("pure-Lambda, analytical", 70.0, 0.0, 1.0, True),
    ]
    failures = []
    print("=" * 70)
    print("  FLRW regression test: kappa_FLRW must vanish")
    print("=" * 70)
    for label, H0, Om, Ol, analytical in cases:
        t0 = time.time()
        kappa, gamma, lamS = _run_case(H0, Om, Ol, analytical)
        dt_s = time.time() - t0
        ok = abs(kappa) < KAPPA_TOL and abs(gamma) < GAMMA_TOL and lamS > 0.0
        flag = "OK" if ok else "FAIL"
        print(f"  [{flag}]  {label:30s}  kappa = {kappa:+.3e}  "
              f"|gamma| = {gamma:.3e}  lambda_S = {lamS:.3e} s  "
              f"({dt_s:.1f}s)")
        if not ok:
            failures.append((label, kappa, gamma, lamS))

    assert not failures, (
        f"FLRW regression: {len(failures)} case(s) exceeded "
        f"|kappa| < {KAPPA_TOL:.0e} or |gamma| < {GAMMA_TOL:.0e}: {failures}"
    )
    print("\n  All FLRW cases pass.")


if __name__ == "__main__":
    test_kappa_flrw_vanishes()
