#!/usr/bin/env python3
"""Diagnostic units/debug script for (perturbed) FLRW.

This mirrors `_tests/test_schwarzschild_units_debug.py` but uses a cartesian-like
FLRW metric (`PerturbedFLRWMetricFast`) with a tiny grid and a constant potential.

This is intentionally a *script* (not collected by pytest) meant to be run
manually for quick sanity checks.
"""

import sys

import numpy as np

from excalibur.core.constants import c, one_Mpc
from excalibur.core.cosmology import LCDM_Cosmology
from excalibur.grid.grid import Grid
from excalibur.grid.interpolator_fast import InterpolatorFast
from excalibur.metrics.perturbed_flrw_metric_fast import PerturbedFLRWMetricFast
from excalibur.photon.photons import Photons
from excalibur.integration.integrator import Integrator


def test_flrw_units():
    print("=== TEST DIAGNOSTIC FLRW (PERTURBÉ) ===\n")

    # Minimal cosmology
    H0 = 70
    Omega_m = 0.3
    Omega_r = 0.0
    Omega_lambda = 0.7
    cosmology = LCDM_Cosmology(H0, Omega_m=Omega_m, Omega_r=Omega_r, Omega_lambda=Omega_lambda)

    # Force-initialize internal eta_at_a1 cache
    _ = cosmology.a_of_eta(1e18)
    eta_present = cosmology._eta_at_a1

    print("1. Cosmologie:")
    print(f"   H0={H0} km/s/Mpc, Om={Omega_m}, Or={Omega_r}, Ol={Omega_lambda}")
    print(f"   eta_present ~ {eta_present:.3e}")

    # Tiny grid (keep observer inside bounds)
    N = 8
    grid_size = 10 * one_Mpc
    spacing = (grid_size / N, grid_size / N, grid_size / N)
    origin3 = np.array([0.0, 0.0, 0.0])
    grid = Grid((N, N, N), spacing, origin3)

    # Phi field: dimensionless Phi/c^2 (in the run scripts we often store Phi/c^2)
    grid.add_field("Phi", np.zeros((N, N, N), dtype=float))

    interpolator = InterpolatorFast(grid)
    metric = PerturbedFLRWMetricFast(
        a_of_eta=cosmology.a_of_eta,
        grid=grid,
        interpolator=interpolator,
        analytical_geodesics=False,
    )

    print("\n2. Grille + métrique:")
    print(f"   grid_size={grid_size/one_Mpc:.2f} Mpc, N={N}")
    print(f"   origin={origin3/one_Mpc} Mpc")

    # Observer inside [0, grid_size]^3
    observer_position = np.array([0.2, 0.3, 0.4]) * grid_size
    observer_pos_4d = np.array([eta_present, *observer_position], dtype=float)

    print("\n3. Position observateur 4D:")
    print(f"   [eta, x, y, z] = {observer_pos_4d}")
    print(f"   en Mpc: [eta, {observer_pos_4d[1]/one_Mpc:.3f}, {observer_pos_4d[2]/one_Mpc:.3f}, {observer_pos_4d[3]/one_Mpc:.3f}]")

    # Metric tensor at observer
    g_obs = metric.metric_tensor(observer_pos_4d)
    print("\n4. Métrique à la position de l'observateur:")
    print(f"   g00 = {g_obs[0,0]:.3e}   (attendu ~ -a(eta)^2 c^2)")
    print(f"   g11 = {g_obs[1,1]:.3e}   (attendu ~ +a(eta)^2)")
    print(f"   g22 = {g_obs[2,2]:.3e}")
    print(f"   g33 = {g_obs[3,3]:.3e}")

    # Generate one photon pointing roughly to the center of the box
    photons = Photons(metric)
    center = np.array([0.5, 0.5, 0.5]) * grid_size
    direction_to_center = center - observer_position
    direction_to_center = direction_to_center / np.linalg.norm(direction_to_center)

    print("\n5. Génération d'un photon de test:")
    print(f"   Direction (cartésienne) vers centre: {direction_to_center}")

    photons.generate_cone_random(
        n_photons=1,
        origin=observer_pos_4d,
        central_direction=direction_to_center,
        cone_angle=0.01,
        direction_basis="cartesian",
    )

    if len(photons.photons) == 0:
        print("❌ Aucun photon généré!")
        return False

    photon = photons.photons[0]
    print("\n6. Photon généré:")
    print(f"   Position x = {photon.x}")
    print(f"   Position (en Mpc) = {photon.x[1:]/one_Mpc}")
    print(f"   Vitesse u = {photon.u}")
    print(f"   Vitesse spatiale / c = {photon.u[1:]/c}")

    null_rel = photon.null_condition_relative_error(metric)
    null_abs = photon.null_condition(metric)
    print(f"   Condition nulle: u.u = {null_abs:.3e}")
    print(f"   Condition nulle (rel): {null_rel:.3e}")

    # Very short integration: just a few RK4 steps
    integrator = Integrator(metric=metric, dt=-1.0, mode="sequential", integrator="rk4")

    print("\n7. Test intégration (3 étapes RK4):")
    print(f"   Position initiale (Mpc): {photon.x[1:]/one_Mpc}")

    try:
        integrator.integrate_single(photon, stop_mode="steps", stop_value=10000)
    except Exception as e:
        print(f"   ❌ Erreur d'intégration: {e}")
        return False

    null_rel_after = photon.null_condition_relative_error(metric)
    print(f"   Position finale (Mpc): {photon.x[1:]/one_Mpc}")
    print(f"   Null rel après: {null_rel_after:.3e}")

    print("\n✅ Script FLRW terminé")
    return True


if __name__ == "__main__":
    ok = test_flrw_units()
    if not ok:
        print("\n❌ TEST ÉCHOUÉ - Problème détecté dans FLRW")
        sys.exit(1)
    print("\n✅ TEST RÉUSSI - FLRW OK")
