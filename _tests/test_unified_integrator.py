#!/usr/bin/env python3
"""
Test the unified Integrator with sequential, parallel, and parallel_chunked modes.
Tests both grid-based (perturbed FLRW) and analytical (Schwarzschild) metrics.
"""
import numpy as np
from scipy import interpolate
import os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from excalibur.grid.grid import Grid
from excalibur.grid.interpolator_fast import InterpolatorFast
from excalibur.metrics.perturbed_flrw_metric_fast import PerturbedFLRWMetricFast
from excalibur.metrics.schwarzschild_metric_cartesian import SchwarzschildMetricCartesian
from excalibur.photon.photon import Photon
from excalibur.integration.integrator import Integrator
from excalibur.core.constants import *
from excalibur.core.cosmology import LCDM_Cosmology
from excalibur.objects.spherical_mass import spherical_mass


def make_grid_metric():
    """Create a grid-based perturbed FLRW metric for testing."""
    N = 32
    grid_size = 2000 * one_Mpc
    dx = dy = dz = grid_size / N
    shape = (N, N, N)
    spacing = (dx, dy, dz)
    origin = (-grid_size/2, -grid_size/2, -grid_size/2)
    grid = Grid(shape, spacing, origin)

    x = y = z = np.linspace(-grid_size/2, grid_size/2, N)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    M = 1e20 * one_Msun
    radius = 10 * one_Mpc
    center = np.array([500.0, 500.0, 500.0]) * one_Mpc
    halo = spherical_mass(M, radius, center)

    phi_field = halo.potential(X, Y, Z)
    grid.add_field("Phi", phi_field)

    H0 = 70
    cosmology = LCDM_Cosmology(H0, Omega_m=0.3, Omega_r=0, Omega_lambda=0.7)
    eta_sample = np.linspace(1.0e25, 5.0e26, 1000)
    a_sample = cosmology.a_of_eta(eta_sample)
    a_of_eta = interpolate.interp1d(eta_sample, a_sample, kind='cubic', fill_value="extrapolate")

    interp = InterpolatorFast(grid)
    metric = PerturbedFLRWMetricFast(a_of_eta, grid, interp)
    return metric, center


def make_analytical_metric():
    """Create a Schwarzschild metric for testing."""
    M = 1e15 * one_Msun
    radius = 2 * G * M / (c**2)
    center = np.array([0.0, 0.0, 0.0])
    metric = SchwarzschildMetricCartesian(mass=M, radius=radius, center=center)
    return metric


def make_photons(n, observer_4d, direction, metric):
    """Create n photons near a direction."""
    photons = []
    for i in range(n):
        # Small angular spread
        angle = (i - n/2) * 0.001
        dir_i = direction.copy()
        dir_i[1] += angle * 0.01
        dir_i = dir_i / np.linalg.norm(dir_i)

        p = Photon(position=observer_4d.copy(), direction=np.concatenate([[1.0], dir_i]))
        p.state_quantities(metric.metric_physical_quantities)
        p.record()
        photons.append(p)
    return photons


def check_photon_results(photons, label):
    """Verify that photons have been integrated (positions changed, history populated)."""
    ok = True
    for i, p in enumerate(photons):
        n_states = len(p.history.states)
        if n_states <= 1:
            print(f"  FAIL [{label}] photon {i}: only {n_states} history states")
            ok = False
    return ok


def test_grid_metric():
    """Test all 3 modes with a grid-based FLRW metric."""
    print("=" * 60)
    print("TEST: Grid-based metric (PerturbedFLRWMetricFast)")
    print("=" * 60)

    metric, center = make_grid_metric()
    observer_eta = 4.4e26
    observer_4d = np.array([observer_eta, 0.0, 0.0, 0.0])
    direction = center / np.linalg.norm(center)

    n_photons = 6
    n_steps = 50
    dt = -1e15

    # --- Sequential ---
    print("\n[1/3] Sequential mode...")
    photons_seq = make_photons(n_photons, observer_4d, direction, metric)
    t0 = time.time()
    integ = Integrator(metric, dt=dt, mode="sequential", integrator="rk4")
    n_ok = integ.integrate(photons_seq, stop_mode="steps", stop_value=n_steps, verbose=False)
    t_seq = time.time() - t0
    seq_ok = check_photon_results(photons_seq, "sequential")
    print(f"  {n_ok}/{n_photons} success, {t_seq:.3f}s, "
          f"history: {len(photons_seq[0].history.states)} states")

    # --- Parallel ---
    print("\n[2/3] Parallel mode...")
    photons_par = make_photons(n_photons, observer_4d, direction, metric)
    t0 = time.time()
    with Integrator(metric, dt=dt, mode="parallel", integrator="rk4", n_workers=2) as integ:
        n_ok = integ.integrate(photons_par, stop_mode="steps", stop_value=n_steps, verbose=False)
    t_par = time.time() - t0
    par_ok = check_photon_results(photons_par, "parallel")
    print(f"  {n_ok}/{n_photons} success, {t_par:.3f}s, "
          f"history: {len(photons_par[0].history.states)} states")

    # --- Parallel chunked ---
    print("\n[3/3] Parallel chunked mode...")
    photons_chu = make_photons(n_photons, observer_4d, direction, metric)
    t0 = time.time()
    with Integrator(metric, dt=dt, mode="parallel_chunked", integrator="rk4",
                    n_workers=2, chunk_size=3) as integ:
        n_ok = integ.integrate(photons_chu, stop_mode="steps", stop_value=n_steps, verbose=False)
    t_chu = time.time() - t0
    chu_ok = check_photon_results(photons_chu, "parallel_chunked")
    print(f"  {n_ok}/{n_photons} success, {t_chu:.3f}s, "
          f"history: {len(photons_chu[0].history.states)} states")

    # --- Compare results ---
    print("\n  Comparing sequential vs parallel final positions...")
    all_close = True
    for i in range(n_photons):
        d_par = np.linalg.norm(photons_seq[i].x - photons_par[i].x)
        d_chu = np.linalg.norm(photons_seq[i].x - photons_chu[i].x)
        if d_par > 1e-6 or d_chu > 1e-6:
            print(f"  photon {i}: seq-par dist={d_par:.2e}, seq-chu dist={d_chu:.2e}")
            all_close = False

    if all_close:
        print("  All positions match between modes!")

    return seq_ok and par_ok and chu_ok


def test_analytical_metric():
    """Test all 3 modes with a Schwarzschild metric."""
    print("\n" + "=" * 60)
    print("TEST: Analytical metric (SchwarzschildMetricCartesian)")
    print("=" * 60)

    metric = make_analytical_metric()
    # Place observer well outside the Schwarzschild radius
    r_s = 2 * G * metric.mass / (c**2)
    r0 = 100 * r_s  # 100x the Schwarzschild radius
    observer_4d = np.array([0.0, r0, 0.0, 0.0])
    direction = np.array([0.0, 0.0, 1.0])  # tangential

    n_photons = 4
    n_steps = 30
    dt = -r0 / (c * 10)  # ~10 steps to traverse r0

    # --- Sequential ---
    print("\n[1/3] Sequential mode...")
    photons_seq = make_photons(n_photons, observer_4d, direction, metric)
    integ = Integrator(metric, dt=dt, mode="sequential", integrator="rk4")
    n_ok = integ.integrate(photons_seq, stop_mode="steps", stop_value=n_steps, verbose=False)
    seq_ok = check_photon_results(photons_seq, "sequential")
    print(f"  {n_ok}/{n_photons} success, history: {len(photons_seq[0].history.states)} states")

    # --- Parallel ---
    print("\n[2/3] Parallel mode...")
    photons_par = make_photons(n_photons, observer_4d, direction, metric)
    with Integrator(metric, dt=dt, mode="parallel", integrator="rk4", n_workers=2) as integ:
        n_ok = integ.integrate(photons_par, stop_mode="steps", stop_value=n_steps, verbose=False)
    par_ok = check_photon_results(photons_par, "parallel")
    print(f"  {n_ok}/{n_photons} success, history: {len(photons_par[0].history.states)} states")

    # --- Parallel chunked ---
    print("\n[3/3] Parallel chunked mode...")
    photons_chu = make_photons(n_photons, observer_4d, direction, metric)
    with Integrator(metric, dt=dt, mode="parallel_chunked", integrator="rk4",
                    n_workers=2, chunk_size=2) as integ:
        n_ok = integ.integrate(photons_chu, stop_mode="steps", stop_value=n_steps, verbose=False)
    chu_ok = check_photon_results(photons_chu, "parallel_chunked")
    print(f"  {n_ok}/{n_photons} success, history: {len(photons_chu[0].history.states)} states")

    # Compare
    print("\n  Comparing sequential vs parallel final positions...")
    all_close = True
    for i in range(n_photons):
        d_par = np.linalg.norm(photons_seq[i].x - photons_par[i].x)
        d_chu = np.linalg.norm(photons_seq[i].x - photons_chu[i].x)
        if d_par > 1e-6 or d_chu > 1e-6:
            print(f"  photon {i}: seq-par dist={d_par:.2e}, seq-chu dist={d_chu:.2e}")
            all_close = False

    if all_close:
        print("  All positions match between modes!")

    return seq_ok and par_ok and chu_ok


if __name__ == '__main__':
    print("Testing unified Integrator\n")

    grid_ok = test_grid_metric()
    analytical_ok = test_analytical_metric()

    print("\n" + "=" * 60)
    if grid_ok and analytical_ok:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        if not grid_ok:
            print("  - Grid metric tests failed")
        if not analytical_ok:
            print("  - Analytical metric tests failed")
    print("=" * 60)
