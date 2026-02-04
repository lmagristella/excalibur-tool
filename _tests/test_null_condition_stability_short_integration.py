import numpy as np

from excalibur.core.constants import c, one_Mpc, one_Gpc, one_Msun
from excalibur.core.coordinates import cartesian_to_spherical
from excalibur.core.cosmology import LCDM_Cosmology
from excalibur.grid.grid import Grid
from excalibur.grid.interpolator_fast import InterpolatorFast
from excalibur.integration.integrator import Integrator
from excalibur.metrics.perturbed_flrw_metric_fast import PerturbedFLRWMetricFast
from excalibur.metrics.schwarzschild_metric import SchwarzschildMetric
from excalibur.photon.photons import Photons


def _max_null_rel_over_steps(
    photon,
    metric,
    integrator,
    n_steps: int,
    *,
    chunk_size: int = 50,
) -> float:
    """Return max null relative error sampled during an integration.

    We integrate in chunks because calling `integrate_single(..., steps=1)` in a
    tight Python loop adds a lot of overhead and is not representative of how
    the integrator is meant to be used.
    """

    if n_steps <= 0:
        return float(photon.null_condition_relative_error(metric))

    chunk_size = max(1, int(chunk_size))
    remaining = int(n_steps)

    max_rel = float(photon.null_condition_relative_error(metric))

    while remaining > 0:
        steps = min(chunk_size, remaining)
        integrator.integrate_single(photon, stop_mode="steps", stop_value=steps)
        max_rel = max(max_rel, float(photon.null_condition_relative_error(metric)))
        remaining -= steps

    return max_rel


def test_schwarzschild_null_condition_stable_over_short_integration():
    """Schwarzschild: null condition should stay small over a short integration.

    This is a smoke test (not a physics validation). It catches coordinate/unit
    mismatches, bad sign conventions, and obvious integrator/metric breakage.
    """

    M = 10**15 * one_Msun
    R = 3 * one_Mpc
    center = np.array([0.5, 0.5, 0.5]) * one_Gpc

    metric = SchwarzschildMetric(mass=M, radius=R, center=center, coords="spherical")

    observer_cart = np.array([0.01, 0.01, 0.01]) * one_Gpc
    observer_sph = cartesian_to_spherical(*(observer_cart - center))
    origin = np.array([0.0, *observer_sph], dtype=float)  # [t, r, theta, phi]

    central_dir = center - observer_cart

    photons = Photons(metric=metric)
    photons.generate_cone_grid(
        n_photons=1,
        origin=origin,
        central_direction=central_dir,
        cone_angle=np.deg2rad(1),
        direction_basis="cartesian",
    )

    photon = photons.photons[0]

    integrator = Integrator(metric=metric, dt=-1.0, mode="sequential", integrator="rk4")

    max_rel = _max_null_rel_over_steps(photon, metric, integrator, n_steps=50, chunk_size=10)

    assert max_rel < 1e-10


def test_flrw_null_condition_stable_over_short_integration():
    """FLRW (fast metric): null condition should stay small over a short integration."""

    cosmology = LCDM_Cosmology(70, Omega_m=0.3, Omega_r=0.0, Omega_lambda=0.7)
    _ = cosmology.a_of_eta(1e18)
    eta = cosmology._eta_at_a1

    # Small grid with Phi=0 everywhere (dimensionless)
    N = 8
    grid_size = 10 * one_Mpc
    spacing = (grid_size / N, grid_size / N, grid_size / N)
    origin3 = np.array([0.0, 0.0, 0.0])
    grid = Grid((N, N, N), spacing, origin3)
    grid.add_field("Phi", np.zeros((N, N, N), dtype=float))

    interp = InterpolatorFast(grid)
    metric = PerturbedFLRWMetricFast(
        a_of_eta=cosmology.a_of_eta,
        adot_of_eta=cosmology.adot_of_eta,
        grid=grid,
        interpolator=interp,
        analytical_geodesics=False,
    )

    observer_position = np.array([0.2, 0.3, 0.4]) * grid_size
    center = np.array([0.5, 0.5, 0.5]) * grid_size
    central_dir = center - observer_position
    central_dir = central_dir / np.linalg.norm(central_dir)

    origin = np.array([eta, *observer_position], dtype=float)  # [eta, x, y, z]

    photons = Photons(metric=metric)
    photons.generate_cone_grid(
        n_photons=1,
        origin=origin,
        central_direction=central_dir,
        cone_angle=np.deg2rad(1),
        direction_basis="cartesian",
    )

    photon = photons.photons[0]

    integrator = Integrator(metric=metric, dt=-1.0, mode="sequential", integrator="rk4")

    # Using chunks makes this both faster and closer to a "real" integration.
    max_rel = _max_null_rel_over_steps(photon, metric, integrator, n_steps=5000, chunk_size=50)

    # Still extremely strict compared to float64 (~1e-16), but avoids the
    # pathological `== 0` requirement.
    assert max_rel < 1e-30
