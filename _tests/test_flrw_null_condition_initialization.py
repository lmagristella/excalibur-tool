import numpy as np

from excalibur.core.constants import one_Mpc
from excalibur.core.cosmology import LCDM_Cosmology
from excalibur.grid.grid import Grid
from excalibur.grid.interpolator_fast import InterpolatorFast
from excalibur.metrics.perturbed_flrw_metric_fast import PerturbedFLRWMetricFast
from excalibur.photon.photons import Photons


def test_flrw_photons_initial_null_condition_small_rel_error():
    """Photons generated in (perturbed) FLRW should start (close to) null.

    This is a smoke/regression test for the photon initialization routine when
    the metric is cartesian-like (origin=[eta,x,y,z]) and we pass
    direction_basis="cartesian".

    We use a tiny grid with Phi=0 everywhere so interpolation is well-defined
    and cheap.
    """

    # Minimal cosmology and scale factor
    cosmology = LCDM_Cosmology(70, Omega_m=0.3, Omega_r=0.0, Omega_lambda=0.7)
    _ = cosmology.a_of_eta(1e18)
    eta_present = cosmology._eta_at_a1

    # Tiny cubic grid with zero potential (dimensionless Phi/c^2)
    N = 8
    grid_size = 10 * one_Mpc
    spacing = (grid_size / N, grid_size / N, grid_size / N)
    origin3 = np.array([0.0, 0.0, 0.0])
    grid = Grid((N, N, N), spacing, origin3)
    grid.add_field("Phi", np.zeros((N, N, N), dtype=float))

    interpolator = InterpolatorFast(grid)
    metric = PerturbedFLRWMetricFast(
        a_of_eta=cosmology.a_of_eta,
        grid=grid,
        interpolator=interpolator,
        analytical_geodesics=False,
    )

    # Observer inside box to avoid interpolator bounds errors
    observer_position = np.array([0.2, 0.3, 0.4]) * grid_size
    center = np.array([0.5, 0.5, 0.5]) * grid_size

    central_dir = center - observer_position
    central_dir = central_dir / np.linalg.norm(central_dir)

    origin = np.array([eta_present, *observer_position], dtype=float)

    photons = Photons(metric=metric)
    photons.generate_cone_grid(
        n_photons=9,
        origin=origin,
        central_direction=central_dir,
        cone_angle=np.deg2rad(5),
        direction_basis="cartesian",
    )

    rel_errors = [p.null_condition_relative_error(metric) for p in photons.photons]
    assert max(rel_errors) < 1e-10
