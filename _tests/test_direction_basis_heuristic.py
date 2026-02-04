import numpy as np

from excalibur.core.constants import c, one_Mpc
from excalibur.core.cosmology import LCDM_Cosmology
from excalibur.grid.grid import Grid
from excalibur.grid.interpolator_fast import InterpolatorFast
from excalibur.metrics.perturbed_flrw_metric_fast import PerturbedFLRWMetricFast
from excalibur.photon.photons import Photons


def test_direction_basis_cartesian_does_not_convert_for_flrw_cartesian_origin():
    """direction_basis='cartesian' must keep (vx,vy,vz) for cartesian-like origins.

    Regression guard: older heuristic converted whenever len(origin)>=4, which would
    silently treat x,y,z as (r,theta,phi) and corrupt the photon initial velocity.
    """

    cosmology = LCDM_Cosmology(70, Omega_m=0.3, Omega_r=0.0, Omega_lambda=0.7)
    _ = cosmology.a_of_eta(1e18)
    eta = cosmology._eta_at_a1

    N = 6
    grid_size = 10.0 * one_Mpc
    spacing = (grid_size / N, grid_size / N, grid_size / N)
    origin3 = np.array([0.0, 0.0, 0.0]) * one_Mpc

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

    # Keep position inside the grid bounds [0, grid_size] (origin3 is [0,0,0]).
    # Use an interior position: the fast interpolator uses central differences
    # and can reject points too close to the boundary.
    origin = np.array([eta, 0.2 * grid_size, 0.3 * grid_size, 0.4 * grid_size], dtype=float)
    central_dir = np.array([1.0, 0.0, 0.0])

    photons = Photons(metric=metric)
    photons.generate_cone_grid(
        n_photons=1,
        origin=origin,
        central_direction=central_dir,
        cone_angle=0.0,
        direction_basis="cartesian",
    )

    # Cone angle 0 -> spatial direction should align with +x.
    p = photons.photons[0]
    assert np.isclose(p.u[1], c, rtol=0, atol=1e-12)
    assert np.isclose(p.u[2], 0.0, rtol=0, atol=1e-12)
    assert np.isclose(p.u[3], 0.0, rtol=0, atol=1e-12)
