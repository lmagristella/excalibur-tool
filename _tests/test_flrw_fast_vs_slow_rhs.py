import numpy as np

from excalibur.core.constants import c, one_Mpc
from excalibur.core.cosmology import LCDM_Cosmology
from excalibur.grid.grid import Grid
from excalibur.grid.interpolator_fast import InterpolatorFast
from excalibur.metrics.perturbed_flrw_metric import PerturbedFLRWMetric
from excalibur.metrics.perturbed_flrw_metric_fast import PerturbedFLRWMetricFast


def test_flrw_fast_matches_slow_rhs_for_phi0():
    """Fast FLRW RHS should match the slow Christoffel-based RHS for Phi=0.

    This guards against subtle bugs in the fast path (notably adot estimation).
    """

    cosmology = LCDM_Cosmology(70, Omega_m=0.3, Omega_r=0.0, Omega_lambda=0.7)
    _ = cosmology.a_of_eta(1e18)
    eta = cosmology._eta_at_a1

    N = 8
    grid_size = 10 * one_Mpc
    spacing = (grid_size / N, grid_size / N, grid_size / N)
    origin3 = np.array([0.0, 0.0, 0.0])

    grid = Grid((N, N, N), spacing, origin3)
    grid.add_field("Phi", np.zeros((N, N, N), dtype=float))

    interp = InterpolatorFast(grid)

    metric_slow = PerturbedFLRWMetric(cosmology=cosmology, grid=grid, interpolator=interp)
    metric_fast = PerturbedFLRWMetricFast(
        a_of_eta=cosmology.a_of_eta,
        adot_of_eta=cosmology.adot_of_eta,
        grid=grid,
        interpolator=interp,
        analytical_geodesics=False,
    )

    x = np.array([eta, 0.2 * grid_size, 0.3 * grid_size, 0.4 * grid_size], float)

    state = np.zeros(8)
    state[:4] = x
    state[4:] = np.array([-0.4, 0.1 * c, 0.2 * c, 0.3 * c])

    rhs_slow = metric_slow.geodesic_equations_tensor(state)
    rhs_fast = metric_fast.geodesic_equations(state)

    # Loose tolerance: slow uses a cached interpolation / different algebra;
    # we mostly care about avoiding the pathological du=0 bug.
    assert np.max(np.abs(rhs_slow - rhs_fast)) < 1e-10
