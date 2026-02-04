import numpy as np


from excalibur.core.constants import one_Msun, one_Mpc, one_Gpc
from excalibur.core.coordinates import cartesian_to_spherical
from excalibur.metrics.schwarzschild_metric import SchwarzschildMetric
from excalibur.photon.photons import Photons


def test_schwarzschild_photons_initial_null_condition_small_rel_error():
    """Photons generated in Schwarzschild metric should start (close to) null.

    This guards against unit/sign mismatches in the metric tensor (notably g00).
    """

    M = 10**15 * one_Msun
    R = 3 * one_Mpc
    center = np.array([0.5, 0.5, 0.5]) * one_Gpc
    metric = SchwarzschildMetric(mass=M, radius=R, center=center)

    observer_position_cart = np.array([0.01, 0.01, 0.01]) * one_Gpc
    observer_sph = cartesian_to_spherical(*(observer_position_cart - center))
    origin = np.array([0.0, *observer_sph], dtype=float)
    central_dir = center - observer_position_cart

    photons = Photons(metric=metric)
    photons.generate_cone_grid(
        n_photons=9,
        origin=origin,
        central_direction=central_dir,
        cone_angle=np.deg2rad(5),
        direction_basis="cartesian",
    )

    # For each photon, relative error should be tiny.
    rel_errors = [p.null_condition_relative_error(metric) for p in photons.photons]
    assert max(rel_errors) < 1e-10
