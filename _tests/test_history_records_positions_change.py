import numpy as np

from excalibur.core.constants import c, one_Mpc, one_Msun
from excalibur.core.coordinates import cartesian_to_spherical
from excalibur.integration.integrator import Integrator
from excalibur.metrics.schwarzschild_metric import SchwarzschildMetric
from excalibur.photon.photons import Photons


def test_history_records_positions_change_schwarzschild():
    """Regression test for a bug where per-step recorded coordinates stayed constant.

    The Schwarzschild runner integrates in spherical coords, so we check that r/theta/phi
    evolve over steps when the photon is not configured to be static.
    """

    mass = 1e15 * one_Msun
    radius = 3.0 * one_Mpc

    # Simple centered geometry (like compare runner)
    L = 1000.0 * one_Mpc
    lens = np.array([0.5 * L, 0.5 * L, 0.5 * L], dtype=float)
    D = 50.0 * one_Mpc
    obs = lens + np.array([-D, 0.0, 0.0], dtype=float)
    central_dir = lens - obs

    metric = SchwarzschildMetric(mass=mass, radius=radius, center=lens, coords="spherical")

    travel = 2.0 * D
    n_steps = 200
    dlambda = (travel / n_steps) / c

    integ = Integrator(metric=metric, dt=dlambda, mode="sequential", integrator="rk4")

    photons = Photons(metric=metric)
    origin_sph = cartesian_to_spherical(*(obs - lens))
    origin = np.array([0.0, *origin_sph], dtype=float)

    photons.generate_cone_grid(
        n_photons=1,
        origin=origin,
        central_direction=central_dir,
        cone_angle=np.deg2rad(5.0),
        direction_basis="cartesian",
    )

    p = photons.photons[0]
    integ.integrate_single(p, stop_mode="steps", stop_value=20, record_every=1)

    # History states may have varying length (also true in production runs).
    # Pad to a rectangular array, like PhotonHistory.save_to_hdf5.
    raw_states = list(p.history.states)
    assert len(raw_states) >= 3
    max_len = max(len(s) for s in raw_states)
    states = np.full((len(raw_states), max_len), np.nan, dtype=float)
    for i, s in enumerate(raw_states):
        arr = np.asarray(s, dtype=float)
        states[i, : arr.shape[0]] = arr

    # (eta, r, theta, phi, ...) in spherical coords
    r = states[:, 1]
    theta = states[:, 2]
    phi = states[:, 3]

    # We expect at least one of these to change for a moving photon.
    # Use nanmax to be robust to any padding.
    dr = float(np.nanmax(r) - np.nanmin(r))
    dtheta = float(np.nanmax(theta) - np.nanmin(theta))
    dphi = float(np.nanmax(phi) - np.nanmin(phi))

    assert (dr > 0.0) or (dtheta > 0.0) or (dphi > 0.0)
