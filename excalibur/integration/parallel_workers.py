"""Parallel worker utilities for photon integration.

Defines top-level functions used by multiprocessing so they are picklable
under spawn/fork start methods and when profiling with cProfile.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from excalibur.integration.integrator import Integrator
from excalibur.photon.photon import Photon


def integrate_single_photon(args: Tuple[int, Dict[str, Any], Dict[str, Any]]):
    """Integrate a single photon (used in multiprocessing Pool).

    Parameters
    ----------
    args : tuple
        (photon_index, photon_data, integrator_params)

    Returns
    -------
    tuple
        (photon_index, integrated_photon, success_flag)
    """
    photon_idx, photon_data, integrator_params = args

    # Local integrator (sequential for a single photon)
    local_integrator = Integrator(
        metric=integrator_params['metric'],
        dt=integrator_params['dt'],
        mode="sequential",
        integrator=integrator_params['integrator'],
        rtol=integrator_params['rtol'],
        atol=integrator_params['atol'],
        dt_min=integrator_params['dt_min'],
        dt_max=integrator_params['dt_max'],
        n_workers=1,
        chunk_size=1,
    )

    photon = Photon(position=photon_data['position'], direction=photon_data['u'])
    photon.record()

    success = local_integrator.integrate(
        [photon],
        stop_mode=integrator_params['stop_mode'],
        stop_value=integrator_params['stop_value'],
        verbose=False,
    )

    return photon_idx, photon, success
