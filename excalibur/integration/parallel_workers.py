"""Unified parallel worker module for photon integration.

Provides top-level picklable functions for multiprocessing workers.
Supports both grid-based metrics (via shared memory) and analytical metrics.

Workers are initialized once per process via Pool initializer functions,
then reused for all photon integrations in that worker.
"""
from __future__ import annotations

import atexit
import numpy as np
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Global worker state (set once per worker process by the initializer)
# ---------------------------------------------------------------------------
_worker_metric = None
_worker_dt = None
_worker_shm_handle = None


# ---------------------------------------------------------------------------
# Worker initializers (called ONCE per worker process at Pool creation)
# ---------------------------------------------------------------------------

def init_worker_grid(
    shm_name: str,
    grid_shape: tuple,
    grid_dtype: str,
    grid_params: dict,
    metric_class: str,
    metric_params: dict,
    dt: float,
):
    """Initialize worker for grid-based metrics using shared memory.

    The grid's Phi field lives in a SharedMemory block created by the parent.
    Each worker attaches to it (zero-copy on the same machine).
    """
    global _worker_metric, _worker_dt, _worker_shm_handle

    from multiprocessing import shared_memory
    from excalibur.grid.grid import Grid
    from excalibur.grid.interpolator_fast import InterpolatorFast

    # Attach to parent's shared memory
    _worker_shm_handle = shared_memory.SharedMemory(name=shm_name)
    shared_phi = np.ndarray(grid_shape, dtype=grid_dtype, buffer=_worker_shm_handle.buf)
    atexit.register(_cleanup_shm)

    # Reconstruct grid + interpolator
    grid = Grid(
        shape=grid_params['shape'],
        spacing=grid_params['spacing'],
        origin=grid_params['origin'],
    )
    grid.add_field("Phi", shared_phi)
    interpolator = InterpolatorFast(grid)

    # Reconstruct metric
    if metric_class == "PerturbedFLRWMetricFast":
        from excalibur.metrics.perturbed_flrw_metric_fast import PerturbedFLRWMetricFast
        _worker_metric = PerturbedFLRWMetricFast(
            metric_params['a_of_eta'], grid, interpolator,
        )
    elif metric_class == "PerturbedFLRWMetric":
        from excalibur.metrics.perturbed_flrw_metric import PerturbedFLRWMetric
        _worker_metric = PerturbedFLRWMetric(
            metric_params['a_of_eta'], grid, interpolator,
        )
    else:
        raise ValueError(f"Unsupported grid metric class: {metric_class}")

    _worker_dt = dt


def init_worker_analytical(
    metric_class: str,
    metric_params: dict,
    dt: float,
):
    """Initialize worker for analytical (non-grid) metrics.

    Analytical metrics are lightweight, so we just recreate them from params.
    """
    global _worker_metric, _worker_dt

    if metric_class == "SchwarzschildMetricCartesian":
        from excalibur.metrics.schwarzschild_metric_cartesian import SchwarzschildMetricCartesian
        _worker_metric = SchwarzschildMetricCartesian(
            mass=metric_params['mass'],
            radius=metric_params['radius'],
            center=metric_params['center'],
        )
    elif metric_class == "SchwarzschildMetric":
        from excalibur.metrics.schwarzschild_metric import SchwarzschildMetric
        _worker_metric = SchwarzschildMetric(
            mass=metric_params['mass'],
            radius=metric_params['radius'],
            center=metric_params['center'],
        )
    elif metric_class == "SchwarzschildMetricFast":
        from excalibur.metrics.schwarzschild_metric_fast import SchwarzschildMetricFast
        _worker_metric = SchwarzschildMetricFast(
            mass=metric_params['mass'],
            radius=metric_params['radius'],
            center=metric_params['center'],
        )
    else:
        raise ValueError(f"Unsupported analytical metric class: {metric_class}")

    _worker_dt = dt


def _cleanup_shm():
    """Close shared memory handle in worker (registered via atexit)."""
    global _worker_shm_handle
    if _worker_shm_handle is not None:
        try:
            _worker_shm_handle.close()
        except Exception:
            pass
        _worker_shm_handle = None


# ---------------------------------------------------------------------------
# Worker integration functions (called for each photon / chunk)
# ---------------------------------------------------------------------------

def integrate_photon_worker(photon_data: tuple) -> tuple:
    """Integrate a single photon using the pre-initialized worker metric.

    Uses RK4 with separate position/momentum updates to reduce catastrophic
    cancellation (same technique as the main RK4 integrator).

    Args:
        photon_data: (x, u, n_steps) where x and u are 4-vectors.

    Returns:
        (success, final_x, final_u, history_states)
    """
    from excalibur.photon.photon import Photon

    x, u, n_steps = photon_data
    photon = Photon(position=x, direction=u)

    # Record initial quantities
    photon.state_quantities(_worker_metric.metric_physical_quantities)
    photon.record()

    # RK4 integration loop
    state = np.concatenate([photon.x, photon.u])
    dt = _worker_dt
    for _ in range(n_steps):
        try:
            k1 = _worker_metric.geodesic_equations(state)
            k2 = _worker_metric.geodesic_equations(state + 0.5 * dt * k1)
            k3 = _worker_metric.geodesic_equations(state + 0.5 * dt * k2)
            k4 = _worker_metric.geodesic_equations(state + dt * k3)
            incr = (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            # Separate position/momentum update to reduce cancellation
            state[:4] += incr[:4]
            state[4:] += incr[4:]
            photon.x = state[:4]
            photon.u = state[4:]
            photon.state_quantities(_worker_metric.metric_physical_quantities)
            photon.record()
        except (ValueError, IndexError, RuntimeError):
            break

    history_states = [np.copy(s) for s in photon.history.states]
    success = len(history_states) > 1
    return (success, photon.x.copy(), photon.u.copy(), history_states)


def integrate_chunk_worker(chunk_data: tuple) -> list:
    """Integrate a chunk of photons.

    Args:
        chunk_data: (photon_list, n_steps) where photon_list is a list of (x, u).

    Returns:
        List of (success, final_x, final_u, history_states).
    """
    photon_list, n_steps = chunk_data
    return [
        integrate_photon_worker((x, u, n_steps))
        for x, u in photon_list
    ]
