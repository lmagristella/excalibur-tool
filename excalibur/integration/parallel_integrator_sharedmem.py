"""
Parallel integrator using shared memory for efficient multiprocessing on Windows.

This version avoids copying the large grid data to each worker by using
shared memory (multiprocessing.shared_memory, Python 3.8+).
"""

import atexit
import numpy as np
from multiprocessing import Pool, shared_memory
import multiprocessing as mp
from typing import List, Tuple

# Global worker state
_shared_grid = None
_worker_metric = None
_worker_interpolator = None
_worker_shm_handle = None  # track for cleanup


def _init_worker_sharedmem(shm_name: str, grid_shape: tuple, metric_params: dict):
    """
    Initialize worker with shared memory access.

    This function is called once per worker process to set up access to
    the shared grid data without copying it.

    Args:
        shm_name: Name of the shared memory block
        grid_shape: Shape of the grid data
        metric_params: Parameters to reconstruct metric (cosmology, etc.)
    """
    global _shared_grid, _worker_metric, _worker_interpolator, _worker_shm_handle

    # Attach to existing shared memory
    _worker_shm_handle = shared_memory.SharedMemory(name=shm_name)
    _shared_grid = np.ndarray(grid_shape, dtype=np.float64, buffer=_worker_shm_handle.buf)

    # Register cleanup so the handle is closed when the worker exits
    atexit.register(_cleanup_worker_shm)

    # Reconstruct interpolator and metric in worker
    from excalibur.grid.grid import Grid
    from excalibur.grid.interpolator_fast import InterpolatorFast
    from excalibur.metrics.perturbed_flrw_metric_fast import PerturbedFLRWMetricFast

    grid = Grid(
        shape=metric_params['grid_shape'],
        spacing=metric_params['grid_spacing'],
        origin=metric_params['grid_origin']
    )
    grid.add_field("Phi", _shared_grid)

    _worker_interpolator = InterpolatorFast(grid)
    _worker_metric = PerturbedFLRWMetricFast(
        metric_params['a_of_eta'],
        grid,
        _worker_interpolator
    )


def _cleanup_worker_shm():
    """Close the shared memory handle in the worker process."""
    global _worker_shm_handle
    if _worker_shm_handle is not None:
        try:
            _worker_shm_handle.close()
        except Exception:
            pass
        _worker_shm_handle = None


def _integrate_photon_sharedmem(args: Tuple):
    """
    Integrate a single photon using shared memory grid.

    Args:
        args: (photon_x, photon_u, n_steps, dt)

    Returns:
        (success, final_x, final_u, history_states)
    """
    from excalibur.photon.photon import Photon

    photon_x, photon_u, n_steps, dt = args

    photon = Photon(position=photon_x, direction=photon_u, weight=1.0)
    photon.state_quantities(_worker_metric.metric_physical_quantities)
    photon.record()

    # Inline RK4 integration using worker's pre-initialized metric
    state = np.concatenate([photon.x, photon.u])
    for step in range(n_steps):
        try:
            k1 = _worker_metric.geodesic_equations(state)
            k2 = _worker_metric.geodesic_equations(state + 0.5 * dt * k1)
            k3 = _worker_metric.geodesic_equations(state + 0.5 * dt * k2)
            k4 = _worker_metric.geodesic_equations(state + dt * k3)
            incr = (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            state[:4] += incr[:4]
            state[4:] += incr[4:]
            photon.x = state[:4]
            photon.u = state[4:]
            photon.state_quantities(_worker_metric.metric_physical_quantities)
            photon.record()
        except (ValueError, IndexError, RuntimeError):
            break

    history_states = [np.copy(s) for s in photon.history.states]
    success = len(photon.history.states) > 1
    return (success, photon.x.copy(), photon.u.copy(), history_states)


def _integrate_chunk_sharedmem(chunk_args: Tuple):
    """
    Integrate a chunk of photons using shared memory grid.

    Args:
        chunk_args: (chunk_data, n_steps, dt) where chunk_data is a list
                    of (photon_x, photon_u) tuples.

    Returns:
        list of (success, final_x, final_u, history_states)
    """
    chunk_data, n_steps, dt = chunk_args
    results = []
    for photon_x, photon_u in chunk_data:
        result = _integrate_photon_sharedmem((photon_x, photon_u, n_steps, dt))
        results.append(result)
    return results


def _copy_results_to_photons(photons, results):
    """Copy worker results back to original photon objects."""
    for photon, (success, final_x, final_u, history_states) in zip(photons, results):
        photon.x = final_x
        photon.u = final_u
        photon.history.states = []
        for state in history_states:
            photon.history.append(state)


class ParallelIntegratorSharedMem:
    """
    Parallel photon integrator using shared memory for efficient Windows multiprocessing.

    This version creates a shared memory block for the grid data, avoiding the
    expensive serialization/deserialization overhead of copying large arrays.

    Performance gain on Windows:
    - Standard multiprocessing: Slower than single-core due to copying overhead
    - Shared memory: Near-linear speedup (Nx faster with N cores)

    Example:
        >>> integrator = ParallelIntegratorSharedMem(
        ...     metric=metric,
        ...     dt=-1e15,
        ...     n_workers=4
        ... )
        >>> success = integrator.integrate_photons_sharedmem(photons, n_steps=1000)
    """

    def __init__(self, metric, dt: float, n_workers: int = None):
        """
        Initialize parallel integrator with shared memory support.

        Args:
            metric: Metric object (must have grid with Phi field)
            dt: Time step for integration
            n_workers: Number of worker processes (default: cpu_count() - 1)
        """
        self.metric = metric
        self.dt = dt
        self.n_workers = n_workers or max(1, mp.cpu_count() - 1)

        # Extract grid data
        self.grid = metric.grid
        self.phi_field = self.grid.fields["Phi"]

        # Prepare metric parameters for workers
        self.metric_params = {
            'a_of_eta': metric.a_of_eta,
            'grid_shape': self.grid.shape,
            'grid_spacing': self.grid.spacing,
            'grid_origin': self.grid.origin
        }

    def integrate_photons_sharedmem(
        self,
        photons,
        n_steps: int,
        verbose: bool = True
    ) -> int:
        """
        Integrate multiple photons in parallel using shared memory.

        Args:
            photons: Photons object containing list of photons
            n_steps: Number of integration steps
            verbose: Print progress information

        Returns:
            Number of successfully integrated photons
        """
        if len(photons) == 0:
            return 0

        if verbose:
            print(f"Starting parallel integration with {self.n_workers} workers (shared memory)")
            print(f"  Photons: {len(photons)}")
            print(f"  Steps: {n_steps}")
            print(f"  Grid size: {self.phi_field.nbytes / 1e6:.1f} MB (shared, not copied)")

        # Create shared memory for grid data
        shm = shared_memory.SharedMemory(create=True, size=self.phi_field.nbytes)

        try:
            # Copy grid data to shared memory
            shared_array = np.ndarray(
                self.phi_field.shape,
                dtype=self.phi_field.dtype,
                buffer=shm.buf
            )
            np.copyto(shared_array, self.phi_field)

            # Prepare photon data (lightweight - positions and velocities only)
            photon_args = [
                (photon.x.copy(), photon.u.copy(), n_steps, self.dt)
                for photon in photons
            ]

            # Create worker pool with shared memory initialization
            with Pool(
                processes=self.n_workers,
                initializer=_init_worker_sharedmem,
                initargs=(shm.name, self.phi_field.shape, self.metric_params)
            ) as pool:
                results = pool.map(_integrate_photon_sharedmem, photon_args)

            # Copy results back to original photons
            _copy_results_to_photons(photons, results)

            success_count = sum(1 for r in results if r[0])

            if verbose:
                print(f"[ok] Completed: {success_count}/{len(photons)} photons successful")

            return success_count

        finally:
            # Clean up shared memory
            shm.close()
            shm.unlink()

    def integrate_photons_chunked_sharedmem(
        self,
        photons,
        n_steps: int,
        chunk_size: int = None,
        verbose: bool = True
    ) -> int:
        """
        Integrate photons in chunks for better load balancing.

        Creates a single pool and shared memory block, then distributes
        chunks of photons to workers.

        Args:
            photons: Photons object
            n_steps: Number of integration steps
            chunk_size: Photons per chunk (default: n_photons / n_workers)
            verbose: Print progress

        Returns:
            Number of successful integrations
        """
        if len(photons) == 0:
            return 0

        if chunk_size is None:
            chunk_size = max(1, len(photons) // self.n_workers)

        if verbose:
            print(f"Starting chunked parallel integration with {self.n_workers} workers (shared memory)")
            print(f"  Processing {len(photons)} photons in chunks of ~{chunk_size}")

        # Create shared memory for grid data
        shm = shared_memory.SharedMemory(create=True, size=self.phi_field.nbytes)

        try:
            # Copy grid data to shared memory
            shared_array = np.ndarray(
                self.phi_field.shape,
                dtype=self.phi_field.dtype,
                buffer=shm.buf
            )
            np.copyto(shared_array, self.phi_field)

            # Prepare chunk arguments
            chunk_args_list = []
            for i in range(0, len(photons), chunk_size):
                chunk_photons = photons[i:i+chunk_size]
                chunk_data = [(p.x.copy(), p.u.copy()) for p in chunk_photons]
                chunk_args_list.append((chunk_data, n_steps, self.dt))

            # Single pool for all chunks
            with Pool(
                processes=self.n_workers,
                initializer=_init_worker_sharedmem,
                initargs=(shm.name, self.phi_field.shape, self.metric_params)
            ) as pool:
                chunk_results = pool.map(_integrate_chunk_sharedmem, chunk_args_list)

            # Flatten and copy back
            all_results = []
            for chunk_result in chunk_results:
                all_results.extend(chunk_result)

            _copy_results_to_photons(photons, all_results)

            total_success = sum(1 for r in all_results if r[0])

            if verbose:
                print(f"[ok] Completed: {total_success}/{len(photons)} photons successful")

            return total_success

        finally:
            shm.close()
            shm.unlink()
