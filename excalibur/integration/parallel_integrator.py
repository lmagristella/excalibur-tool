# integration/parallel_integrator.py
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial


def integrate_single_photon(args):
    """
    Wrapper function for integrating a single photon.
    Used for multiprocessing. Must be a top-level function for pickling.

    Parameters:
    -----------
    args : tuple
        (photon_x, photon_u, photon_quantities, metric, dt, n_steps)

    Returns:
    --------
    tuple: (success, final_x, final_u, history_states)
    """
    photon_x, photon_u, photon_quantities, metric, dt, n_steps = args

    from excalibur.integration.integrator_old import Integrator
    from excalibur.photon.photon import Photon

    photon = Photon(position=photon_x, direction=photon_u)
    photon.state_quantities(metric.metric_physical_quantities)
    photon.record()

    integrator = Integrator(metric, dt=dt)

    try:
        integrator.integrate(photon, n_steps)
        history_states = [np.copy(s) for s in photon.history.states]
        return (True, photon.x.copy(), photon.u.copy(), history_states)
    except Exception as e:
        print(f"Error integrating photon: {e}")
        history_states = [np.copy(s) for s in photon.history.states]
        return (False, photon.x.copy(), photon.u.copy(), history_states)


def _integrate_chunk(chunk_args):
    """
    Integrate a chunk of photons sequentially.
    Top-level function for pickling compatibility.

    Parameters:
    -----------
    chunk_args : tuple
        (chunk_data, metric, dt, n_steps) where chunk_data is a list of
        (photon_x, photon_u, photon_quantities) tuples.

    Returns:
    --------
    list of (success, final_x, final_u, history_states)
    """
    chunk_data, metric, dt, n_steps = chunk_args

    from excalibur.integration.integrator_old import Integrator
    from excalibur.photon.photon import Photon

    integrator = Integrator(metric, dt=dt)
    results = []

    for photon_x, photon_u, photon_quantities in chunk_data:
        photon = Photon(position=photon_x, direction=photon_u)
        photon.state_quantities(metric.metric_physical_quantities)
        photon.record()

        try:
            integrator.integrate(photon, n_steps)
            history_states = [np.copy(s) for s in photon.history.states]
            results.append((True, photon.x.copy(), photon.u.copy(), history_states))
        except Exception:
            history_states = [np.copy(s) for s in photon.history.states]
            results.append((False, photon.x.copy(), photon.u.copy(), history_states))

    return results


def _copy_results_to_photons(photons, results):
    """Copy worker results back to original photon objects."""
    for photon, (success, final_x, final_u, history_states) in zip(photons, results):
        photon.x = final_x
        photon.u = final_u
        photon.history.states = []
        for state in history_states:
            photon.history.append(state)


class ParallelIntegrator:
    """
    Parallel integrator for multiple photons using multiprocessing.
    Provides significant speedup on multi-core systems.

    NOTE: On Windows, this sends the metric object to each worker via pickling.
    For grid-based metrics (large data), prefer PersistentPoolIntegrator or
    ParallelIntegratorSharedMem which avoid this overhead.
    """
    def __init__(self, metric, dt=1e-3, n_workers=None):
        self.metric = metric
        self.dt = dt
        self.n_workers = n_workers if n_workers is not None else max(1, cpu_count() - 1)

    def integrate_photons(self, photons, n_steps, verbose=True):
        """
        Integrate multiple photons in parallel.

        Parameters:
        -----------
        photons : list of Photon
            Photons to integrate
        n_steps : int
            Number of integration steps
        verbose : bool
            Print progress information

        Returns:
        --------
        success_count : int
            Number of successfully integrated photons
        """
        if verbose:
            print(f"   Parallel integration using {self.n_workers} workers...")

        # Prepare arguments for each photon
        args_list = [
            (photon.x.copy(), photon.u.copy(), photon.quantities.copy(),
             self.metric, self.dt, n_steps)
            for photon in photons
        ]

        # Use multiprocessing pool
        with Pool(processes=self.n_workers) as pool:
            results = pool.map(integrate_single_photon, args_list)

        # Copy results back to original photons
        _copy_results_to_photons(photons, results)

        # Count successes
        success_count = sum(1 for r in results if r[0])

        if verbose:
            print(f"   Successfully integrated {success_count}/{len(photons)} photons")

        return success_count


class ParallelIntegratorChunked:
    """
    Parallel integrator that processes photons in chunks to reduce overhead.
    More efficient for large numbers of photons.

    NOTE: On Windows, this sends the metric object to each worker via pickling.
    For grid-based metrics (large data), prefer PersistentPoolIntegrator.
    """
    def __init__(self, metric, dt=1e-3, n_workers=None, chunk_size=10):
        self.metric = metric
        self.dt = dt
        self.n_workers = n_workers if n_workers is not None else max(1, cpu_count() - 1)
        self.chunk_size = chunk_size

    def integrate_photons_chunked(self, photons, n_steps, verbose=True):
        """
        Integrate photons in parallel with chunked processing.

        Parameters:
        -----------
        photons : list of Photon
            Photons to integrate
        n_steps : int
            Number of integration steps
        verbose : bool
            Print progress information

        Returns:
        --------
        success_count : int
            Number of successfully integrated photons
        """
        if verbose:
            print(f"   Chunked parallel integration using {self.n_workers} workers...")
            print(f"   Chunk size: {self.chunk_size} photons per chunk")

        # Split photons into chunks and prepare data
        chunk_args_list = []
        for i in range(0, len(photons), self.chunk_size):
            chunk_photons = photons[i:i+self.chunk_size]
            chunk_data = [
                (p.x.copy(), p.u.copy(), p.quantities.copy())
                for p in chunk_photons
            ]
            chunk_args_list.append((chunk_data, self.metric, self.dt, n_steps))

        if verbose:
            print(f"   Processing {len(chunk_args_list)} chunks...")

        # Process chunks in parallel
        with Pool(processes=self.n_workers) as pool:
            chunk_results = pool.map(_integrate_chunk, chunk_args_list)

        # Flatten results and copy back to photons
        all_results = []
        for chunk_result in chunk_results:
            all_results.extend(chunk_result)

        _copy_results_to_photons(photons, all_results)

        success_count = sum(1 for r in all_results if r[0])

        if verbose:
            print(f"   Successfully integrated {success_count}/{len(photons)} photons")

        return success_count
