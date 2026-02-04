"""
Parallel integrator using shared memory for efficient multiprocessing on Windows.

This version avoids copying the large grid data to each worker by using
shared memory (multiprocessing.shared_memory, Python 3.8+).
"""

import numpy as np
from multiprocessing import Pool, shared_memory
import multiprocessing as mp
from typing import List, Tuple


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
    global _shared_grid, _worker_metric, _worker_interpolator
    
    # Attach to existing shared memory
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    _shared_grid = np.ndarray(grid_shape, dtype=np.float64, buffer=existing_shm.buf)
    
    # Reconstruct interpolator and metric in worker
    # (These are lightweight compared to the grid data)
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


def _integrate_photon_sharedmem(args: Tuple):
    """
    Integrate a single photon using shared memory grid.
    
    Args:
        args: (photon_x, photon_u, photon_quantities, n_steps, dt)
    
    Returns:
        True if successful, False otherwise
    """
    from excalibur.integration.integrator_old import Integrator
    from excalibur.photon.photon import Photon
    
    photon_x, photon_u, photon_quantities, n_steps, dt = args
    
    # Create temporary photon object
    # Photon.__init__ expects (position_4d, direction_4d, weight)
    photon = Photon(position=photon_x, direction=photon_u, weight=1.0)
    photon.quantities = photon_quantities
    photon.record()
    
    # Use worker's metric (which references shared memory grid)
    integrator = Integrator(_worker_metric, dt=dt)
    
    try:
        integrator.integrate(photon, n_steps)
        return True
    except Exception as e:
        return False


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
            photon_args = []
            for photon in photons:
                photon_args.append((
                    photon.x.copy(),
                    photon.u.copy(),
                    photon.quantities.copy() if hasattr(photon, 'quantities') else {},
                    n_steps,
                    self.dt
                ))
            
            # Create worker pool with shared memory initialization
            with Pool(
                processes=self.n_workers,
                initializer=_init_worker_sharedmem,
                initargs=(shm.name, self.phi_field.shape, self.metric_params)
            ) as pool:
                results = pool.map(_integrate_photon_sharedmem, photon_args)
            
            success_count = sum(results)
            
            if verbose:
                print(f"✓ Completed: {success_count}/{len(photons)} photons successful")
            
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
        
        For large numbers of photons, chunking reduces overhead by processing
        multiple photons per task submission.
        
        Args:
            photons: Photons object
            n_steps: Number of integration steps
            chunk_size: Photons per chunk (default: n_photons / n_workers)
            verbose: Print progress
        
        Returns:
            Number of successful integrations
        """
        if chunk_size is None:
            chunk_size = max(1, len(photons) // self.n_workers)
        
        # Split photons into chunks
        chunks = []
        for i in range(0, len(photons), chunk_size):
            chunk = photons[i:i+chunk_size]
            chunks.append(chunk)
        
        if verbose:
            print(f"Processing {len(chunks)} chunks of ~{chunk_size} photons each")
        
        # Process each chunk (can parallelize this too if needed)
        total_success = 0
        for i, chunk in enumerate(chunks):
            success = self.integrate_photons_sharedmem(chunk, n_steps, verbose=False)
            total_success += success
            if verbose:
                print(f"  Chunk {i+1}/{len(chunks)}: {success}/{len(chunk)} successful")
        
        return total_success
