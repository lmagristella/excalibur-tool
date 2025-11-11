# integration/parallel_integrator.py
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial

def integrate_single_photon(args):
    """
    Wrapper function for integrating a single photon.
    Used for multiprocessing.
    
    Parameters:
    -----------
    args : tuple
        (photon, metric, dt, n_steps)
    """
    photon, metric, dt, n_steps = args
    
    # Create local integrator
    from excalibur.integration.integrator import Integrator
    integrator = Integrator(metric, dt=dt)
    
    try:
        integrator.integrate(photon, n_steps)
        return photon, True
    except Exception as e:
        print(f"Error integrating photon: {e}")
        return photon, False


class ParallelIntegrator:
    """
    Parallel integrator for multiple photons using multiprocessing.
    Provides significant speedup on multi-core systems.
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
        args_list = [(photon, self.metric, self.dt, n_steps) for photon in photons]
        
        # Use multiprocessing pool
        with Pool(processes=self.n_workers) as pool:
            results = pool.map(integrate_single_photon, args_list)
        
        # Count successes
        success_count = sum(1 for _, success in results if success)
        
        if verbose:
            print(f"   Successfully integrated {success_count}/{len(photons)} photons")
        
        return success_count


class ParallelIntegratorChunked:
    """
    Parallel integrator that processes photons in chunks to reduce overhead.
    More efficient for large numbers of photons.
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
        
        # Split photons into chunks
        chunks = []
        for i in range(0, len(photons), self.chunk_size):
            chunks.append(photons[i:i+self.chunk_size])
        
        if verbose:
            print(f"   Processing {len(chunks)} chunks...")
        
        # Process chunks in parallel
        partial_func = partial(self._integrate_chunk, n_steps=n_steps)
        
        with Pool(processes=self.n_workers) as pool:
            results = pool.map(partial_func, chunks)
        
        # Aggregate results
        success_count = sum(results)
        
        if verbose:
            print(f"   Successfully integrated {success_count}/{len(photons)} photons")
        
        return success_count
    
    def _integrate_chunk(self, photon_chunk, n_steps):
        """Integrate a chunk of photons sequentially."""
        from excalibur.integration.integrator import Integrator
        integrator = Integrator(self.metric, dt=self.dt)
        
        success_count = 0
        for photon in photon_chunk:
            try:
                integrator.integrate(photon, n_steps)
                success_count += 1
            except Exception as e:
                pass  # Photon failed, continue with others
        
        return success_count
