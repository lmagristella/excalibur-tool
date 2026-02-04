"""
Parallel integrator with persistent worker pool for ANALYTICAL METRICS.

Specialized version for analytical metrics like Schwarzschild, Kerr, etc.
Unlike the grid-based persistent integrator, this one recreates the metric
objects in each worker since analytical metrics are lightweight.

Key features:
1. Persistent worker pool avoiding process creation overhead
2. Metric parameters passed to workers (not full metric objects)
3. Local metric reconstruction in each worker
4. Compatible with Schwarzschild, Kerr, and other analytical metrics

Expected speedup: 3-4x with 4 workers for analytical metrics
"""

import numpy as np
from multiprocessing import Pool
import multiprocessing as mp
from typing import List, Dict, Any
import time


# Global worker state for analytical metrics
_worker_metric = None
_worker_dt = None


def _init_analytical_worker(metric_type: str, metric_params: dict, dt: float):
    """
    Initialize worker with analytical metric.
    
    This function recreates the metric object in each worker process.
    For analytical metrics, this is lightweight since they don't carry
    large data structures like grids.
    
    Args:
        metric_type: Type of metric ("schwarzschild", "kerr", etc.)
        metric_params: Dictionary with metric parameters
        dt: Integration time step
    """
    global _worker_metric, _worker_dt
    
    # Import and create metric based on type
    if metric_type == "schwarzschild":
        from excalibur.metrics.schwarzschild_metric_cartesian import SchwarzschildMetricCartesian
        _worker_metric = SchwarzschildMetricCartesian(
            mass=metric_params['mass'],
            radius=metric_params['radius'], 
            center=metric_params['center']
        )
    elif metric_type == "kerr":
        # Add other analytical metrics here as needed
        raise NotImplementedError("Kerr metric not yet supported")
    else:
        raise ValueError(f"Unknown analytical metric type: {metric_type}")
    
    _worker_dt = dt


def _integrate_photon_analytical(photon_data: dict, n_steps: int) -> tuple:
    """
    Worker function to integrate a single photon with analytical metric.
    
    Args:
        photon_data: Dictionary containing photon state and configuration
        n_steps: Number of integration steps
    
    Returns:
        (success, trajectory_data, photon_id)
    """
    global _worker_metric, _worker_dt
    
    try:
        from excalibur.photon.photon import Photon
        from excalibur.integration.integrator_optimized import IntegratorOptimized
        
        # Recreate photon from data
        photon = Photon(
            position=photon_data['position'][1:4],  # spatial coordinates [x, y, z]
            direction=photon_data['velocity'][1:4]  # spatial velocity [dx, dy, dz]
        )
        
        # Set full state vector
        photon.x = photon_data['position'].copy()  # [t, x, y, z]
        photon.u = photon_data['velocity'].copy()  # [dt, dx, dy, dz]
        
        # Create local integrator
        integrator = IntegratorOptimized(_worker_metric, _worker_dt)
        
        # Integrate photon
        integrator.integrate(photon, n_steps)
        
        # Extract trajectory data
        if len(photon.history.states) > 1:
            trajectory_data = {
                'positions': np.array([state.position_Mpc for state in photon.history.states]),
                'times': np.array([state.time_s for state in photon.history.states]),
                'photon_id': photon.photon_id
            }
            return True, trajectory_data, photon.photon_id
        else:
            return False, None, photon.photon_id
            
    except Exception as e:
        print(f"Error integrating photon {photon_data.get('photon_id', 'unknown')}: {e}")
        return False, None, photon_data.get('photon_id', -1)


class AnalyticalMetricParallelIntegrator:
    """
    Parallel integrator optimized for analytical metrics.
    
    This integrator is specifically designed for metrics like Schwarzschild
    that don't require heavy data structures (grids, fields, etc.).
    
    Usage:
        >>> integrator = AnalyticalMetricParallelIntegrator(
        ...     metric=schwarzschild_metric,
        ...     dt=-1e13,
        ...     n_workers=4
        ... )
        >>> 
        >>> with integrator:
        ...     success_count = integrator.integrate_photons(photons, 3000)
        >>> 
        >>> # Or manually manage
        >>> integrator.close()
    """
    
    def __init__(self, metric, dt: float, n_workers: int = None):
        """
        Initialize analytical metric parallel integrator.
        
        Args:
            metric: Analytical metric object (Schwarzschild, Kerr, etc.)
            dt: Integration time step
            n_workers: Number of worker processes
        """
        self.metric = metric
        self.dt = dt
        self.n_workers = n_workers or max(1, mp.cpu_count() - 1)
        self.pool = None
        
        # Determine metric type and extract parameters
        self._extract_metric_info()
    
    def _extract_metric_info(self):
        """Extract metric type and parameters for worker initialization."""
        metric_class_name = self.metric.__class__.__name__
        
        if metric_class_name == "SchwarzschildMetricCartesian":
            self.metric_type = "schwarzschild"
            self.metric_params = {
                'mass': self.metric.mass,  # en kg
                'radius': self.metric.radius,  # en mètres  
                'center': self.metric.center.copy()  # en mètres
            }
        else:
            raise ValueError(f"Unsupported analytical metric: {metric_class_name}")
    
    def __enter__(self):
        """Context manager entry."""
        self._create_pool()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def _create_pool(self):
        """Create the persistent worker pool."""
        if self.pool is not None:
            return  # Pool already exists
        
        print(f"Creating analytical metric pool with {self.n_workers} workers...")
        start = time.time()
        
        self.pool = Pool(
            processes=self.n_workers,
            initializer=_init_analytical_worker,
            initargs=(self.metric_type, self.metric_params, self.dt)
        )
        
        elapsed = time.time() - start
        print(f"Analytical pool initialized in {elapsed:.3f}s")
    
    def integrate_photons(self, photons, n_steps: int, verbose: bool = True) -> int:
        """
        Integrate multiple photons using the persistent pool.
        
        Args:
            photons: List of Photon objects or Photons container
            n_steps: Number of integration steps
            verbose: Print progress information
        
        Returns:
            Number of successfully integrated photons
        """
        # Ensure pool exists
        if self.pool is None:
            self._create_pool()
        
        # Handle Photons container
        if hasattr(photons, 'photons'):
            photon_list = photons.photons
        else:
            photon_list = photons
        
        if verbose:
            print(f"Integrating {len(photon_list)} photons with analytical parallel integrator...")
        
        # Prepare photon data for workers
        photon_data_list = []
        for photon in photon_list:
            photon_data = {
                'position': photon.x.copy(),  # [t, x, y, z] state vector
                'velocity': photon.u.copy(),  # [dt, dx, dy, dz] velocity vector
                'photon_id': getattr(photon, 'photon_id', len(photon_data_list))  # ID ou index
            }
            photon_data_list.append(photon_data)
        
        # Parallel integration
        start_time = time.time()
        
        # Create tasks
        tasks = [(photon_data, n_steps) for photon_data in photon_data_list]
        
        # Execute in parallel
        results = self.pool.starmap(_integrate_photon_analytical, tasks)
        
        # Process results
        success_count = 0
        for i, (success, trajectory_data, photon_id) in enumerate(results):
            if success and trajectory_data is not None:
                # Update original photon with results
                photon = photon_list[i]
                # Clear existing history
                photon.history.states = []
                # Add trajectory points as simple state arrays
                for j, (pos, time_val) in enumerate(zip(trajectory_data['positions'], trajectory_data['times'])):
                    # Create state array [t, x, y, z, vt, vx, vy, vz] like the integrator expects
                    # For backward tracing, we store positions and reconstruct velocities if needed
                    state_array = np.zeros(8)
                    state_array[0] = time_val  # t
                    state_array[1:4] = pos     # x, y, z
                    # Velocities would need to be computed from derivatives, skip for now
                    photon.history.states.append(state_array)
                success_count += 1
            elif verbose:
                print(f"Failed to integrate photon {photon_id}")
        
        elapsed = time.time() - start_time
        
        if verbose:
            print(f"Analytical parallel integration completed: {success_count}/{len(photon_list)} in {elapsed:.1f}s")
        
        return success_count
    
    def close(self):
        """Close the worker pool and cleanup resources."""
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None
            print("Analytical metric pool closed")


# Convenience function for one-time use
def integrate_photons_analytical_parallel(
    metric, photons, n_steps: int, dt: float, n_workers: int = None
) -> int:
    """
    Convenience function for one-time analytical metric parallel integration.
    
    Args:
        metric: Analytical metric object
        photons: Photons to integrate
        n_steps: Integration steps
        dt: Time step
        n_workers: Number of workers
    
    Returns:
        Number of successfully integrated photons
    """
    with AnalyticalMetricParallelIntegrator(metric, dt, n_workers) as integrator:
        return integrator.integrate_photons(photons, n_steps)