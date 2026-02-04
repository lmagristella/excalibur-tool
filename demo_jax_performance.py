#!/usr/bin/env python3
"""
JAX Performance Demo for Excalibur Ray Tracing

This script demonstrates the performance gains achievable with JAX
for cosmological ray tracing calculations.
"""

import sys
import os
sys.path.append('/home/magri/excalibur_project')

import numpy as np
import time
from excalibur.metrics.perturbed_flrw_metric import PerturbedFLRWMetric
from excalibur.metrics.perturbed_flrw_metric_fast import PerturbedFLRWMetricFast

# Try to import JAX - fallback gracefully if not available
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap
    jax.config.update("jax_enable_x64", True)
    JAX_AVAILABLE = True
    print("JAX successfully loaded!")
except ImportError:
    print("JAX not available - running standard demo")
    JAX_AVAILABLE = False
    # Define dummy decorators for graceful fallback
    def jit(func):
        return func
    def vmap(func, in_axes=None):
        return func
    jnp = np

def create_test_metric_data():
    """Create synthetic metric data for testing."""
    print("Creating synthetic test data...")
    
    # Synthetic scale factor function
    def a_of_eta(eta):
        return 1.0 + 0.1 * eta  # Simple linear expansion
    
    def adot_of_eta(eta):
        return 0.1  # Constant expansion rate
    
    # Mock cosmology object
    class MockCosmology:
        def __init__(self):
            self.a_of_eta = a_of_eta
            self.adot_of_eta = adot_of_eta
    
    # Mock interpolator
    class MockInterpolator:
        def value_gradient_and_time_derivative(self, pos, field, eta):
            x, y, z = pos
            # Synthetic gravitational potential (weak field)
            phi = -1e10 * (x**2 + y**2 + z**2) / 1e18  # m²/s²
            grad_phi = np.array([-2e10 * x / 1e18, -2e10 * y / 1e18, -2e10 * z / 1e18])
            phi_dot = 0.0  # Static field
            return phi, grad_phi, phi_dot
    
    return MockCosmology(), MockInterpolator()

@jit
def jax_geodesic_acceleration_pure(u, a, adot, phi_norm, grad_phi_si, phi_dot_norm):
    """Pure JAX implementation of geodesic acceleration."""
    if not JAX_AVAILABLE:
        return numpy_geodesic_acceleration(u, a, adot, phi_norm, grad_phi_si, phi_dot_norm)
        
    c = 299792458.0
    c2 = c * c
    c4 = c2 * c2
    a2 = a * a
    adot_a = adot / a
    
    u0, u1, u2, u3 = u[0], u[1], u[2], u[3]
    u_spatial_sq = u1*u1 + u2*u2 + u3*u3
    
    # Newtonian gauge: psi = phi
    psi = phi_norm
    
    # du0/dlambda (temporal acceleration)
    Gamma_0ij_term = (a*adot/c2 + 2*a*adot/c4 * (phi_norm + psi) - a2/c4 * phi_dot_norm) * u_spatial_sq
    Gamma_00i_term = 2 * ((grad_phi_si[0]/c2) * u0 * u1 +
                          (grad_phi_si[1]/c2) * u0 * u2 +
                          (grad_phi_si[2]/c2) * u0 * u3)
    
    du0 = -(Gamma_0ij_term + Gamma_00i_term)
    
    # du1/dlambda (x acceleration)
    Gamma_100_term = (grad_phi_si[0] / c2) / a2 * u0*u0
    Gamma_10i_term = 2 * (adot_a - phi_dot_norm/c2) * u0 * u1
    Gamma_1ii_term = (-(grad_phi_si[0]/c2) * u1*u1 +
                      (grad_phi_si[0]/c2) * (u2*u2 + u3*u3) +
                      (-(grad_phi_si[1]/c2)) * u1*u2 +
                      (-(grad_phi_si[2]/c2)) * u1*u3)
    
    du1 = -(Gamma_100_term + Gamma_10i_term + Gamma_1ii_term)
    
    # du2/dlambda (y acceleration)
    Gamma_200_term = (grad_phi_si[1] / c2) / a2 * u0*u0
    Gamma_20i_term = 2 * (adot_a - phi_dot_norm/c2) * u0 * u2
    Gamma_2ii_term = (-(grad_phi_si[1]/c2) * u2*u2 +
                      (grad_phi_si[1]/c2) * (u1*u1 + u3*u3) +
                      (-(grad_phi_si[0]/c2)) * u2*u1 +
                      (-(grad_phi_si[2]/c2)) * u2*u3)
    
    du2 = -(Gamma_200_term + Gamma_20i_term + Gamma_2ii_term)
    
    # du3/dlambda (z acceleration)
    Gamma_300_term = (grad_phi_si[2] / c2) / a2 * u0*u0
    Gamma_30i_term = 2 * (adot_a - phi_dot_norm/c2) * u0 * u3
    Gamma_3ii_term = (-(grad_phi_si[2]/c2) * u3*u3 +
                      (grad_phi_si[2]/c2) * (u1*u1 + u2*u2) +
                      (-(grad_phi_si[0]/c2)) * u3*u1 +
                      (-(grad_phi_si[1]/c2)) * u3*u2)
    
    du3 = -(Gamma_300_term + Gamma_30i_term + Gamma_3ii_term)
    
    return jnp.array([du0, du1, du2, du3])

def numpy_geodesic_acceleration(u, a, adot, phi_norm, grad_phi_si, phi_dot_norm):
    """NumPy reference implementation."""
    c = 299792458.0
    c2 = c * c
    c4 = c2 * c2
    a2 = a * a
    adot_a = adot / a
    
    u0, u1, u2, u3 = u[0], u[1], u[2], u[3]
    u_spatial_sq = u1*u1 + u2*u2 + u3*u3
    
    psi = phi_norm
    
    # Same calculation as JAX version but with NumPy
    Gamma_0ij_term = (a*adot/c2 + 2*a*adot/c4 * (phi_norm + psi) - a2/c4 * phi_dot_norm) * u_spatial_sq
    Gamma_00i_term = 2 * ((grad_phi_si[0]/c2) * u0 * u1 +
                          (grad_phi_si[1]/c2) * u0 * u2 +
                          (grad_phi_si[2]/c2) * u0 * u3)
    
    du0 = -(Gamma_0ij_term + Gamma_00i_term)
    
    # Spatial components (simplified for demo)
    Gamma_100_term = (grad_phi_si[0] / c2) / a2 * u0*u0
    du1 = -Gamma_100_term
    du2 = -(grad_phi_si[1] / c2) / a2 * u0*u0
    du3 = -(grad_phi_si[2] / c2) / a2 * u0*u0
    
    return np.array([du0, du1, du2, du3])

def benchmark_jax_vs_numpy(n_iterations=10000):
    """Benchmark JAX vs NumPy performance."""
    print(f"\n{'='*60}")
    print("JAX vs NumPy Performance Benchmark")
    print(f"{'='*60}")
    
    # Create test data
    cosmology, interpolator = create_test_metric_data()
    
    # Test parameters
    eta = 0.0
    pos = np.array([1e26, 1e26, 1e26])  # 100 Mpc coordinates
    u = np.array([1.0, 0.1, 0.1, 0.1])  # 4-velocity
    
    # Get metric data
    phi_si, grad_phi_si, phi_dot_si = interpolator.value_gradient_and_time_derivative(pos, "Phi", eta)
    a = cosmology.a_of_eta(eta)
    adot = cosmology.adot_of_eta(eta)
    
    c = 299792458.0
    phi_norm = phi_si / (c**2)
    phi_dot_norm = phi_dot_si / (c**2)
    
    print(f"Test parameters:")
    print(f"  Position: {pos[0]/1e24:.1f}, {pos[1]/1e24:.1f}, {pos[2]/1e24:.1f} Mpc")
    print(f"  Potential: {phi_si:.2e} m²/s²")
    print(f"  Scale factor: {a:.3f}")
    print(f"  Iterations: {n_iterations:,}")
    
    if JAX_AVAILABLE:
        # Warm-up JAX (compilation)
        print("\nWarming up JAX (JIT compilation)...")
        u_jax = jnp.array(u)
        grad_phi_jax = jnp.array(grad_phi_si)
        
        for _ in range(10):
            _ = jax_geodesic_acceleration_pure(u_jax, a, adot, phi_norm, grad_phi_jax, phi_dot_norm)
        
        # Benchmark JAX
        print("Benchmarking JAX...")
        start_time = time.time()
        for _ in range(n_iterations):
            result_jax = jax_geodesic_acceleration_pure(u_jax, a, adot, phi_norm, grad_phi_jax, phi_dot_norm)
        jax_time = time.time() - start_time
        
        # Benchmark NumPy
        print("Benchmarking NumPy...")
        start_time = time.time()
        for _ in range(n_iterations):
            result_numpy = numpy_geodesic_acceleration(u, a, adot, phi_norm, grad_phi_si, phi_dot_norm)
        numpy_time = time.time() - start_time
        
        # Results
        print(f"\n{'Results':<20} {'NumPy':<15} {'JAX':<15} {'Speedup':<10}")
        print("-" * 65)
        print(f"{'Time (s)':<20} {numpy_time:<15.4f} {jax_time:<15.4f} {numpy_time/jax_time:<10.1f}x")
        print(f"{'Steps/sec':<20} {n_iterations/numpy_time:<15.0f} {n_iterations/jax_time:<15.0f}")
        print(f"{'μs/step':<20} {numpy_time/n_iterations*1e6:<15.1f} {jax_time/n_iterations*1e6:<15.1f}")
        
        # Verify results are equivalent
        diff = np.abs(np.array(result_jax) - result_numpy)
        max_diff = np.max(diff)
        print(f"\nNumerical accuracy:")
        print(f"  Maximum difference: {max_diff:.2e}")
        print(f"  Results match: {'✓' if max_diff < 1e-12 else '✗'}")
        
        return numpy_time / jax_time
    
    else:
        print("JAX not available - skipping benchmark")
        return None

def benchmark_vectorization(batch_sizes=[1, 10, 100, 1000]):
    """Benchmark JAX vectorization for batch processing."""
    if not JAX_AVAILABLE:
        print("JAX not available - skipping vectorization benchmark")
        return
    
    print(f"\n{'='*60}")
    print("JAX Vectorization Benchmark")
    print(f"{'='*60}")
    
    cosmology, interpolator = create_test_metric_data()
    
    # Create vectorized function
    jax_geodesic_vmap = vmap(jax_geodesic_acceleration_pure, in_axes=(0, None, None, None, None, None))
    
    eta = 0.0
    pos = np.array([1e26, 1e26, 1e26])
    phi_si, grad_phi_si, phi_dot_si = interpolator.value_gradient_and_time_derivative(pos, "Phi", eta)
    a = cosmology.a_of_eta(eta)
    adot = cosmology.adot_of_eta(eta)
    
    c = 299792458.0
    phi_norm = phi_si / (c**2)
    phi_dot_norm = phi_dot_si / (c**2)
    grad_phi_jax = jnp.array(grad_phi_si)
    
    print(f"{'Batch Size':<12} {'Time (s)':<12} {'Steps/sec':<15} {'Efficiency':<12}")
    print("-" * 55)
    
    for batch_size in batch_sizes:
        # Create batch of random 4-velocities
        u_batch = np.random.normal(0, 0.1, (batch_size, 4))
        u_batch[:, 0] = 1.0  # Set temporal component
        u_batch_jax = jnp.array(u_batch)
        
        # Warm-up
        for _ in range(5):
            _ = jax_geodesic_vmap(u_batch_jax, a, adot, phi_norm, grad_phi_jax, phi_dot_norm)
        
        # Benchmark
        n_iterations = max(100, 1000 // batch_size)
        start_time = time.time()
        for _ in range(n_iterations):
            _ = jax_geodesic_vmap(u_batch_jax, a, adot, phi_norm, grad_phi_jax, phi_dot_norm)
        total_time = time.time() - start_time
        
        total_steps = n_iterations * batch_size
        steps_per_sec = total_steps / total_time
        efficiency = steps_per_sec / (batch_size * 1000)  # Normalized efficiency
        
        print(f"{batch_size:<12} {total_time:<12.4f} {steps_per_sec:<15.0f} {efficiency:<12.2f}")

def demonstrate_automatic_differentiation():
    """Demonstrate JAX automatic differentiation for geodesic calculations."""
    if not JAX_AVAILABLE:
        print("JAX not available - skipping autodiff demo")
        return
    
    print(f"\n{'='*60}")
    print("JAX Automatic Differentiation Demo")
    print(f"{'='*60}")
    
    cosmology, interpolator = create_test_metric_data()
    
    # Test parameters
    eta = 0.0
    pos = np.array([1e26, 1e26, 1e26])
    u = jnp.array([1.0, 0.1, 0.1, 0.1])
    
    phi_si, grad_phi_si, phi_dot_si = interpolator.value_gradient_and_time_derivative(pos, "Phi", eta)
    a = cosmology.a_of_eta(eta)
    adot = cosmology.adot_of_eta(eta)
    
    c = 299792458.0
    phi_norm = phi_si / (c**2)
    phi_dot_norm = phi_dot_si / (c**2)
    grad_phi_jax = jnp.array(grad_phi_si)
    
    # Define a function to differentiate
    def geodesic_norm(u_vec):
        """Compute the norm of geodesic acceleration."""
        accel = jax_geodesic_acceleration_pure(u_vec, a, adot, phi_norm, grad_phi_jax, phi_dot_norm)
        return jnp.linalg.norm(accel)
    
    # Compute derivatives automatically
    grad_func = jax.grad(geodesic_norm)
    jacobian_func = jax.jacfwd(geodesic_norm)
    
    # Evaluate
    acceleration = jax_geodesic_acceleration_pure(u, a, adot, phi_norm, grad_phi_jax, phi_dot_norm)
    norm_value = geodesic_norm(u)
    gradient = grad_func(u)
    jacobian = jacobian_func(u)
    
    print("Geodesic acceleration:")
    print(f"  du/dλ = {acceleration}")
    print(f"  ||du/dλ|| = {norm_value:.6e}")
    
    print("\nAutomatic derivatives:")
    print(f"  ∇(||du/dλ||) = {gradient}")
    print(f"  Jacobian shape: {jacobian.shape}")
    
    print("\nThis enables:")
    print("  • Automatic error estimation")
    print("  • Adaptive step size control")
    print("  • Optimization of photon paths")
    print("  • Stability analysis")

if __name__ == "__main__":
    print("Excalibur JAX Performance Demo")
    print("=" * 60)
    
    if JAX_AVAILABLE:
        print("✓ JAX is available and configured")
        print(f"✓ JAX version: {jax.__version__}")
        print(f"✓ JAX backend: {jax.default_backend()}")
        print(f"✓ 64-bit precision: {jax.config.jax_enable_x64}")
    else:
        print("✗ JAX is not available")
        print("  Install with: pip install jax jaxlib")
        
    # Run benchmarks
    speedup = benchmark_jax_vs_numpy(n_iterations=50000)
    
    if speedup:
        print(f"\n🚀 JAX provides {speedup:.1f}x speedup for geodesic calculations!")
        
        if speedup > 10:
            print("   Excellent performance - ready for production use")
        elif speedup > 5:
            print("   Good performance - significant improvement")
        else:
            print("   Modest improvement - consider larger problem sizes")
    
    # Additional demos
    benchmark_vectorization()
    demonstrate_automatic_differentiation()
    
    print(f"\n{'='*60}")
    print("Integration Strategy for Excalibur + JAX:")
    print("✓ Use JAX for geodesic acceleration calculations")
    print("✓ Vectorize photon processing with vmap")  
    print("✓ JIT compile integration loops")
    print("✓ Leverage autodiff for adaptive error control")
    print("✓ Ready for GPU acceleration when needed")
    print("=" * 60)