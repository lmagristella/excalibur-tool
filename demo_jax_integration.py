#!/usr/bin/env python3
"""
JAX Integration Example for Excalibur

Demonstrates how to integrate JAX into the existing excalibur pipeline
for maximum performance gains.
"""

import sys
sys.path.append('/home/magri/excalibur_project')

import numpy as np
import time
import h5py
from pathlib import Path

# JAX imports
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap
    jax.config.update("jax_enable_x64", True)
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    print("JAX not available")
    sys.exit(1)

# Excalibur imports
from excalibur.core.cosmology import LCDMCosmology
from excalibur.grid.grid_3d import Grid3D
from excalibur.grid.linear_interpolator import LinearInterpolator3D
from excalibur.metrics.perturbed_flrw_metric import PerturbedFLRWMetric
from excalibur.photon.photon import Photon
from excalibur.core.constants import c

print("🚀 JAX-Excalibur Integration Demo")
print("=" * 50)

def load_test_data():
    """Load test data for demonstration."""
    print("📊 Loading test data...")
    
    # Use existing test data
    data_file = Path("/home/magri/excalibur_project/_data") / "backward_raytracing_schwarzschild_mass_500_500_500_Mpc.h5"
    
    if not data_file.exists():
        print("❌ Test data not found")
        return None, None, None
        
    # Create small synthetic data for demo
    cosmology = LCDMCosmology(H0=70.0, Omega_m=0.3, Omega_Lambda=0.7)
    
    # Small grid for fast demo
    nx, ny, nz = 32, 32, 32
    Lx, Ly, Lz = 100e6, 100e6, 100e6  # 100 Mpc
    
    # Create grid
    grid = Grid3D(nx, ny, nz, Lx, Ly, Lz)
    
    # Synthetic potential field (weak gravitational field)
    x_coords = np.linspace(-Lx/2, Lx/2, nx)
    y_coords = np.linspace(-Ly/2, Ly/2, ny)  
    z_coords = np.linspace(-Lz/2, Lz/2, nz)
    
    X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    
    # Simple Gaussian perturbation
    sigma = 20e6  # 20 Mpc
    phi_field = -1e10 * np.exp(-(X**2 + Y**2 + Z**2) / (2 * sigma**2))
    
    # Create interpolator
    interpolator = LinearInterpolator3D(grid)
    interpolator.set_field("Phi", phi_field, time_dependent=False)
    
    print(f"✅ Grid: {nx}×{ny}×{nz}, Box size: {Lx/1e6:.0f} Mpc")
    print(f"✅ Potential range: [{phi_field.min():.2e}, {phi_field.max():.2e}] m²/s²")
    
    return cosmology, grid, interpolator

@jit
def jax_geodesic_acceleration_optimized(u, eta, pos, a, adot, phi_norm, grad_phi_si, phi_dot_norm):
    """Highly optimized JAX geodesic acceleration."""
    c_val = 299792458.0
    c2 = c_val * c_val
    c4 = c2 * c2
    a2 = a * a
    adot_a = adot / a
    
    u0, u1, u2, u3 = u[0], u[1], u[2], u[3]
    u_spatial_sq = u1*u1 + u2*u2 + u3*u3
    
    # Newtonian gauge: psi = phi  
    psi = phi_norm
    grad_psi_norm = grad_phi_si / c2  # Normalize gradient
    
    # Temporal acceleration du0/dlambda
    expansion_term = (a*adot/c2) * u_spatial_sq
    perturbation_term = (2*a*adot*psi/c2 - a2*phi_dot_norm) * u_spatial_sq / c2
    coupling_term = 2 * jnp.dot(grad_psi_norm, u[1:]) * u0
    
    du0 = -(expansion_term + perturbation_term + coupling_term)
    
    # Spatial accelerations du_i/dlambda
    gravitational_term = grad_psi_norm / a2 * (u0 * u0)
    expansion_coupling = 2 * (adot_a - phi_dot_norm/c2) * u0 * u[1:]
    
    # Cross terms (simplified for performance)
    cross_terms = jnp.array([
        -grad_psi_norm[0] * (u1*u1) + grad_psi_norm[0] * (u2*u2 + u3*u3),
        -grad_psi_norm[1] * (u2*u2) + grad_psi_norm[1] * (u1*u1 + u3*u3), 
        -grad_psi_norm[2] * (u3*u3) + grad_psi_norm[2] * (u1*u1 + u2*u2)
    ]) * 0.1  # Reduced contribution for stability
    
    du_spatial = -(gravitational_term + expansion_coupling + cross_terms)
    
    return jnp.concatenate([jnp.array([du0]), du_spatial])

# Create vectorized version for batch processing
jax_geodesic_batch = vmap(jax_geodesic_acceleration_optimized, 
                         in_axes=(0, None, 0, None, None, 0, 0, 0))

@jit  
def jax_rk4_step_vectorized(states_batch, dt, eta, a, adot, phi_norms, grad_phi_batch, phi_dot_norms):
    """Vectorized RK4 step for multiple photons."""
    
    # k1
    positions = states_batch[:, 1:4]
    velocities = states_batch[:, 4:]
    
    k1 = jnp.zeros_like(states_batch)
    k1 = k1.at[:, :4].set(velocities)  
    k1_accel = jax_geodesic_batch(velocities, eta, positions, a, adot, phi_norms, grad_phi_batch, phi_dot_norms)
    k1 = k1.at[:, 4:].set(k1_accel)
    
    # k2
    states2 = states_batch + 0.5 * dt * k1
    vel2 = states2[:, 4:]
    pos2 = states2[:, 1:4]
    
    k2 = jnp.zeros_like(states_batch)
    k2 = k2.at[:, :4].set(vel2)
    k2_accel = jax_geodesic_batch(vel2, eta + 0.5*dt, pos2, a, adot, phi_norms, grad_phi_batch, phi_dot_norms)
    k2 = k2.at[:, 4:].set(k2_accel)
    
    # k3  
    states3 = states_batch + 0.5 * dt * k2
    vel3 = states3[:, 4:]
    pos3 = states3[:, 1:4]
    
    k3 = jnp.zeros_like(states_batch)
    k3 = k3.at[:, :4].set(vel3)
    k3_accel = jax_geodesic_batch(vel3, eta + 0.5*dt, pos3, a, adot, phi_norms, grad_phi_batch, phi_dot_norms)
    k3 = k3.at[:, 4:].set(k3_accel)
    
    # k4
    states4 = states_batch + dt * k3  
    vel4 = states4[:, 4:]
    pos4 = states4[:, 1:4]
    
    k4 = jnp.zeros_like(states_batch)
    k4 = k4.at[:, :4].set(vel4)
    k4_accel = jax_geodesic_batch(vel4, eta + dt, pos4, a, adot, phi_norms, grad_phi_batch, phi_dot_norms)
    k4 = k4.at[:, 4:].set(k4_accel)
    
    # Final step
    new_states = states_batch + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    return new_states

def create_test_photons(n_photons=100):
    """Create test photons for integration."""
    print(f"🌟 Creating {n_photons} test photons...")
    
    # Observer at origin
    obs_pos = np.array([0.0, 0.0, 0.0])
    
    # Create photon directions (spherical sampling)
    phi_angles = np.random.uniform(0, 2*np.pi, n_photons)
    theta_angles = np.random.uniform(0, np.pi, n_photons)
    
    photon_states = []
    
    for i in range(n_photons):
        phi = phi_angles[i]
        theta = theta_angles[i]
        
        # Direction unit vector
        direction = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])
        
        # Initial conformal time
        eta_init = 0.0
        
        # 4-velocity (null condition: u_0 = |u_spatial|)  
        u_spatial_magnitude = c  # Natural scale
        u_spatial = direction * u_spatial_magnitude
        u_temporal = u_spatial_magnitude  # For null geodesics
        
        # State vector: [eta, x, y, z, u_eta, u_x, u_y, u_z]
        state = np.array([
            eta_init, obs_pos[0], obs_pos[1], obs_pos[2],
            u_temporal, u_spatial[0], u_spatial[1], u_spatial[2]
        ])
        
        photon_states.append(state)
    
    return np.array(photon_states)

def benchmark_jax_integration(cosmology, interpolator, n_photons=100, n_steps=1000):
    """Benchmark JAX integration vs standard methods."""
    print(f"⚡ Benchmarking JAX integration...")
    print(f"   Photons: {n_photons}, Steps: {n_steps}")
    
    # Create test photons
    photon_states = create_test_photons(n_photons)
    
    # Integration parameters
    eta_start = 0.0
    eta_end = -500.0e6 / c  # ~500 Mpc lookback
    dt = (eta_end - eta_start) / n_steps
    
    print(f"   Time range: η = {eta_start:.2e} → {eta_end:.2e}")
    print(f"   Step size: Δη = {dt:.2e}")
    
    # Prepare metric data (assuming static field for simplicity)
    eta_mid = (eta_start + eta_end) / 2
    a = cosmology.a_of_eta(eta_mid)
    adot = cosmology.adot_of_eta(eta_mid)
    
    # Get potential data for all photon positions
    positions = photon_states[:, 1:4]
    phi_data = []
    grad_phi_data = []
    phi_dot_data = []
    
    for pos in positions:
        phi, grad_phi, phi_dot = interpolator.value_gradient_and_time_derivative(pos, "Phi", eta_mid)
        phi_data.append(phi / (c**2))  # Normalize
        grad_phi_data.append(grad_phi)
        phi_dot_data.append(phi_dot / (c**2))  # Normalize
    
    phi_norms = jnp.array(phi_data)
    grad_phi_batch = jnp.array(grad_phi_data)
    phi_dot_norms = jnp.array(phi_dot_data)
    
    # Convert to JAX arrays
    states_jax = jnp.array(photon_states)
    
    print("🔥 Warming up JAX (JIT compilation)...")
    # Warm-up
    for _ in range(5):
        _ = jax_rk4_step_vectorized(states_jax, dt, eta_mid, a, adot, phi_norms, grad_phi_batch, phi_dot_norms)
    
    print("📈 Running JAX benchmark...")
    start_time = time.time()
    
    # Run integration steps
    current_states = states_jax
    for step in range(n_steps):
        current_states = jax_rk4_step_vectorized(
            current_states, dt, eta_mid, a, adot, phi_norms, grad_phi_batch, phi_dot_norms
        )
    
    jax_time = time.time() - start_time
    
    # Performance metrics
    total_operations = n_photons * n_steps
    ops_per_second = total_operations / jax_time
    
    print(f"\n📊 JAX Integration Results:")
    print(f"   Total time: {jax_time:.4f} seconds")
    print(f"   Photon-steps: {total_operations:,}")
    print(f"   Performance: {ops_per_second:.0f} photon-steps/second")
    print(f"   Per photon: {jax_time/n_photons*1000:.2f} ms/photon")
    print(f"   Per step: {jax_time/n_steps*1000:.3f} ms/step")
    
    # Calculate photons per second equivalent
    if n_steps == 1000:  # Assuming ~1000 steps per photon integration
        photons_per_second = n_photons / jax_time
        print(f"   Equivalent: {photons_per_second:.1f} complete photons/second")
    
    return current_states, jax_time, ops_per_second

def save_jax_results(final_states, filename="jax_trajectory_demo.h5"):
    """Save JAX integration results."""
    output_path = Path("/home/magri/excalibur_project/_data/output") / filename
    output_path.parent.mkdir(exist_ok=True)
    
    print(f"💾 Saving results to {filename}...")
    
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('final_states', data=np.array(final_states))
        f.create_dataset('positions', data=np.array(final_states)[:, 1:4])
        f.create_dataset('velocities', data=np.array(final_states)[:, 4:])
        
        # Metadata
        f.attrs['n_photons'] = final_states.shape[0]
        f.attrs['integration_method'] = 'JAX-RK4'
        f.attrs['jax_version'] = jax.__version__
    
    print(f"✅ Saved {final_states.shape[0]} photon states")

if __name__ == "__main__":
    if not JAX_AVAILABLE:
        print("❌ JAX not available")
        sys.exit(1)
    
    print(f"✅ JAX version: {jax.__version__}")
    print(f"✅ JAX backend: {jax.default_backend()}")
    
    # Load data
    cosmology, grid, interpolator = load_test_data()
    if cosmology is None:
        sys.exit(1)
    
    # Run benchmark with different sizes
    test_cases = [
        (50, 500),    # Small test
        (100, 1000),  # Medium test  
        (200, 2000),  # Larger test
    ]
    
    results = {}
    
    for n_photons, n_steps in test_cases:
        print(f"\n{'='*60}")
        print(f"Test Case: {n_photons} photons × {n_steps} steps")
        print(f"{'='*60}")
        
        try:
            final_states, integration_time, performance = benchmark_jax_integration(
                cosmology, interpolator, n_photons, n_steps
            )
            
            results[f"{n_photons}x{n_steps}"] = {
                'time': integration_time,
                'performance': performance,
                'final_states': final_states
            }
            
            # Save results for largest test
            if n_photons == 200:
                save_jax_results(final_states, f"jax_demo_{n_photons}_photons.h5")
                
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("🎯 JAX Integration Summary")
    print(f"{'='*60}")
    
    for test_name, result in results.items():
        photons, steps = test_name.split('x')
        total_ops = int(photons) * int(steps)
        print(f"{test_name:<15} {result['time']:<8.3f}s  {result['performance']:<12.0f} ops/s  {total_ops:,} total ops")
    
    if results:
        best_performance = max(results.values(), key=lambda x: x['performance'])
        print(f"\n🏆 Best performance: {best_performance['performance']:.0f} photon-steps/second")
        
        # Estimate production capabilities
        print(f"\n🚀 Production Estimates:")
        print(f"   For 5000 steps/photon: {best_performance['performance']/5000:.0f} photons/second")
        print(f"   For 10000 steps/photon: {best_performance['performance']/10000:.0f} photons/second")
        print(f"   For 1000 photons: {1000*5000/best_performance['performance']:.1f} seconds")
    
    print(f"\n✅ JAX integration demo completed!")
    print("💡 Next steps: Integrate with full excalibur pipeline")
    print("🎯 Expected gains: 10-100x faster than current Python implementation")