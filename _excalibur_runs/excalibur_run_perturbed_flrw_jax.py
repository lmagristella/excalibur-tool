#!/usr/bin/env python3
"""Backward ray tracing on perturbed FLRW using JAX backend.

This script mirrors `integrate_photons_on_perturbed_flrw_integrator.py`
 but replaces the integration core by `excalibur.jax_backend` (JAX RK4).

Pipeline:
- build cosmology, grid, interpolator, metric (same as integrator script)
- generate photons in a cone at observer
- convert photons to a (N, 8) state array for JAX
- call integrate_photons_jax
- save final states to HDF5 in `_data/output/`

Note: this first version saves only final states, not full trajectories.
We can later extend it to full histories if needed.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import interpolate

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from excalibur.grid.grid import Grid
from excalibur.grid.interpolator_fast import InterpolatorFast
from excalibur.metrics.perturbed_flrw_metric_fast import PerturbedFLRWMetricFast
from excalibur.photon.photons import Photons
from excalibur.photon.photon import Photon
from excalibur.core.constants import *
from excalibur.core.cosmology import LCDM_Cosmology
from excalibur.objects.spherical_mass import spherical_mass
from excalibur.io.filename_utils import generate_trajectory_filename

from excalibur.jax_backend.batch_pipeline import integrate_photons_jax

try:
    import h5py
except Exception:
    h5py = None


def build_cosmology():
    H0 = 70  # km/s/Mpc
    Omega_m = 0.3
    Omega_r = 0.0
    Omega_lambda = 0.7
    cosmology = LCDM_Cosmology(H0, Omega_m=Omega_m, Omega_r=Omega_r, Omega_lambda=Omega_lambda)

    _ = cosmology.a_of_eta(1e18)
    eta_present = cosmology._eta_at_a1

    eta_start = eta_present
    eta_end = 0.5 * eta_present
    eta_sample = np.linspace(eta_start, eta_end, 1000)
    a_sample = cosmology.a_of_eta(eta_sample)
    a_of_eta = cosmology.a_of_eta  # use cosmology method directly

    return cosmology, eta_present, a_of_eta, eta_start, eta_end, a_sample


def build_grid_and_potential():
    N = 256  # reduce for first JAX tests
    grid_size = 500 * one_Mpc
    dx = dy = dz = grid_size / N

    shape = (N, N, N)
    spacing = (dx, dy, dz)
    origin = np.array([0, 0, 0]) * grid_size
    grid = Grid(shape, spacing, origin)

    x = y = z = np.linspace(0, grid_size, N)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    M = 1e15 * one_Msun
    radius = 5 * one_Mpc
    center = np.array([0.5, 0.5, 0.5]) * grid_size
    spherical_halo = spherical_mass(M, radius, center)

    phi_field = spherical_halo.potential(X, Y, Z)

    # NEW: work with dimensionless potential Phi/c^2 to keep numbers ~1e-5
    phi_field = phi_field / c**2

    grid.add_field("Phi", phi_field)

    return grid, phi_field, M, radius, center


def generate_photon_states(metric, cosmology, grid, center, eta_present, n_photons=200):
    observer_eta = eta_present
    observer_position = np.array([1e-12, 1e-12, 1e-12]) * one_Mpc

    direction_to_mass = center - observer_position
    direction_to_mass = direction_to_mass / np.linalg.norm(direction_to_mass)

    photons = Photons(metric=metric)
    observer_4d_position = np.array([observer_eta, *observer_position])

    n_theta = int(np.sqrt(n_photons))
    n_phi = int(np.sqrt(n_photons))

    photons.generate_cone_grid(
        n_theta=n_theta,
        n_phi=n_phi,
        origin=observer_4d_position,
        central_direction=direction_to_mass,
        cone_angle=np.pi / 24,
    )

    for photon in photons:
        photon.u = -photon.u
        photon.record()

    # Convert photons to (N, 8) state array
    states = []
    for photon in photons:
        x = photon.x
        u = photon.u
        state = np.array([x[0], x[1], x[2], x[3], u[0], u[1], u[2], u[3]])
        states.append(state)

    initial_states = np.array(states)
    return photons, initial_states, observer_position


def save_final_states_hdf5(filename, final_states, M, radius, center, observer_position, metadata=None):
    if h5py is None:
        print("[WARNING] h5py not available, skipping HDF5 save.")
        return

    output_path = Path("../_data/output") / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("final_states", data=final_states)
        f.create_dataset("final_positions", data=final_states[:, 1:4])
        f.create_dataset("final_velocities", data=final_states[:, 4:])

        f.attrs["mass_kg"] = M
        f.attrs["radius_m"] = radius
        f.attrs["mass_position_m"] = center
        f.attrs["observer_position_m"] = observer_position
        if metadata:
            for k, v in metadata.items():
                f.attrs[k] = v

    print(f"   [OK] Saved final states to {output_path}")


def main():
    print("=== Backward Ray Tracing with Excalibur + JAX (perturbed FLRW) ===")

    t0 = time.time()

    print("1. Cosmology setup...")
    cosmology, eta_present, a_of_eta, eta_start, eta_end, a_sample = build_cosmology()

    print("2. Grid and potential...")
    grid, phi_field, M, radius, center = build_grid_and_potential()

    print("3. Metric and interpolator (host-side)...")
    interpolator = InterpolatorFast(grid)
    metric = PerturbedFLRWMetricFast(cosmology.a_of_eta, grid, interpolator)

    print("4. Photon generation...")
    photons, initial_states, observer_position = generate_photon_states(
        metric, cosmology, grid, center, eta_present, n_photons=200
    )

    print(f"   Generated {len(photons)} photons for JAX integration")

    print("5. Integration parameters...")
    a_obs = a_of_eta(eta_present)
    box_comoving_distance = grid.shape[0] * grid.spacing[0]
    dt_initial = grid.spacing[0] / (10 * c)
    dt = -dt_initial
    n_steps = int(box_comoving_distance / (c * abs(dt_initial)))

    print(f"   dt = {dt:.2e} s, n_steps ≈ {n_steps}")

    # --- Debug: inspect metric inputs for the first photon ---
    sample_state = initial_states[0]
    eta_sample = sample_state[3]
    x_sample = sample_state[1:4]

    # Cosmology values
    a_eta = cosmology.a_of_eta(eta_sample)
    try:
        adot_eta = cosmology.adot_of_eta(eta_sample)
    except AttributeError:
        # Fallback: finite-difference derivative if explicit adot is unavailable
        d_eta_fd = 1e-3 * abs(eta_sample) if eta_sample != 0 else 1e-3
        a_plus = cosmology.a_of_eta(eta_sample + d_eta_fd)
        a_minus = cosmology.a_of_eta(eta_sample - d_eta_fd)
        adot_eta = (a_plus - a_minus) / (2.0 * d_eta_fd)

    # Potential values (InterpolatorFast expects x, field, t)
    phi_val, grad_phi_val, phi_dot_val = interpolator.value_gradient_and_time_derivative(
        x_sample, "Phi", eta_sample
    )

    print("[DEBUG] sample eta:", eta_sample)
    print("[DEBUG] a(eta):", a_eta, "adot(eta):", adot_eta)
    print("[DEBUG] phi:", phi_val)
    print("[DEBUG] grad_phi:", grad_phi_val)
    print("[DEBUG] phi_dot:", phi_dot_val)

    print("6. JAX integration...")
    t_int0 = time.time()

    traj = integrate_photons_jax(
        initial_states,
        cosmology,
        interpolator,
        n_steps=n_steps,
        dt=dt,
        field_name="Phi",
        eta_mode="midpoint",
    )

    t_int = time.time() - t_int0
    print(f"   JAX integration done in {t_int:.2f} s")

    final_states = traj[-1]

    print("7. Saving results (final states only)...")
    filename = "excalibur_run_perturbed_flrw_jax_final_states.h5"

    save_final_states_hdf5(
        filename,
        final_states,
        M,
        radius,
        center,
        observer_position,
        metadata={
            "n_photons": final_states.shape[0],
            "n_steps": n_steps,
            "dt": dt,
            "integrator": "jax_rk4_batch",
        },
    )

    total_time = time.time() - t0
    print("=== Summary ===")
    print(f"Total time: {total_time:.2f} s")
    print(f"Photons: {final_states.shape[0]}, steps: {n_steps}")
    print(f"Approx photons/s (kernel): {len(photons) * n_steps / max(t_int, 1e-6):.1f}")


if __name__ == "__main__":
    main()
