#!/usr/bin/env python3
"""Quick smoke test for the JAX backend.

This script uses a mock cosmology + interpolator to verify that
`excalibur.jax_backend` works end-to-end on a small batch of photons.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from excalibur.jax_backend.batch_pipeline import integrate_photons_jax
from excalibur.core.constants import c


class MockCosmology:
    def a_of_eta(self, eta):
        return 1.0 + 0.1 * eta

    def adot_of_eta(self, eta):
        return 0.1


class MockInterpolator:
    def value_gradient_and_time_derivative(self, pos, field, eta):
        x, y, z = pos
        # Simple quadratic potential
        phi = -1e10 * (x**2 + y**2 + z**2) / 1e18
        grad_phi = np.array([-2e10 * x / 1e18, -2e10 * y / 1e18, -2e10 * z / 1e18])
        phi_dot = 0.0
        return phi, grad_phi, phi_dot


def create_test_states(n_photons=10):
    """Create a small batch of test photon states."""
    states = []
    rng = np.random.default_rng(0)

    for _ in range(n_photons):
        eta0 = 0.0
        pos0 = np.zeros(3)

        # Random direction on sphere
        phi = rng.uniform(0, 2 * np.pi)
        cost = rng.uniform(-1, 1)
        sint = np.sqrt(1 - cost**2)
        direction = np.array([sint * np.cos(phi), sint * np.sin(phi), cost])

        u_spatial = direction * c
        u0 = c

        state = np.array([
            eta0,
            pos0[0], pos0[1], pos0[2],
            u0,
            u_spatial[0], u_spatial[1], u_spatial[2],
        ])
        states.append(state)

    return np.array(states)


if __name__ == "__main__":
    print("Running JAX backend smoke test...")

    cosmo = MockCosmology()
    interp = MockInterpolator()

    initial_states = create_test_states(32)

    n_steps = 100
    dt = -1e-3

    traj = integrate_photons_jax(
        initial_states,
        cosmo,
        interp,
        n_steps,
        dt,
    )

    print("Trajectory shape:", traj.shape)
    print("Final positions (first 3 photons):")
    print(traj[-1, :3, 1:4])
    print("JAX backend smoke test completed.")
