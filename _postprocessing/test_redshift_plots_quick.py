#!/usr/bin/env python3
"""
Quick test of redshift visualization (faster version).
"""

import numpy as np
import h5py
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from excalibur.observables.redshift import RedshiftCalculator
from excalibur.observables.redshift_plots import plot_redshift_evolution


def load_trajectories(filename):
    """Load photon trajectories from HDF5 file."""
    trajectories = []
    
    with h5py.File(filename, 'r') as f:
        n_photons = f.attrs['n_photons']
        
        for i in range(n_photons):
            dataset_name = f"photon_{i}_states"
            if dataset_name in f:
                states = f[dataset_name][:]
                valid_mask = ~np.isnan(states).any(axis=1)
                clean_states = states[valid_mask]
                trajectories.append(clean_states)
    
    return trajectories


def main():
    """Quick test of single trajectory plot."""
    
    filename = "backward_raytracing_trajectories_OPTIMAL_mass_500_500_500_Mpc.h5"
    
    if not os.path.exists(filename):
        print(f"Error: File not found: {filename}")
        return
    
    print("Loading trajectories...")
    trajectories = load_trajectories(filename)
    print(f"Loaded {len(trajectories)} trajectories")
    
    # Test with first trajectory
    print("\nCreating RedshiftCalculator...")
    calc = RedshiftCalculator(trajectories[0])
    
    print("\nPlotting evolution (100 sample points)...")
    plot_redshift_evolution(
        calc,
        save_path="test_redshift_evolution.png",
        show=False,
        n_samples=100
    )
    
    print("\nDone! Saved to: test_redshift_evolution.png")


if __name__ == "__main__":
    main()
