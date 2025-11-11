#!/usr/bin/env python3
"""
Example script demonstrating redshift visualization tools.

This script shows how to create various plots analyzing redshift evolution
along photon trajectories in perturbed FLRW cosmology.
"""

import numpy as np
import h5py
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from excalibur.observables.redshift import RedshiftCalculator
from excalibur.observables.redshift_plots import (
    plot_redshift_evolution,
    plot_redshift_statistics,
    plot_redshift_vs_quantity
)


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
    """Main function demonstrating all visualization tools."""
    
    # Input file
    filename = "backward_raytracing_trajectories_OPTIMAL_mass_500_500_500_Mpc.h5"
    
    if not os.path.exists(filename):
        print(f"Error: File not found: {filename}")
        print("Please run integrate_photons_on_perturbed_flrw_OPTIMAL.py first")
        return
    
    print("="*70)
    print("REDSHIFT VISUALIZATION EXAMPLES")
    print("="*70)
    
    # Load trajectories
    print(f"\n1. Loading trajectories from {filename}...")
    trajectories = load_trajectories(filename)
    print(f"   Loaded {len(trajectories)} photon trajectories")
    
    # Check if quantities are present
    has_quantities = (trajectories[0].shape[1] >= 14)
    if has_quantities:
        print("   ✓ Trajectories contain pre-computed quantities")
    else:
        print("   ✗ Trajectories do not contain quantities (will need callables)")
        return
    
    # Create RedshiftCalculators for all trajectories
    print("\n2. Creating RedshiftCalculators...")
    calculators = []
    for i, traj in enumerate(trajectories):
        calc = RedshiftCalculator(traj)
        calculators.append(calc)
        if (i + 1) % 10 == 0:
            print(f"   Progress: {i+1}/{len(trajectories)}")
    
    print(f"   ✓ Created {len(calculators)} calculators")
    
    # Example 1: Single trajectory evolution
    print("\n3. Plotting redshift evolution for first trajectory...")
    print("   This shows how each redshift component evolves along one photon path")
    plot_redshift_evolution(
        calculators[0],
        save_path="redshift_evolution_single_trajectory.png",
        show=False
    )
    
    # Example 2: Statistical analysis of all trajectories
    print("\n4. Plotting statistical analysis for all trajectories...")
    print("   This shows mean evolution, distributions, and correlations")
    plot_redshift_statistics(
        calculators,
        save_path="redshift_statistics_all_trajectories.png",
        show=False
    )
    
    # Example 3: Redshift vs scale factor for all trajectories
    print("\n5. Plotting total redshift vs scale factor...")
    plot_redshift_vs_quantity(
        calculators,
        quantity='a',
        component='total',
        save_path="redshift_vs_scale_factor.png",
        show=False
    )
    
    # Example 4: Redshift vs distance
    print("\n6. Plotting total redshift vs distance from observer...")
    plot_redshift_vs_quantity(
        calculators,
        quantity='distance',
        component='total',
        save_path="redshift_vs_distance.png",
        show=False
    )
    
    # Example 5: Individual components vs scale factor
    print("\n7. Plotting individual components vs scale factor...")
    
    for component in ['H', 'SW', 'ISW']:
        print(f"   Component: z_{component}")
        plot_redshift_vs_quantity(
            calculators,
            quantity='a',
            component=component,
            save_path=f"redshift_{component}_vs_scale_factor.png",
            show=False
        )
    
    # Example 6: Redshift vs conformal time
    print("\n8. Plotting total redshift vs conformal time...")
    plot_redshift_vs_quantity(
        calculators,
        quantity='eta',
        component='total',
        save_path="redshift_vs_conformal_time.png",
        show=False
    )
    
    # Example 7: Redshift vs proper distance
    print("\n9. Plotting total redshift vs proper distance traveled...")
    plot_redshift_vs_quantity(
        calculators,
        quantity='proper_distance',
        component='total',
        save_path="redshift_vs_proper_distance.png",
        show=False
    )
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - redshift_evolution_single_trajectory.png")
    print("  - redshift_statistics_all_trajectories.png")
    print("  - redshift_vs_scale_factor.png")
    print("  - redshift_vs_distance.png")
    print("  - redshift_H_vs_scale_factor.png")
    print("  - redshift_SW_vs_scale_factor.png")
    print("  - redshift_ISW_vs_scale_factor.png")
    print("  - redshift_vs_conformal_time.png")
    print("  - redshift_vs_proper_distance.png")
    print("="*70)


if __name__ == "__main__":
    main()
