#!/usr/bin/env python3
"""
Validate homogeneous redshift against FLRW theoretical prediction.

This script compares the measured redshift from backward ray-traced trajectories
with the theoretical homogeneous redshift formula:

    1 + z_H = a_obs / a_em

In pure FLRW (no perturbations), this should be exact. With perturbations,
the homogeneous component should still match the theoretical value since
it only depends on the scale factor evolution.
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from excalibur.core.constants import c, one_Mpc
from excalibur.observables.redshift import RedshiftCalculator


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


def compute_theoretical_redshift(a_obs, a_em):
    """
    Compute theoretical homogeneous redshift in FLRW.
    
    Formula: 1 + z = a_obs / a_em
    
    Parameters:
    -----------
    a_obs : float
        Scale factor at observation
    a_em : float
        Scale factor at emission
    
    Returns:
    --------
    z_theory : float
        Theoretical redshift
    """
    return (a_obs / a_em) - 1.0


def validate_redshift(trajectories):
    """
    Validate redshift measurements against theory.
    
    Returns:
    --------
    results : dict
        Dictionary with validation results
    """
    n_photons = len(trajectories)
    
    # Arrays to store results
    z_measured = np.zeros(n_photons)
    z_theory = np.zeros(n_photons)
    a_obs_array = np.zeros(n_photons)
    a_em_array = np.zeros(n_photons)
    eta_obs_array = np.zeros(n_photons)
    eta_em_array = np.zeros(n_photons)
    
    print(f"\nValidating redshift for {n_photons} trajectories...")
    
    for i, traj in enumerate(trajectories):
        # Create calculator
        calc = RedshiftCalculator(traj)
        
        # Measured redshift from trajectory
        z_measured[i] = calc.compute_homogeneous_redshift()
        
        # Extract scale factors
        a_obs_array[i] = calc.a_obs
        a_em_array[i] = calc.a_em
        eta_obs_array[i] = calc.eta_obs
        eta_em_array[i] = calc.eta_em
        
        # Theoretical redshift
        z_theory[i] = compute_theoretical_redshift(calc.a_obs, calc.a_em)
        
        if (i + 1) % 10 == 0:
            print(f"   Progress: {i+1}/{n_photons}")
    
    # Compute errors
    absolute_error = z_measured - z_theory
    relative_error = (z_measured - z_theory) / z_theory
    
    results = {
        'z_measured': z_measured,
        'z_theory': z_theory,
        'absolute_error': absolute_error,
        'relative_error': relative_error,
        'a_obs': a_obs_array,
        'a_em': a_em_array,
        'eta_obs': eta_obs_array,
        'eta_em': eta_em_array
    }
    
    return results


def print_validation_summary(results):
    """Print summary of validation results."""
    
    z_measured = results['z_measured']
    z_theory = results['z_theory']
    abs_err = results['absolute_error']
    rel_err = results['relative_error']
    
    print("\n" + "="*70)
    print("REDSHIFT VALIDATION SUMMARY")
    print("="*70)
    
    print("\nScale Factors:")
    print(f"  a_obs (observation): mean = {np.mean(results['a_obs']):.6f}, std = {np.std(results['a_obs']):.6e}")
    print(f"  a_em (emission):     mean = {np.mean(results['a_em']):.6f}, std = {np.std(results['a_em']):.6e}")
    
    print("\nConformal Times:")
    print(f"  η_obs: mean = {np.mean(results['eta_obs']):.6e} s")
    print(f"  η_em:  mean = {np.mean(results['eta_em']):.6e} s")
    print(f"  Δη:    mean = {np.mean(results['eta_obs'] - results['eta_em']):.6e} s")
    
    print("\nRedshift Measurements:")
    print(f"  z_measured (from trajectories):")
    print(f"    Mean:   {np.mean(z_measured):.10f}")
    print(f"    Std:    {np.std(z_measured):.10e}")
    print(f"    Min:    {np.min(z_measured):.10f}")
    print(f"    Max:    {np.max(z_measured):.10f}")
    
    print(f"\n  z_theory (FLRW formula):")
    print(f"    Mean:   {np.mean(z_theory):.10f}")
    print(f"    Std:    {np.std(z_theory):.10e}")
    print(f"    Min:    {np.min(z_theory):.10f}")
    print(f"    Max:    {np.max(z_theory):.10f}")
    
    print("\nErrors:")
    print(f"  Absolute error (z_measured - z_theory):")
    print(f"    Mean:   {np.mean(abs_err):.10e}")
    print(f"    Std:    {np.std(abs_err):.10e}")
    print(f"    Min:    {np.min(abs_err):.10e}")
    print(f"    Max:    {np.max(abs_err):.10e}")
    print(f"    RMS:    {np.sqrt(np.mean(abs_err**2)):.10e}")
    
    print(f"\n  Relative error ((z_measured - z_theory) / z_theory):")
    print(f"    Mean:   {np.mean(rel_err):.10e}")
    print(f"    Std:    {np.std(rel_err):.10e}")
    print(f"    Min:    {np.min(rel_err):.10e}")
    print(f"    Max:    {np.max(rel_err):.10e}")
    print(f"    RMS:    {np.sqrt(np.mean(rel_err**2)):.10e}")
    
    # Validation criteria
    print("\n" + "="*70)
    print("VALIDATION CRITERIA")
    print("="*70)
    
    rms_rel_error = np.sqrt(np.mean(rel_err**2))
    max_abs_rel_error = np.max(np.abs(rel_err))
    
    print(f"\n  RMS relative error:     {rms_rel_error:.2e}")
    print(f"  Max |relative error|:   {max_abs_rel_error:.2e}")
    
    # Check criteria
    tolerance_excellent = 1e-10  # 0.00000001% error
    tolerance_good = 1e-8        # 0.000001% error
    tolerance_acceptable = 1e-6  # 0.0001% error
    
    if max_abs_rel_error < tolerance_excellent:
        print(f"\n  ✅ EXCELLENT: Max relative error < {tolerance_excellent:.0e} (numerical precision)")
    elif max_abs_rel_error < tolerance_good:
        print(f"\n  ✅ GOOD: Max relative error < {tolerance_good:.0e} (high accuracy)")
    elif max_abs_rel_error < tolerance_acceptable:
        print(f"\n  ✅ ACCEPTABLE: Max relative error < {tolerance_acceptable:.0e}")
    else:
        print(f"\n  ⚠️  WARNING: Max relative error > {tolerance_acceptable:.0e}")
        print("     This may indicate integration errors or incorrect scale factor evolution")
    
    print("\n" + "="*70)
    
    return rms_rel_error < tolerance_acceptable


def plot_validation(results, save_path=None):
    """Create validation plots."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Homogeneous Redshift Validation Against FLRW Theory', 
                 fontsize=16, fontweight='bold')
    
    z_measured = results['z_measured']
    z_theory = results['z_theory']
    abs_err = results['absolute_error']
    rel_err = results['relative_error']
    a_em = results['a_em']
    
    # Panel 1: Measured vs Theory (scatter)
    ax = axes[0, 0]
    ax.scatter(z_theory, z_measured, alpha=0.7, s=100, edgecolors='k', linewidth=0.5)
    
    # Perfect agreement line
    z_range = [np.min(z_theory), np.max(z_theory)]
    ax.plot(z_range, z_range, 'r--', lw=2, label='Perfect agreement')
    
    ax.set_xlabel('z_theory (FLRW)', fontsize=12)
    ax.set_ylabel('z_measured (trajectory)', fontsize=12)
    ax.set_title('Measured vs Theoretical Redshift', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Panel 2: Absolute error vs z_theory
    ax = axes[0, 1]
    ax.scatter(z_theory, abs_err, alpha=0.7, s=100, c=abs_err, 
              cmap='RdBu_r', edgecolors='k', linewidth=0.5)
    ax.axhline(0, color='red', linestyle='--', lw=2, label='Zero error')
    ax.set_xlabel('z_theory', fontsize=12)
    ax.set_ylabel('Absolute error (z_meas - z_theory)', fontsize=12)
    ax.set_title('Absolute Error', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Relative error vs z_theory
    ax = axes[0, 2]
    ax.scatter(z_theory, rel_err * 100, alpha=0.7, s=100, c=rel_err, 
              cmap='RdBu_r', edgecolors='k', linewidth=0.5)
    ax.axhline(0, color='red', linestyle='--', lw=2, label='Zero error')
    ax.set_xlabel('z_theory', fontsize=12)
    ax.set_ylabel('Relative error (%)', fontsize=12)
    ax.set_title('Relative Error', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Histogram of absolute errors
    ax = axes[1, 0]
    ax.hist(abs_err, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(np.mean(abs_err), color='red', linestyle='--', lw=2,
               label=f'Mean: {np.mean(abs_err):.2e}')
    ax.axvline(0, color='green', linestyle='-', lw=2, label='Zero error')
    ax.set_xlabel('Absolute error', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Absolute Errors', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 5: Histogram of relative errors
    ax = axes[1, 1]
    ax.hist(rel_err * 100, bins=20, alpha=0.7, color='orange', edgecolor='black')
    ax.axvline(np.mean(rel_err) * 100, color='red', linestyle='--', lw=2,
               label=f'Mean: {np.mean(rel_err)*100:.4e}%')
    ax.axvline(0, color='green', linestyle='-', lw=2, label='Zero error')
    ax.set_xlabel('Relative error (%)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Relative Errors', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 6: Error vs emission scale factor
    ax = axes[1, 2]
    scatter = ax.scatter(a_em, rel_err * 100, alpha=0.7, s=100, 
                        c=rel_err, cmap='RdBu_r', 
                        edgecolors='k', linewidth=0.5)
    ax.axhline(0, color='red', linestyle='--', lw=2, label='Zero error')
    ax.set_xlabel('a_em (emission scale factor)', fontsize=12)
    ax.set_ylabel('Relative error (%)', fontsize=12)
    ax.set_title('Error vs Emission Scale Factor', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax, label='Relative error')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n   Saved validation plot to: {save_path}")
    
    plt.show()


def main():
    """Main validation function."""
    
    print("="*70)
    print("HOMOGENEOUS REDSHIFT VALIDATION")
    print("="*70)
    print("\nComparing measured redshift from ray-traced trajectories")
    print("with theoretical FLRW prediction: 1 + z = a_obs / a_em")
    
    # Input file
    filename = "backward_raytracing_trajectories_OPTIMAL_mass_500_500_500_Mpc.h5"
    
    if not os.path.exists(filename):
        print(f"\nError: File not found: {filename}")
        print("Please run integrate_photons_on_perturbed_flrw_OPTIMAL.py first")
        return
    
    # Load trajectories
    print(f"\n1. Loading trajectories from {filename}...")
    trajectories = load_trajectories(filename)
    print(f"   Loaded {len(trajectories)} photon trajectories")
    
    # Check if quantities are present
    has_quantities = (trajectories[0].shape[1] >= 14)
    if not has_quantities:
        print("\n   ⚠️  Trajectories do not contain quantities!")
        print("      Please re-run integration with quantities enabled")
        return
    
    print(f"   ✓ Trajectories contain pre-computed quantities (shape: {trajectories[0].shape})")
    
    # Validate redshift
    print("\n2. Computing redshift from trajectories and comparing with theory...")
    results = validate_redshift(trajectories)
    
    # Print summary
    print("\n3. Validation summary:")
    is_valid = print_validation_summary(results)
    
    # Plot results
    print("\n4. Creating validation plots...")
    plot_validation(results, save_path="redshift_validation.png")
    
    # Final verdict
    print("\n" + "="*70)
    if is_valid:
        print("VALIDATION PASSED: Homogeneous redshift matches FLRW theory")
    else:
        print("VALIDATION FAILED: Significant deviation from theory detected")
    print("="*70)


if __name__ == "__main__":
    main()
