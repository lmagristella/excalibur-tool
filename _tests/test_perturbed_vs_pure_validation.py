#!/usr/bin/env python3
"""
Complete validation comparing perturbed vs pure FLRW trajectories.

This script loads trajectories from both:
1. Perturbed FLRW simulation (with gravitational potential Phi)
2. Pure FLRW simulation (Phi = 0)

And performs a comprehensive comparison of:
- Redshift components (z_H, z_SW, z_ISW, z_total)
- Scale factors and emission times
- Spatial trajectories
- Statistical distributions

This validates that:
- Pure FLRW: z_total = z_H, z_SW = 0, z_ISW = 0
- Perturbed: z_total = z_H + z_SW + z_ISW ≠ z_H
- Differences are physical (due to Phi) not numerical errors
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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


def compute_redshift_components(trajectories):
    """Compute all redshift components for all trajectories."""
    
    n_photons = len(trajectories)
    
    results = {
        'z_H': np.zeros(n_photons),
        'z_SW': np.zeros(n_photons),
        'z_ISW': np.zeros(n_photons),
        'z_total': np.zeros(n_photons),
        'a_obs': np.zeros(n_photons),
        'a_em': np.zeros(n_photons),
        'eta_obs': np.zeros(n_photons),
        'eta_em': np.zeros(n_photons),
        'distance': np.zeros(n_photons),
        'proper_distance': np.zeros(n_photons)
    }
    
    for i, traj in enumerate(trajectories):
        calc = RedshiftCalculator(traj)
        
        # Compute components
        results['z_H'][i] = calc.compute_homogeneous_redshift()
        
        # compute_sachs_wolfe_redshift returns (z_SW, Phi_obs, Phi_em)
        z_SW, _, _ = calc.compute_sachs_wolfe_redshift()
        results['z_SW'][i] = z_SW
        
        # compute_integrated_sachs_wolfe_redshift returns (z_ISW, dPhi_deta_array)
        z_ISW, _ = calc.compute_integrated_sachs_wolfe_redshift()
        results['z_ISW'][i] = z_ISW
        
        # Total redshift = sum of components (linear approximation)
        results['z_total'][i] = results['z_H'][i] + z_SW + z_ISW
        
        # Store parameters
        results['a_obs'][i] = calc.a_obs
        results['a_em'][i] = calc.a_em
        results['eta_obs'][i] = calc.eta_obs
        results['eta_em'][i] = calc.eta_em
        
        # Compute distances
        x_em, y_em, z_em = traj[-1, 1:4]
        x_obs, y_obs, z_obs = traj[0, 1:4]
        results['distance'][i] = np.sqrt((x_obs - x_em)**2 + (y_obs - y_em)**2 + (z_obs - z_em)**2)
        results['proper_distance'][i] = results['distance'][i] * calc.a_obs
    
    return results


def print_comparison_summary(results_perturbed, results_pure):
    """Print detailed comparison summary."""
    
    print("\n" + "="*80)
    print("PERTURBED vs PURE FLRW - COMPREHENSIVE COMPARISON")
    print("="*80)
    
    n = len(results_perturbed['z_total'])
    
    # Scale factors comparison
    print("\n1. SCALE FACTORS")
    print("-" * 80)
    print(f"{'Parameter':<25} {'Perturbed':<25} {'Pure FLRW':<25}")
    print("-" * 80)
    
    for key in ['a_obs', 'a_em']:
        p_mean = np.mean(results_perturbed[key])
        p_std = np.std(results_perturbed[key])
        f_mean = np.mean(results_pure[key])
        f_std = np.std(results_pure[key])
        
        print(f"{key:<25} {p_mean:.8f} ± {p_std:.2e}   {f_mean:.8f} ± {f_std:.2e}")
    
    # Conformal times
    print("\n2. CONFORMAL TIMES")
    print("-" * 80)
    
    for key in ['eta_obs', 'eta_em']:
        p_mean = np.mean(results_perturbed[key])
        f_mean = np.mean(results_pure[key])
        
        print(f"{key:<25} {p_mean:.6e} s        {f_mean:.6e} s")
    
    # Redshift components - THE KEY COMPARISON
    print("\n3. REDSHIFT COMPONENTS (THE KEY TEST)")
    print("="*80)
    
    components = ['z_H', 'z_SW', 'z_ISW', 'z_total']
    
    print(f"{'Component':<15} {'Perturbed':<30} {'Pure FLRW':<30} {'Difference'}")
    print("-" * 80)
    
    for comp in components:
        p_mean = np.mean(results_perturbed[comp])
        p_std = np.std(results_perturbed[comp])
        f_mean = np.mean(results_pure[comp])
        f_std = np.std(results_pure[comp])
        diff = p_mean - f_mean
        
        print(f"{comp:<15} {p_mean:+.10f} ± {p_std:.2e}   {f_mean:+.10f} ± {f_std:.2e}   {diff:+.2e}")
    
    # Physical interpretation
    print("\n4. PHYSICAL INTERPRETATION")
    print("="*80)
    
    # Check pure FLRW expectations
    pure_SW_rms = np.sqrt(np.mean(results_pure['z_SW']**2))
    pure_ISW_rms = np.sqrt(np.mean(results_pure['z_ISW']**2))
    pure_diff_total_H = np.mean(np.abs(results_pure['z_total'] - results_pure['z_H']))
    
    print("\nPure FLRW expectations (should all be ~ 0):")
    print(f"  z_SW (RMS):           {pure_SW_rms:.2e}  {'✅' if pure_SW_rms < 1e-10 else '❌'}")
    print(f"  z_ISW (RMS):          {pure_ISW_rms:.2e}  {'✅' if pure_ISW_rms < 1e-10 else '❌'}")
    print(f"  |z_total - z_H|:      {pure_diff_total_H:.2e}  {'✅' if pure_diff_total_H < 1e-10 else '❌'}")
    
    # Check perturbed deviations
    pert_SW_rms = np.sqrt(np.mean(results_perturbed['z_SW']**2))
    pert_ISW_rms = np.sqrt(np.mean(results_perturbed['z_ISW']**2))
    pert_diff_total_H = np.mean(np.abs(results_perturbed['z_total'] - results_perturbed['z_H']))
    
    print("\nPerturbed FLRW effects (should be NON-ZERO):")
    print(f"  z_SW (RMS):           {pert_SW_rms:.2e}  {'✅' if pert_SW_rms > 1e-10 else '❌'}")
    print(f"  z_ISW (RMS):          {pert_ISW_rms:.2e}  {'✅' if pert_ISW_rms > 1e-10 else '❌'}")
    print(f"  |z_total - z_H|:      {pert_diff_total_H:.2e}  {'✅' if pert_diff_total_H > 1e-10 else '❌'}")
    
    # Perturbation effects
    print("\n5. PERTURBATION EFFECTS (Perturbed - Pure)")
    print("="*80)
    
    delta_z_H = np.mean(results_perturbed['z_H']) - np.mean(results_pure['z_H'])
    delta_z_SW = np.mean(results_perturbed['z_SW']) - np.mean(results_pure['z_SW'])
    delta_z_ISW = np.mean(results_perturbed['z_ISW']) - np.mean(results_pure['z_ISW'])
    delta_z_total = np.mean(results_perturbed['z_total']) - np.mean(results_pure['z_total'])
    
    print(f"  Δ<z_H>:       {delta_z_H:+.2e}")
    print(f"  Δ<z_SW>:      {delta_z_SW:+.2e}  (should equal perturbed z_SW)")
    print(f"  Δ<z_ISW>:     {delta_z_ISW:+.2e}  (should equal perturbed z_ISW)")
    print(f"  Δ<z_total>:   {delta_z_total:+.2e}  (net effect)")
    
    # Relative effects
    mean_z = np.mean(results_pure['z_H'])
    print(f"\nRelative to <z_H> = {mean_z:.4f}:")
    print(f"  Δz_H / z_H:       {delta_z_H / mean_z * 100:.4f}%")
    print(f"  Δz_SW / z_H:      {delta_z_SW / mean_z * 100:.4f}%")
    print(f"  Δz_ISW / z_H:     {delta_z_ISW / mean_z * 100:.4f}%")
    print(f"  Δz_total / z_H:   {delta_z_total / mean_z * 100:.4f}%")
    
    # Distances
    print("\n6. DISTANCES")
    print("="*80)
    
    print(f"{'Distance type':<25} {'Perturbed':<25} {'Pure FLRW':<25}")
    print("-" * 80)
    
    for key in ['distance', 'proper_distance']:
        p_mean = np.mean(results_perturbed[key])
        f_mean = np.mean(results_pure[key])
        
        if 'proper' in key:
            print(f"{key:<25} {p_mean/one_Mpc:.2f} Mpc             {f_mean/one_Mpc:.2f} Mpc")
        else:
            print(f"{key:<25} {p_mean/one_Mpc:.2f} Mpc             {f_mean/one_Mpc:.2f} Mpc")
    
    # Validation verdict
    print("\n" + "="*80)
    print("VALIDATION VERDICT")
    print("="*80)
    
    all_checks_pass = True
    
    # Check 1: Pure FLRW has no SW/ISW
    check1 = (pure_SW_rms < 1e-10) and (pure_ISW_rms < 1e-10) and (pure_diff_total_H < 1e-10)
    print(f"\n✓ Check 1: Pure FLRW has z_SW=0, z_ISW=0, z_total=z_H ... {'✅ PASS' if check1 else '❌ FAIL'}")
    all_checks_pass = all_checks_pass and check1
    
    # Check 2: Perturbed has non-zero SW/ISW
    check2 = (pert_SW_rms > 1e-10) and (pert_ISW_rms > 1e-10)
    print(f"✓ Check 2: Perturbed has non-zero z_SW, z_ISW ... {'✅ PASS' if check2 else '❌ FAIL'}")
    all_checks_pass = all_checks_pass and check2
    
    # Check 3: z_H is approximately same (small variations from Phi affecting a_em)
    check3 = np.abs(delta_z_H / mean_z) < 0.01  # Within 1%
    print(f"✓ Check 3: z_H similar in both (<1% difference) ... {'✅ PASS' if check3 else '❌ FAIL'}")
    all_checks_pass = all_checks_pass and check3
    
    # Check 4: Perturbations introduce measurable effect
    check4 = np.abs(delta_z_total) > 1e-8
    print(f"✓ Check 4: Total redshift differs (Δz > 10⁻⁸) ... {'✅ PASS' if check4 else '❌ FAIL'}")
    all_checks_pass = all_checks_pass and check4
    
    print("\n" + "="*80)
    if all_checks_pass:
        print("🎉 ALL VALIDATION CHECKS PASSED!")
        print("   The code correctly distinguishes perturbed from pure FLRW.")
    else:
        print("⚠️  SOME VALIDATION CHECKS FAILED")
        print("   Review the results above for discrepancies.")
    print("="*80)
    
    return all_checks_pass


def plot_comprehensive_comparison(results_perturbed, results_pure, save_path=None):
    """Create comprehensive comparison plots."""
    
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Perturbed vs Pure FLRW - Comprehensive Validation', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Color scheme
    color_pert = 'red'
    color_pure = 'blue'
    
    # =========================================================================
    # Row 1: Redshift components scatter plots
    # =========================================================================
    
    # Panel 1: z_H comparison
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(results_pure['z_H'], results_perturbed['z_H'], 
              alpha=0.7, s=80, c=color_pert, edgecolors='k', linewidth=0.5, label='Data')
    
    # Perfect agreement line
    z_range = [min(np.min(results_pure['z_H']), np.min(results_perturbed['z_H'])),
               max(np.max(results_pure['z_H']), np.max(results_perturbed['z_H']))]
    ax.plot(z_range, z_range, 'k--', lw=2, label='1:1 line')
    
    ax.set_xlabel('z_H (Pure FLRW)', fontsize=11)
    ax.set_ylabel('z_H (Perturbed)', fontsize=11)
    ax.set_title('Homogeneous Redshift', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Panel 2: z_SW comparison (should be 0 for pure)
    ax = fig.add_subplot(gs[0, 1])
    x_pos = np.arange(2)
    z_SW_means = [np.mean(results_pure['z_SW']), np.mean(results_perturbed['z_SW'])]
    z_SW_stds = [np.std(results_pure['z_SW']), np.std(results_perturbed['z_SW'])]
    
    ax.bar(x_pos, z_SW_means, yerr=z_SW_stds, 
          color=[color_pure, color_pert], alpha=0.7, 
          edgecolor='black', linewidth=1.5, capsize=5)
    ax.axhline(0, color='black', linestyle='--', lw=2)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Pure FLRW', 'Perturbed'], fontsize=10)
    ax.set_ylabel('z_SW', fontsize=11)
    ax.set_title('Sachs-Wolfe Effect', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 3: z_ISW comparison (should be 0 for pure)
    ax = fig.add_subplot(gs[0, 2])
    z_ISW_means = [np.mean(results_pure['z_ISW']), np.mean(results_perturbed['z_ISW'])]
    z_ISW_stds = [np.std(results_pure['z_ISW']), np.std(results_perturbed['z_ISW'])]
    
    ax.bar(x_pos, z_ISW_means, yerr=z_ISW_stds, 
          color=[color_pure, color_pert], alpha=0.7, 
          edgecolor='black', linewidth=1.5, capsize=5)
    ax.axhline(0, color='black', linestyle='--', lw=2)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Pure FLRW', 'Perturbed'], fontsize=10)
    ax.set_ylabel('z_ISW', fontsize=11)
    ax.set_title('Integrated SW Effect', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 4: z_total comparison
    ax = fig.add_subplot(gs[0, 3])
    z_total_means = [np.mean(results_pure['z_total']), np.mean(results_perturbed['z_total'])]
    z_total_stds = [np.std(results_pure['z_total']), np.std(results_perturbed['z_total'])]
    
    ax.bar(x_pos, z_total_means, yerr=z_total_stds, 
          color=[color_pure, color_pert], alpha=0.7, 
          edgecolor='black', linewidth=1.5, capsize=5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Pure FLRW', 'Perturbed'], fontsize=10)
    ax.set_ylabel('z_total', fontsize=11)
    ax.set_title('Total Redshift', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # =========================================================================
    # Row 2: Distributions
    # =========================================================================
    
    # Panel 5: z_H distributions
    ax = fig.add_subplot(gs[1, 0])
    ax.hist(results_pure['z_H'], bins=15, alpha=0.6, color=color_pure, 
           edgecolor='black', label='Pure FLRW', density=True)
    ax.hist(results_perturbed['z_H'], bins=15, alpha=0.6, color=color_pert, 
           edgecolor='black', label='Perturbed', density=True)
    ax.set_xlabel('z_H', fontsize=11)
    ax.set_ylabel('Probability density', fontsize=11)
    ax.set_title('z_H Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 6: z_SW distributions
    ax = fig.add_subplot(gs[1, 1])
    ax.hist(results_pure['z_SW'], bins=15, alpha=0.6, color=color_pure, 
           edgecolor='black', label='Pure FLRW', density=True)
    ax.hist(results_perturbed['z_SW'], bins=15, alpha=0.6, color=color_pert, 
           edgecolor='black', label='Perturbed', density=True)
    ax.axvline(0, color='black', linestyle='--', lw=2)
    ax.set_xlabel('z_SW', fontsize=11)
    ax.set_ylabel('Probability density', fontsize=11)
    ax.set_title('z_SW Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 7: z_ISW distributions
    ax = fig.add_subplot(gs[1, 2])
    ax.hist(results_pure['z_ISW'], bins=15, alpha=0.6, color=color_pure, 
           edgecolor='black', label='Pure FLRW', density=True)
    ax.hist(results_perturbed['z_ISW'], bins=15, alpha=0.6, color=color_pert, 
           edgecolor='black', label='Perturbed', density=True)
    ax.axvline(0, color='black', linestyle='--', lw=2)
    ax.set_xlabel('z_ISW', fontsize=11)
    ax.set_ylabel('Probability density', fontsize=11)
    ax.set_title('z_ISW Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 8: z_total distributions
    ax = fig.add_subplot(gs[1, 3])
    ax.hist(results_pure['z_total'], bins=15, alpha=0.6, color=color_pure, 
           edgecolor='black', label='Pure FLRW', density=True)
    ax.hist(results_perturbed['z_total'], bins=15, alpha=0.6, color=color_pert, 
           edgecolor='black', label='Perturbed', density=True)
    ax.set_xlabel('z_total', fontsize=11)
    ax.set_ylabel('Probability density', fontsize=11)
    ax.set_title('z_total Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # =========================================================================
    # Row 3: Advanced analysis
    # =========================================================================
    
    # Panel 9: Difference z_total - z_H
    ax = fig.add_subplot(gs[2, 0])
    diff_pure = results_pure['z_total'] - results_pure['z_H']
    diff_pert = results_perturbed['z_total'] - results_perturbed['z_H']
    
    ax.scatter(np.arange(len(diff_pure)), diff_pure, 
              alpha=0.7, s=80, c=color_pure, label='Pure FLRW', 
              edgecolors='k', linewidth=0.5, marker='o')
    ax.scatter(np.arange(len(diff_pert)), diff_pert, 
              alpha=0.7, s=80, c=color_pert, label='Perturbed', 
              edgecolors='k', linewidth=0.5, marker='s')
    ax.axhline(0, color='black', linestyle='--', lw=2)
    ax.set_xlabel('Photon index', fontsize=11)
    ax.set_ylabel('z_total - z_H', fontsize=11)
    ax.set_title('Total vs Homogeneous', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Panel 10: Component contributions (perturbed only)
    ax = fig.add_subplot(gs[2, 1])
    components_mean = [
        np.mean(results_perturbed['z_H']),
        np.mean(results_perturbed['z_SW']),
        np.mean(results_perturbed['z_ISW'])
    ]
    components_std = [
        np.std(results_perturbed['z_H']),
        np.std(results_perturbed['z_SW']),
        np.std(results_perturbed['z_ISW'])
    ]
    
    x_pos = np.arange(3)
    bars = ax.bar(x_pos, components_mean, yerr=components_std, 
                 color=['orange', 'green', 'purple'], alpha=0.7, 
                 edgecolor='black', linewidth=1.5, capsize=5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['z_H', 'z_SW', 'z_ISW'], fontsize=10)
    ax.set_ylabel('Redshift', fontsize=11)
    ax.set_title('Components (Perturbed)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 11: Scale factor at emission
    ax = fig.add_subplot(gs[2, 2])
    ax.scatter(results_pure['a_em'], results_pure['z_H'], 
              alpha=0.7, s=80, c=color_pure, label='Pure FLRW', 
              edgecolors='k', linewidth=0.5, marker='o')
    ax.scatter(results_perturbed['a_em'], results_perturbed['z_H'], 
              alpha=0.7, s=80, c=color_pert, label='Perturbed', 
              edgecolors='k', linewidth=0.5, marker='s')
    ax.set_xlabel('a_em (emission scale factor)', fontsize=11)
    ax.set_ylabel('z_H', fontsize=11)
    ax.set_title('z_H vs Scale Factor', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Panel 12: Relative effect of perturbations
    ax = fig.add_subplot(gs[2, 3])
    rel_effect = (results_perturbed['z_total'] - results_pure['z_total']) / results_pure['z_total'] * 100
    
    ax.hist(rel_effect, bins=20, alpha=0.7, color='darkgreen', 
           edgecolor='black', linewidth=1.5)
    ax.axvline(np.mean(rel_effect), color='red', linestyle='--', lw=2,
               label=f'Mean: {np.mean(rel_effect):.4f}%')
    ax.axvline(0, color='black', linestyle='-', lw=1)
    ax.set_xlabel('(z_pert - z_pure) / z_pure (%)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Relative Perturbation Effect', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n   Saved comparison plot to: {save_path}")
    
    plt.show()


def main():
    """Main validation function."""
    
    print("="*80)
    print("COMPREHENSIVE VALIDATION: PERTURBED vs PURE FLRW")
    print("="*80)
    print("\nThis script compares trajectories with and without perturbations")
    print("to validate the physical effects of the gravitational potential.")
    
    # Input files
    file_perturbed = "backward_raytracing_trajectories_OPTIMAL_mass_500_500_500_Mpc.h5"
    file_pure = "backward_raytracing_trajectories_FLRW_pure.h5"
    
    # Check files exist
    if not os.path.exists(file_perturbed):
        print(f"\n❌ Error: Perturbed file not found: {file_perturbed}")
        print("   Please run integrate_photons_on_perturbed_flrw_OPTIMAL.py first")
        return
    
    if not os.path.exists(file_pure):
        print(f"\n❌ Error: Pure FLRW file not found: {file_pure}")
        print("   Please run integrate_photons_FLRW_pure.py first")
        return
    
    # Load trajectories
    print(f"\n1. Loading perturbed trajectories from {file_perturbed}...")
    trajectories_perturbed = load_trajectories(file_perturbed)
    print(f"   Loaded {len(trajectories_perturbed)} trajectories")
    
    print(f"\n2. Loading pure FLRW trajectories from {file_pure}...")
    trajectories_pure = load_trajectories(file_pure)
    print(f"   Loaded {len(trajectories_pure)} trajectories")
    
    # Check consistency
    if len(trajectories_perturbed) != len(trajectories_pure):
        print(f"\n⚠️  Warning: Different number of photons ({len(trajectories_perturbed)} vs {len(trajectories_pure)})")
        print(f"   Will compare only the first {min(len(trajectories_perturbed), len(trajectories_pure))} photons")
        
        # Truncate to same length
        n_compare = min(len(trajectories_perturbed), len(trajectories_pure))
        trajectories_perturbed = trajectories_perturbed[:n_compare]
        trajectories_pure = trajectories_pure[:n_compare]
    
    # Compute redshift components
    print("\n3. Computing redshift components for perturbed trajectories...")
    results_perturbed = compute_redshift_components(trajectories_perturbed)
    
    print("\n4. Computing redshift components for pure FLRW trajectories...")
    results_pure = compute_redshift_components(trajectories_pure)
    
    # Print comparison
    print("\n5. Comparison analysis...")
    is_valid = print_comparison_summary(results_perturbed, results_pure)
    
    # Plot results
    print("\n6. Creating comparison plots...")
    plot_comprehensive_comparison(results_perturbed, results_pure, 
                                  save_path="validation_perturbed_vs_pure.png")
    
    # Final message
    print("\n" + "="*80)
    if is_valid:
        print("✅ VALIDATION COMPLETE: Results are physically consistent")
    else:
        print("⚠️  VALIDATION CONCERNS: Review results carefully")
    print("="*80)


if __name__ == "__main__":
    main()
