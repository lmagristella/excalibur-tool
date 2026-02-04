#!/usr/bin/env python3
"""
Statistical visualization of redshift components across all trajectories.

Creates plots with shaded uncertainty bands (mean ± std) to visualize
the statistical distribution of redshift effects across multiple photon paths.
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


def compute_redshift_evolution(trajectories, n_samples=1000):
    """
    Compute redshift evolution for all trajectories on a common grid.
    
    Parameters:
    -----------
    trajectories : list of arrays
        List of trajectory arrays
    n_samples : int
        Number of sample points along each trajectory
    
    Returns:
    --------
    results : dict
        Dictionary with arrays for statistical analysis
    """
    n_photons = len(trajectories)
    
    # Storage for each trajectory
    all_z_H = []
    all_z_SW = []
    all_z_ISW = []
    all_z_total = []
    all_a = []
    all_distance = []
    all_proper_distance = []
    
    print(f"\nComputing redshift evolution for {n_photons} trajectories...")
    
    for i, traj in enumerate(trajectories):
        n_steps = len(traj)
        
        # Sub-sample trajectory if too long
        if n_steps > n_samples:
            indices = np.linspace(0, n_steps-1, n_samples, dtype=int)
            traj_sampled = traj[indices]
        else:
            traj_sampled = traj
            indices = np.arange(n_steps)
        
        n_sampled = len(traj_sampled)
        
        # Arrays for this trajectory
        z_H = np.zeros(n_sampled)
        z_SW = np.zeros(n_sampled)
        z_ISW = np.zeros(n_sampled)
        z_total = np.zeros(n_sampled)
        a = np.zeros(n_sampled)
        distance = np.zeros(n_sampled)
        proper_distance = np.zeros(n_sampled)
        
        # Observer position
        x_obs, y_obs, z_obs = traj[0, 1:4]
        
        # Compute redshift at each point
        for j, idx in enumerate(indices):
            # Create sub-trajectory from observer to this point
            sub_traj = traj[:idx+1]
            
            if len(sub_traj) < 2:
                continue
            
            # Create calculator for this segment
            calc = RedshiftCalculator(sub_traj)
            
            # Compute components
            z_H[j] = calc.compute_homogeneous_redshift()
            z_SW_val, _, _ = calc.compute_sachs_wolfe_redshift()
            z_SW[j] = z_SW_val
            z_ISW_val, _ = calc.compute_integrated_sachs_wolfe_redshift()
            z_ISW[j] = z_ISW_val
            z_total[j] = z_H[j] + z_SW[j] + z_ISW[j]
            
            # Scale factor at this point
            a[j] = calc.a_em
            
            # Distance to this point
            x, y, z = traj_sampled[j, 1:4]
            distance[j] = np.sqrt((x - x_obs)**2 + (y - y_obs)**2 + (z - z_obs)**2)
            proper_distance[j] = distance[j] * calc.a_obs
        
        all_z_H.append(z_H)
        all_z_SW.append(z_SW)
        all_z_ISW.append(z_ISW)
        all_z_total.append(z_total)
        all_a.append(a)
        all_distance.append(distance)
        all_proper_distance.append(proper_distance)
        
        if (i + 1) % 10 == 0:
            print(f"   Progress: {i+1}/{n_photons}")
    
    print("   Done!")
    
    # Convert to arrays
    results = {
        'z_H': np.array(all_z_H),
        'z_SW': np.array(all_z_SW),
        'z_ISW': np.array(all_z_ISW),
        'z_total': np.array(all_z_total),
        'a': np.array(all_a),
        'distance': np.array(all_distance),
        'proper_distance': np.array(all_proper_distance),
        'n_photons': n_photons,
        'n_samples': n_samples
    }
    
    return results


def plot_shaded_statistics(results, save_path=None):
    """
    Create comprehensive shaded statistical plots.
    
    Parameters:
    -----------
    results : dict
        Results from compute_redshift_evolution
    save_path : str, optional
        Path to save figure
    """
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    fig.suptitle('Redshift Statistics Across All Trajectories (Perturbed FLRW)', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    n_photons = results['n_photons']
    n_samples = results['n_samples']
    
    # Color scheme
    color_H = 'blue'
    color_SW = 'green'
    color_ISW = 'purple'
    color_total = 'red'
    
    # =========================================================================
    # Row 1: Redshift vs scale factor
    # =========================================================================
    
    # Panel 1: z_H vs a
    ax = fig.add_subplot(gs[0, 0])
    a_array = results['a']
    z_H_array = results['z_H']
    
    # Plot all individual trajectories in light gray
    for i in range(n_photons):
        ax.plot(a_array[i], z_H_array[i], color='gray', alpha=0.1, lw=0.5)
    
    # Compute statistics on common a grid
    a_min = np.min(a_array[a_array > 0])
    a_max = np.max(a_array)
    a_grid = np.linspace(a_min, a_max, 100)
    
    z_H_mean = np.zeros(len(a_grid))
    z_H_std = np.zeros(len(a_grid))
    
    for j, a_val in enumerate(a_grid):
        # Find closest a value in each trajectory
        z_values = []
        for i in range(n_photons):
            if len(a_array[i][a_array[i] > 0]) > 0:
                idx = np.argmin(np.abs(a_array[i] - a_val))
                if a_array[i][idx] > 0:
                    z_values.append(z_H_array[i][idx])
        
        if len(z_values) > 0:
            z_H_mean[j] = np.mean(z_values)
            z_H_std[j] = np.std(z_values)
    
    # Plot mean with shaded std
    ax.plot(a_grid, z_H_mean, color=color_H, lw=3, label=f'Mean ({n_photons} trajectories)', zorder=10)
    ax.fill_between(a_grid, z_H_mean - z_H_std, z_H_mean + z_H_std, 
                     color=color_H, alpha=0.3, label='±1σ', zorder=5)
    
    ax.set_xlabel('Scale factor a(η)', fontsize=12, fontweight='bold')
    ax.set_ylabel('z_H (homogeneous)', fontsize=12, fontweight='bold')
    ax.set_title('Homogeneous Redshift vs Scale Factor', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # Earlier times (smaller a) on right
    
    # Panel 2: z_SW vs a
    ax = fig.add_subplot(gs[0, 1])
    z_SW_array = results['z_SW']
    
    # Individual trajectories
    for i in range(n_photons):
        ax.plot(a_array[i], z_SW_array[i], color='gray', alpha=0.1, lw=0.5)
    
    # Statistics
    z_SW_mean = np.zeros(len(a_grid))
    z_SW_std = np.zeros(len(a_grid))
    
    for j, a_val in enumerate(a_grid):
        z_values = []
        for i in range(n_photons):
            if len(a_array[i][a_array[i] > 0]) > 0:
                idx = np.argmin(np.abs(a_array[i] - a_val))
                if a_array[i][idx] > 0:
                    z_values.append(z_SW_array[i][idx])
        
        if len(z_values) > 0:
            z_SW_mean[j] = np.mean(z_values)
            z_SW_std[j] = np.std(z_values)
    
    ax.plot(a_grid, z_SW_mean, color=color_SW, lw=3, label=f'Mean', zorder=10)
    ax.fill_between(a_grid, z_SW_mean - z_SW_std, z_SW_mean + z_SW_std, 
                     color=color_SW, alpha=0.3, label='±1σ', zorder=5)
    #ax.axhline(0, color='black', linestyle='--', lw=1, alpha=0.5)
    
    ax.set_xlabel('Scale factor a(η)', fontsize=12, fontweight='bold')
    ax.set_ylabel('z_SW (Sachs-Wolfe)', fontsize=12, fontweight='bold')
    ax.set_title('Sachs-Wolfe Effect vs Scale Factor', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    
    # Panel 3: z_total vs a
    ax = fig.add_subplot(gs[0, 2])
    z_total_array = results['z_total']
    
    # Individual trajectories
    for i in range(n_photons):
        ax.plot(a_array[i], z_total_array[i], color='gray', alpha=0.1, lw=0.5)
    
    # Statistics
    z_total_mean = np.zeros(len(a_grid))
    z_total_std = np.zeros(len(a_grid))
    
    for j, a_val in enumerate(a_grid):
        z_values = []
        for i in range(n_photons):
            if len(a_array[i][a_array[i] > 0]) > 0:
                idx = np.argmin(np.abs(a_array[i] - a_val))
                if a_array[i][idx] > 0:
                    z_values.append(z_total_array[i][idx])
        
        if len(z_values) > 0:
            z_total_mean[j] = np.mean(z_values)
            z_total_std[j] = np.std(z_values)
    
    ax.plot(a_grid, z_total_mean, color=color_total, lw=3, label=f'Mean', zorder=10)
    ax.fill_between(a_grid, z_total_mean - z_total_std, z_total_mean + z_total_std, 
                     color=color_total, alpha=0.3, label='±1σ', zorder=5)
    
    ax.set_xlabel('Scale factor a(η)', fontsize=12, fontweight='bold')
    ax.set_ylabel('z_total', fontsize=12, fontweight='bold')
    ax.set_title('Total Redshift vs Scale Factor', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    
    # =========================================================================
    # Row 2: Redshift vs distance
    # =========================================================================
    
    distance_array = results['distance'] / one_Mpc  # Convert to Mpc
    
    # Panel 4: z_H vs distance
    ax = fig.add_subplot(gs[1, 0])
    
    for i in range(n_photons):
        ax.plot(distance_array[i], z_H_array[i], color='gray', alpha=0.1, lw=0.5)
    
    # Statistics on distance grid
    d_min = np.min(distance_array[distance_array > 0])
    d_max = np.max(distance_array)
    d_grid = np.linspace(d_min, d_max, 100)
    
    z_H_mean_d = np.zeros(len(d_grid))
    z_H_std_d = np.zeros(len(d_grid))
    
    for j, d_val in enumerate(d_grid):
        z_values = []
        for i in range(n_photons):
            if len(distance_array[i][distance_array[i] > 0]) > 0:
                idx = np.argmin(np.abs(distance_array[i] - d_val))
                if distance_array[i][idx] > 0:
                    z_values.append(z_H_array[i][idx])
        
        if len(z_values) > 0:
            z_H_mean_d[j] = np.mean(z_values)
            z_H_std_d[j] = np.std(z_values)
    
    ax.plot(d_grid, z_H_mean_d, color=color_H, lw=3, label='Mean', zorder=10)
    ax.fill_between(d_grid, z_H_mean_d - z_H_std_d, z_H_mean_d + z_H_std_d, 
                     color=color_H, alpha=0.3, label='±1σ', zorder=5)
    
    ax.set_xlabel('Comoving distance (Mpc)', fontsize=12, fontweight='bold')
    ax.set_ylabel('z_H', fontsize=12, fontweight='bold')
    ax.set_title('z_H vs Distance', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Panel 5: z_SW vs distance
    ax = fig.add_subplot(gs[1, 1])
    
    for i in range(n_photons):
        ax.plot(distance_array[i], z_SW_array[i], color='gray', alpha=0.1, lw=0.5)
    
    z_SW_mean_d = np.zeros(len(d_grid))
    z_SW_std_d = np.zeros(len(d_grid))
    
    for j, d_val in enumerate(d_grid):
        z_values = []
        for i in range(n_photons):
            if len(distance_array[i][distance_array[i] > 0]) > 0:
                idx = np.argmin(np.abs(distance_array[i] - d_val))
                if distance_array[i][idx] > 0:
                    z_values.append(z_SW_array[i][idx])
        
        if len(z_values) > 0:
            z_SW_mean_d[j] = np.mean(z_values)
            z_SW_std_d[j] = np.std(z_values)
    
    ax.plot(d_grid, z_SW_mean_d, color=color_SW, lw=3, label='Mean', zorder=10)
    ax.fill_between(d_grid, z_SW_mean_d - z_SW_std_d, z_SW_mean_d + z_SW_std_d, 
                     color=color_SW, alpha=0.3, label='±1σ', zorder=5)
    #ax.axhline(0, color='black', linestyle='--', lw=1, alpha=0.5)
    ax.axvline(np.linalg.norm(np.array([714,714,714])), color='red', linestyle=':', lw=1.5, alpha=0.7,
               label='Halo Center Distance')
    ax.set_xlabel('Comoving distance (Mpc)', fontsize=12, fontweight='bold')
    ax.set_ylabel('z_SW', fontsize=12, fontweight='bold')
    ax.set_title('z_SW vs Distance', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Panel 6: z_total vs distance
    ax = fig.add_subplot(gs[1, 2])
    
    for i in range(n_photons):
        ax.plot(distance_array[i], z_total_array[i], color='gray', alpha=0.1, lw=0.5)
    
    z_total_mean_d = np.zeros(len(d_grid))
    z_total_std_d = np.zeros(len(d_grid))
    
    for j, d_val in enumerate(d_grid):
        z_values = []
        for i in range(n_photons):
            if len(distance_array[i][distance_array[i] > 0]) > 0:
                idx = np.argmin(np.abs(distance_array[i] - d_val))
                if distance_array[i][idx] > 0:
                    z_values.append(z_total_array[i][idx])
        
        if len(z_values) > 0:
            z_total_mean_d[j] = np.mean(z_values)
            z_total_std_d[j] = np.std(z_values)
    
    ax.plot(d_grid, z_total_mean_d, color=color_total, lw=3, label='Mean', zorder=10)
    ax.fill_between(d_grid, z_total_mean_d - z_total_std_d, z_total_mean_d + z_total_std_d, 
                     color=color_total, alpha=0.3, label='±1σ', zorder=5)
    
    ax.set_xlabel('Comoving distance (Mpc)', fontsize=12, fontweight='bold')
    ax.set_ylabel('z_total', fontsize=12, fontweight='bold')
    ax.set_title('z_total vs Distance', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # =========================================================================
    # Row 3: Perturbation effects and final distributions
    # =========================================================================
    
    # Panel 7: Perturbation contribution (z_SW + z_ISW)
    ax = fig.add_subplot(gs[2, 0])
    
    z_pert_array = z_SW_array + results['z_ISW']
    
    for i in range(n_photons):
        ax.plot(distance_array[i], z_pert_array[i], color='gray', alpha=0.1, lw=0.5)
    
    z_pert_mean_d = np.zeros(len(d_grid))
    z_pert_std_d = np.zeros(len(d_grid))
    
    for j, d_val in enumerate(d_grid):
        z_values = []
        for i in range(n_photons):
            if len(distance_array[i][distance_array[i] > 0]) > 0:
                idx = np.argmin(np.abs(distance_array[i] - d_val))
                if distance_array[i][idx] > 0:
                    z_values.append(z_pert_array[i][idx])
        
        if len(z_values) > 0:
            z_pert_mean_d[j] = np.mean(z_values)
            z_pert_std_d[j] = np.std(z_values)
    
    ax.plot(d_grid, z_pert_mean_d, color='orange', lw=3, label='Mean (z_SW + z_ISW)', zorder=10)
    ax.fill_between(d_grid, z_pert_mean_d - z_pert_std_d, z_pert_mean_d + z_pert_std_d, 
                     color='orange', alpha=0.3, label='±1σ', zorder=5)
    #ax.axhline(0, color='black', linestyle='--', lw=1, alpha=0.5)
    
    ax.set_xlabel('Comoving distance (Mpc)', fontsize=12, fontweight='bold')
    ax.set_ylabel('z_SW + z_ISW', fontsize=12, fontweight='bold')
    ax.set_title('Perturbation Effects vs Distance', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Panel 8: Final redshift distribution
    ax = fig.add_subplot(gs[2, 1])
    
    # Get final values (last point of each trajectory)
    z_H_final = [z_H_array[i][-1] for i in range(n_photons) if len(z_H_array[i]) > 0]
    z_SW_final = [z_SW_array[i][-1] for i in range(n_photons) if len(z_SW_array[i]) > 0]
    z_total_final = [z_total_array[i][-1] for i in range(n_photons) if len(z_total_array[i]) > 0]
    
    ax.hist(z_H_final, bins=20, alpha=0.6, color=color_H, label='z_H', density=True, edgecolor='black')
    ax.hist(z_total_final, bins=20, alpha=0.6, color=color_total, label='z_total', density=True, edgecolor='black')
    
    ax.axvline(np.mean(z_H_final), color=color_H, linestyle='--', lw=2, 
               label=f'Mean z_H: {np.mean(z_H_final):.6f}')
    ax.axvline(np.mean(z_total_final), color=color_total, linestyle='--', lw=2,
               label=f'Mean z_tot: {np.mean(z_total_final):.6f}')
    
    ax.set_xlabel('Redshift', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probability density', fontsize=12, fontweight='bold')
    ax.set_title('Final Redshift Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 9: Statistics summary
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')
    
    # Compute statistics
    stats_text = "STATISTICAL SUMMARY\n" + "="*40 + "\n\n"
    stats_text += f"Number of trajectories: {n_photons}\n"
    stats_text += f"Sample points per traj: {n_samples}\n\n"
    
    stats_text += "Final values (at emission):\n"
    stats_text += f"  z_H:     {np.mean(z_H_final):.6f} ± {np.std(z_H_final):.2e}\n"
    stats_text += f"  z_SW:    {np.mean(z_SW_final):.6e} ± {np.std(z_SW_final):.2e}\n"
    stats_text += f"  z_total: {np.mean(z_total_final):.6f} ± {np.std(z_total_final):.2e}\n\n"
    
    stats_text += "Relative perturbation effect:\n"
    rel_effect = (np.mean(z_total_final) - np.mean(z_H_final)) / np.mean(z_H_final) * 100
    stats_text += f"  (z_tot - z_H) / z_H: {rel_effect:.4f}%\n\n"
    
    stats_text += "Distance statistics:\n"
    d_final = [distance_array[i][-1] for i in range(n_photons) if len(distance_array[i]) > 0]
    stats_text += f"  Mean: {np.mean(d_final):.2f} Mpc\n"
    stats_text += f"  Std:  {np.std(d_final):.2f} Mpc\n"
    stats_text += f"  Range: [{np.min(d_final):.2f}, {np.max(d_final):.2f}] Mpc\n"
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved plot to: {save_path}")
    
    plt.show()


def main():
    """Main function."""
    
    print("="*70)
    print("REDSHIFT STATISTICS WITH SHADED UNCERTAINTY BANDS")
    print("="*70)
    
    # Input file
    filename = "/home/magri/_data/output/excalibur_run_perturbed_flrw_M1.0e15_R5.0_mass714_714_714_obs3_3_3_N91_sequential_S5119_Mpc.h5"
    
    if not os.path.exists(filename):
        print(f"\n❌ Error: File not found: {filename}")
        return
    
    # Ensure output directory exists
    output_dir = "_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load trajectories
    print(f"\n1. Loading trajectories from {filename}...")
    trajectories = load_trajectories(filename)
    print(f"   Loaded {len(trajectories)} trajectories")
    
    # Compute evolution
    print("\n2. Computing redshift evolution along trajectories...")
    results = compute_redshift_evolution(trajectories, n_samples=5000)
    
    # Plot
    print("\n3. Creating shaded statistical plots...")
    output_path = os.path.join(output_dir, "redshift_statistics_shaded.png")
    plot_shaded_statistics(results, save_path=output_path)
    
    print("\n" + "="*70)
    print("✅ DONE!")
    print("="*70)


if __name__ == "__main__":
    main()
