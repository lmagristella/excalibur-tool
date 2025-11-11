#!/usr/bin/env python3
"""
Redshift visualization tools.

Provides plotting functions for analyzing redshift evolution along photon
trajectories in perturbed FLRW cosmology.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from excalibur.core.constants import c, one_Mpc


def plot_redshift_evolution(calculator, save_path=None, show=True, n_samples=100):
    """
    Plot redshift evolution along a single photon trajectory.
    
    Creates a multi-panel figure showing how different redshift components
    evolve as a function of:
    - Scale factor a
    - Conformal time η
    - Distance from observer
    - Proper distance traveled
    
    Parameters:
    -----------
    calculator : RedshiftCalculator
        Calculator with trajectory data
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display the figure
    n_samples : int
        Number of points to sample along trajectory (default 100)
    """
    traj = calculator.trajectory
    n_steps = len(traj)
    
    # Subsample trajectory for efficiency
    if n_steps > n_samples:
        indices = np.linspace(0, n_steps-1, n_samples, dtype=int)
        traj_sampled = traj[indices]
    else:
        traj_sampled = traj
        indices = np.arange(n_steps)
    
    n_sampled = len(traj_sampled)
    
    n_sampled = len(traj_sampled)
    
    # Extract trajectory data
    eta_array = traj_sampled[:, 0]
    positions = traj_sampled[:, 1:4]
    
    # Compute derived quantities
    if calculator.has_quantities:
        a_array = traj_sampled[:, 8]
    else:
        a_array = np.array([calculator.a_of_eta(eta) for eta in eta_array])
    
    # Distances
    observer_pos = positions[0]
    distances_from_obs = np.linalg.norm(positions - observer_pos, axis=1)
    
    # Proper distance traveled (cumulative)
    segment_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    proper_distance = np.concatenate([[0], np.cumsum(segment_lengths)])
    
    # Compute redshift at each point along trajectory
    z_H_array = np.zeros(n_sampled)
    z_SW_array = np.zeros(n_sampled)
    z_ISW_array = np.zeros(n_sampled)
    
    for i in range(n_sampled):
        # Create sub-trajectory from observer to this point
        sub_traj = traj[:indices[i]+1]
        if len(sub_traj) < 2:
            continue
        
        # Create temporary calculator for this sub-trajectory
        from excalibur.observables.redshift import RedshiftCalculator
        temp_calc = RedshiftCalculator(
            sub_traj,
            a_of_eta=calculator.a_of_eta if not calculator.has_quantities else None,
            potential_func=calculator.potential_func if not calculator.has_quantities else None
        )
        
        # Compute components
        z_H_array[i] = temp_calc.compute_homogeneous_redshift()
        z_SW, _, _ = temp_calc.compute_sachs_wolfe_redshift()
        z_SW_array[i] = z_SW
        z_ISW, _ = temp_calc.compute_integrated_sachs_wolfe_redshift()
        z_ISW_array[i] = z_ISW
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Convert to physical units
    distances_Mpc = distances_from_obs / one_Mpc
    proper_distance_Mpc = proper_distance / one_Mpc
    
    # Panel 1: Redshift vs scale factor
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(a_array, z_H_array, label='z_H (expansion)', lw=2)
    ax1.plot(a_array, z_SW_array, label='z_SW (Sachs-Wolfe)', lw=2, alpha=0.8)
    ax1.plot(a_array, z_ISW_array, label='z_ISW (Integrated)', lw=2, alpha=0.8)
    ax1.plot(a_array, z_H_array + z_SW_array + z_ISW_array, 
             label='z_total', lw=2, ls='--', color='k')
    ax1.set_xlabel('Scale factor a', fontsize=12)
    ax1.set_ylabel('Redshift z', fontsize=12)
    ax1.set_title('Redshift vs Scale Factor', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Redshift vs conformal time
    ax2 = fig.add_subplot(gs[0, 1])
    eta_Gyr = eta_array / (1e9 * 365.25 * 24 * 3600)  # Convert to Gyr
    ax2.plot(eta_Gyr, z_H_array, label='z_H', lw=2)
    ax2.plot(eta_Gyr, z_SW_array, label='z_SW', lw=2, alpha=0.8)
    ax2.plot(eta_Gyr, z_ISW_array, label='z_ISW', lw=2, alpha=0.8)
    ax2.plot(eta_Gyr, z_H_array + z_SW_array + z_ISW_array, 
             label='z_total', lw=2, ls='--', color='k')
    ax2.set_xlabel('Conformal time η (Gyr)', fontsize=12)
    ax2.set_ylabel('Redshift z', fontsize=12)
    ax2.set_title('Redshift vs Conformal Time', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Redshift vs distance from observer
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(distances_Mpc, z_H_array, label='z_H', lw=2)
    ax3.plot(distances_Mpc, z_SW_array, label='z_SW', lw=2, alpha=0.8)
    ax3.plot(distances_Mpc, z_ISW_array, label='z_ISW', lw=2, alpha=0.8)
    ax3.plot(distances_Mpc, z_H_array + z_SW_array + z_ISW_array, 
             label='z_total', lw=2, ls='--', color='k')
    ax3.set_xlabel('Distance from observer (Mpc)', fontsize=12)
    ax3.set_ylabel('Redshift z', fontsize=12)
    ax3.set_title('Redshift vs Distance from Observer', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Redshift vs proper distance traveled
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(proper_distance_Mpc, z_H_array, label='z_H', lw=2)
    ax4.plot(proper_distance_Mpc, z_SW_array, label='z_SW', lw=2, alpha=0.8)
    ax4.plot(proper_distance_Mpc, z_ISW_array, label='z_ISW', lw=2, alpha=0.8)
    ax4.plot(proper_distance_Mpc, z_H_array + z_SW_array + z_ISW_array, 
             label='z_total', lw=2, ls='--', color='k')
    ax4.set_xlabel('Proper distance traveled (Mpc)', fontsize=12)
    ax4.set_ylabel('Redshift z', fontsize=12)
    ax4.set_title('Redshift vs Proper Distance', fontsize=14, fontweight='bold')
    ax4.legend(loc='best', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # Panel 5: Scale factor evolution
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(eta_Gyr, a_array, lw=2, color='purple')
    ax5.set_xlabel('Conformal time η (Gyr)', fontsize=12)
    ax5.set_ylabel('Scale factor a', fontsize=12)
    ax5.set_title('Scale Factor Evolution', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Panel 6: Trajectory in space
    ax6 = fig.add_subplot(gs[2, 1], projection='3d')
    ax6.plot(positions[:, 0]/one_Mpc, positions[:, 1]/one_Mpc, 
             positions[:, 2]/one_Mpc, lw=2, alpha=0.7)
    ax6.scatter(positions[0, 0]/one_Mpc, positions[0, 1]/one_Mpc, 
                positions[0, 2]/one_Mpc, c='green', s=100, marker='o', 
                label='Observer', edgecolors='k', linewidth=2)
    ax6.scatter(positions[-1, 0]/one_Mpc, positions[-1, 1]/one_Mpc, 
                positions[-1, 2]/one_Mpc, c='red', s=100, marker='*', 
                label='Source', edgecolors='k', linewidth=2)
    ax6.set_xlabel('X (Mpc)', fontsize=10)
    ax6.set_ylabel('Y (Mpc)', fontsize=10)
    ax6.set_zlabel('Z (Mpc)', fontsize=10)
    ax6.set_title('3D Trajectory', fontsize=14, fontweight='bold')
    ax6.legend(loc='best', fontsize=10)
    
    plt.suptitle('Redshift Evolution Along Photon Trajectory', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   Saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_redshift_statistics(calculators, save_path=None, show=True):
    """
    Plot statistical analysis of redshift for multiple trajectories.
    
    Creates visualizations showing:
    - Mean redshift evolution with error bands
    - Redshift distribution at final point
    - Component contributions
    - Scatter plots
    
    Parameters:
    -----------
    calculators : list of RedshiftCalculator
        List of calculators for different photon trajectories
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display the figure
    """
    n_photons = len(calculators)
    
    # Find common scale factor range
    a_min = min([calc.a_em for calc in calculators])
    a_max = max([calc.a_obs for calc in calculators])
    a_common = np.linspace(a_min, a_max, 200)
    
    # Arrays to store interpolated values
    z_H_interp = np.zeros((n_photons, len(a_common)))
    z_SW_interp = np.zeros((n_photons, len(a_common)))
    z_ISW_interp = np.zeros((n_photons, len(a_common)))
    z_total_interp = np.zeros((n_photons, len(a_common)))
    
    # Final redshifts
    z_H_final = np.zeros(n_photons)
    z_SW_final = np.zeros(n_photons)
    z_ISW_final = np.zeros(n_photons)
    z_total_final = np.zeros(n_photons)
    
    print(f"Computing redshift evolution for {n_photons} trajectories...")
    
    for i, calc in enumerate(calculators):
        traj = calc.trajectory
        n_steps = len(traj)
        
        # Get scale factors
        if calc.has_quantities:
            a_array = calc.a_array
        else:
            eta_array = traj[:, 0]
            a_array = np.array([calc.a_of_eta(eta) for eta in eta_array])
        
        # Compute redshift at each point
        z_H_traj = np.zeros(n_steps)
        z_SW_traj = np.zeros(n_steps)
        z_ISW_traj = np.zeros(n_steps)
        
        for j in range(n_steps):
            sub_traj = traj[:j+1]
            if len(sub_traj) < 2:
                continue
            
            from excalibur.observables.redshift import RedshiftCalculator
            temp_calc = RedshiftCalculator(
                sub_traj,
                a_of_eta=calc.a_of_eta if not calc.has_quantities else None,
                potential_func=calc.potential_func if not calc.has_quantities else None
            )
            
            z_H_traj[j] = temp_calc.compute_homogeneous_redshift()
            z_SW, _, _ = temp_calc.compute_sachs_wolfe_redshift()
            z_SW_traj[j] = z_SW
            z_ISW, _ = temp_calc.compute_integrated_sachs_wolfe_redshift()
            z_ISW_traj[j] = z_ISW
        
        z_total_traj = z_H_traj + z_SW_traj + z_ISW_traj
        
        # Interpolate to common scale factor grid
        z_H_interp[i] = np.interp(a_common, a_array, z_H_traj)
        z_SW_interp[i] = np.interp(a_common, a_array, z_SW_traj)
        z_ISW_interp[i] = np.interp(a_common, a_array, z_ISW_traj)
        z_total_interp[i] = np.interp(a_common, a_array, z_total_traj)
        
        # Final values
        z_H_final[i] = z_H_traj[-1]
        z_SW_final[i] = z_SW_traj[-1]
        z_ISW_final[i] = z_ISW_traj[-1]
        z_total_final[i] = z_total_traj[-1]
        
        if (i + 1) % 10 == 0:
            print(f"   Progress: {i+1}/{n_photons}")
    
    # Compute statistics
    z_H_mean = np.mean(z_H_interp, axis=0)
    z_H_std = np.std(z_H_interp, axis=0)
    z_SW_mean = np.mean(z_SW_interp, axis=0)
    z_SW_std = np.std(z_SW_interp, axis=0)
    z_ISW_mean = np.mean(z_ISW_interp, axis=0)
    z_ISW_std = np.std(z_ISW_interp, axis=0)
    z_total_mean = np.mean(z_total_interp, axis=0)
    z_total_std = np.std(z_total_interp, axis=0)
    
    # Create figure
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)
    
    # Panel 1: Mean evolution with error bands
    ax1 = fig.add_subplot(gs[0, :])
    
    # Plot individual trajectories (faint)
    for i in range(min(n_photons, 20)):  # Limit to 20 for visibility
        ax1.plot(a_common, z_total_interp[i], alpha=0.15, color='gray', lw=0.5)
    
    # Plot means with error bands
    ax1.fill_between(a_common, z_total_mean - z_total_std, z_total_mean + z_total_std,
                     alpha=0.3, color='black', label='Total (±1σ)')
    ax1.plot(a_common, z_total_mean, lw=3, color='black', label='Total (mean)')
    
    ax1.fill_between(a_common, z_H_mean - z_H_std, z_H_mean + z_H_std,
                     alpha=0.3, color='C0', label='z_H (±1σ)')
    ax1.plot(a_common, z_H_mean, lw=2, color='C0', label='z_H (mean)')
    
    ax1.fill_between(a_common, z_SW_mean - z_SW_std, z_SW_mean + z_SW_std,
                     alpha=0.3, color='C1', label='z_SW (±1σ)')
    ax1.plot(a_common, z_SW_mean, lw=2, color='C1', label='z_SW (mean)')
    
    ax1.set_xlabel('Scale factor a', fontsize=12)
    ax1.set_ylabel('Redshift z', fontsize=12)
    ax1.set_title(f'Mean Redshift Evolution ({n_photons} trajectories)', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10, ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Histogram of z_H
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(z_H_final, bins=20, alpha=0.7, color='C0', edgecolor='black')
    ax2.axvline(np.mean(z_H_final), color='red', linestyle='--', lw=2, 
                label=f'Mean: {np.mean(z_H_final):.4f}')
    ax2.set_xlabel('z_H (expansion)', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Distribution of z_H', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Panel 3: Histogram of z_SW
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(z_SW_final, bins=20, alpha=0.7, color='C1', edgecolor='black')
    ax3.axvline(np.mean(z_SW_final), color='red', linestyle='--', lw=2,
                label=f'Mean: {np.mean(z_SW_final):.4e}')
    ax3.set_xlabel('z_SW (Sachs-Wolfe)', fontsize=11)
    ax3.set_ylabel('Count', fontsize=11)
    ax3.set_title('Distribution of z_SW', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Panel 4: Histogram of z_total
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.hist(z_total_final, bins=20, alpha=0.7, color='black', edgecolor='black')
    ax4.axvline(np.mean(z_total_final), color='red', linestyle='--', lw=2,
                label=f'Mean: {np.mean(z_total_final):.4e}')
    ax4.set_xlabel('z_total', fontsize=11)
    ax4.set_ylabel('Count', fontsize=11)
    ax4.set_title('Distribution of z_total', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Panel 5: Scatter z_H vs z_SW
    ax5 = fig.add_subplot(gs[2, 0])
    scatter = ax5.scatter(z_H_final, z_SW_final, c=z_total_final, 
                         s=80, alpha=0.7, edgecolors='k', linewidth=0.5,
                         cmap='viridis')
    ax5.set_xlabel('z_H (expansion)', fontsize=11)
    ax5.set_ylabel('z_SW (Sachs-Wolfe)', fontsize=11)
    ax5.set_title('z_H vs z_SW', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax5, label='z_total')
    
    # Panel 6: Component contributions
    ax6 = fig.add_subplot(gs[2, 1])
    components = ['z_H', 'z_SW', 'z_ISW']
    means = [np.mean(z_H_final), np.mean(z_SW_final), np.mean(z_ISW_final)]
    stds = [np.std(z_H_final), np.std(z_SW_final), np.std(z_ISW_final)]
    colors = ['C0', 'C1', 'C2']
    
    x_pos = np.arange(len(components))
    bars = ax6.bar(x_pos, means, yerr=stds, alpha=0.7, color=colors,
                   edgecolor='black', capsize=5, error_kw={'linewidth': 2})
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(components, fontsize=11)
    ax6.set_ylabel('Mean redshift', fontsize=11)
    ax6.set_title('Component Contributions (mean ± std)', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.2e}\n±{std:.2e}',
                ha='center', va='bottom', fontsize=8)
    
    # Panel 7: Box plot
    ax7 = fig.add_subplot(gs[2, 2])
    box_data = [z_H_final, z_SW_final, z_total_final]
    bp = ax7.boxplot(box_data, labels=['z_H', 'z_SW', 'z_total'],
                     patch_artist=True, showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
    
    for patch, color in zip(bp['boxes'], ['C0', 'C1', 'black']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax7.set_ylabel('Redshift', fontsize=11)
    ax7.set_title('Redshift Distributions', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Statistical Analysis of Redshift ({n_photons} trajectories)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   Saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_redshift_vs_quantity(calculators, quantity='distance', component='total',
                              save_path=None, show=True):
    """
    Plot redshift as function of a specific quantity for all trajectories.
    
    Parameters:
    -----------
    calculators : list of RedshiftCalculator
        List of calculators for different photon trajectories
    quantity : str
        Quantity to plot on x-axis: 'distance', 'a', 'eta', 'proper_distance'
    component : str
        Redshift component: 'total', 'H', 'SW', 'ISW', 'all'
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display the figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    n_photons = len(calculators)
    print(f"Plotting {component} redshift vs {quantity} for {n_photons} trajectories...")
    
    for i, calc in enumerate(calculators):
        traj = calc.trajectory
        n_steps = len(traj)
        
        # Extract trajectory data
        eta_array = traj[:, 0]
        positions = traj[:, 1:4]
        
        # Get scale factors
        if calc.has_quantities:
            a_array = calc.a_array
        else:
            a_array = np.array([calc.a_of_eta(eta) for eta in eta_array])
        
        # Compute x-axis quantity
        if quantity == 'a':
            x_data = a_array
            xlabel = 'Scale factor a'
        elif quantity == 'eta':
            x_data = eta_array / (1e9 * 365.25 * 24 * 3600)  # Gyr
            xlabel = 'Conformal time η (Gyr)'
        elif quantity == 'distance':
            observer_pos = positions[0]
            x_data = np.linalg.norm(positions - observer_pos, axis=1) / one_Mpc
            xlabel = 'Distance from observer (Mpc)'
        elif quantity == 'proper_distance':
            segment_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
            x_data = np.concatenate([[0], np.cumsum(segment_lengths)]) / one_Mpc
            xlabel = 'Proper distance traveled (Mpc)'
        else:
            raise ValueError(f"Unknown quantity: {quantity}")
        
        # Compute redshift evolution
        z_traj = np.zeros(n_steps)
        
        for j in range(n_steps):
            sub_traj = traj[:j+1]
            if len(sub_traj) < 2:
                continue
            
            from excalibur.observables.redshift import RedshiftCalculator
            temp_calc = RedshiftCalculator(
                sub_traj,
                a_of_eta=calc.a_of_eta if not calc.has_quantities else None,
                potential_func=calc.potential_func if not calc.has_quantities else None
            )
            
            if component == 'total':
                z_H = temp_calc.compute_homogeneous_redshift()
                z_SW, _, _ = temp_calc.compute_sachs_wolfe_redshift()
                z_ISW, _ = temp_calc.compute_integrated_sachs_wolfe_redshift()
                z_traj[j] = z_H + z_SW + z_ISW
            elif component == 'H':
                z_traj[j] = temp_calc.compute_homogeneous_redshift()
            elif component == 'SW':
                z_SW, _, _ = temp_calc.compute_sachs_wolfe_redshift()
                z_traj[j] = z_SW
            elif component == 'ISW':
                z_ISW, _ = temp_calc.compute_integrated_sachs_wolfe_redshift()
                z_traj[j] = z_ISW
        
        # Plot
        ax.plot(x_data, z_traj, alpha=0.5, lw=1.5)
        
        if (i + 1) % 10 == 0:
            print(f"   Progress: {i+1}/{n_photons}")
    
    # Styling
    component_names = {
        'total': 'Total',
        'H': 'Homogeneous (expansion)',
        'SW': 'Sachs-Wolfe',
        'ISW': 'Integrated Sachs-Wolfe'
    }
    
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(f'Redshift z_{component.upper()}', fontsize=13)
    ax.set_title(f'{component_names.get(component, component)} Redshift vs {quantity.replace("_", " ").title()}\n'
                 f'({n_photons} trajectories)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   Saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
