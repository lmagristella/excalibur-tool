import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import h5py
import os

from excalibur.core.constants import *
from skimage import measure


def _ensure_visualization_dir():
    """Ensure _visualizations directory exists and return its path."""
    viz_dir = "_visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    return viz_dir


def _get_save_path(filename):
    """
    Get proper save path for visualization files.
    
    If filename is just a basename, save to _visualizations/.
    If filename is an absolute path, use it as-is.
    """
    if filename is None:
        return None
    
    # If it's an absolute path or already includes a directory, use as-is
    if os.path.isabs(filename) or os.path.dirname(filename):
        return filename
    
    # Otherwise, save to _visualizations/
    viz_dir = _ensure_visualization_dir()
    return os.path.join(viz_dir, filename)



class GridVisualizer:
    """ Class for visualizing various aspects of the cosmological grid, such as slices of the metric, the potential or density.."""
    def __init__(self, grid_data=None, grid_file=None):
        """Initialize GridVisualizer with grid data or from file."""
        if grid_data is not None:
            self.grid_data = grid_data
        elif grid_file is not None:
            self.load_grid_data(grid_file)
        else:
            self.grid_data = None
            print("Warning: No grid data provided. Use load_grid_data() to load data.")
        
        if self.grid_data is not None:
            self.setup_grid_info()

    def load_grid_data(self, filename):
        """Load grid data from HDF5 file."""
        print(f"Loading grid data from {filename}...")
        
        with h5py.File(filename, "r") as f:
            self.grid_data = {}
            
            # Load grid dimensions and coordinates
            if 'grid' in f:
                grid_group = f['grid']
                for key in grid_group.keys():
                    self.grid_data[key] = grid_group[key][:]
            
            # Load metadata if available
            self.metadata = {}
            for attr_name in f.attrs:
                self.metadata[attr_name] = f.attrs[attr_name]
        
        self.setup_grid_info()
        print(f"   Grid data loaded successfully")

    def setup_grid_info(self):
        """Extract grid information and statistics."""
        if 'x' in self.grid_data and 'y' in self.grid_data and 'z' in self.grid_data:
            self.x_coords = self.grid_data['x']
            self.y_coords = self.grid_data['y'] 
            self.z_coords = self.grid_data['z']
            
            self.nx, self.ny, self.nz = len(self.x_coords), len(self.y_coords), len(self.z_coords)
            
            # Coordinate ranges
            self.x_range = [self.x_coords.min(), self.x_coords.max()]
            self.y_range = [self.y_coords.min(), self.y_coords.max()]
            self.z_range = [self.z_coords.min(), self.z_coords.max()]
            
            print(f"   Grid dimensions: {self.nx} × {self.ny} × {self.nz}")
            print(f"   X range: [{self.x_range[0]/one_Mpc:.1f}, {self.x_range[1]/one_Mpc:.1f}] Mpc")
            print(f"   Y range: [{self.y_range[0]/one_Mpc:.1f}, {self.y_range[1]/one_Mpc:.1f}] Mpc")
            print(f"   Z range: [{self.z_range[0]/one_Mpc:.1f}, {self.z_range[1]/one_Mpc:.1f}] Mpc")

    def plot_2d_slice(self, field_name, slice_axis='z', slice_index=None, slice_value=None, 
                      log_scale=False, save_file=None, cmap='viridis'):
        """Plot a 2D slice of a 3D field."""
        if field_name not in self.grid_data:
            print(f"Error: Field '{field_name}' not found in grid data")
            return
        
        field = self.grid_data[field_name]
        
        # Determine slice index
        if slice_index is None and slice_value is not None:
            if slice_axis == 'x':
                slice_index = np.argmin(np.abs(self.x_coords - slice_value))
            elif slice_axis == 'y':
                slice_index = np.argmin(np.abs(self.y_coords - slice_value))
            elif slice_axis == 'z':
                slice_index = np.argmin(np.abs(self.z_coords - slice_value))
        elif slice_index is None:
            # Default to middle slice
            if slice_axis == 'x':
                slice_index = self.nx // 2
            elif slice_axis == 'y':
                slice_index = self.ny // 2
            elif slice_axis == 'z':
                slice_index = self.nz // 2
        
        # Extract slice
        if slice_axis == 'x':
            data_slice = field[slice_index, :, :]
            x_coords, y_coords = self.y_coords, self.z_coords
            xlabel, ylabel = 'Y [Mpc]', 'Z [Mpc]'
            slice_coord = self.x_coords[slice_index]
            title = f'{field_name} at X = {slice_coord/one_Mpc:.1f} Mpc'
        elif slice_axis == 'y':
            data_slice = field[:, slice_index, :]
            x_coords, y_coords = self.x_coords, self.z_coords
            xlabel, ylabel = 'X [Mpc]', 'Z [Mpc]'
            slice_coord = self.y_coords[slice_index]
            title = f'{field_name} at Y = {slice_coord/one_Mpc:.1f} Mpc'
        elif slice_axis == 'z':
            data_slice = field[:, :, slice_index]
            x_coords, y_coords = self.x_coords, self.y_coords
            xlabel, ylabel = 'X [Mpc]', 'Y [Mpc]'
            slice_coord = self.z_coords[slice_index]
            title = f'{field_name} at Z = {slice_coord/one_Mpc:.1f} Mpc'
        
        # Convert coordinates to Mpc
        x_coords_mpc = x_coords / one_Mpc
        y_coords_mpc = y_coords / one_Mpc
        
        # Apply log scale if requested
        plot_data = data_slice.T  # Transpose for correct orientation
        if log_scale:
            plot_data = np.log10(np.abs(plot_data) + 1e-10)
            title += ' (log scale)'
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(plot_data, extent=[x_coords_mpc.min(), x_coords_mpc.max(),
                                         y_coords_mpc.min(), y_coords_mpc.max()],
                       origin='lower', cmap=cmap, aspect='equal')
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        if log_scale:
            cbar.set_label(f'log₁₀({field_name})')
        else:
            cbar.set_label(field_name)
        
        plt.tight_layout()
        
        if save_file:
            save_path = _get_save_path(save_file)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   Saved 2D slice plot to {save_path}")
        plt.show()

    def plot_3d_isosurface(self, field_name, iso_value=None, opacity=0.3, save_file=None):
        """Plot 3D isosurface of a field."""
        if field_name not in self.grid_data:
            print(f"Error: Field '{field_name}' not found in grid data")
            return
        
        field = self.grid_data[field_name]
        
        if iso_value is None:
            iso_value = np.mean(field)
        
        print(f"Creating isosurface for {field_name} at value {iso_value}")
        
        # Generate isosurface
        verts, faces, _, _ = measure.marching_cubes(field, iso_value)
        
        # Convert vertices to physical coordinates
        verts[:, 0] = self.x_coords[0] + verts[:, 0] * (self.x_coords[-1] - self.x_coords[0]) / (self.nx - 1)
        verts[:, 1] = self.y_coords[0] + verts[:, 1] * (self.y_coords[-1] - self.y_coords[0]) / (self.ny - 1)
        verts[:, 2] = self.z_coords[0] + verts[:, 2] * (self.z_coords[-1] - self.z_coords[0]) / (self.nz - 1)
        
        # Convert to Mpc
        verts = verts / one_Mpc
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot isosurface
        ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], 
                        triangles=faces, alpha=opacity, cmap='viridis')
        
        ax.set_xlabel('X [Mpc]')
        ax.set_ylabel('Y [Mpc]')
        ax.set_zlabel('Z [Mpc]')
        ax.set_title(f'Isosurface of {field_name} = {iso_value}')
        
        plt.tight_layout()
        
        if save_file:
            save_path = _get_save_path(save_file); plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   Saved to {save_path}")
        
        plt.show()

    def plot_field_histogram(self, field_name, bins=50, log_scale=False, save_file=None):
        """Plot histogram of field values."""
        if field_name not in self.grid_data:
            print(f"Error: Field '{field_name}' not found in grid data")
            return
        
        field = self.grid_data[field_name].flatten()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        plot_data = field
        if log_scale:
            plot_data = np.log10(np.abs(field) + 1e-10)
            xlabel = f'log₁₀({field_name})'
        else:
            xlabel = field_name
        
        ax.hist(plot_data, bins=bins, alpha=0.7, edgecolor='black')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {field_name}')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = np.mean(plot_data)
        std_val = np.std(plot_data)
        ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3e}')
        ax.axvline(mean_val + std_val, color='orange', linestyle='--', alpha=0.7, label=f'±1σ: {std_val:.3e}')
        ax.axvline(mean_val - std_val, color='orange', linestyle='--', alpha=0.7)
        ax.legend()
        
        plt.tight_layout()
        
        if save_file:
            save_path = _get_save_path(save_file); plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   Saved to {save_path}")
        
        plt.show()

    def plot_metric_components(self, save_file=None):
        """Plot spatial slices of metric tensor components."""
        metric_components = ['g00', 'g11', 'g22', 'g33', 'g01', 'g02', 'g03', 'g12', 'g13', 'g23']
        available_components = [comp for comp in metric_components if comp in self.grid_data]
        
        if not available_components:
            print("Error: No metric components found in grid data")
            return
        
        n_components = len(available_components)
        n_cols = min(4, n_components)
        n_rows = (n_components + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        if n_components == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, comp in enumerate(available_components):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            # Plot middle z-slice
            z_mid = self.nz // 2
            data = self.grid_data[comp][:, :, z_mid].T
            
            im = ax.imshow(data, extent=[self.x_range[0]/one_Mpc, self.x_range[1]/one_Mpc,
                                        self.y_range[0]/one_Mpc, self.y_range[1]/one_Mpc],
                          origin='lower', cmap='RdBu_r', aspect='equal')
            
            ax.set_title(comp)
            ax.set_xlabel('X [Mpc]')
            ax.set_ylabel('Y [Mpc]')
            
            plt.colorbar(im, ax=ax)
        
        # Hide unused subplots
        for i in range(n_components, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.set_visible(False)
        
        fig.suptitle('Metric Tensor Components (middle z-slice)', fontsize=14)
        plt.tight_layout()
        
        if save_file:
            save_path = _get_save_path(save_file); plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   Saved to {save_path}")
        
        plt.show()

    def plot_field_comparison(self, field_names, slice_axis='z', slice_index=None, 
                             log_scale=False, save_file=None):
        """Compare multiple fields side by side."""
        available_fields = [name for name in field_names if name in self.grid_data]
        
        if not available_fields:
            print("Error: None of the specified fields found in grid data")
            return
        
        n_fields = len(available_fields)
        fig, axes = plt.subplots(1, n_fields, figsize=(5*n_fields, 4))
        if n_fields == 1:
            axes = [axes]
        
        if slice_index is None:
            if slice_axis == 'x':
                slice_index = self.nx // 2
            elif slice_axis == 'y':
                slice_index = self.ny // 2
            elif slice_axis == 'z':
                slice_index = self.nz // 2
        
        for i, field_name in enumerate(available_fields):
            field = self.grid_data[field_name]
            
            # Extract slice
            if slice_axis == 'x':
                data_slice = field[slice_index, :, :].T
                extent = [self.y_range[0]/one_Mpc, self.y_range[1]/one_Mpc,
                         self.z_range[0]/one_Mpc, self.z_range[1]/one_Mpc]
                xlabel, ylabel = 'Y [Mpc]', 'Z [Mpc]'
            elif slice_axis == 'y':
                data_slice = field[:, slice_index, :].T
                extent = [self.x_range[0]/one_Mpc, self.x_range[1]/one_Mpc,
                         self.z_range[0]/one_Mpc, self.z_range[1]/one_Mpc]
                xlabel, ylabel = 'X [Mpc]', 'Y [Mpc]'
            elif slice_axis == 'z':
                data_slice = field[:, :, slice_index].T
                extent = [self.x_range[0]/one_Mpc, self.x_range[1]/one_Mpc,
                         self.y_range[0]/one_Mpc, self.y_range[1]/one_Mpc]
                xlabel, ylabel = 'X [Mpc]', 'Y [Mpc]'
            
            if log_scale:
                data_slice = np.log10(np.abs(data_slice) + 1e-10)
            
            im = axes[i].imshow(data_slice, extent=extent, origin='lower', 
                               cmap='viridis', aspect='equal')
            
            axes[i].set_xlabel(xlabel)
            axes[i].set_ylabel(ylabel)
            title = field_name
            if log_scale:
                title += ' (log)'
            axes[i].set_title(title)
            
            plt.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        
        if save_file:
            save_path = _get_save_path(save_file); plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   Saved to {save_path}")
        
        plt.show()

class TrajectoryVisualizer:
    """Class for visualizing photon trajectories from HDF5 files."""
    
    def __init__(self, filename):
        """Load trajectory data from HDF5 file."""
        self.filename = filename
        self.parse_mass_position()
        self.load_data()
    
    def parse_mass_position(self):
        """Parse mass position from filename if available."""
        import re
        # Pattern: backward_raytracing_trajectories_mass_X_Y_Z_Mpc.h5
        pattern = r'mass_(\d+)_(\d+)_(\d+)_Mpc'
        match = re.search(pattern, self.filename)
        
        if match:
            self.mass_position = np.array([
                float(match.group(1)),
                float(match.group(2)),
                float(match.group(3))
            ])  # In Mpc
            print(f"   Parsed mass position: [{self.mass_position[0]:.0f}, {self.mass_position[1]:.0f}, {self.mass_position[2]:.0f}] Mpc")
        else:
            self.mass_position = None
            print("   No mass position found in filename")
    
    def load_data(self):
        """Load all trajectory data from the HDF5 file."""
        print(f"Loading trajectory data from {self.filename}...")
        
        self.trajectories = []
        self.metadata = {}
        
        with h5py.File(self.filename, "r") as f:
            # Get number of photons
            self.n_photons = f.attrs['n_photons']
            print(f"   Found {self.n_photons} photon trajectories")
            
            # Load metadata if available
            if 'photon_info' in f:
                photon_info = f['photon_info'][:]
                for info in photon_info:
                    self.metadata[info['id']] = {
                        'n_states': info['n_states'],
                        'initial_position': info['initial_position'],
                        'initial_velocity': info['initial_velocity'],
                        'weight': info['weight']
                    }
            
            # Load each trajectory
            for i in range(self.n_photons):
                dataset_name = f"photon_{i}_states"
                if dataset_name in f:
                    states_data = f[dataset_name][:]
                    
                    # Remove NaN padding and extract essential data (position + velocity)
                    trajectory = []
                    for state_data in states_data:
                        # Remove NaN values
                        valid_indices = ~np.isnan(state_data)
                        if np.any(valid_indices):
                            state = state_data[valid_indices]
                            # Take only the first 8 components (4D position + 4D velocity)
                            if len(state) >= 8:
                                trajectory.append(state[:8])  # [η, x, y, z, u_η, u_x, u_y, u_z]
                    
                    if trajectory:
                        self.trajectories.append(np.array(trajectory))
                    else:
                        self.trajectories.append(np.array([]).reshape(0, 8))
        
        # Calculate statistics
        self.calculate_statistics()
        print(f"   Loaded {len([t for t in self.trajectories if len(t) > 0])} valid trajectories")
    
    def calculate_statistics(self):
        """Calculate trajectory statistics."""
        valid_trajectories = [t for t in self.trajectories if len(t) > 0]
        
        if not valid_trajectories:
            print("   Warning: No valid trajectories found!")
            return
        
        # Time ranges
        self.eta_min = min(t[:, 0].min() for t in valid_trajectories)
        self.eta_max = max(t[:, 0].max() for t in valid_trajectories)
        
        # Spatial ranges
        all_positions = np.vstack([t[:, 1:4] for t in valid_trajectories])
        self.pos_min = all_positions.min(axis=0)
        self.pos_max = all_positions.max(axis=0)
        self.pos_center = (self.pos_min + self.pos_max) / 2
        self.pos_range = self.pos_max - self.pos_min
        
        print(f"   Time range: η ∈ [{self.eta_min:.2f}, {self.eta_max:.2f}]")
        print(f"   Spatial range: x ∈ [{self.pos_min[0]/one_Mpc:.1f}, {self.pos_max[0]/one_Mpc:.1f}] Mpc")
        print(f"   Spatial range: y ∈ [{self.pos_min[1]/one_Mpc:.1f}, {self.pos_max[1]/one_Mpc:.1f}] Mpc")
        print(f"   Spatial range: z ∈ [{self.pos_min[2]/one_Mpc:.1f}, {self.pos_max[2]/one_Mpc:.1f}] Mpc")
    
    def plot_3d_trajectories(self, max_trajectories=None, save_file=None):
        """Create a 3D plot of all trajectories."""
        print("\nCreating 3D trajectory plot...")
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        valid_trajectories = [t for t in self.trajectories if len(t) > 0]
        if max_trajectories:
            valid_trajectories = valid_trajectories[:max_trajectories]
        
        # Plot trajectories
        colors = plt.cm.viridis(np.linspace(0, 1, len(valid_trajectories)))
        
        for i, trajectory in enumerate(valid_trajectories):
            x = trajectory[:, 1] / one_Mpc  # Convert to Mpc
            y = trajectory[:, 2] / one_Mpc
            z = trajectory[:, 3] / one_Mpc
            
            ax.plot(x, y, z, color=colors[i], alpha=0.7, linewidth=1)
            
            # Mark start and end points
            ax.scatter(x[0], y[0], z[0], color='red', s=30, alpha=0.8)  # Start (observer)
            ax.scatter(x[-1], y[-1], z[-1], color='blue', s=30, alpha=0.8)  # End (source)
        
        # Set labels and title
        ax.set_xlabel('X [Mpc]')
        ax.set_ylabel('Y [Mpc]')
        ax.set_zlabel('Z [Mpc]')
        ax.set_title(f'Backward Ray Tracing Trajectories\n{len(valid_trajectories)} photons')
        
        # Plot mass position if available
        if self.mass_position is not None:
            ax.scatter(self.mass_position[0], self.mass_position[1], self.mass_position[2],
                      color='gold', s=200, marker='*', edgecolors='black', linewidths=2,
                      label='Mass', zorder=100)
        
        # Add legend
        ax.scatter([], [], [], color='red', s=30, label='Observer (start)')
        ax.scatter([], [], [], color='blue', s=30, label='Source (end)')
        ax.legend()
        
        # Set equal aspect ratio
        ax.set_box_aspect([1,1,1])
        
        plt.tight_layout()
        
        if save_file:
            save_path = _get_save_path(save_file); plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   Saved to {save_path}")
        
        plt.show()
    
    def plot_2d_projections(self, save_file=None):
        """Create 2D projections of trajectories."""
        print("\nCreating 2D projection plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        valid_trajectories = [t for t in self.trajectories if len(t) > 0]
        colors = plt.cm.viridis(np.linspace(0, 1, len(valid_trajectories)))
        
        # XY projection
        ax = axes[0, 0]
        for i, trajectory in enumerate(valid_trajectories):
            x = trajectory[:, 1] / one_Mpc
            y = trajectory[:, 2] / one_Mpc
            ax.plot(x, y, color=colors[i], alpha=0.7, linewidth=1)
            ax.scatter(x[0], y[0], color='red', s=10, alpha=0.8)
            ax.scatter(x[-1], y[-1], color='blue', s=10, alpha=0.8)
        # Plot mass position
        if self.mass_position is not None:
            ax.scatter(self.mass_position[0], self.mass_position[1], color='gold', 
                      s=200, marker='*', edgecolors='black', linewidths=2, zorder=100)
        ax.set_xlabel('X [Mpc]')
        ax.set_ylabel('Y [Mpc]')
        ax.set_title('XY Projection')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # XZ projection
        ax = axes[0, 1]
        for i, trajectory in enumerate(valid_trajectories):
            x = trajectory[:, 1] / one_Mpc
            z = trajectory[:, 3] / one_Mpc
            ax.plot(x, z, color=colors[i], alpha=0.7, linewidth=1)
            ax.scatter(x[0], z[0], color='red', s=10, alpha=0.8)
            ax.scatter(x[-1], z[-1], color='blue', s=10, alpha=0.8)
        # Plot mass position
        if self.mass_position is not None:
            ax.scatter(self.mass_position[0], self.mass_position[2], color='gold',
                      s=200, marker='*', edgecolors='black', linewidths=2, zorder=100)
        ax.set_xlabel('X [Mpc]')
        ax.set_ylabel('Z [Mpc]')
        ax.set_title('XZ Projection')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # YZ projection
        ax = axes[1, 0]
        for i, trajectory in enumerate(valid_trajectories):
            y = trajectory[:, 2] / one_Mpc
            z = trajectory[:, 3] / one_Mpc
            ax.plot(y, z, color=colors[i], alpha=0.7, linewidth=1)
            ax.scatter(y[0], z[0], color='red', s=10, alpha=0.8)
            ax.scatter(y[-1], z[-1], color='blue', s=10, alpha=0.8)
        # Plot mass position
        if self.mass_position is not None:
            ax.scatter(self.mass_position[1], self.mass_position[2], color='gold',
                      s=200, marker='*', edgecolors='black', linewidths=2, zorder=100)
        ax.set_xlabel('Y [Mpc]')
        ax.set_ylabel('Z [Mpc]')
        ax.set_title('YZ Projection')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Time evolution (X vs η)
        ax = axes[1, 1]
        for i, trajectory in enumerate(valid_trajectories):
            eta = trajectory[:, 0]
            x = trajectory[:, 1] / one_Mpc
            ax.plot(eta, x, color=colors[i], alpha=0.7, linewidth=1)
        ax.set_xlabel('Conformal Time η')
        ax.set_ylabel('X Position [Mpc]')
        ax.set_title('Time Evolution (X coordinate)')
        ax.grid(True, alpha=0.3)
        
        # Add overall legend
        fig.suptitle(f'Backward Ray Tracing - 2D Projections ({len(valid_trajectories)} photons)', fontsize=14)
        
        # Create custom legend
        import matplotlib.patches as mpatches
        red_patch = mpatches.Patch(color='red', label='Observer (start)')
        blue_patch = mpatches.Patch(color='blue', label='Source (end)')
        legend_handles = [red_patch, blue_patch]
        
        if self.mass_position is not None:
            from matplotlib.lines import Line2D
            mass_marker = Line2D([0], [0], marker='*', color='w', markerfacecolor='gold',
                                markersize=15, markeredgecolor='black', markeredgewidth=2,
                                label=f'Mass @[{self.mass_position[0]:.0f}, {self.mass_position[1]:.0f}, {self.mass_position[2]:.0f}] Mpc')
            legend_handles.append(mass_marker)
        
        fig.legend(handles=legend_handles, loc='center', bbox_to_anchor=(0.5, 0.02), ncol=3)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        
        if save_file:
            save_path = _get_save_path(save_file); plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   Saved to {save_path}")
        
        plt.show()
    
    def plot_time_evolution(self, save_file=None):
        """Plot the time evolution of trajectories."""
        print("\nCreating time evolution plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        valid_trajectories = [t for t in self.trajectories if len(t) > 0]
        colors = plt.cm.viridis(np.linspace(0, 1, len(valid_trajectories)))
        
        # Position components vs time
        components = ['X', 'Y', 'Z']
        for i, comp in enumerate(components):
            ax = axes[i//2, i%2]
            for j, trajectory in enumerate(valid_trajectories):
                eta = trajectory[:, 0]
                pos = trajectory[:, i+1] / one_Mpc
                ax.plot(eta, pos, color=colors[j], alpha=0.7, linewidth=1)
            ax.set_xlabel('Conformal Time η')
            ax.set_ylabel(f'{comp} Position [Mpc]')
            ax.set_title(f'{comp} Position vs Time')
            ax.grid(True, alpha=0.3)
        
        # Distance from initial position
        ax = axes[1, 1]
        for j, trajectory in enumerate(valid_trajectories):
            eta = trajectory[:, 0]
            positions = trajectory[:, 1:4]
            initial_pos = positions[0]
            distances = np.linalg.norm(positions - initial_pos, axis=1) / one_Mpc
            ax.plot(eta, distances, color=colors[j], alpha=0.7, linewidth=1)
        ax.set_xlabel('Conformal Time η')
        ax.set_ylabel('Distance from Observer [Mpc]')
        ax.set_title('Distance from Observer vs Time')
        ax.grid(True, alpha=0.3)
        
        fig.suptitle(f'Time Evolution Analysis ({len(valid_trajectories)} photons)', fontsize=14)
        plt.tight_layout()
        
        if save_file:
            save_path = _get_save_path(save_file); plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   Saved to {save_path}")
        
        plt.show()
    
    def plot_statistics(self, save_file=None):
        """Plot statistical analysis of trajectories."""
        print("\nCreating statistical analysis plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        valid_trajectories = [t for t in self.trajectories if len(t) > 0]
        
        # Trajectory lengths
        ax = axes[0, 0]
        lengths = [len(t) for t in valid_trajectories]
        ax.hist(lengths, bins=20, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Number of Time Steps')
        ax.set_ylabel('Number of Trajectories')
        ax.set_title('Trajectory Length Distribution')
        ax.grid(True, alpha=0.3)
        
        # Final positions (source distribution)
        ax = axes[0, 1]
        final_positions = np.array([t[-1, 1:4] for t in valid_trajectories]) / one_Mpc
        ax.scatter(final_positions[:, 0], final_positions[:, 1], alpha=0.7, s=30)
        ax.set_xlabel('X [Mpc]')
        ax.set_ylabel('Y [Mpc]')
        ax.set_title('Source Positions (Final Positions)')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Distance distribution
        ax = axes[1, 0]
        observer_pos = valid_trajectories[0][0, 1:4]  # First trajectory's initial position
        final_distances = [np.linalg.norm(t[-1, 1:4] - observer_pos) / one_Mpc for t in valid_trajectories]
        ax.hist(final_distances, bins=20, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Distance from Observer [Mpc]')
        ax.set_ylabel('Number of Trajectories')
        ax.set_title('Source Distance Distribution')
        ax.grid(True, alpha=0.3)
        
        # Time span distribution
        ax = axes[1, 1]
        time_spans = [t[-1, 0] - t[0, 0] for t in valid_trajectories]
        ax.hist(time_spans, bins=20, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Time Span Δη')
        ax.set_ylabel('Number of Trajectories')
        ax.set_title('Time Span Distribution')
        ax.grid(True, alpha=0.3)
        
        fig.suptitle(f'Statistical Analysis ({len(valid_trajectories)} photons)', fontsize=14)
        plt.tight_layout()
        
        if save_file:
            save_path = _get_save_path(save_file); plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   Saved to {save_path}")
        
        plt.show()
    
    def create_animation(self, save_file=None, interval=100, max_trajectories=20):
        """Create an animation showing trajectory evolution."""
        print(f"\nCreating trajectory animation...")
        
        valid_trajectories = [t for t in self.trajectories if len(t) > 0]
        if max_trajectories:
            valid_trajectories = valid_trajectories[:max_trajectories]
        
        # Find maximum trajectory length for animation
        max_length = max(len(t) for t in valid_trajectories)
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Initialize line objects for each trajectory
        lines = []
        colors = plt.cm.viridis(np.linspace(0, 1, len(valid_trajectories)))
        
        for i in range(len(valid_trajectories)):
            line, = ax.plot([], [], [], color=colors[i], alpha=0.7, linewidth=2)
            lines.append(line)
        
        def animate(frame):
            ax.clear()
            
            # Set up the plot
            ax.set_xlabel('X [Mpc]')
            ax.set_ylabel('Y [Mpc]')
            ax.set_zlabel('Z [Mpc]')
            ax.set_title(f'Backward Ray Tracing Animation\nTime Step: {frame}/{max_length-1}')
            
            # Plot each trajectory up to current frame
            for i, trajectory in enumerate(valid_trajectories):
                if frame < len(trajectory):
                    x = trajectory[:frame+1, 1] / one_Mpc
                    y = trajectory[:frame+1, 2] / one_Mpc
                    z = trajectory[:frame+1, 3] / one_Mpc
                    
                    ax.plot(x, y, z, color=colors[i], alpha=0.7, linewidth=2)
                    
                    # Mark current position
                    if len(x) > 0:
                        ax.scatter(x[-1], y[-1], z[-1], color=colors[i], s=50, alpha=0.9)
            
            # Plot mass position if available
            if self.mass_position is not None:
                ax.scatter(self.mass_position[0], self.mass_position[1], self.mass_position[2],
                          color='gold', s=300, marker='*', edgecolors='black', linewidths=2,
                          label='Mass', zorder=100)
            
            # Set consistent axis limits
            margin = 50  # Mpc
            ax.set_xlim(self.pos_min[0]/one_Mpc - margin, self.pos_max[0]/one_Mpc + margin)
            ax.set_ylim(self.pos_min[1]/one_Mpc - margin, self.pos_max[1]/one_Mpc + margin)
            ax.set_zlim(self.pos_min[2]/one_Mpc - margin, self.pos_max[2]/one_Mpc + margin)
            
            return lines
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=max_length, 
                                     interval=interval, blit=False, repeat=True)
        
        if save_file:
            print(f"   Saving animation to {save_file} (this may take a while)...")
            anim.save(save_file, writer='pillow', fps=10)
            print(f"   Animation saved to {save_file}")
        else:
            plt.show()
        
        return anim

