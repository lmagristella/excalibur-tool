# photon/photons.py
import numpy as np
import h5py
from .photon import Photon
from excalibur.core.constants import c

class Photons:

    """Class for handling multiple photons with collective operations."""
    def __init__(self, metric):
        self.photons = []
        self.metric = metric
    
    def add_photon(self, photon):
        """Add a single photon to the collection."""
        self.photons.append(photon)
    
    def generate_cone_random(self, n_photons, origin, central_direction, cone_angle):
        """
        Generate n_photons with random directions within a cone.
        
        Parameters:
        -----------
        n_photons : int
            Number of photons to generate
        origin : array-like
            4D starting position [eta, x, y, z]
        central_direction : array-like
            Central direction vector (will be normalized)
        cone_angle : float
            Half-angle of the cone in radians
        """
        origin = np.asarray(origin, dtype=float)
        central_dir = np.asarray(central_direction, dtype=float)
        central_dir = central_dir / np.linalg.norm(central_dir)  # Normalize

        # Get scale factor at initial time
        eta_init = origin[0]
        a_init = self.metric.a_of_eta(eta_init) if hasattr(self.metric, 'a_of_eta') else 1.0
        
        for i in range(n_photons):
            # Generate random direction within cone using spherical coordinates
            # Random angle within cone
            theta = np.random.uniform(0, cone_angle)
            # Random azimuthal angle
            phi = np.random.uniform(0, 2*np.pi)
            
            # Convert to 3D direction in cone coordinate system
            direction_cone = np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta)
            ])
            
            # Rotate to align with central direction
            direction_3d = self._rotate_to_direction(direction_cone, central_dir)
            
            # Normalize to unit vector
            direction_3d_normalized = direction_3d / np.linalg.norm(direction_3d)
            
            # CORRECT INITIALIZATION FOR PHOTON 4-VELOCITY in FLRW + perturbations
            # Metric: ds² = -a²(1+2ψ)c²dη² + a²(1-2φ)(dx²+dy²+dz²)
            # For null geodesic: g_μν u^μ u^ν = 0
            # This gives: -a²(1+2ψ)c²(u⁰)² + a²(1-2φ)(u^i u^i) = 0
            # So: (u⁰)² = (1-2φ)/(1+2ψ) × (u^i u^i)/c²
            # In weak field limit (φ,ψ << 1): u⁰ ≈ ±|u_spatial|/c
            
            # For spatial components: u^i (contravariant) in coordinate basis
            # These should be O(c) in magnitude for photons
            u_spatial = direction_3d_normalized * c
            
            # Temporal component: solve null condition
            # For forward-time photons: u⁰ > 0, for backward: u⁰ < 0
            # Here we initialize for FORWARD, will invert later for backward tracing
            u0 = np.linalg.norm(u_spatial) / c  # Positive for forward time
            
            # Build 4-velocity (contravariant components)
            direction_4d = np.array([u0, u_spatial[0], u_spatial[1], u_spatial[2]])
            
            photon = Photon(origin.copy(), direction_4d)
            self.add_photon(photon)
            
            # Verify null condition with relative error
            if hasattr(photon, 'null_condition_relative_error'):
                rel_error = photon.null_condition_relative_error(metric=self.metric)
                if rel_error > 1e-5:  # Relaxed threshold for practical purposes
                    u_u = photon.null_condition(metric=self.metric)
                    print(f"Warning: Photon {i} null condition violated: u.u = {u_u:.3e}, rel_error = {rel_error:.3e}")

    
    def generate_cone_grid(self, n_theta, n_phi, origin, central_direction, cone_angle):
        """
        Generate photons on a regular grid within a cone.
        
        Parameters:
        -----------
        n_theta : int
            Number of polar angle divisions
        n_phi : int
            Number of azimuthal angle divisions
        origin : array-like
            4D starting position [eta, x, y, z]
        central_direction : array-like
            Central direction vector (will be normalized)
        cone_angle : float
            Half-angle of the cone in radians
        """
        origin = np.asarray(origin, dtype=float)
        central_dir = np.asarray(central_direction, dtype=float)
        central_dir = central_dir / np.linalg.norm(central_dir)  # Normalize
        
        # Get scale factor at initial time
        eta_init = origin[0]
        a_init = self.metric.a_of_eta(eta_init) if hasattr(self.metric, 'a_of_eta') else 1.0
        
        # Create grid in spherical coordinates
        theta_values = np.linspace(0, cone_angle, n_theta)
        phi_values = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
        
        photon_count = 0
        for theta in theta_values:
            for phi in phi_values:
                # Skip the central point duplication
                if theta == 0 and phi != phi_values[0]:
                    continue
                
                # Convert to 3D direction in cone coordinate system
                direction_cone = np.array([
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta)
                ])
                
                # Rotate to align with central direction
                direction_3d = self._rotate_to_direction(direction_cone, central_dir)
                
                # Normalize to unit vector
                direction_3d_normalized = direction_3d / np.linalg.norm(direction_3d)
                
                # CORRECT INITIALIZATION FOR PHOTON 4-VELOCITY in FLRW + perturbations
                # Metric: ds² = -a²(1+2ψ)c²dη² + a²(1-2φ)(dx²+dy²+dz²)
                # For null geodesic: g_μν u^μ u^ν = 0
                # This gives: -a²(1+2ψ)c²(u⁰)² + a²(1-2φ)(u^i u^i) = 0
                # So: (u⁰)² = (1-2φ)/(1+2ψ) × (u^i u^i)/c²
                # In weak field limit (φ,ψ << 1): u⁰ ≈ ±|u_spatial|/c
                
                # For spatial components: u^i (contravariant) in coordinate basis
                # These should be O(c) in magnitude for photons
                u_spatial = direction_3d_normalized * c
                
                # Temporal component: solve null condition
                # For forward-time photons: u⁰ > 0, for backward: u⁰ < 0
                # Here we initialize for FORWARD, will invert later for backward tracing
                u0 = np.linalg.norm(u_spatial) / c  # Positive for forward time
                
                # Build 4-velocity (contravariant components)
                direction_4d = np.array([-u0, u_spatial[0], u_spatial[1], u_spatial[2]])
                
                photon = Photon(origin.copy(), direction_4d)
                self.add_photon(photon)
                
                # Verify null condition with relative error
                if hasattr(photon, 'null_condition_relative_error'):
                    rel_error = photon.null_condition_relative_error(metric=self.metric)
                    if rel_error > 1e-5:  # Relaxed threshold for practical purposes
                        u_u = photon.null_condition(metric=self.metric)
                        print(f"Warning: Photon {photon_count} null condition violated: u.u = {u_u:.3e}, rel_error = {rel_error:.3e}")
                
                photon_count += 1
    
    def _rotate_to_direction(self, vector, target_direction):
        """
        Rotate a vector to align with a target direction.
        Uses Rodrigues' rotation formula.
        """
        target_direction = target_direction / np.linalg.norm(target_direction)
        
        # If target is already along z-axis, no rotation needed
        z_axis = np.array([0, 0, 1])
        if np.allclose(target_direction, z_axis):
            return vector
        
        # If target is opposite to z-axis, rotate 180° around x-axis
        if np.allclose(target_direction, -z_axis):
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]
            ])
            return rotation_matrix @ vector
        
        # General rotation using Rodrigues' formula
        k = np.cross(z_axis, target_direction)
        k = k / np.linalg.norm(k)
        
        cos_angle = np.dot(z_axis, target_direction)
        sin_angle = np.linalg.norm(np.cross(z_axis, target_direction))
        
        # Rodrigues' rotation formula
        rotated = (vector * cos_angle + 
                  np.cross(k, vector) * sin_angle + 
                  k * np.dot(k, vector) * (1 - cos_angle))
        
        return rotated
    
    def record_all(self):
        """Record current state for all photons."""
        for photon in self.photons:
            photon.record()
    
    def update_quantities_all(self, relevant_quantities_func):
        """Update quantities for all photons using the given function."""
        for photon in self.photons:
            photon.state_quantities(relevant_quantities_func)

    def save_to_binary(self, filename):
        """
        Save all photon data to a binary file using numpy's binary format.
        
        Parameters:
        -----------
        filename : str
            Path to the binary file to save
        """
        # Collect all data
        data_to_save = {
            'n_photons': len(self.photons),
            'photon_data': []
        }
        
        for i, photon in enumerate(self.photons):
            photon_data = {
                'weight': photon.weight,
                'initial_position': photon.position,
                'initial_velocity': photon.velocity,
                'history_states': []
            }
            
            # Collect history states
            if photon.history.states:
                for state in photon.history.states:
                    photon_data['history_states'].append(np.asarray(state))
            
            data_to_save['photon_data'].append(photon_data)
        
        # Save to binary file
        np.savez_compressed(filename, **data_to_save)
    
    def load_from_binary(self, filename):
        """
        Load photon data from a binary file created with save_to_binary.
        
        Parameters:
        -----------
        filename : str
            Path to the binary file to load
        """
        self.photons = []
        
        # Load data
        data = np.load(filename, allow_pickle=True)
        
        n_photons = int(data['n_photons'])
        photon_data_list = data['photon_data']
        
        for i in range(n_photons):
            photon_data = photon_data_list[i]
            
            # Extract photon parameters
            weight = float(photon_data['weight'])
            initial_position = np.asarray(photon_data['initial_position'])
            initial_velocity = np.asarray(photon_data['initial_velocity'])
            
            # Create photon
            photon = Photon(initial_position, initial_velocity, weight)
            
            # Reconstruct history
            history_states = photon_data['history_states']
            for state in history_states:
                photon.history.append(np.asarray(state))
            
            self.add_photon(photon)
   
    
    def save_all_histories(self, filename):
        """
        Save all photon histories to a single HDF5 file.
        
        Creates datasets:
        - 'photon_{i}_states': trajectory for photon i
        - 'n_photons': total number of photons
        - 'photon_info': metadata about each photon
        """
        with h5py.File(filename, "w") as f:
            # Save number of photons
            f.attrs['n_photons'] = len(self.photons)
            
            # Save each photon's trajectory
            photon_info = []
            for i, photon in enumerate(self.photons):
                # Save trajectory
                if photon.history.states:
                    max_length = max(len(state) for state in photon.history.states)
                    states_array = np.full((len(photon.history.states), max_length), np.nan)
                    for j, state in enumerate(photon.history.states):
                        states_array[j, :len(state)] = state
                    f.create_dataset(f"photon_{i}_states", data=states_array)
                    
                    # Store metadata
                    photon_info.append({
                        'id': i,
                        'n_states': len(photon.history.states),
                        'initial_position': photon.history.states[0][:4] if photon.history.states else np.nan,
                        'initial_velocity': photon.history.states[0][4:8] if len(photon.history.states[0]) >= 8 else np.nan,
                        'weight': photon.weight
                    })
                else:
                    f.create_dataset(f"photon_{i}_states", data=np.array([]))
                    photon_info.append({
                        'id': i,
                        'n_states': 0,
                        'initial_position': np.nan,
                        'initial_velocity': np.nan,
                        'weight': photon.weight
                    })
            
            # Save photon metadata
            if photon_info:
                # Create structured array for metadata
                dtype = [
                    ('id', 'i4'),
                    ('n_states', 'i4'), 
                    ('initial_position', '4f8'),
                    ('initial_velocity', '4f8'),
                    ('weight', 'f8')
                ]
                
                metadata_array = np.empty(len(photon_info), dtype=dtype)
                for i, info in enumerate(photon_info):
                    metadata_array[i] = (
                        info['id'],
                        info['n_states'],
                        info['initial_position'],
                        info['initial_velocity'], 
                        info['weight']
                    )
                
                f.create_dataset('photon_info', data=metadata_array)
    
    def load_from_hdf5(self, filename):
        """
        Load photon trajectories from an HDF5 file.
        Reconstructs the photon objects with their histories.
        """
        self.photons = []
        
        with h5py.File(filename, "r") as f:
            n_photons = f.attrs['n_photons']
            
            # Load metadata if available
            metadata = {}
            if 'photon_info' in f:
                photon_info = f['photon_info'][:]
                for info in photon_info:
                    metadata[info['id']] = {
                        'n_states': info['n_states'],
                        'initial_position': info['initial_position'],
                        'initial_velocity': info['initial_velocity'],
                        'weight': info['weight']
                    }
            
            # Load each photon
            for i in range(n_photons):
                dataset_name = f"photon_{i}_states"
                if dataset_name in f:
                    states_data = f[dataset_name][:]
                    
                    # Get initial conditions from metadata or first state
                    if i in metadata:
                        initial_pos = metadata[i]['initial_position']
                        initial_vel = metadata[i]['initial_velocity']
                        weight = metadata[i]['weight']
                    else:
                        # Fallback: use first state
                        if len(states_data) > 0:
                            first_state = states_data[0]
                            # Remove NaN values
                            first_state = first_state[~np.isnan(first_state)]
                            initial_pos = first_state[:4] if len(first_state) >= 4 else np.zeros(4)
                            initial_vel = first_state[4:8] if len(first_state) >= 8 else np.array([1,1,0,0])
                            weight = 1.0
                        else:
                            initial_pos = np.zeros(4)
                            initial_vel = np.array([1,1,0,0])
                            weight = 1.0
                    
                    # Create photon
                    photon = Photon(initial_pos, initial_vel, weight)
                    
                    # Reconstruct history
                    for state_data in states_data:
                        # Remove NaN padding
                        state = state_data[~np.isnan(state_data)]
                        if len(state) > 0:
                            photon.history.append(state)
                    
                    self.add_photon(photon)
    
    @property
    def n_photons(self):
        """Return the number of photons."""
        return len(self.photons)
    
    def __len__(self):
        """Return the number of photons."""
        return len(self.photons)
    
    def __getitem__(self, index):
        """Get photon by index."""
        return self.photons[index]
    
    def __iter__(self):
        """Iterate over photons."""
        return iter(self.photons)
