# photon/photons.py
import numpy as np
from .photon import Photon
from excalibur.core.constants import c
from excalibur.core.coordinates import cartesian_velocity_to_spherical
try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover
    # h5py is only required for trajectory I/O; keep it optional so core raytracing
    # and tests (photon init, integration) don't fail if h5py isn't available or
    # is binary-incompatible with the current numpy.
    h5py = None

class Photons:

    """Class for handling multiple photons with collective operations."""
    def __init__(self, metric):
        self.photons = []
        self.metric = metric
    
    def positions(self):
        """Return array of photon positions."""
        return np.array([photon.x for photon in self.photons])
    
    def directions(self):
        """Return array of photon directions (4-velocities)."""
        return np.array([photon.u for photon in self.photons])
    
    def add_photon(self, photon):
        """Add a single photon to the collection."""
        self.photons.append(photon)
    
    def generate_cone_random(
        self,
        n_photons,
        origin,
        central_direction,
        cone_angle,
        direction_basis="coordinates",
        *,
        direction_coords=None,
    ):
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
        # Backward-compatible alias:
        #   direction_coords='metric'    -> direction_basis='coordinates'
        #   direction_coords='cartesian' -> direction_basis='cartesian'
        if direction_coords is not None:
            if direction_coords == "metric":
                direction_basis = "coordinates"
            elif direction_coords == "cartesian":
                direction_basis = "cartesian"
            else:
                raise ValueError("direction_coords must be 'metric' or 'cartesian'")

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
            
            # Calculate temporal component using null condition
            # For FLRW + perturbations: ds² = -a²(1+2ψ)c²dη² + a²(1-2φ)(dx²+dy²+dz²)
            # Null conditi on: g_μν u^μ u^ν = 0
            
            # Get metric components at origin
            g_components = self.metric.metric_tensor(origin)
            g00 = g_components[0, 0]  #-a²(1+2ψ)c²
            g11 = g_components[1, 1]  # a²(1-2φ)
            g22 = g_components[2, 2]  # a²(1-2φ)
            g33 = g_components[3, 3]  # a²(1-2φ)
            
            # Set spatial components (in physical units)
            u_spatial_cart = direction_3d_normalized * c

            if direction_basis == "coordinates":
                u_spatial = u_spatial_cart
            elif direction_basis == "cartesian":
                # Interpret generated direction as cartesian (vx,vy,vz).
                # Convert ONLY if the coordinate system is truly spherical-like
                # (origin=[t,r,theta,phi]) rather than cartesian-like (origin=[eta,x,y,z]).
                if origin.shape[0] >= 4 and self._origin_looks_spherical(origin):
                    r0, theta0, phi0 = float(origin[1]), float(origin[2]), float(origin[3])
                    x0 = r0 * np.sin(theta0) * np.cos(phi0)
                    y0 = r0 * np.sin(theta0) * np.sin(phi0)
                    z0 = r0 * np.cos(theta0)
                    vr, vtheta, vphi = cartesian_velocity_to_spherical(
                        x0,
                        y0,
                        z0,
                        u_spatial_cart[0],
                        u_spatial_cart[1],
                        u_spatial_cart[2],
                    )
                    u_spatial = np.array([vr, vtheta, vphi], dtype=float)
                else:
                    # For cartesian-coordinate metrics like FLRW, keep (vx,vy,vz).
                    u_spatial = u_spatial_cart
            else:
                raise ValueError("direction_basis must be 'coordinates' or 'cartesian'")
            
            # Solve null condition for temporal component
            # g00 (u⁰)² + g11 (u¹)² + g22 (u²)² + g33 (u³)² = 0
            # (u⁰)² = -(g11 (u¹)² + g22 (u²)² + g33 (u³)²) / g00
            spatial_term = g11 * u_spatial[0]**2 + g22 * u_spatial[1]**2 + g33 * u_spatial[2]**2
            u0_squared = -spatial_term / g00
            
            # Choose positive temporal component for forward time evolution
            u0 = np.sqrt(abs(u0_squared))
            
            # Build 4-velocity (contravariant components)
            direction_4d = np.array([u0, u_spatial[0], u_spatial[1], u_spatial[2]])
            
            photon = Photon(origin.copy(), direction_4d)
            self.add_photon(photon)
            
            # Verify null condition
            if hasattr(photon, 'null_condition_relative_error'):
                rel_error = photon.null_condition_relative_error(metric=self.metric)
                if rel_error > 1e-5:
                    u_u = photon.null_condition(metric=self.metric)
                    print(f"Warning: Photon {i} null condition violated: u.u = {u_u:.3e}, rel_error = {rel_error:.3e}")

    
    def generate_cone_grid_metric_tensor(
        self,
        n_photons,
        origin,
        central_direction,
        cone_angle,
        direction_basis="coordinates",
        *,
        direction_coords=None,
    ):
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
        if type(n_photons) == int:
            n_phi = int(np.floor(np.sqrt(n_photons)))
            n_theta = int(np.floor(np.sqrt(n_photons)))
        else :
            n_phi = n_photons[1]
            n_theta = n_photons[0]

        # Backward-compatible alias:
        #   direction_coords='metric'    -> direction_basis='coordinates'
        #   direction_coords='cartesian' -> direction_basis='cartesian'
        if direction_coords is not None:
            if direction_coords == "metric":
                direction_basis = "coordinates"
            elif direction_coords == "cartesian":
                direction_basis = "cartesian"
            else:
                raise ValueError("direction_coords must be 'metric' or 'cartesian'")

        origin = np.asarray(origin, dtype=float)
        central_dir = np.asarray(central_direction, dtype=float)
        central_dir = central_dir / np.linalg.norm(central_dir)  # Normalize
                
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
                
                # Calculate temporal component using null condition
                # For FLRW + perturbations: ds² = -a²(1+2ψ)c²dη² + a²(1-2φ)(dx²+dy²+dz²)
                # Null condition: g_μν u^μ u^ν = 0
                
                # Get metric components at origin
                g_components = self.metric.metric_tensor(origin)
                g00 = g_components[0, 0] 
                g11 = g_components[1, 1]  
                g22 = g_components[2, 2]  
                g33 = g_components[3, 3]  
                
                # Set spatial components (in physical units)
                u_spatial_cart = direction_3d_normalized * c

                if direction_basis == "coordinates":
                    u_spatial = u_spatial_cart
                elif direction_basis == "cartesian":
                    # Interpret generated direction as cartesian (vx,vy,vz).
                    # Convert ONLY if the coordinate system is truly spherical-like
                    # (origin=[t,r,theta,phi]) rather than cartesian-like (origin=[eta,x,y,z]).
                    if origin.shape[0] >= 4 and self._origin_looks_spherical(origin):
                        r0, theta0, phi0 = float(origin[1]), float(origin[2]), float(origin[3])
                        x0 = r0 * np.sin(theta0) * np.cos(phi0)
                        y0 = r0 * np.sin(theta0) * np.sin(phi0)
                        z0 = r0 * np.cos(theta0)
                        vr, vtheta, vphi = cartesian_velocity_to_spherical(
                            x0,
                            y0,
                            z0,
                            u_spatial_cart[0],
                            u_spatial_cart[1],
                            u_spatial_cart[2],
                        )
                        u_spatial = np.array([vr, vtheta, vphi], dtype=float)
                    else:
                        # For cartesian-coordinate metrics like FLRW, keep (vx,vy,vz).
                        u_spatial = u_spatial_cart
                else:
                    raise ValueError("direction_basis must be 'coordinates' or 'cartesian'")
                
                # Solve null condition for temporal component
                # g00 (u⁰)² + g11 (u¹)² + g22 (u²)² + g33 (u³)² = 0
                # (u⁰)² = -(g11 (u¹)² + g22 (u²)² + g33 (u³)²) / g00
                spatial_term = g11 * u_spatial[0]**2 + g22 * u_spatial[1]**2 + g33 * u_spatial[2]**2
                u0_squared = -spatial_term / g00
                
                # Choose negative temporal component for backward time evolution (ray tracing)
                u0 = -np.sqrt(abs(u0_squared))
                
                # Build 4-velocity (contravariant components)
                direction_4d = np.array([u0, u_spatial[0], u_spatial[1], u_spatial[2]])
                
                photon = Photon(origin.copy(), direction_4d)
                self.add_photon(photon)
                
                # Verify null condition
                if hasattr(photon, 'null_condition_relative_error'):
                    rel_error = photon.null_condition_relative_error(metric=self.metric)
                    if rel_error > 1e-5:
                        u_u = photon.null_condition(metric=self.metric)
                        print(f"Warning: Photon {photon_count} null condition violated: u.u = {u_u:.3e}, rel_error = {rel_error:.3e}")
                
                photon_count += 1

    def generate_cone_grid(
    self,
    n_photons,
    origin,
    central_direction,
    cone_angle,
    direction_basis="coordinates",
    *,
    direction_coords=None,
):
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
        if type(n_photons) == int:
            n_phi = int(np.floor(np.sqrt(n_photons)))
            n_theta = int(np.floor(np.sqrt(n_photons)))
        else :
            n_phi = n_photons[1]
            n_theta = n_photons[0]

        # Backward-compatible alias:
        #   direction_coords='metric'    -> direction_basis='coordinates'
        #   direction_coords='cartesian' -> direction_basis='cartesian'
        if direction_coords is not None:
            if direction_coords == "metric":
                direction_basis = "coordinates"
            elif direction_coords == "cartesian":
                direction_basis = "cartesian"
            else:
                raise ValueError("direction_coords must be 'metric' or 'cartesian'")

        origin = np.asarray(origin, dtype=float)
        central_dir = np.asarray(central_direction, dtype=float)
        central_dir = central_dir / np.linalg.norm(central_dir)  # Normalize
                
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
                
                # Calculate temporal component using null condition
                # For FLRW + perturbations: ds² = -a²(1+2ψ)c²dη² + a²(1-2φ)(dx²+dy²+dz²)
                # Null condition: g_μν u^μ u^ν = 0
                
                # Spatial components in physical units
                u_spatial_cart = direction_3d_normalized * c

                if direction_basis == "coordinates":
                    u_spatial = u_spatial_cart
                elif direction_basis == "cartesian":
                    # Only convert (vx,vy,vz)->(vr,vtheta,vphi) if origin is really spherical coords
                    if origin.shape[0] >= 4 and self._origin_looks_spherical(origin):
                        r0, theta0, phi0 = float(origin[1]), float(origin[2]), float(origin[3])
                        x0 = r0 * np.sin(theta0) * np.cos(phi0)
                        y0 = r0 * np.sin(theta0) * np.sin(phi0)
                        z0 = r0 * np.cos(theta0)
                        vr, vtheta, vphi = cartesian_velocity_to_spherical(
                            x0, y0, z0,
                            u_spatial_cart[0], u_spatial_cart[1], u_spatial_cart[2]
                        )
                        u_spatial = np.array([vr, vtheta, vphi], dtype=float)
                    else:
                        u_spatial = u_spatial_cart
                else:
                    raise ValueError("direction_basis must be 'coordinates' or 'cartesian'")

                # Robust init for u0:
                # ray-tracing convention (backward time)
                if getattr(self.metric, "internal_coords", None) == "spherical" and self._origin_looks_spherical(origin):
                    g = self.metric.metric_tensor(origin)
                    g00, g11, g22, g33 = g[0,0], g[1,1], g[2,2], g[3,3]
                    spatial_term = g11*u_spatial[0]**2 + g22*u_spatial[1]**2 + g33*u_spatial[2]**2
                    u0 = -np.sqrt(abs(-spatial_term / g00))
                else:
                    u0 = -1.0
                    
                direction_4d = np.array([u0, u_spatial[0], u_spatial[1], u_spatial[2]], dtype=float)

                
                # Build 4-velocity (contravariant components)
                direction_4d = np.array([u0, u_spatial[0], u_spatial[1], u_spatial[2]])
                
                photon = Photon(origin.copy(), direction_4d)
                self.add_photon(photon)
                
                # Verify null condition
                if hasattr(photon, 'null_condition_relative_error'):
                    rel_error = photon.null_condition_relative_error(metric=self.metric)
                    if rel_error > 1e-5:
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

    @staticmethod
    def _origin_looks_spherical(origin: np.ndarray) -> bool:
        """Heuristic to disambiguate spherical [t,r,theta,phi] from cartesian [eta,x,y,z].

        We only need this when direction_basis='cartesian':
          - For Schwarzschild (spherical), we must convert (vx,vy,vz) -> (vr,vtheta,vphi).
          - For FLRW (cartesian), we must NOT convert.

        This avoids accidental conversion when x/y/z happen to be finite numbers.
        """
        if origin.shape[0] < 4:
            return False
        r0, theta0, phi0 = float(origin[1]), float(origin[2]), float(origin[3])
        if not np.isfinite(r0) or not np.isfinite(theta0) or not np.isfinite(phi0):
            return False
        if r0 <= 0:
            return False
        # theta in [0,pi] (with a little slack)
        if not (-1e-12 <= theta0 <= np.pi + 1e-12):
            return False
        # phi is typically in [-pi,pi] or [0,2pi); accept both ranges.
        if not (-2 * np.pi - 1e-12 <= phi0 <= 2 * np.pi + 1e-12):
            return False
        return True
    
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
        if h5py is None:
            raise RuntimeError("h5py is not available; cannot save photon histories. Install h5py or skip saving.")

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

        if h5py is None:
            raise RuntimeError("h5py is not available; cannot load photon histories. Install h5py or skip loading.")

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
