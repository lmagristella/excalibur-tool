import numpy as np
from .base_metric import Metric
from excalibur.core.constants import *


class SchwarzschildMetricCartesian(Metric):
    """
    Schwarzschild metric in ISOTROPIC Cartesian coordinates (t, x, y, z).
    
    In isotropic coordinates, the metric is diagonal:
    ds^2 = -A^2(r) c^2 dt^2 + B^4(r) (dx^2 + dy^2 + dz^2)

    where:
    - r = |position - center| (isotropic radial coordinate)
    - r_s = 2GM/c^2 (Schwarzschild radius)
    - rho = r_s/(4r)
    - A(r) = (1 - rho)/(1 + rho) = (1 - r_s/4r)/(1 + r_s/4r)
    - B(r) = 1 + rho = 1 + r_s/(4r)
    
    This gives:
    g_00 = -A^2(r) c^2 = -[(1 - r_s/4r)/(1 + r_s/4r)]^2 c^2
    g_ij = B^4(r) delta_ij = [1 + r_s/(4r)]^4 delta_ij
    
    The link between radial coordinates and isotropic radius is:
    r_spherical = r * (1 + r_s/(4r))^2
    """
    
    def __init__(self, mass, radius, center):
        """
        Args:
            mass: Total mass in kg
            radius: Characteristic radius (for potential cutoff) in meters
            center: 3D position of mass center in meters (array-like)
        """
        self.mass = mass
        self.radius = radius
        self.center = np.asarray(center, dtype=float)
        self.r_s = 2 * G * mass / c**2  # Schwarzschild radius
        
    def metric_tensor(self, x):
        """
        Compute metric tensor at spacetime position x = [t, x, y, z].
        Uses isotropic coordinates for diagonal form.
        
        Returns:
            4x4 DIAGONAL metric tensor g_mu_nu in isotropic Cartesian coordinates
        """

        if len(x) == 4:
            t = x[0]
            pos = x[1:4]
        else:
            pos = x
        
        # Position relative to center (isotropic radial coordinate)
        r_vec = pos - self.center
        r = np.linalg.norm(r_vec)
        
        # Avoid singularity at r=0
        if r < 1e-10:
            r = 1e-10
            
        # Isotropic coordinate functions
        # rho = r_s/(4r) is the dimensionless parameter
        rho = self.r_s / (4.0 * r)
        
        # A(r) = (1 - rho)/(1 + rho)
        A = (1.0 - rho) / (1.0 + rho)
        
        # B(r) = 1 + rho  
        B = 1.0 + rho
        
        # Build DIAGONAL metric tensor
        g = np.zeros((4, 4))
        
        # Time-time component: g_00 = -A^2 c^2
        g[0, 0] = -A**2 * c**2
        
        # Spatial components: g_ij = B^4 delta_ij (DIAGONAL!)
        B4 = B**4
        g[1, 1] = B4
        g[2, 2] = B4
        g[3, 3] = B4
                
        return g
    
    def christoffel(self, x):
        """
        Compute Christoffel symbols in isotropic Cartesian coordinates.
        
        With diagonal metric g_00 = -A^2c^2, g_ij = B^4delta_ij, the non-zero components are:
        Gamma^0_0i = Gamma^0_i0 = (1/A) dA/dx^i = (dA/dr) (x_i/r) / A
        Gamma^i_00 = (c^2A/B^4) dA/dx^i = (c^2A/B^4) (dA/dr) (x_i/r)
        Gamma^i_jj = (1/B^4) dB^4/dx^i = (4/B) dB/dx^i = (4/B) (dB/dr) (x_i/r) for j!=i (no sum)
        Gamma^i_ji = (1/B^4) dB^4/dx^j = (4/B) (dB/dr) (x_j/r) for j!=i
        Gamma^i_ii = (1/B^4) dB^4/dx^i = (4/B) (dB/dr) (x_i/r) (extra factor from diagonal)
        
        Actually simpler: For diagonal metric, use
        Gamma^mu_alphabeta = (1/2) g^mumu [d_alpha g_mubeta + d_beta g_mualpha - d_mu g_alphabeta] (no sum on mu)
        """
        t = x[0]
        pos = x[1:4]
        
        # Position relative to center
        r_vec = pos - self.center
        r = np.linalg.norm(r_vec)
        
        # Avoid singularity
        if r < 1e-10:
            r = 1e-10
            r_vec = np.array([1e-10, 0, 0])
            
        # Check if inside Schwarzschild radius
        if r <= self.r_s:
            raise ValueError(f"Inside Schwarzschild radius: r={r/1e3:.2e} km, r_s={self.r_s/1e3:.2e} km")
            
        # Isotropic parameters
        rho = self.r_s / (4.0 * r)
        A = (1.0 - rho) / (1.0 + rho)
        B = 1.0 + rho
        
        # Derivatives with respect to r
        # drho/dr = -r_s/(4r^2)
        drho_dr = -self.r_s / (4.0 * r**2)
        
        # dA/dr = d/dr[(1-rho)/(1+rho)] = -2/(1+rho)^2 x drho/dr
        dA_dr = -2.0 / (1.0 + rho)**2 * drho_dr
        
        # dB/dr = drho/dr
        dB_dr = drho_dr
        
        # Unit radial vector
        n = r_vec / r
        
        # Initialize Christoffel symbols
        Gamma = np.zeros((4, 4, 4))
        
        # Gamma^0_0i = Gamma^0_i0 = (1/A) dA/dr x (x_i/r)
        for i in range(3):
            Gamma[0, 0, i+1] = (dA_dr / A) * n[i]
            Gamma[0, i+1, 0] = Gamma[0, 0, i+1]
            
        # Gamma^i_00 = (c^2A/B^4) dA/dr x (x_i/r)
        B4 = B**4
        for i in range(3):
            Gamma[i+1, 0, 0] = (c**2 * A / B4) * dA_dr * n[i]
            
        # Spatial Christoffel symbols
        # For diagonal metric: Gamma^i_jk = (1/g_ii) x (1/2)[d_j g_ik + d_k g_ij - d_i g_jk]
        # With g_ii = B^4 (same for all i) and g_ij = 0 for i!=j:
        # Gamma^i_jk = (4/B) dB/dr x n[i] x delta_jk  (if j=k)
        #        - (4/B) dB/dr x n[j] x delta_ik  (if i=k)
        #        - (4/B) dB/dr x n[k] x delta_ij  (if i=j)
        
        # Actually, for diagonal isotropic metric with g_ij = B^4 delta_ij:
        # Gamma^i_jk = (2/B) dB/dr x [n[i]delta_jk + n[j]delta_ik + n[k]delta_ij - 2n[i]delta_jk]
        #        = (2/B) dB/dr x [n[j]delta_ik + n[k]delta_ij - n[i]delta_jk]
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    term = 0.0
                    if j == k:
                        term -= n[i]  # -n[i]delta_jk
                    if i == k:
                        term += n[j]  # +n[j]delta_ik  
                    if i == j:
                        term += n[k]  # +n[k]delta_ij
                    
                    Gamma[i+1, j+1, k+1] = (2.0 / B) * dB_dr * term
                        
        return Gamma
    
    def geodesic_equations(self, state):
        """
        Compute d/dlambda (x^mu, u^mu) = (u^mu, du^mu/dlambda)
        
        where du^mu/dlambda = -Gamma^mu_alphabeta u^alpha u^beta
        """
        x = state[:4]
        u = state[4:]
        
        # Get Christoffel symbols
        Gamma = self.christoffel(x)
        
        # Compute acceleration
        du = np.zeros(4)
        for mu in range(4):
            for alpha in range(4):
                for beta in range(4):
                    du[mu] -= Gamma[mu, alpha, beta] * u[alpha] * u[beta]
                    
        return np.concatenate([u, du])
    
    def metric_physical_quantities(self, state):
        """
        Return relevant physical quantities at given state.
        
        Returns: [r, mass, radius, r_s]
        """
        t = state[0]
        pos = state[1:4]
        
        r_vec = pos - self.center
        r = np.linalg.norm(r_vec)
        
        return np.array([r, self.mass, self.radius, self.r_s])
