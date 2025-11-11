import numpy as np
from .base_metric import Metric
from excalibur.core.constants import *


class SchwarzschildMetricCartesian(Metric):
    """
    Schwarzschild metric in ISOTROPIC Cartesian coordinates (t, x, y, z).
    
    In isotropic coordinates, the metric is diagonal:
    ds² = -A²(r) c² dt² + B⁴(r) (dx² + dy² + dz²)
    
    where:
    - r = |position - center| (isotropic radial coordinate)
    - r_s = 2GM/c² (Schwarzschild radius)
    - ρ = r_s/(4r)
    - A(r) = (1 - ρ)/(1 + ρ) = (1 - r_s/4r)/(1 + r_s/4r)
    - B(r) = 1 + ρ = 1 + r_s/(4r)
    
    This gives:
    g_00 = -A²(r) c² = -[(1 - r_s/4r)/(1 + r_s/4r)]² c²
    g_ij = B⁴(r) δ_ij = [1 + r_s/(4r)]⁴ δ_ij
    
    The metric is DIAGONAL in Cartesian coordinates, making it much simpler!
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
            4x4 DIAGONAL metric tensor g_μν in isotropic Cartesian coordinates
        """
        t = x[0]
        pos = x[1:4]
        
        # Position relative to center (isotropic radial coordinate)
        r_vec = pos - self.center
        r = np.linalg.norm(r_vec)
        
        # Avoid singularity at r=0
        if r < 1e-10:
            r = 1e-10
            
        # Isotropic coordinate functions
        # ρ = r_s/(4r) is the dimensionless parameter
        rho = self.r_s / (4.0 * r)
        
        # A(r) = (1 - ρ)/(1 + ρ)
        A = (1.0 - rho) / (1.0 + rho)
        
        # B(r) = 1 + ρ  
        B = 1.0 + rho
        
        # Build DIAGONAL metric tensor
        g = np.zeros((4, 4))
        
        # Time-time component: g_00 = -A² c²
        g[0, 0] = -A**2 * c**2
        
        # Spatial components: g_ij = B⁴ δ_ij (DIAGONAL!)
        B4 = B**4
        g[1, 1] = B4
        g[2, 2] = B4
        g[3, 3] = B4
                
        return g
    
    def christoffel(self, x):
        """
        Compute Christoffel symbols in isotropic Cartesian coordinates.
        
        With diagonal metric g_00 = -A²c², g_ij = B⁴δ_ij, the non-zero components are:
        Γ^0_0i = Γ^0_i0 = (1/A) dA/dx^i = (dA/dr) (x_i/r) / A
        Γ^i_00 = (c²A/B⁴) dA/dx^i = (c²A/B⁴) (dA/dr) (x_i/r)
        Γ^i_jj = (1/B⁴) dB⁴/dx^i = (4/B) dB/dx^i = (4/B) (dB/dr) (x_i/r) for j≠i (no sum)
        Γ^i_ji = (1/B⁴) dB⁴/dx^j = (4/B) (dB/dr) (x_j/r) for j≠i
        Γ^i_ii = (1/B⁴) dB⁴/dx^i = (4/B) (dB/dr) (x_i/r) (extra factor from diagonal)
        
        Actually simpler: For diagonal metric, use
        Γ^μ_αβ = (1/2) g^μμ [∂_α g_μβ + ∂_β g_μα - ∂_μ g_αβ] (no sum on μ)
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
        # dρ/dr = -r_s/(4r²)
        drho_dr = -self.r_s / (4.0 * r**2)
        
        # dA/dr = d/dr[(1-ρ)/(1+ρ)] = -2/(1+ρ)² × dρ/dr
        dA_dr = -2.0 / (1.0 + rho)**2 * drho_dr
        
        # dB/dr = dρ/dr
        dB_dr = drho_dr
        
        # Unit radial vector
        n = r_vec / r
        
        # Initialize Christoffel symbols
        Γ = np.zeros((4, 4, 4))
        
        # Γ^0_0i = Γ^0_i0 = (1/A) dA/dr × (x_i/r)
        for i in range(3):
            Γ[0, 0, i+1] = (dA_dr / A) * n[i]
            Γ[0, i+1, 0] = Γ[0, 0, i+1]
            
        # Γ^i_00 = (c²A/B⁴) dA/dr × (x_i/r)
        B4 = B**4
        for i in range(3):
            Γ[i+1, 0, 0] = (c**2 * A / B4) * dA_dr * n[i]
            
        # Spatial Christoffel symbols
        # For diagonal metric: Γ^i_jk = (1/g_ii) × (1/2)[∂_j g_ik + ∂_k g_ij - ∂_i g_jk]
        # With g_ii = B⁴ (same for all i) and g_ij = 0 for i≠j:
        # Γ^i_jk = (4/B) dB/dr × n[i] × δ_jk  (if j=k)
        #        - (4/B) dB/dr × n[j] × δ_ik  (if i=k)
        #        - (4/B) dB/dr × n[k] × δ_ij  (if i=j)
        
        # Actually, for diagonal isotropic metric with g_ij = B⁴ δ_ij:
        # Γ^i_jk = (2/B) dB/dr × [n[i]δ_jk + n[j]δ_ik + n[k]δ_ij - 2n[i]δ_jk]
        #        = (2/B) dB/dr × [n[j]δ_ik + n[k]δ_ij - n[i]δ_jk]
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    term = 0.0
                    if j == k:
                        term -= n[i]  # -n[i]δ_jk
                    if i == k:
                        term += n[j]  # +n[j]δ_ik  
                    if i == j:
                        term += n[k]  # +n[k]δ_ij
                    
                    Γ[i+1, j+1, k+1] = (2.0 / B) * dB_dr * term
                        
        return Γ
    
    def geodesic_equations(self, state):
        """
        Compute d/dλ (x^μ, u^μ) = (u^μ, du^μ/dλ)
        
        where du^μ/dλ = -Γ^μ_αβ u^α u^β
        """
        x = state[:4]
        u = state[4:]
        
        # Get Christoffel symbols
        Γ = self.christoffel(x)
        
        # Compute acceleration
        du = np.zeros(4)
        for μ in range(4):
            for α in range(4):
                for β in range(4):
                    du[μ] -= Γ[μ, α, β] * u[α] * u[β]
                    
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
