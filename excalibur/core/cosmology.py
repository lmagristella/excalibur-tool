import numpy as np
from scipy.integrate import quad
from excalibur.core.constants import *


class LCDM_Cosmology:
    def __init__(self, H0, Omega_m, Omega_r, Omega_lambda, Omega_k = 0):
        """
        Initialize LCDM cosmology.
        
        Parameters:
        -----------
        H0 : float
            Hubble constant in km/s/Mpc (will be converted to SI units internally)
        Omega_m : float
            Matter density parameter
        Omega_r : float
            Radiation density parameter
        Omega_lambda : float
            Dark energy density parameter
        Omega_k : float, optional
            Curvature parameter (default: 0)
        """
        # Convert H0 from km/s/Mpc to SI units (s^-1)
        self.H0_original = H0  # Store original value in km/s/Mpc for reference
        self.H0 = H0 * 1000 / (1e6 * one_pc)  # Convert to s^-1
        self.Omega_m = Omega_m
        self.Omega_lambda = Omega_lambda
        self.Omega_r = Omega_r
        self.Omega_k = 0


    def E(self, z):
        """Fonction de Hubble normalisée."""
        return np.sqrt(self.Omega_m * (1 + z)**3 + self.Omega_r * (1 + z)**4 + self.Omega_k * (1 + z)**2 + self.Omega_lambda)
    
    
    def H_of_z(self, z):
        """Calcule le paramètre de Hubble H à partir du décalage vers le rouge z."""
        return self.H0 * self.E(z)
    
    def H_of_eta(self, eta):
        """Calcule le paramètre de Hubble H à partir du temps conforme η."""
        a = self.a_of_eta(eta)
        z = 1.0 / a - 1.0
        return self.H_of_z(z) / a**2

    def a_of_z(self, z):
        """Calcule le facteur d'échelle a à partir du décalage vers le rouge z."""
        return 1.0 / (1.0 + z)
    
    def a_of_t(self, t):
        """Calcule le facteur d'échelle a à partir du temps cosmologique t (approximation numérique)."""
        def integrand(a):
            return 1.0 / (a * self.E(1.0/a - 1.0))
        
        integral, _ = quad(integrand, 0, 1)
        t0 = integral / self.H0
        
        def time_to_a(a):
            integral, _ = quad(integrand, 0, a)
            return integral / self.H0
        
        # Handle both scalar and array inputs
        t = np.asarray(t)
        scalar_input = t.ndim == 0
        t = np.atleast_1d(t)
        
        a_result = np.zeros_like(t)
        
        for i, t_val in enumerate(t):
            # Recherche numérique de a pour lequel time_to_a(a) = t_val
            a_guess = 1.0
            for _ in range(100):
                t_guess = time_to_a(a_guess)
                if abs(t_guess - t_val) < 1e-6:
                    break
                a_guess *= t_val / t_guess
            a_result[i] = a_guess
        
        return a_result.item() if scalar_input else a_result
    
    def a_of_eta(self, eta, eta_present_seconds=None):
        """Calcule le facteur d'échelle a à partir du temps conforme η (approximation numérique).
        
        Le temps conforme est défini par: dη = dt/a = da/(a²H)
        où H = H0·E(z) = H0·E(1/a - 1)
        
        Donc: η(a) = ∫[0 to a] da'/(a'²·H0·E(1/a' - 1))
        
        Pour ΛCDM avec H0=70 km/s/Mpc, on a η(a=1) ≈ 46 milliards d'années ≈ 1.46e18 s.
        
        Parameters:
        -----------
        eta : float or array
            Conformal time in seconds
        eta_present_seconds : float, optional
            Present conformal time in seconds. If None, will use the physical value
            computed from the cosmology (η at a=1). This is typically ~1.46e18 s 
            (46 Gyr) for ΛCDM.
            
        Returns:
        --------
        a : float or array
            Scale factor(s)
        """
        from scipy.interpolate import interp1d
        
        # Build lookup table if not cached
        if not hasattr(self, '_eta_to_a_base_computed'):
            # Build lookup table: a_values -> eta_values
            a_values = np.logspace(-4, 0.5, 500)  # From a=0.0001 to a~3
            eta_values = np.zeros_like(a_values)
            
            def integrand(a_val):
                # Integrand for conformal time: 1 / (a² H0 E(z))
                # Note: dη = dt/a = da/(a²H), so η = ∫ da/(a²H)
                return 1.0 / (a_val**2 * self.H0 * self.E(1.0/a_val - 1.0))
            
            # Compute eta for each a value
            for i, a_val in enumerate(a_values):
                integral, _ = quad(integrand, 1e-6, a_val)
                eta_values[i] = integral
            
            # Store base arrays
            self._eta_values_base = eta_values
            self._a_values_base = a_values
            self._eta_to_a_base_computed = True
            
            # Find eta at a=1 (present day)
            idx_a1 = np.argmin(np.abs(a_values - 1.0))
            self._eta_at_a1 = eta_values[idx_a1]
        
        # Determine normalization
        if eta_present_seconds is None:
            # Use physical value: no shift needed, use base arrays
            eta_values_normalized = self._eta_values_base
        else:
            # Apply shift to make a(eta_present_seconds) = 1
            eta_shift = eta_present_seconds - self._eta_at_a1
            eta_values_normalized = self._eta_values_base + eta_shift
        
        # Create interpolator
        interp = interp1d(eta_values_normalized, self._a_values_base, 
                         kind='cubic', 
                         fill_value='extrapolate',
                         bounds_error=False)
        
        return interp(eta)

    def comoving_distance(self, z):
        """Calcule la distance comobile jusqu'à un décalage vers le rouge z."""
        integral, _ = quad(lambda zp: 1.0 / self.E(zp), 0, z)
        return (c / self.H0) * integral
    
