import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from excalibur.core.constants import *
from numba import njit


@njit(cache=True, fastmath=True)
def a_interp_numba(eta, eta_min, eta_step, a_grid):
    """
    linear interpolation on a regular η grid.
    """
    # clamp to bounds
    if eta <= eta_min:
        return a_grid[0]
    max_eta = eta_min + eta_step * (len(a_grid)-1)
    if eta >= max_eta:
        return a_grid[-1]

    # compute index = (η−η0)/Δη
    i = int((eta - eta_min) / eta_step)

    # fractional part
    t = (eta - (eta_min + i*eta_step)) / eta_step
    return (1-t)*a_grid[i] + t*a_grid[i+1]


@njit(cache=True, fastmath=True)
def a_interp_numba_vec(etas, eta_min, eta_step, a_grid):
    """
    Vectorized wrapper over a_interp_numba for 1D arrays of eta.
    """
    out = np.empty_like(etas)
    for i in range(etas.size):
        out[i] = a_interp_numba(etas[i], eta_min, eta_step, a_grid)
    return out


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
        self.Omega_k = Omega_k
        self._build_eta_to_a_interpolator()
        self._build_eta_grid_fast()
        


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

    def adot_of_eta(self, eta):
        """
        PERFORMANCE OPTIMIZATION: Analytical da/dt calculation.
        Much faster than numerical differentiation: da/dt = a * H(t)
        In conformal time: da/deta = a^2 * H(t)
        """
        a = self.a_of_eta(eta)
        z = 1.0 / a - 1.0
        H = self.H_of_z(z)
        return a * a * H  # da/deta in conformal time

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
    
    def _build_eta_grid_fast(self):
        """
        Build a regular η-grid and store corresponding a(η) values
        for ultra-fast numba interpolation.
        """
        # a grid for computing eta(a)
        a_values = np.logspace(-4, 0.5, 2000)  # high-resolution a-grid
        eta_values = np.zeros_like(a_values)

        def integrand(a):
            return 1.0 / (a*a * self.H0 * self.E(1/a - 1))

        # compute eta(a)
        for i, a in enumerate(a_values):
            val, _ = quad(integrand, a_values[0], a)
            eta_values[i] = val

        self._a_base = a_values
        self._eta_base = eta_values

        # now build a **regular eta-grid**
        self.eta_min = float(eta_values[0])
        self.eta_max = float(eta_values[-1])
        self.N_eta = 5000  # number of points — you can increase

        self.eta_grid = np.linspace(self.eta_min, self.eta_max, self.N_eta)
        self.a_grid = np.interp(self.eta_grid, eta_values, a_values)

        # precompute spacing for O(1) lookup
        self.eta_step = (self.eta_max - self.eta_min) / (self.N_eta - 1)



    def _build_eta_to_a_interpolator(self):
        """Precompute the η(a) table and build the interpolator
        so that a_of_eta() is extremely fast.
        """
        # Build lookup table only once
        a_values = np.logspace(-4, 0.5, 500)  # From a=1e-4 to a≈3
        eta_values = np.zeros_like(a_values)

        def integrand(a_val):
            return 1.0 / (a_val**2 * self.H0 * self.E(1.0/a_val - 1.0))

        # Compute η(a)
        for i, a_val in enumerate(a_values):
            val, _ = quad(integrand, 1e-6, a_val)
            eta_values[i] = val

        # Save arrays
        self._a_values_base = a_values
        self._eta_values_base = eta_values

        # Present conformal time
        idx = np.argmin(np.abs(a_values - 1.0))
        self._eta_at_a1 = eta_values[idx]

        # Build the interpolator and store it
        self._interp_eta_to_a = interp1d(
            eta_values,
            a_values,
            kind='cubic',
            fill_value='extrapolate',
            bounds_error=False
        )

    # ------------------------------------------------------------------
    # Public API expected across the codebase
    # ------------------------------------------------------------------
    def a_of_eta(self, eta, eta_present_seconds=None):
        if eta_present_seconds is not None:
            return self.a_of_eta_old(eta, eta_present_seconds=eta_present_seconds)

        if isinstance(eta, np.ndarray):
            return a_interp_numba_vec(eta, self.eta_min, self.eta_step, self.a_grid)
        else:
            return a_interp_numba(float(eta), self.eta_min, self.eta_step, self.a_grid)



    def a_of_eta_old(self, eta, eta_present_seconds=None):
        """Fast version: uses a precomputed interpolator."""
        
        # Default normalization (physical)
        if eta_present_seconds is None:
            interp = self._interp_eta_to_a
            return interp(eta)

        # If user wants custom normalization:
        # build a *shifted* interpolator lazily (rare case)
        eta_shift = eta_present_seconds - self._eta_at_a1
        if not hasattr(self, "_interp_shifted") or self._last_eta_shift != eta_shift:

            eta_shifted = self._eta_values_base + eta_shift
            self._interp_shifted = interp1d(
                eta_shifted,
                self._a_values_base,
                kind='cubic',
                fill_value='extrapolate',
                bounds_error=False
            )
            self._last_eta_shift = eta_shift

        return self._interp_shifted(eta)


class StaticCosmology:
    r"""A minimal cosmology model with constant scale factor.

    This is useful to run "FLRW"-style metrics in a static limit, i.e.
    $a(\eta)=1$ and $\dot a(\eta)=0$.

    The rest of the codebase expects a cosmology-like object exposing
    ``a_of_eta(eta)`` and ``adot_of_eta(eta)``.
    """

    def a_of_eta(self, eta):
        if isinstance(eta, np.ndarray):
            return np.ones_like(eta, dtype=float)
        return 1.0

    def adot_of_eta(self, eta):
        if isinstance(eta, np.ndarray):
            return np.zeros_like(eta, dtype=float)
        return 0.0

    def a_and_adot(self, eta):
        return self.a_of_eta(eta), self.adot_of_eta(eta)

    
