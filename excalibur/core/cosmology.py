import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from excalibur.core.constants import *
from numba import njit


@njit(cache=True, fastmath=True)
def a_interp_numba(eta, eta_min, eta_step, a_grid):
    """
    linear interpolation on a regular eta grid.
    """
    # clamp to bounds
    if eta <= eta_min:
        return a_grid[0]
    max_eta = eta_min + eta_step * (len(a_grid)-1)
    if eta >= max_eta:
        return a_grid[-1]

    # compute index = (eta-eta0)/Deltaeta
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
        """Fonction de Hubble normalisee."""
        return np.sqrt(self.Omega_m * (1 + z)**3 + self.Omega_r * (1 + z)**4 + self.Omega_k * (1 + z)**2 + self.Omega_lambda)
    
    
    def H_of_z(self, z):
        """Calcule le parametre de Hubble H a partir du decalage vers le rouge z."""
        return self.H0 * self.E(z)
    
    def H_of_eta(self, eta):
        """Calcule le parametre de Hubble H a partir du temps conforme eta."""
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

    def conformal_hubble(self, eta):
        r"""Conformal Hubble parameter  H = a'/a = a H(t)  in s^{-1}."""
        a = self.a_of_eta(eta)
        z = 1.0 / a - 1.0
        return a * self.H_of_z(z)

    def conformal_hubble_prime(self, eta):
        r"""Conformal-time derivative of the conformal Hubble parameter.

        H' = dH/d\eta  where  H = a'/a  (conformal Hubble).

        Derivation
        ----------
        H = a H_phys, so

            H' = a^2 H_phys^2  +  a^2 \dot{H}_phys

        where  \dot{H}_phys = H_0^2/2 (-3 Omega_m/a^3 - 4 Omega_r/a^4
                                        - 2 Omega_k/a^2).

        Combining with  H^2 = H_0^2 (Omega_m/a + Omega_r/a^2
                                      + Omega_Lambda a^2 + Omega_k):

            H' = H_0^2 ( -Omega_m/(2a) - Omega_r/a^2
                          + Omega_Lambda a^2 )

        This is an *exact* algebraic expression  --  no numerical derivative.
        Units: s^{-2} (when H_0 is in s^{-1}).
        """
        a = self.a_of_eta(eta)
        H0sq = self.H0 * self.H0
        return H0sq * (-0.5 * self.Omega_m / a
                       - self.Omega_r / (a * a)
                       + self.Omega_lambda * a * a)

    # ------------------------------------------------------------------
    #  Distance measures
    # ------------------------------------------------------------------

    def comoving_distance(self, z):
        r"""Comoving distance (line-of-sight) to redshift *z*.

        .. math::
            \chi(z) = \frac{c}{H_0} \int_0^z \frac{\mathrm{d}z'}{E(z')}

        Parameters
        ----------
        z : float or array_like
            Redshift(s).

        Returns
        -------
        chi : float or ndarray
            Comoving distance in **metres** (SI) or the active unit system.
        """
        z = np.asarray(z, dtype=float)
        scalar = z.ndim == 0
        z = np.atleast_1d(z)

        from excalibur.core.constants import c as c_light

        chi = np.empty_like(z)
        for i, zi in enumerate(z):
            val, _ = quad(lambda zp: 1.0 / self.E(zp), 0.0, zi)
            chi[i] = (c_light / self.H0) * val

        return chi.item() if scalar else chi

    def angular_diameter_distance(self, z):
        r"""Angular diameter distance to redshift *z* (background FLRW).

        .. math::
            D_A(z) = \frac{\chi(z)}{1 + z}

        For flat cosmology (:math:`\Omega_k = 0`).

        Returns
        -------
        D_A : float or ndarray
            Angular diameter distance in the active unit system (metres for SI).
        """
        return self.comoving_distance(z) / (1.0 + np.asarray(z, dtype=float))

    def luminosity_distance(self, z):
        r"""Luminosity distance to redshift *z*.

        .. math::
            D_L(z) = (1 + z)\,\chi(z)

        Returns
        -------
        D_L : float or ndarray
            Luminosity distance in the active unit system.
        """
        return self.comoving_distance(z) * (1.0 + np.asarray(z, dtype=float))

    def angular_diameter_distance_z1z2(self, z1, z2):
        r"""Angular diameter distance between two redshifts (flat cosmology).

        .. math::
            D_A(z_1, z_2) = \frac{\chi(z_2) - \chi(z_1)}{1 + z_2}

        Valid only for :math:`\Omega_k = 0` and :math:`z_2 > z_1`.

        Returns
        -------
        D_A_12 : float or ndarray
        """
        return (self.comoving_distance(z2) - self.comoving_distance(z1)) / (1.0 + np.asarray(z2, dtype=float))

    # ------------------------------------------------------------------

    def a_of_z(self, z):
        """Calcule le facteur d'echelle a a partir du decalage vers le rouge z."""
        return 1.0 / (1.0 + z)
    
    def a_of_t(self, t):
        """Calcule le facteur d'echelle a a partir du temps cosmologique t (approximation numerique)."""
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
            # Recherche numerique de a pour lequel time_to_a(a) = t_val
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
        Build a regular eta-grid and store corresponding a(eta) values
        for ultra-fast numba interpolation.

        IMPORTANT: The integration lower bound MUST match
        ``_build_eta_to_a_interpolator`` (1e-6) so that the two eta
        tables share the same zero-point.  A mismatch here causes a
        systematic offset in a(eta)  --  see Bug #1 diagnosis.
        """
        # a grid for computing eta(a)
        a_values = np.logspace(-4, 0.5, 2000)  # high-resolution a-grid
        eta_values = np.zeros_like(a_values)

        def integrand(a):
            return 1.0 / (a*a * self.H0 * self.E(1/a - 1))

        # compute eta(a)  --  lower bound = 1e-6 (same as _build_eta_to_a_interpolator)
        for i, a in enumerate(a_values):
            val, _ = quad(integrand, 1e-6, a)
            eta_values[i] = val

        self._a_base = a_values
        self._eta_base = eta_values

        # now build a **regular eta-grid**
        self.eta_min = float(eta_values[0])
        self.eta_max = float(eta_values[-1])
        self.N_eta = 5000  # number of points  --  you can increase

        self.eta_grid = np.linspace(self.eta_min, self.eta_max, self.N_eta)
        self.a_grid = np.interp(self.eta_grid, eta_values, a_values)

        # precompute spacing for O(1) lookup
        self.eta_step = (self.eta_max - self.eta_min) / (self.N_eta - 1)

        # Compute _eta_at_a1 from the fast grid itself (authoritative source)
        idx_a1 = int(np.argmin(np.abs(self.a_grid - 1.0)))
        self._eta_at_a1 = float(self.eta_grid[idx_a1])



    def _build_eta_to_a_interpolator(self):
        """Precompute the eta(a) table and build the interpolator
        so that a_of_eta() is extremely fast.
        """
        # Build lookup table only once
        a_values = np.logspace(-4, 0.5, 500)  # From a=1e-4 to a~3
        eta_values = np.zeros_like(a_values)

        def integrand(a_val):
            return 1.0 / (a_val**2 * self.H0 * self.E(1.0/a_val - 1.0))

        # Compute eta(a)
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

    def conformal_hubble(self, eta):
        return 0.0

    def conformal_hubble_prime(self, eta):
        return 0.0
