# photon/photon.py
import numpy as np
from .photon_history import PhotonHistory

class Photon:
    """Objet représentant l’état instantané d’un photon."""
    def __init__(self, position, direction, weight=1.0):
        self.x = np.asarray(position, dtype=float)
        self.u = np.asarray(direction, dtype=float)
        self.quantities = np.array([])
        self.weight = weight
        self.history = PhotonHistory()

    def null_condition(self, metric):
        """
        Compute g_μν u^μ u^ν for the photon.
        Should be zero (or very small) for null geodesics.
        
        Note: In SI units with FLRW metric, g_μν ~ O(a²c²) ~ O(10^52) at a~1,
        so even with relative errors of 10^-6, absolute values can be large.
        Use relative error instead: |g_μν u^μ u^ν| / |g_μμ u^μ u^μ|
        """
        g = metric.metric_tensor(self.x)
        norm = np.dot(self.u, np.dot(g, self.u))
        return norm
    
    def photon_norm(self, metric):
        r"""
        Scale-free diagnostic for how close the photon is to being null.

        Historically this returned ``sqrt(|g_μν u^μ u^ν|)`` (an absolute quantity).
        In SI units this absolute value can be *huge* even when the null condition is
        satisfied at machine precision (because individual terms are ~O(r^2 c^2)).

        This method therefore returns the absolute *relative* null-condition error,
        which should be \~0 for a valid null geodesic.
    """
        return self.null_condition_relative_error(metric)

    def photon_norm_abs(self, metric):
        r"""Absolute $\sqrt{|g_{\mu\nu}u^\mu u^\nu|}$ (mostly for debugging)."""
        g = metric.metric_tensor(self.x)
        norm = np.einsum('ij,i,j->', g, self.u, self.u)
        return np.sqrt(abs(norm))
    
    def null_condition_relative_error(self, metric, *, debug: bool = False):
        """
        Compute the relative error in the null condition.
        
        Returns |g_μν u^μ u^ν| / (|g₀₀ (u⁰)²| + |g_ii (u^i)²|)
        This should be << 1 for a valid null geodesic.
        """
        g = metric.metric_tensor(self.x)
        norm = np.dot(self.u, np.dot(g, self.u))
        
        g00_term = abs(g[0,0] * self.u[0]**2)
        g_spatial_terms = sum(abs(g[i,i] * self.u[i]**2) for i in range(1, 4))
        normalization = g00_term + g_spatial_terms
        
        if normalization > 0:
            result = abs(norm) / normalization
            if debug and result < 1e-18:
                print("Extremely small relative null condition error:")
                print(abs(norm), normalization)
            return result

        else:
            return abs(norm)

    def state_quantities(self,relevant_quantities):
        self.quantities = relevant_quantities(self.x)

    @property
    def state(self):
        return np.concatenate([self.x, self.u, self.quantities])

    def record(self):
        self.history.append(self.state)
