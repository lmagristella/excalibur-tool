# metrics/base_metric.py
from abc import ABC, abstractmethod
import numpy as np

class Metric(ABC):

    @abstractmethod
    def metric_tensor(self, x):
        """Retourne le tenseur métrique évalué à x"""
        pass

    @abstractmethod
    def christoffel(self, x):
        """Retourne Γ^μ_{νσ} sous forme de tenseur [μ,ν,σ]."""
        pass

    @abstractmethod
    def geodesic_equations(self, state):
        """Retourne d/dλ [x^μ, u^μ]."""
        pass

    @abstractmethod
    def metric_physical_quantities(self, state):
        """Prend en entrée un état de photon, et renvoie une liste des quantités physiques intéressantes propres à la métrique"""
        pass