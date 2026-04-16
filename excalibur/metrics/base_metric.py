# metrics/base_metric.py
from abc import ABC, abstractmethod
import numpy as np

class Metric(ABC):

    @abstractmethod
    def metric_tensor(self, x):
        """Retourne le tenseur metrique evalue a x"""
        pass

    @abstractmethod
    def christoffel(self, x):
        """Retourne Gamma^mu_{nusigma} sous forme de tenseur [mu,nu,sigma]."""
        pass

    @abstractmethod
    def geodesic_equations(self, state):
        """Retourne d/dlambda [x^mu, u^mu]."""
        pass

    @abstractmethod
    def metric_physical_quantities(self, state):
        """Prend en entree un etat de photon, et renvoie une liste des quantites physiques interessantes propres a la metrique"""
        pass