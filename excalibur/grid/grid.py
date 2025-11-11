import numpy as np

class Grid:
    """
    Stocke les champs physiques (potentiels, densités...) sur une grille régulière.
    """
    def __init__(self, shape, spacing, origin=(0, 0, 0), cycle=False):
        self.shape = np.array(shape)
        self.spacing = np.array(spacing)
        self.origin = np.array(origin)
        self.fields = {}  # ex: {"Psi": array, "Phi": array}
        self.cycle = cycle

    def add_field(self, name, data):
        assert data.shape == tuple(self.shape), "Mauvaise dimension du champ"
        self.fields[name] = data

    def get_value(self, name, i, j, k):
        return self.fields[name][i, j, k]

    def physical_position(self, indices):
        return self.origin + indices * self.spacing
    
    def indices_from_position(self, position):
        relative_pos = (np.array(position) - self.origin) / self.spacing
        indices = np.floor(relative_pos).astype(int)
        
        if self.cycle:
            indices = indices % self.shape
        
        return indices
    