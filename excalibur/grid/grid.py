import numpy as np
from multiprocessing import shared_memory
import atexit

class Grid_test:
    _instance = None

    @staticmethod
    def instance():
        if Grid._instance is None:
            Grid._instance = Grid()
        return Grid._instance

    def __init__(self):
        if Grid._instance is not None:
            raise RuntimeError("Grid is a singleton, use Grid.instance()")
        self.fields = {}           # dictionnaire {name: np.array}
        self._shared_fields = {}   # dictionnaire {name: SharedMemory}
        self._shapes = {}          # shape pour chaque champ

    def add_field(self, name, shape=(128,128,128), precompute=False):
        """Ajoute un champ en mémoire standard"""
        arr = np.zeros(shape, dtype=np.float64)
        self.fields[name] = arr
        self._shapes[name] = shape
        if precompute:
            self.compute_field(name)

    def compute_field(self, name):
        """Exemple: Phi généré comme un champ test"""
        if name == "Phi":
            self.fields[name][:] = np.random.rand(*self._shapes[name])

    def attach_shared_memory(self):
        """Crée et attache tous les champs à la mémoire partagée"""
        for name, arr in self.fields.items():
            shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
            shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
            np.copyto(shm_arr, arr)  # copier le contenu
            self.fields[name] = shm_arr
            self._shared_fields[name] = shm
            atexit.register(shm.close)
            atexit.register(shm.unlink)
        print(f"Grid: attached {len(self._shared_fields)} shared fields")

    def reconnect_shared(self):
        """Reconnecte tous les champs partagés (pour les workers)"""
        for name, shm in self._shared_fields.items():
            arr = np.ndarray(self._shapes[name], dtype=np.float64, buffer=shm.buf)
            self.fields[name] = arr

    def get_field(self, name):
        """Accès sécurisé au champ"""
        return self.fields[name]


class Grid:
    """
    Stocke les champs physiques (potentiels, densités...) sur une grille régulière.
    Supporte shared_memory pour multiprocessing.
    """
    
    def __init__(self, shape, spacing, origin=(0, 0, 0), cycle=False):
        self.shape = np.array(shape)
        self.spacing = np.array(spacing)
        self.origin = np.array(origin)
        self.fields = {}            # accès normal: {"Psi": ndarray}
        self._shared_fields = {}    # accès shared_memory: {"Psi": SharedMemory}
        self.cycle = cycle

    def add_field(self, name, data, shared=False):
        assert data.shape == tuple(self.shape), "Mauvaise dimension du champ"

        if shared:
            # Créer un buffer en shared_memory et copier les données
            shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
            shared_array = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
            shared_array[:] = data[:]
            self.fields[name] = shared_array
            self._shared_fields[name] = shm
        else:
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

    def reconnect_shared(self):
        """
        À appeler dans un worker pour reconnecter tous les tableaux partagés.
        """
        for name, shm in self._shared_fields.items():
            if name not in self.fields:
                arr = np.ndarray(self.shape, dtype=self._dtype_fields[name], buffer=shm.buf)
                self.fields[name] = arr

    def close_shared(self):
        for shm in self._shared_fields.values():
            shm.close()
            shm.unlink()
        self._shared_fields.clear()
