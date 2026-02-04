"""Photon history utilities.

This module intentionally treats ``h5py`` as an optional dependency so that
the core photon machinery can be imported even when the system ``h5py``
installation is broken or missing (for example after upgrading ``numpy`` for
JAX experiments).

The ``save_to_hdf5`` method will raise a clear error if ``h5py`` is not
available instead of failing at import time.
"""

import numpy as np

try:  # soft import to avoid hard dependency on a potentially broken h5py
    import h5py  # type: ignore
except Exception:  # pragma: no cover - allow running without h5py
    h5py = None

class PhotonHistory:
    """Historique d’états successifs d’un photon."""
    def __init__(self):
        self.states = []

    def append(self, state):
        self.states.append(state.copy())

    def save_to_hdf5(self, filename):
        if h5py is None:
            raise RuntimeError(
                "h5py is not available; cannot save PhotonHistory to HDF5. "
                "Install a working h5py or run without calling save_to_hdf5()."
            )

        with h5py.File(filename, "w") as f:
            # Convert list of states to a 2D array
            # All states should have the same length, but let's be safe
            if self.states:
                max_length = max(len(state) for state in self.states)
                states_array = np.full((len(self.states), max_length), np.nan)
                for i, state in enumerate(self.states):
                    states_array[i, :len(state)] = state
                f.create_dataset("states", data=states_array)
            else:
                f.create_dataset("states", data=np.array([]))
