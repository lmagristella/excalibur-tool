# photon/photon_history.py
import numpy as np
import h5py

class PhotonHistory:
    """Historique d’états successifs d’un photon."""
    def __init__(self):
        self.states = []

    def append(self, state):
        self.states.append(state.copy())

    def save_to_hdf5(self, filename):
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
