import numpy as np
import h5py

### This script to check and open photon trajectory files
# Open and examine the HDF5 file
with h5py.File('./photon_trajectory.h5', 'r') as f:
    print("File keys:", list(f.keys()))
    
    # Function to recursively explore HDF5 structure
    def explore_h5(name, obj):
        print(f"{name}: {type(obj)}")
        if isinstance(obj, h5py.Dataset):
            print(f"  Shape: {obj.shape}, Dtype: {obj.dtype}")
    
    # Explore the file structure
    f.visititems(explore_h5)
    
    # Print some sample data if datasets exist
    for key in f.keys():
        if isinstance(f[key], h5py.Dataset):
            print(f"\nSample data from '{key}':")
            print(f[key][:20])  # Show first 5 elements
