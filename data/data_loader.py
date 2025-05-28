# ==========================================
# data/data_loader.py
# ==========================================
"""
Data loading utilities for IASC dataset.
Simple functions to load preprocessed pickle files.
"""

import pickle
import numpy as np
import os


def load_iasc_data(data_path="data/iasc_dataset/processed", data_type="1D", channels=2):
    """
    Load IASC dataset from pickle files.
    
    Args:
        data_path: Path to processed data directory
        data_type: Type of data to load ("1D" or "park" for 2D)
        channels: Channel configuration (0, 1, or 2 for both channels)
        
    Returns:
        X, y: Features and labels
    """
    if data_type == "1D":
        X_path = os.path.join(data_path, "X_IASC_1D.pickle")
        y_path = os.path.join(data_path, "y_IASC_1D.pickle")
    elif data_type == "park":
        X_path = os.path.join(data_path, "X_IASC_park.pickle")
        y_path = os.path.join(data_path, "y_IASC_park.pickle")
    else:
        raise ValueError("data_type must be '1D' or 'park'")
    
    with open(X_path, 'rb') as f:
        X = pickle.load(f)
    with open(y_path, 'rb') as f:
        y = pickle.load(f)
    
    # Handle channel selection for 1D data
    if data_type == "1D":
        if channels == 0:
            X = X[:, :, 0].reshape(-1, X.shape[1], 1)
        elif channels == 1:
            X = X[:, :, 1].reshape(-1, X.shape[1], 1)
        # channels == 2 keeps both channels
    
    # For 2D data (park), channel handling would be different if needed
    # Currently keeping all channels for 2D data
    
    return X, y