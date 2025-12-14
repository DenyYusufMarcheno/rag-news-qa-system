"""Utility functions and constants for RAG News QA System."""

import os

# Default data paths
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
DEFAULT_DATASET_FILENAME = 'News_Category_Dataset_v3.json'
DEFAULT_DATASET_PATH = os.path.join(DEFAULT_DATA_DIR, DEFAULT_DATASET_FILENAME)


def get_dataset_path(filename: str = DEFAULT_DATASET_FILENAME) -> str:
    """
    Get the path to the dataset file.
    
    Args:
        filename: Name of the dataset file
        
    Returns:
        Absolute path to the dataset file
    """
    return os.path.join(DEFAULT_DATA_DIR, filename)


def ensure_data_dir() -> str:
    """
    Ensure the data directory exists.
    
    Returns:
        Path to the data directory
    """
    os.makedirs(DEFAULT_DATA_DIR, exist_ok=True)
    return DEFAULT_DATA_DIR
