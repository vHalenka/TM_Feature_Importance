"""
Serialization and normalization utilities.
"""
import numpy as np
import json


def minmax_norm(arr):
    """
    Normalize array to [0, 1] using min-max scaling.
    
    Args:
        arr: Input array
        
    Returns:
        Normalized array (or array of 0.5 if min == max)
    """
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn) if mx > mn else np.full_like(arr, 0.5, dtype=float)


def convert_to_serializable(obj):
    """
    Convert numpy arrays and other non-serializable objects to JSON-compatible types.
    
    Args:
        obj: Object to convert (can be nested dict/list/array)
        
    Returns:
        JSON-serializable object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj


def save_json(data, filepath, indent=4):
    """
    Save data to JSON file with automatic serialization.
    
    Args:
        data: Data to save (will be converted to serializable format)
        filepath: Path to output file
        indent: JSON indentation level
    """
    serializable_data = convert_to_serializable(data)
    with open(filepath, 'w') as f:
        json.dump(serializable_data, f, indent=indent)


def load_json(filepath):
    """
    Load data from JSON file.
    
    Args:
        filepath: Path to input file
        
    Returns:
        Loaded data
    """
    with open(filepath, 'r') as f:
        return json.load(f)

