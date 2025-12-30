"""
Synthetic dataset generators for feature selection experiments.
"""
import numpy as np


def generate_increasing_parity_dataset(n_samples=2000, d=10, L=5):
    """
    Generate Increasing Parity Complexity dataset.
    
    Args:
        n_samples: Number of samples
        d: Total number of features
        L: Number of important features (first L features)
        
    Returns:
        X: Binary feature matrix (uint32)
        Y: Binary target vector (uint32)
        dataset_name: Name of the dataset
        d: Number of features
        important_indices: Indices of important features
    """
    print("Generating Increasing Parity Complexity dataset...")
    X = np.random.randint(0, 2, size=(n_samples, d)).astype(np.uint32)
    Y = np.mod(np.sum(X[:, :L], axis=1), 2).astype(np.uint32)
    dataset_name = "Increasing_Parity_Complexity"
    important_indices = np.arange(L)
    return X, Y, dataset_name, d, important_indices


def generate_hierarchical_boolean_dataset(n_samples=500, d=20, n_groups=10):
    """
    Generate Hierarchical Boolean Rules dataset.
    
    Args:
        n_samples: Number of samples
        d: Total number of features
        n_groups: Number of feature groups
        
    Returns:
        X: Binary feature matrix (uint32)
        Y: Binary target vector (uint32)
        dataset_name: Name of the dataset
        d: Number of features
        important_indices: Indices of important features
    """
    print("Generating Hierarchical Boolean Rules dataset...")
    X = np.random.randint(0, 2, size=(n_samples, d)).astype(np.uint32)
    group_size = d // n_groups
    important_indices = []
    XORs = np.zeros((n_samples, n_groups), dtype=np.uint8)
    # For a fixed rule, choose the first two indices of each group as important
    for g in range(n_groups):
        i1 = g * group_size
        i2 = g * group_size + 1
        important_indices.extend([i1, i2])
        XORs[:, g] = np.logical_xor(X[:, i1], X[:, i2]).astype(np.uint8)
    threshold = n_groups // 2
    Y = (np.sum(XORs, axis=1) >= threshold).astype(np.uint32)
    dataset_name = "Hierarchical_Boolean_Rules"
    return X, Y, dataset_name, d, np.array(important_indices)


def generate_progressive_interaction_dataset(n_samples=3000, d=20, k=10):
    """
    Generate Progressive Feature Interaction dataset.
    
    Args:
        n_samples: Number of samples
        d: Total number of features
        k: Number of important features
        
    Returns:
        X: Binary feature matrix (uint32)
        Y: Binary target vector (uint32)
        dataset_name: Name of the dataset
        d: Number of features
        important_indices: Indices of important features
    """
    print("Generating Progressive Feature Interaction dataset...")
    X = np.random.randint(0, 2, size=(n_samples, d)).astype(np.uint32)
    np.random.seed(42)  # Seed for reproducibility of important features
    important_indices = np.sort(np.random.choice(d, k, replace=False))
    Y = np.mod(np.sum(X[:, important_indices], axis=1), 2).astype(np.uint32)
    dataset_name = "Progressive_Feature_Interaction"
    return X, Y, dataset_name, d, important_indices

