"""
Data loading and preprocessing utilities.
"""
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_breast_cancer, load_iris, load_digits, fetch_openml


def load_dataset(name):
    """
    Load a dataset by name.
    
    Args:
        name: Dataset name (e.g., 'breast_cancer', 'iris', 'pima', etc.)
        
    Returns:
        X: Feature matrix (float)
        y: Target vector (uint32)
    """
    if name == "breast_cancer":
        ds = load_breast_cancer(as_frame=False)
        X, y = ds.data, ds.target
    elif name == "iris":
        ds = load_iris(as_frame=False)
        X, y = ds.data, ds.target
    elif name == "digits":
        ds = load_digits(as_frame=False)
        X, y = ds.data, ds.target
    else:
        oml_map = {
            "pima": "diabetes",
            "ionosphere": "ionosphere",
            "sonar": "sonar",
            "heart": "heart",
            "wine": "wine",
            "glass": "glass",
            "vehicle": "vehicle",
            "steel": "steel-plates-fault",
            "spambase": "spambase",
            "ecoli": "ecoli",
            "balance_scale": "balance-scale",
            "banknote": "banknote-authentication",
            "transfusion": "blood-transfusion-service-center",
            "lymphography": "lymphography",
            "madelon": "madelon",
            "arcene": "arcene",
            "leukemia": "leukemia-golub",
        }
        key = oml_map.get(name)
        if key is None:
            raise ValueError(f"Unknown dataset: {name}")
        try:
            ds = fetch_openml(key, version=1, as_frame=False)
        except:
            # Fallback without version for robustness
            ds = fetch_openml(key, as_frame=False, parser='auto')
        X, y = ds.data, ds.target
    
    # Convert sparse to dense if needed
    if hasattr(X, "toarray"):
        X = X.toarray()
    
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if y.dtype.kind in ("U", "S", "O"):
        y = LabelEncoder().fit_transform(y)
    return X, y.astype(np.uint32)


def preprocess_data(X, y, max_bins=10):
    """
    Thermometer-encode each feature in X adaptively.
    
    For each feature, chooses bins_i = min(max_bins, unique_values_i),
    discretizes by quantile thresholds, and encodes each bin as thermometer bits.
    
    Args:
        X: Feature matrix (float)
        y: Target vector
        max_bins: Maximum number of bins per feature
        
    Returns:
        Xb: Binarized feature matrix (uint32)
        y: Target vector (uint32)
    """
    n_samples, n_feats = X.shape
    cols = []
    for i in range(n_feats):
        col = X[:, i]
        unique = np.unique(col)
        bins_i = max(1, min(max_bins, len(unique)))
        if bins_i > 1:
            cuts = np.quantile(col, np.linspace(0, 1, bins_i + 1)[1:-1])
            ords = np.digitize(col, cuts)
        else:
            ords = np.zeros(n_samples, int)
        for b in range(bins_i):
            cols.append((ords >= b).astype(np.uint32))
    Xb = np.stack(cols, axis=1).astype(np.uint32)
    return Xb, y

