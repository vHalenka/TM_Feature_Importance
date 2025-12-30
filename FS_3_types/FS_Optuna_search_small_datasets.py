import json
import numpy as np
import optuna
from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tmu.models.classification.vanilla_classifier import TMClassifier

from sklearn.datasets import (
    load_breast_cancer, load_iris, load_digits, fetch_openml
)
from sklearn.preprocessing import LabelEncoder
import numpy as np

def load_dataset(name):
    """
    Returns X (float) and y (uint32) for UCI/OpenML and built-ins.
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
        oml = {
            "pima":          "diabetes",
            "ionosphere":    "ionosphere",
            "sonar":         "sonar",
            "heart":         "heart",
            "wine":          "wine",
            "glass":         "glass",
            "vehicle":       "vehicle",
            "steel":         "steel-plates-fault",
            "spambase":      "spambase",
            "ecoli":         "ecoli",
            "lymphography":  "61",
            "balance_scale": "balance-scale",
            "banknote":      "banknote-authentication",
            "transfusion":   "blood-transfusion-service-center",
            "madelon":       "madelon",
            "arcene":        "arcene",
        }
        if name not in oml:
            raise ValueError(f"Unknown dataset: {name}")
        # Remove version=1 to fetch the default version, as specific versions can be removed or changed on OpenML
        # For 'transfusion', version 1 might no longer be available or was never the primary one.
        # For other datasets, version=1 is usually fine, but this makes it more robust.
        ds = fetch_openml(oml[name], as_frame=False, parser='auto') # Added parser='auto' for robustness
        X, y = ds.data, ds.target

    # Ensure X is a dense array if it's sparse
    if hasattr(X, "toarray"):
        X = X.toarray()

    # cast & encode
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if y.dtype.kind in ("U", "S", "O"):
        y = LabelEncoder().fit_transform(y)
    return X, y.astype(np.uint32)

# Adaptive thermometer encoding & cast to uint32
def preprocess_data(X, y, max_bins=10):
    """
    Thermometer-encode each feature in X adaptively:
      - For feature i, choose bins_i = min(max_bins, unique_values_i)
      - Discretize by quantile thresholds and encode each bin as thermometer bits
    Returns X_bin (uint32) and y_uint32.
    """
    n_samples, n_feats = X.shape
    X_bin_cols = []
    for i in range(n_feats):
        col = X[:, i]
        # Determine adaptive bin count
        unique_vals = np.unique(col)
        bins_i = min(max_bins, len(unique_vals))
        bins_i = max(1, bins_i)
        # Compute quantile thresholds for bins_i
        if bins_i > 1:
            cuts = np.quantile(col, np.linspace(0, 1, bins_i + 1)[1:-1])
            ordinal = np.digitize(col, cuts)
        else:
            ordinal = np.zeros(n_samples, dtype=int)
        # Thermometer bits for this feature
        for b in range(bins_i):
            bit_col = (ordinal >= b).astype(np.uint32)
            X_bin_cols.append(bit_col)
    # Stack columns and cast
    X_bin = np.stack(X_bin_cols, axis=1).astype(np.uint32)
    y_uint = np.asarray(y, dtype=np.uint32)
    return X_bin, y_uint

# Objective for Optuna
def make_objective(X_train, y_train, X_val, y_val, clauses, epochs, patience):
    def objective(trial):
        s = trial.suggest_float("s", 0.9, 20.0)
        T = trial.suggest_categorical("T", [50, 200, 300, 500, 800])
        tm = TMClassifier(
            number_of_clauses=clauses,
            T=T,
            s=s,
            max_included_literals=20,
            platform='CPU'
        )
        best_val = 0.0
        wait = 0
        for ep in range(epochs):
            tm.fit(X_train, y_train, epochs=1)
            acc = (tm.predict(X_val) == y_val).mean()
            if acc > best_val + 1e-6:
                best_val = acc
                wait = 0
            else:
                wait += 1
            if wait >= patience:
                break
        return best_val
    return objective

# Main HPO loop
if __name__ == "__main__":
    datasets = [
        #"breast_cancer", "pima", "ionosphere",
        #"sonar", 
        #"heart", "wine", "glass",
        #"vehicle", "steel", "iris", "digits",

        # new UCI/OpenML benchmarks
        #"spambase", "ecoli", 
        # "balance_scale", "banknote", 
        # "madelon", #not done
        
        #"transfusion",
         
        "leukemia"
    ]
    clauses = 500
    epochs = 30
    patience = 3
    trials = 100
    max_bins = 10

    for name in datasets:
        print(f"\n=== Tuning {name} ===")
        X_raw, y_raw = load_dataset(name)
        X_bin, y = preprocess_data(X_raw, y_raw, max_bins=max_bins)
        X_train, X_val, y_train, y_val = train_test_split(
            X_bin, y, test_size=0.2, random_state=42, stratify=y
        )
        study = optuna.create_study(direction="maximize")
        study.optimize(
            make_objective(X_train, y_train, X_val, y_val, clauses, epochs, patience),
            n_trials=trials
        )
        best = study.best_trial
        result = {
            "dataset": name,
            "best_s": best.params["s"],
            "best_T": best.params["T"],
            "best_val_accuracy": best.value
        }
        filename = f"best_params_{name}.json"
        with open(filename, 'w') as f:
            json.dump(result, f, indent=4)
        print(f"Best params for {name}: {best.value}")
        print(f"Saved best params to {filename}")
