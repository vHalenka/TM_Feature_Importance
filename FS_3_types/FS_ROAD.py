import time
import warnings
import json # Added for JSON output
import numpy as np
import matplotlib.pyplot as plt

# Switch to Wisconsin Breast Cancer Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.metrics import f1_score, balanced_accuracy_score, matthews_corrcoef

from numpy import trapz
from tmu.models.classification.vanilla_classifier import TMClassifier
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.feature_selection import chi2, VarianceThreshold # Added for new filters


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
            "lymphography":  "lymphography",
            "balance_scale": "balance-scale",
            "banknote":      "banknote-authentication",
            "transfusion":   "blood-transfusion-service-center",
            "madelon":       "madelon",
            "arcene":        "arcene",
            "leukemia":      "leukemia-golub",
        }
        if name not in oml:
            raise ValueError(f"Unknown dataset: {name}")
        ds = fetch_openml(oml[name], version=1, as_frame=False)
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

# --- Synthetic Dataset Generators ---
def generate_increasing_parity_dataset(n_samples=500, d=20, L=10):
    print("Generating Increasing Parity Complexity dataset...")
    X = np.random.randint(0, 2, size=(n_samples, d)).astype(np.uint32)
    Y = np.mod(np.sum(X[:, :L], axis=1), 2).astype(np.uint32)
    dataset_name = "Increasing_Parity_Complexity"
    important_indices = np.arange(L)  # the first L features are important
    return X, Y, dataset_name, d, important_indices

def generate_hierarchical_boolean_dataset(n_samples=500, d=20, n_groups=10):
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

def generate_progressive_interaction_dataset(n_samples=500, d=20, k=10):
    print("Generating Progressive Feature Interaction dataset...")
    X = np.random.randint(0, 2, size=(n_samples, d)).astype(np.uint32)
    np.random.seed(42) # Seed for reproducibility of important features
    important_indices = np.sort(np.random.choice(d, k, replace=False))
    Y = np.mod(np.sum(X[:, important_indices], axis=1), 2).astype(np.uint32)
    dataset_name = "Progressive_Feature_Interaction"
    return X, Y, dataset_name, d, important_indices

def count_ta_states(tm):
    """Extract TA states: (n_classes, 2, n_clauses/2, n_features)."""
    n_classes = tm.number_of_classes
    n_clauses = tm.number_of_clauses // 2
    n_feats   = tm.clause_banks[0].number_of_features
    arr = np.zeros((n_classes, 2, n_clauses, n_feats), dtype=int)
    for c in range(n_classes):
        for pol in (0,1):
            for cl in range(n_clauses):
                for f in range(n_feats):
                    arr[c,pol,cl,f] = tm.get_ta_action(
                        clause=cl, ta=f, the_class=c, polarity=pol
                    )
    return arr

def count_clause_weights(tm, ta_states=None):
    """Sum clause weights per class & feature."""
    if ta_states is None:
        ta_states = count_ta_states(tm)
    n_classes, _, n_clauses, n_feats = ta_states.shape
    pos_w = np.zeros((n_classes, n_feats), dtype=float)
    neg_w = np.zeros((n_classes, n_feats), dtype=float)
    for c in range(n_classes):
        for pol in (0,1):
            for cl in range(n_clauses):
                w = tm.get_weight(the_class=c, polarity=pol, clause=cl)
                active = (ta_states[c,pol,cl] != 0)
                if pol == 0:
                    pos_w[c] += w * active
                else:
                    neg_w[c] -= w * active
    return pos_w, neg_w

def count_clause_weights_posneg(tm, ta_states=None):
    """
    Sum clause weights per class & feature, adding negative weights.
    This is an alternative calculation to count_clause_weights.
    """
    if ta_states is None:
        ta_states = count_ta_states(tm)
    n_classes, _, n_clauses, n_feats = ta_states.shape
    pos_w = np.zeros((n_classes, n_feats), dtype=float)
    neg_w = np.zeros((n_classes, n_feats), dtype=float)
    for c in range(n_classes):
        for pol in (0,1):
            for cl in range(n_clauses):
                w = tm.get_weight(the_class=c, polarity=pol, clause=cl)
                active = (ta_states[c,pol,cl] != 0)
                if pol == 0:
                    pos_w[c] += w * active
                else:
                    # Here we add the negative weight instead of subtracting
                    neg_w[c] += w * active
    return pos_w, neg_w

def minmax_norm(arr):
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        return (arr - mn) / (mx - mn)
    else:
        return np.full_like(arr, 0.5)

def predict_proba(model, X_input_uint32, Y_context):
    """
    Generates probability-like scores for LIME/SHAP.
    X_input_uint32: Processed input data (e.g., binarized uint32).
    Y_context:      Labels corresponding to the original training/validation set for context.
    """
    try:
        # Get raw predictions and class sums
        predictions, class_sums = model.predict(X_input_uint32, return_class_sums=True)
        
        # Convert predictions to one-hot encoding
        n_classes = len(np.unique(Y_context))
        n_samples = len(predictions)
        probas = np.zeros((n_samples, n_classes), dtype=float)
        
        # For each sample, set the predicted class to 1.0
        for i, pred in enumerate(predictions):
            if 0 <= pred < n_classes:
                probas[i, pred] = 1.0
            else:
                # If prediction is out of bounds, use uniform distribution
                probas[i, :] = 1.0 / n_classes
        
        return probas
        
    except Exception as e:
        print(f"Error in predict_proba: {str(e)}")
        # Fallback to uniform distribution
        n_classes = len(np.unique(Y_context))
        n_samples = len(X_input_uint32)
        return np.ones((n_samples, n_classes)) / n_classes

def shap_predict(X):
    """
    SHAP prediction function that ensures binary input for Tsetlin Machine.
    """
    # Ensure X is binary
    X_binary = (X > 0.5).astype(np.uint32)
    
    # Get predictions
    predictions = model.predict(X_binary)
    
    # Convert to one-hot encoding
    n_classes = model.number_of_classes
    n_samples = len(predictions)
    probas = np.zeros((n_samples, n_classes), dtype=float)
    
    for i, pred in enumerate(predictions):
        if 0 <= pred < n_classes:
            probas[i, pred] = 1.0
        else:
            probas[i, :] = 1.0 / n_classes
    
    return probas

def lime_pred(X):
    """
    LIME prediction function that handles float inputs for Tsetlin Machine.
    """
    # Convert float inputs to binary
    X_binary = (X > 0.5).astype(np.uint32)
    
    # Get predictions
    predictions = model.predict(X_binary)
    
    # Convert to one-hot encoding
    n_classes = model.number_of_classes
    n_samples = len(predictions)
    probas = np.zeros((n_samples, n_classes), dtype=float)
    
    for i, pred in enumerate(predictions):
        if 0 <= pred < n_classes:
            probas[i, pred] = 1.0
        else:
            probas[i, :] = 1.0 / n_classes
    
    return probas

def evaluate_feature_importance(X, y, feature_indices, method_name):
    """
    Evaluate feature importance using Tsetlin Machine.
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train Tsetlin Machine
    tm = TMClassifier(
        number_of_clauses=100,
        number_of_features=X.shape[1],
        number_of_states=100,
        s=3.0,
        threshold=15,
        number_of_classes=len(np.unique(y))
    )
    
    try:
        # Train the model
        tm.fit(X_train, y_train, epochs=100)
        
        # Get predictions
        y_pred = tm.predict(X_test)
        
        # Calculate metrics
        accuracy = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        mcc = matthews_corrcoef(y_test, y_pred)
        
        return {
            "method": method_name,
            "accuracy": float(accuracy),
            "f1_score": float(f1),
            "mcc": float(mcc),
            "selected_features": feature_indices.tolist()
        }
        
    except Exception as e:
        print(f"Error in evaluate_feature_importance: {str(e)}")
        return {
            "method": method_name,
            "error": str(e),
            "selected_features": feature_indices.tolist()
        }

def main():
    # Create results directory if it doesn't exist
    os.makedirs("ROAD_results", exist_ok=True)
    
    # List of datasets to process
    datasets = [
        "breast_cancer", "iris", "digits", "pima", "ionosphere",
        "sonar", "heart", "wine", "glass", "vehicle", "steel",
        "spambase", "ecoli", "lymphography", "balance_scale",
        "banknote", "transfusion"
    ]
    
    # Add synthetic datasets
    synthetic_datasets = [
        ("increasing_parity", generate_increasing_parity_dataset),
        ("hierarchical_boolean", generate_hierarchical_boolean_dataset),
        ("progressive_interaction", generate_progressive_interaction_dataset)
    ]
    
    for dataset_name in datasets:
        print(f"\nProcessing {dataset_name}...")
        try:
            # Load dataset
            X, y = load_dataset(dataset_name)
            
            # Preprocess data
            X_bin, y_bin = preprocess_data(X, y)
            
            # Number of features to select (20% of total features)
            n_features = max(1, int(X.shape[1] * 0.2))
            
            # Run feature selection methods
            results = {}
            
            # Chi-squared selection
            try:
                from sklearn.feature_selection import SelectKBest, chi2
                selector = SelectKBest(chi2, k=n_features)
                X_new = selector.fit_transform(X, y)
                selected_features = selector.get_support(indices=True)
                results["chi_squared"] = evaluate_feature_importance(
                    X_bin, y_bin, selected_features, "chi_squared"
                )
            except Exception as e:
                print(f"Error in chi-squared selection: {str(e)}")
            
            # Mutual Information selection
            try:
                from sklearn.feature_selection import SelectKBest, mutual_info_classif
                selector = SelectKBest(mutual_info_classif, k=n_features)
                X_new = selector.fit_transform(X, y)
                selected_features = selector.get_support(indices=True)
                results["mutual_info"] = evaluate_feature_importance(
                    X_bin, y_bin, selected_features, "mutual_info"
                )
            except Exception as e:
                print(f"Error in mutual info selection: {str(e)}")
            
            # ReliefF selection
            try:
                from skrebate import ReliefF
                selector = ReliefF(n_features_to_select=n_features)
                X_new = selector.fit_transform(X, y)
                selected_features = selector.top_features_[:n_features]
                results["relieff"] = evaluate_feature_importance(
                    X_bin, y_bin, selected_features, "relieff"
                )
            except Exception as e:
                print(f"Error in ReliefF selection: {str(e)}")
            
            # Save results
            with open(f"ROAD_results/{dataset_name}_results.json", 'w') as f:
                json.dump(results, f, indent=4)
                
        except Exception as e:
            print(f"Error processing {dataset_name}: {str(e)}")
    
    # Process synthetic datasets
    for dataset_name, generator in synthetic_datasets:
        print(f"\nProcessing {dataset_name}...")
        try:
            # Generate dataset
            X, y, _, _, _ = generator()
            
            # Preprocess data
            X_bin, y_bin = preprocess_data(X, y)
            
            # Number of features to select (20% of total features)
            n_features = max(1, int(X.shape[1] * 0.2))
            
            # Run feature selection methods
            results = {}
            
            # Chi-squared selection
            try:
                from sklearn.feature_selection import SelectKBest, chi2
                selector = SelectKBest(chi2, k=n_features)
                X_new = selector.fit_transform(X, y)
                selected_features = selector.get_support(indices=True)
                results["chi_squared"] = evaluate_feature_importance(
                    X_bin, y_bin, selected_features, "chi_squared"
                )
            except Exception as e:
                print(f"Error in chi-squared selection: {str(e)}")
            
            # Mutual Information selection
            try:
                from sklearn.feature_selection import SelectKBest, mutual_info_classif
                selector = SelectKBest(mutual_info_classif, k=n_features)
                X_new = selector.fit_transform(X, y)
                selected_features = selector.get_support(indices=True)
                results["mutual_info"] = evaluate_feature_importance(
                    X_bin, y_bin, selected_features, "mutual_info"
                )
            except Exception as e:
                print(f"Error in mutual info selection: {str(e)}")
            
            # ReliefF selection
            try:
                from skrebate import ReliefF
                selector = ReliefF(n_features_to_select=n_features)
                X_new = selector.fit_transform(X, y)
                selected_features = selector.top_features_[:n_features]
                results["relieff"] = evaluate_feature_importance(
                    X_bin, y_bin, selected_features, "relieff"
                )
            except Exception as e:
                print(f"Error in ReliefF selection: {str(e)}")
            
            # Save results
            with open(f"ROAD_results/{dataset_name}_results.json", 'w') as f:
                json.dump(results, f, indent=4)
                
        except Exception as e:
            print(f"Error processing {dataset_name}: {str(e)}")

if __name__ == "__main__":
    main()

