
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

def predict_proba(model, X_input_uint32, Y_context): # X_input_uint32 is already processed
    """
    Generates probability-like scores for LIME/SHAP.
    X_input_uint32: Processed input data (e.g., binarized uint32).
    Y_context:      Labels corresponding to the original training/validation set for context (e.g., num_classes).
    """
    try:
        raw_predictions_tuple = model.predict(X_input_uint32, return_class_sums=True)
        
        if isinstance(raw_predictions_tuple, (list, tuple)) and len(raw_predictions_tuple) == 2:
            class_sums = raw_predictions_tuple[1] 
            preds_from_sums = raw_predictions_tuple[0]

            min_val_per_sample = np.min(class_sums, axis=1, keepdims=True)
            shifted_sums = class_sums - min_val_per_sample
            sum_of_shifted_sums = np.sum(shifted_sums, axis=1, keepdims=True)
            
            probas = np.zeros_like(shifted_sums, dtype=float)
            # Mask for samples where sum_of_shifted_sums is not zero
            valid_sums_mask = (sum_of_shifted_sums > 1e-9).flatten()

            if np.any(valid_sums_mask):
                probas[valid_sums_mask] = shifted_sums[valid_sums_mask] / sum_of_shifted_sums[valid_sums_mask]
            
            # For samples where all sums were equal (shifted_sums are all zero), use one-hot from prediction
            invalid_sums_mask = ~valid_sums_mask
            if np.any(invalid_sums_mask):
                if hasattr(model, 'number_of_classes'):
                    n_classes_model = model.number_of_classes
                else:
                    n_classes_model = len(np.unique(Y_context))
                
                for i in np.where(invalid_sums_mask)[0]:
                    pred_val = preds_from_sums[i]
                    if 0 <= pred_val < n_classes_model:
                        probas[i, pred_val] = 1.0
                    else: # Fallback to uniform if prediction is out of bounds
                        probas[i, :] = 1.0 / n_classes_model
            return probas
        else: 
            # If return_class_sums=True didn't return a tuple, assume it returned only predictions
            preds = raw_predictions_tuple # This is now just the predictions array
            # Fall through to one-hot based on these predictions
    except Exception:
        # Fallback if return_class_sums fails or is not supported
        preds = model.predict(X_input_uint32) # Get simple predictions
        # Fall through to one-hot

    # Fallback to one-hot encoding based on `preds`
    if hasattr(model, 'number_of_classes'):
        n_classes_model = model.number_of_classes
    else:
        n_classes_model = len(np.unique(Y_context))

    P = np.zeros((X_input_uint32.shape[0], n_classes_model), dtype=float)
    for i, p_val in enumerate(preds):
        if 0 <= p_val < n_classes_model:
            P[i, p_val] = 1.0
    return P

if __name__ == "__main__":
    # at the top of your __main__ block
    datasets = [
        #"breast_cancer", "pima", "ionosphere",
        #"sonar", 
        #"heart", "wine", "glass",
        #"vehicle", "steel", "iris", "digits",

        # new UCI/OpenML benchmarks
        #"spambase", "ecoli",
        #"balance_scale", "banknote",
        "transfusion",

        # your synthetic toy sets
        "Increasing_Parity_Complexity",
        "Hierarchical_Boolean_Rules",
        "Progressive_Feature_Interaction",
    ]


    clauses = 500
    epochs = 30
    max_lit = 32
    max_bins_param = 10 # Renamed to avoid conflict with local max_bins if any
    # Load best parameters from JSON file
    try:
        with open("all_best_tm_params.json", 'r') as f:
            best_params_data = json.load(f)
        print("Loaded best parameters from all_best_tm_params.json")
    except FileNotFoundError:
        print("Error: all_best_tm_params.json not found. Please run FS_Optuna_search_small_datasets.py first.")
        exit()

    

    # Dictionary to store results for each dataset
    all_dataset_results = {}

    for current_dataset_name in datasets:
        # -------------------------
        # 1) Load Dataset & preprocess
        # -------------------------
        rng = np.random.default_rng(42)
        current_dataset_timings = {} # To store timings per dataset
        ground_truth_features = None # For synthetic datasets

        print(f"\n=== Processing Dataset: {current_dataset_name} ===")
        if current_dataset_name == "Increasing_Parity_Complexity":
            X_bin, y_processed, _, _, ground_truth_features = generate_increasing_parity_dataset()
            dataset_type_info = " (Pre-binarized Synthetic)"
        elif current_dataset_name == "Hierarchical_Boolean_Rules":
            X_bin, y_processed, _, _, ground_truth_features = generate_hierarchical_boolean_dataset()
            dataset_type_info = " (Pre-binarized Synthetic)"
        elif current_dataset_name == "Progressive_Feature_Interaction":
            X_bin, y_processed, _, _, ground_truth_features = generate_progressive_interaction_dataset()
            dataset_type_info = " (Pre-binarized Synthetic)"
        else:
            X_raw, y_raw = load_dataset(current_dataset_name)
            X_bin, y_processed = preprocess_data(X_raw, y_raw, max_bins=max_bins_param)
            dataset_type_info = f" (Thermometer Encoded, max_bins={max_bins_param})"

        # Adjusted data splitting: 60% train, 20% validation, 20% test
        X_train_val_pool, X_test_bin, y_train_val_pool, y_test = train_test_split(
            X_bin, y_processed, test_size=0.2, random_state=42, stratify=y_processed
        )
        X_train_bin, X_val_bin, y_train, y_val = train_test_split(
            X_train_val_pool, y_train_val_pool, test_size=0.25, random_state=42, stratify=y_train_val_pool # 0.25 * 0.8 = 0.2
        )
        
        print(f"Dataset shapes: X_train_bin: {X_train_bin.shape}, X_val_bin: {X_val_bin.shape}, X_test_bin: {X_test_bin.shape}")
        # Retrieve s and T for the current dataset from loaded best_params_data
        dataset_specific_params = best_params_data.get(current_dataset_name)
        if dataset_specific_params:
            s_current_dataset = dataset_specific_params["best_s"]
            T_current_dataset = dataset_specific_params["best_T"]
            print(f"Using Optuna best s={s_current_dataset:.4f}, T={T_current_dataset} for dataset {current_dataset_name}")
        else:
            print(f"Warning: Parameters for dataset '{current_dataset_name}' not found in all_best_tm_params.json. Using default s=3.0, T=600.")
            s_current_dataset = 3.0  # Default s
            T_current_dataset = 600  # Default T

        # -------------------------
        # 2) Train TM & record weight history
        # -------------------------
        

        tm = TMClassifier(
            number_of_clauses      = clauses,
            T                      = T_current_dataset,
            s                      = s_current_dataset,
            max_included_literals  = max_lit,
            platform               = 'CPU'
        )

        pos_w_history = []
        neg_w_history = []
        for ep in range(epochs):
            tm.fit(X_train_bin, y_train, epochs=1)
            ta = count_ta_states(tm)
            pos, neg = count_clause_weights(tm, ta)
            pos_w_history.append(pos)
            neg_w_history.append(neg)
            y_pred_val = tm.predict(X_val_bin)
            full_acc_val = 100 * (y_pred_val == y_val).mean()
            print(f"Epoch {ep+1}/{epochs} for feature scoring model on {current_dataset_name}, Validation Set: {full_acc_val:.2f}%")
        pos_w_history, neg_w_history = np.stack(pos_w_history, axis=0), np.stack(neg_w_history, axis=0)

        # Accuracy on validation set (used for subsequent calculations for feature scoring)
        print(f"Full‐feature Accuracy on {current_dataset_name} Validation Set: {full_acc_val:.2f}%\n")

        # -------------------------
        # 3) Base quantities
        # -------------------------
        net_w = pos_w_history[-1] - neg_w_history[-1] # (n_classes, n_feats)
        abs_w = np.abs(net_w)
        net_w_posneg = pos_w_history[-1] + neg_w_history[-1] # (n_classes, n_feats) - NEW
        abs_w_posneg = np.abs(net_w_posneg) # NEW
        n_classes, n_feats = net_w.shape # n_feats is based on X_train_bin

        # per‐class validation accuracies → class_w (used for CW-Sum, Support-CW-Sum etc.)
        class_acc = np.array([
            (y_pred_val[y_val==c] == c).mean() if np.any(y_val==c) else 0.0 # Use y_pred_val and y_val
            for c in range(n_classes)
        ])
        class_w = class_acc / (class_acc.sum() + 1e-12)

        # -------------------------
        # 4) Pure TM‐only scores
        # -------------------------
        # 1a) TM‐weight (max abs)
        t0 = time.perf_counter()
        weight_score = abs_w.max(axis=0)
        tm_weight_norm      = minmax_norm(weight_score)
        current_dataset_timings['TM-Weight'] = time.perf_counter() - t0
        print(f"Time for TM-Weight: {time.perf_counter() - t0:.4f}s")

        # 1b) TM‐weight (max abs) - PosNeg version
        t0 = time.perf_counter()
        weight_score_posneg = abs_w_posneg.max(axis=0)
        tm_weight_posneg_norm = minmax_norm(weight_score_posneg)
        current_dataset_timings['TM-Weight-PosNeg'] = time.perf_counter() - t0

        # 2a) Class‐weighted sum (CW-Sum)
        t0 = time.perf_counter()
        cw_sum       = (class_w[:,None] * abs_w).sum(axis=0)
        cw_sum_norm  = minmax_norm(cw_sum)
        current_dataset_timings['CW-Sum'] = time.perf_counter() - t0
        print(f"Time for CW-Sum: {time.perf_counter() - t0:.4f}s")

        # -------------------------
        # 2b) Class‐weighted sum (CW-Sum) - PosNeg version
        t0 = time.perf_counter()
        cw_sum_posneg = (class_w[:,None] * abs_w_posneg).sum(axis=0)
        cw_sum_posneg_norm = minmax_norm(cw_sum_posneg)
        current_dataset_timings['CW-Sum-PosNeg'] = time.perf_counter() - t0

        # 2c) Per‐feature CW (CW-Feat) - Original
        # -------------------------
        t0 = time.perf_counter() # New timer for CW-Feat
        freq      = ta.sum(axis=(1,2)) # (C, F)
        alpha     = freq/(freq.sum(axis=0,keepdims=True)+1e-12)
        cw_feat   = (alpha * abs_w).sum(axis=0)
        cw_feat_n = minmax_norm(cw_feat)
        current_dataset_timings['CW-Feat'] = time.perf_counter() - t0
        print(f"Time for CW-Feat: {time.perf_counter() - t0:.4f}s")

        # 2d) Per‐feature CW (CW-Feat) - PosNeg version
        t0 = time.perf_counter()
        # freq is the same, alpha is the same
        cw_feat_posneg = (alpha * abs_w_posneg).sum(axis=0)
        cw_feat_posneg_n = minmax_norm(cw_feat_posneg)
        current_dataset_timings['CW-Feat-PosNeg'] = time.perf_counter() - t0

        # 2e) Supportive Class-Weighted Sum (Prioritizes features for low-accuracy classes) - Original
        t0 = time.perf_counter()
        class_error_rate = 1.0 - class_acc # Error rate for each class
        class_error_weight = class_error_rate / (class_error_rate.sum() + 1e-12)
        support_cw_sum = (class_error_weight[:, None] * abs_w).sum(axis=0)
        support_cw_sum_norm = minmax_norm(support_cw_sum)
        current_dataset_timings['Support-CW-Sum'] = time.perf_counter() - t0
        print(f"Time for Support-CW-Sum: {time.perf_counter() - t0:.4f}s")

        # 2f) Supportive Class-Weighted Sum - PosNeg version
        t0 = time.perf_counter()
        support_cw_sum_posneg = (class_error_weight[:, None] * abs_w_posneg).sum(axis=0)
        support_cw_sum_posneg_norm = minmax_norm(support_cw_sum_posneg)
        current_dataset_timings['Support-CW-Sum-PosNeg'] = time.perf_counter() - t0

        # 3a) Margin top vs runner‐up
        t0 = time.perf_counter() 
        sorted_abs   = np.sort(abs_w, axis=0)
        margin       = sorted_abs[-1] - sorted_abs[-2]
        margin_norm  = minmax_norm(margin) 
        current_dataset_timings['Margin'] = time.perf_counter() - t0 
        print(f"Time for Margin: {time.perf_counter() - t0:.4f}s")

        # 3b) Margin top vs runner‐up - PosNeg version
        t0 = time.perf_counter()
        sorted_abs_posneg = np.sort(abs_w_posneg, axis=0)
        margin_posneg = sorted_abs_posneg[-1] - sorted_abs_posneg[-2]
        margin_posneg_norm = minmax_norm(margin_posneg)
        current_dataset_timings['Margin-PosNeg'] = time.perf_counter() - t0

        # 4a) Entropy‐based 
        t0 = time.perf_counter() # New timer for Entropy
        p = abs_w / (abs_w.sum(axis=0, keepdims=True) + 1e-12)
        entropy = - (p * np.log(p + 1e-12)).sum(axis=0)
        entropy_score = 1.0 - (entropy / np.log(n_classes))
        #entropy_score = np.log(n_classes) - entropy # Invert entropy
        entropy_norm = minmax_norm(entropy_score)
        current_dataset_timings['Entropy'] = time.perf_counter() - t0 
        print(f"Time for Entropy: {time.perf_counter() - t0:.4f}s")

        # 4b) Entropy‐based (invert so low‐entropy→high score)
        t0 = time.perf_counter() # New timer for Entropy
        p = abs_w / (abs_w.sum(axis=0, keepdims=True) + 1e-12)
        entropy = - (p * np.log(p + 1e-12)).sum(axis=0)
        #entropy_score = 1.0 - (entropy / np.log(n_classes))
        entropy_score = np.log(n_classes) - entropy # Invert entropy
        entropy_norm = minmax_norm(entropy_score)
        current_dataset_timings['Entropy-inv'] = time.perf_counter() - t0 

        # 4c) Entropy‐based - PosNeg version
        t0 = time.perf_counter()
        p_posneg = abs_w_posneg / (abs_w_posneg.sum(axis=0, keepdims=True) + 1e-12)
        entropy_posneg = - (p_posneg * np.log(p_posneg + 1e-12)).sum(axis=0)
        entropy_score_posneg = 1.0 - (entropy_posneg / np.log(n_classes))
        #entropy_score_posneg = np.log(n_classes) - entropy_posneg # Invert entropy
        entropy_posneg_norm = minmax_norm(entropy_score_posneg)
        current_dataset_timings['Entropy-PosNeg'] = time.perf_counter() - t0
        
        # 4d) Entropy‐based - PosNeg version
        t0 = time.perf_counter()
        p_posneg = abs_w_posneg / (abs_w_posneg.sum(axis=0, keepdims=True) + 1e-12)
        entropy_posneg = - (p_posneg * np.log(p_posneg + 1e-12)).sum(axis=0)
        entropy_score_posneg = np.log(n_classes) - entropy_posneg # Invert entropy
        entropy_posneg_norm = minmax_norm(entropy_score_posneg)
        current_dataset_timings['Entropy-PosNeg-inv'] = time.perf_counter() - t0
        
        # 5a) Gini‐based
        t0 = time.perf_counter() # New timer for Gini
        gini_score = (1.0 - (p**2).sum(axis=0))  # = sum(p^2)
        gini_norm  = minmax_norm(gini_score)
        current_dataset_timings['Gini'] = time.perf_counter() - t0 
        print(f"Time for Gini: {time.perf_counter() - t0:.4f}s")

        # 5b) Gini‐based (invert so mass concentrated→high score)
        t0 = time.perf_counter() # New timer for Gini
        gini_score = 1.0 - (1.0 - (p**2).sum(axis=0))  # = sum(p^2)
        gini_norm  = minmax_norm(gini_score)
        current_dataset_timings['Gini-inv'] = time.perf_counter() - t0 

        # 5) Gini‐based - PosNeg version
        t0 = time.perf_counter()
        gini_score_posneg = (1.0 - (p_posneg**2).sum(axis=0))
        gini_posneg_norm = minmax_norm(gini_score_posneg)
        current_dataset_timings['Gini-PosNeg'] = time.perf_counter() - t0

        # 5) Gini‐based - PosNeg version (invert)
        t0 = time.perf_counter()
        gini_score_posneg = 1.0 - (1.0 - (p_posneg**2).sum(axis=0))
        gini_posneg_norm = minmax_norm(gini_score_posneg)
        current_dataset_timings['Gini-PosNeg-inv'] = time.perf_counter() - t0

        # 6a) Stability across epochs - Original
        t0 = time.perf_counter()
        net_w_history = pos_w_history - neg_w_history # Recalculate history for original net_w
        max_abs_hist = np.max(np.abs(net_w_history), axis=1) # (epochs, n_feats)
        std_hist     = max_abs_hist.std(axis=0)
        mean_abs     = max_abs_hist.mean(axis=0)
        stab_score   = mean_abs / (std_hist + 1e-6)
        stab_norm    = minmax_norm(stab_score)
        current_dataset_timings['Stability'] = time.perf_counter() - t0
        print(f"Time for Stability: {time.perf_counter() - t0:.4f}s")

        # 6b) Stability across epochs - PosNeg version
        t0 = time.perf_counter()
        net_w_posneg_history = pos_w_history + neg_w_history # History for PosNeg net_w
        max_abs_hist_posneg = np.max(np.abs(net_w_posneg_history), axis=1)
        std_hist_posneg = max_abs_hist_posneg.std(axis=0)
        mean_abs_posneg = max_abs_hist_posneg.mean(axis=0)
        stab_score_posneg = mean_abs_posneg / (std_hist_posneg + 1e-6)
        stab_posneg_norm = minmax_norm(stab_score_posneg)
        current_dataset_timings['Stability-PosNeg'] = time.perf_counter() - t0

        # 7) Dropout sensitivity
        t0 = time.perf_counter() # New timer for Dropout
        drop = np.zeros(n_feats)
        for f in range(n_feats):
            Xm = X_val_bin.copy() 
            Xm[:,f] = 0
            drop[f] = full_acc_val - 100*(tm.predict(Xm)==y_val).mean() 
        drop_norm = minmax_norm(drop)
        current_dataset_timings['Dropout'] = time.perf_counter() - t0
        print(f"Time for Dropout: {time.perf_counter() - t0:.4f}s")

        # 8) Mutual information
        t0 = time.perf_counter() # New timer for MutualInfo
        correct_val = (y_pred_val == y_val).astype(int) # Correctness on validation set
        mi      = np.array([mutual_info_score(X_val_bin[:,f], correct_val) for f in range(n_feats)]) # Use validation set
        mi_norm = minmax_norm(mi)
        current_dataset_timings['MutualInfo'] = time.perf_counter() - t0 # Corrected timing name
        print(f"Time for MutualInfo: {time.perf_counter() - t0:.4f}s")

        # 9a) Relevance
        t0 = time.perf_counter() # New timer for Relevance
        norm_abs   = abs_w / (abs_w.sum(axis=1, keepdims=True) + 1e-12)
        relevance  = (class_w[:,None] * norm_abs).sum(axis=0) # Original
        relevance_n= minmax_norm(relevance)
        current_dataset_timings['Relevance'] = time.perf_counter() - t0
        print(f"Time for Relevance: {time.perf_counter() - t0:.4f}s")

        # 9b) Relevance - PosNeg version
        t0 = time.perf_counter()
        norm_abs_posneg = abs_w_posneg / (abs_w_posneg.sum(axis=1, keepdims=True) + 1e-12)
        relevance_posneg = (class_w[:,None] * norm_abs_posneg).sum(axis=0)
        relevance_posneg_n = minmax_norm(relevance_posneg)
        current_dataset_timings['Relevance-PosNeg'] = time.perf_counter() - t0
        
        # 10) Random Ordering (Baseline)
        t0 = time.perf_counter()
        random_scores = rng.random(n_feats)
        random_norm = minmax_norm(random_scores)
        current_dataset_timings['Random'] = time.perf_counter() - t0 
        print(f"Time for Random: {time.perf_counter() - t0:.4f}s")

        # 11a) Group Lasso Score
        t0 = time.perf_counter()
        img_dim = int(np.sqrt(n_feats))
        
        if np.isnan(abs_w).any():
            print(f"WARNING: NaNs found in abs_w for dataset {current_dataset_name} before GroupLasso calculation!")

        if img_dim * img_dim == n_feats and n_feats > 0: # Check if features form a square image and n_feats > 0
            group_size = 4 # Define group size (e.g., 4x4 patches)
            abs_w_reshaped = abs_w.reshape(n_classes, img_dim, img_dim)
            
            num_groups_h = img_dim // group_size
            num_groups_w = img_dim // group_size
            
            group_l2_norms_per_class = np.zeros((n_classes, num_groups_h, num_groups_w))

            for r_group in range(num_groups_h):
                for c_group in range(num_groups_w):
                    r_start, r_end = r_group * group_size, (r_group + 1) * group_size
                    c_start, c_end = c_group * group_size, (c_group + 1) * group_size
                    
                    group_weights = abs_w_reshaped[:, r_start:r_end, c_start:c_end] # (n_classes, group_size, group_size)
                    l2_norm_group = np.sqrt(np.sum(group_weights**2, axis=(1,2))) # (n_classes,)
                    group_l2_norms_per_class[:, r_group, c_group] = l2_norm_group
            
            mean_group_l2_norms = group_l2_norms_per_class.mean(axis=0) # (num_groups_h, num_groups_w)
            
            group_lasso_feature_scores = np.zeros(n_feats)
            for r_pixel in range(img_dim):
                for c_pixel in range(img_dim):
                    r_group, c_group = r_pixel // group_size, c_pixel // group_size
                    if r_group < num_groups_h and c_group < num_groups_w: # Ensure group index is valid
                        feature_idx = r_pixel * img_dim + c_pixel
                        group_lasso_feature_scores[feature_idx] = mean_group_l2_norms[r_group, c_group]
            group_lasso_norm = minmax_norm(group_lasso_feature_scores)
        elif n_feats == 0:
            print(f"Warning: Group Lasso score not computed for {current_dataset_name} as n_feats is 0.")
            group_lasso_norm = np.array([0.5]) # Consistent with minmax_norm on single value
        else: # n_feats > 0 but not a perfect square
            print(f"Warning: Group Lasso score not computed for {current_dataset_name} as n_feats ({n_feats}) is not a perfect square. Assigning default.")
            group_lasso_norm = np.full(n_feats, 0.5) # Default if not applicable
        current_dataset_timings['GroupLasso'] = time.perf_counter() - t0
        print(f"Time for GroupLasso: {time.perf_counter() - t0:.4f}s")

        # 11b) Group Lasso Score - PosNeg version
        t0 = time.perf_counter()
        if img_dim * img_dim == n_feats and n_feats > 0: # Check if features form a square image and n_feats > 0
            # group_size is the same
            abs_w_posneg_reshaped = abs_w_posneg.reshape(n_classes, img_dim, img_dim)
            
            # num_groups_h, num_groups_w are the same
            
            group_l2_norms_posneg_per_class = np.zeros((n_classes, num_groups_h, num_groups_w))

            for r_group in range(num_groups_h):
                for c_group in range(num_groups_w):
                    r_start, r_end = r_group * group_size, (r_group + 1) * group_size
                    c_start, c_end = c_group * group_size, (c_group + 1) * group_size
                    
                    group_weights_posneg = abs_w_posneg_reshaped[:, r_start:r_end, c_start:c_end] # (n_classes, group_size, group_size)
                    l2_norm_group_posneg = np.sqrt(np.sum(group_weights_posneg**2, axis=(1,2))) # (n_classes,)
                    group_l2_norms_posneg_per_class[:, r_group, c_group] = l2_norm_group_posneg
            
            mean_group_l2_norms_posneg = group_l2_norms_posneg_per_class.mean(axis=0) # (num_groups_h, num_groups_w)
            
            group_lasso_feature_scores_posneg = np.zeros(n_feats)
            for r_pixel in range(img_dim):
                for c_pixel in range(img_dim):
                    r_group, c_group = r_pixel // group_size, c_pixel // group_size
                    if r_group < num_groups_h and c_group < num_groups_w: # Ensure group index is valid
                        feature_idx = r_pixel * img_dim + c_pixel
                        group_lasso_feature_scores_posneg[feature_idx] = mean_group_l2_norms_posneg[r_group, c_group]
            group_lasso_posneg_norm = minmax_norm(group_lasso_feature_scores_posneg)
        elif n_feats == 0:
            print(f"Warning: Group Lasso PosNeg score not computed for {current_dataset_name} as n_feats is 0.")
            group_lasso_posneg_norm = np.array([0.5])
        else: # n_feats > 0 but not a perfect square
            print(f"Warning: Group Lasso PosNeg score not computed for {current_dataset_name} as n_feats ({n_feats}) is not a perfect square. Assigning default.")
            group_lasso_posneg_norm = np.full(n_feats, 0.5) # Default if not applicable
        current_dataset_timings['GroupLasso-PosNeg'] = time.perf_counter() - t0

        # 12) Taylor Criteria Score (First-Order Perturbation Impact)
        t0 = time.perf_counter()
        original_class_sums_val = tm.predict(X_val_bin, return_class_sums=True)[1]
        original_scores_for_true_class_val = original_class_sums_val[np.arange(len(y_val)), y_val]
        taylor_scores = np.zeros(n_feats)
        for f_idx in range(n_feats):
            X_perturbed = X_val_bin.copy()
            X_perturbed[:, f_idx] = 1 - X_perturbed[:, f_idx] # Flip bits for feature f_idx
            perturbed_class_sums_val = tm.predict(X_perturbed, return_class_sums=True)[1]
            perturbed_scores_for_true_class_val = perturbed_class_sums_val[np.arange(len(y_val)), y_val]
            taylor_scores[f_idx] = np.mean(np.abs(original_scores_for_true_class_val - perturbed_scores_for_true_class_val))
        taylor_norm = minmax_norm(taylor_scores)
        current_dataset_timings['TaylorCrit'] = time.perf_counter() - t0
        print(f"Time for TaylorCrit: {time.perf_counter() - t0:.4f}s")

        # 13) Variational Dropout Score (TM Adaptation)
        t0 = time.perf_counter()
        N_MASKS_VAR_DROPOUT = 10 # Number of random masks per feature
        var_dropout_scores = np.zeros(n_feats)
        original_sums_val_for_var_dropout = tm.predict(X_val_bin, return_class_sums=True)[1]
        original_true_class_scores_val_var_dropout = original_sums_val_for_var_dropout[np.arange(len(y_val)), y_val]
        for f_idx in range(n_feats):
            score_changes_for_f = []
            for _ in range(N_MASKS_VAR_DROPOUT):
                X_val_temp = X_val_bin.copy()
                mask_f = np.random.randint(0, 2, size=X_val_bin.shape[0]) # Random 0/1 mask for feature f
                X_val_temp[:, f_idx] = X_val_temp[:, f_idx] * mask_f # Apply mask
                
                perturbed_sums_val = tm.predict(X_val_temp, return_class_sums=True)[1]
                perturbed_true_class_scores_val = perturbed_sums_val[np.arange(len(y_val)), y_val]
                score_changes_for_f.append(np.mean(np.abs(original_true_class_scores_val_var_dropout - perturbed_true_class_scores_val)))
            var_dropout_scores[f_idx] = np.mean(score_changes_for_f)
        var_dropout_norm = minmax_norm(var_dropout_scores)
        print(f"Time for VarDropout: {time.perf_counter() - t0:.4f}s")
        current_dataset_timings['VarDropout'] = time.perf_counter() - t0

        # 14) Ablation Impact Score (TM Adaptation)
        t0 = time.perf_counter()
        ablation_impact_scores = np.zeros(n_feats)
        # original_sums_val_for_ablation already computed as original_sums_val_for_var_dropout
        # original_true_class_scores_val_ablation already computed as original_true_class_scores_val_var_dropout
        for f_idx in range(n_feats):
            X_val_ablated_f = X_val_bin.copy()
            X_val_ablated_f[:, f_idx] = 0 # Systematically set feature f to 0
            ablated_sums_val = tm.predict(X_val_ablated_f, return_class_sums=True)[1]
            ablated_true_class_scores_val = ablated_sums_val[np.arange(len(y_val)), y_val]
            ablation_impact_scores[f_idx] = np.mean(np.abs(original_true_class_scores_val_var_dropout - ablated_true_class_scores_val))
        ablation_impact_norm = minmax_norm(ablation_impact_scores)
        print(f"Time for AblationImpact: {time.perf_counter() - t0:.4f}s")
        current_dataset_timings['AblationImpact'] = time.perf_counter() - t0

        # 15) Smooth Output Stability Score (TM Adaptation)
        t0_smooth_stability = time.perf_counter()
        N_NOISE_SAMPLES_SMOOTH = 10 # Number of noisy samples
        NOISE_LEVEL = 0.05 # Percentage of bits to flip
        smooth_stability_scores = np.zeros(n_feats)
        for f_idx in range(n_feats):
            score_diffs_for_f = []
            for _ in range(N_NOISE_SAMPLES_SMOOTH):
                noise = np.random.choice([0, 1], size=X_val_bin.shape, p=[1-NOISE_LEVEL, NOISE_LEVEL])
                X_val_noisy = X_val_bin ^ noise # XOR to flip bits
                noisy_sums1 = tm.predict(X_val_noisy, return_class_sums=True)[1][np.arange(len(y_val)), y_val]
                X_val_noisy_f_flipped = X_val_noisy.copy()
                X_val_noisy_f_flipped[:, f_idx] = 1 - X_val_noisy_f_flipped[:, f_idx] # Flip feature f
                noisy_sums2_f_flipped = tm.predict(X_val_noisy_f_flipped, return_class_sums=True)[1][np.arange(len(y_val)), y_val]
                score_diffs_for_f.append(np.mean(np.abs(noisy_sums1 - noisy_sums2_f_flipped)))
            smooth_stability_scores[f_idx] = np.mean(score_diffs_for_f)
        smooth_stability_norm = minmax_norm(smooth_stability_scores)
        current_dataset_timings['SmoothStabil'] = time.perf_counter() - t0_smooth_stability
        print(f"Time for SmoothStabil: {time.perf_counter() - t0_smooth_stability:.4f}s")

        # 16) Chi-squared (Filter) - Calculated on training data
        t0 = time.perf_counter()
        # chi2 requires non-negative features, which X_train_bin is.
        chi2_scores, _ = chi2(X_train_bin, y_train)
        chi2_norm = minmax_norm(np.nan_to_num(chi2_scores)) # Handle potential NaNs if a feature is constant
        current_dataset_timings['Chi2'] = time.perf_counter() - t0
        print(f"Time for Chi2: {time.perf_counter() - t0:.4f}s")

        # 14) Variance Threshold (Filter) - Score is the variance - Calculated on training data
        t0 = time.perf_counter()
        selector_var = VarianceThreshold()
        try:
            selector_var.fit(X_train_bin)
            variance_scores = selector_var.variances_
        except ValueError as e:
                if "No feature in X meets the variance threshold" in str(e): # Python 3.10+ message
                    print(f"Warning: All features in X_train_bin for dataset {current_dataset_name} have zero variance. Setting variance_scores to 0 for all features.")
                variance_scores = np.zeros(X_train_bin.shape[1])
        variance_norm = minmax_norm(variance_scores)
        current_dataset_timings['Variance'] = time.perf_counter() - t0
        print(f"Time for Variance: {time.perf_counter() - t0:.4f}s")

        # -------------------------
        # 17) SHAP
        # -------------------------
        
        # SHAP Explanation
        print("Computing SHAP explanation...")
        t0 = time.perf_counter()
        sample_idx_for_shap = 0
        sample_shap = X_val_bin[sample_idx_for_shap].reshape(1, -1) # Sample from validation set
        background_size = min(100, X_train_bin.shape[0]) # Adjust background size based on actual train set
        background_indices = np.random.choice(X_train_bin.shape[0], background_size, replace=False)
        background = X_train_bin[background_indices] # Background from training set
        def shap_predict(X):
            # SHAP's masker should provide binary data if background is binary.
            # Add a check and binarization just in case.
            if X.dtype != np.uint32 or X.min() < 0 or X.max() > 1:
                X_processed = np.nan_to_num(X, nan=0.0)
                X_binarized = (X_processed > 0.5).astype(np.uint32)
                return predict_proba(tm, X_binarized, y_val)
            return predict_proba(tm, X, y_val) # Use y_val as context for SHAP (explaining X_val_bin)

        masker = shap.maskers.Independent(X_train_bin) # Masker uses training data
        shap_explainer = shap.KernelExplainer(shap_predict, background, masker=masker)
        shap_values = shap_explainer.shap_values(sample_shap, nsamples=min(50, X_val_bin.shape[0])) # nsamples from X_val_bin
        predicted_class = tm.predict(sample_shap)[0]

        if isinstance(shap_values, list):
            shap_exp = shap_values[predicted_class] if predicted_class < len(shap_values) else shap_values[0]
        else:
            shap_exp = shap_values[predicted_class] if shap_values.shape[0] > 1 else shap_values
        shap_exp = np.atleast_3d(shap_exp)
        shap_scores = np.mean(np.abs(shap_exp), axis=2).flatten()
        shap_norm = minmax_norm(shap_scores)
        current_dataset_timings['SHAP'] = time.perf_counter() - t0
        print(f"Time for SHAP: {time.perf_counter() - t0:.4f}s")


        # -------------------------
        # 18) LIME
        # -------------------------
        print("Computing LIME explanation…")
        t0 = time.perf_counter()
        feat_names = [f"f{i}" for i in range(n_feats)]
        class_names= [str(c)    for c in range(n_classes)]
        lime_expl   = LimeTabularExplainer(
            training_data       = X_train_bin[:min(2000, X_train_bin.shape[0])], # LIME explainer trained on X_train_bin
            feature_names       = feat_names,
            class_names         = class_names,
            mode                = 'classification',
            discretize_continuous= False,
            verbose             = False,
            random_state        = 42
        )
        def lime_pred(X):
            # LIME generates float samples. Binarize them for the TM.
            X_processed = np.nan_to_num(X, nan=0.0) # Replace NaNs with 0.0
            X_binarized = (X_processed > 0.5).astype(np.uint32) # Binarize and cast
            return predict_proba(tm, X_binarized, y_train) # Use y_train as context for LIME
        lime_imp = np.zeros(n_feats)

        num_lime_samples = min(20, X_val_bin.shape[0]) # Explain instances from validation set
        for i in range(num_lime_samples):
            exp = lime_expl.explain_instance(
                data_row    = X_val_bin[i], # Explain validation data row
                predict_fn  = lime_pred,
                num_features= n_feats,
                num_samples = min(1000, X_train_bin.shape[0]), # Perturbation samples
            )
            for feat_str, w in exp.as_list():
                idx = int(feat_str.split()[0][1:])
                lime_imp[idx] += abs(w)
        lime_norm = minmax_norm(lime_imp)
        current_dataset_timings['LIME'] = time.perf_counter() - t0
        print(f"Time for LIME: {time.perf_counter() - t0:.4f}s")

        # 19) Permutation Importance (Wrapper) - Calculated on validation data using the trained TM
        t0 = time.perf_counter()
        perm_importance_scores = np.zeros(n_feats)
        # Get baseline true class scores on validation set
        baseline_sums_val_perm = tm.predict(X_val_bin, return_class_sums=True)[1]
        baseline_true_class_scores_val_perm = baseline_sums_val_perm[np.arange(len(y_val)), y_val]
        
        for f_idx in range(n_feats):
            X_val_permuted = X_val_bin.copy()
            np.random.shuffle(X_val_permuted[:, f_idx]) # Shuffle feature f_idx
            permuted_sums_val = tm.predict(X_val_permuted, return_class_sums=True)[1]
            permuted_true_class_scores_val = permuted_sums_val[np.arange(len(y_val)), y_val]
            # Importance is the drop in true class score
            perm_importance_scores[f_idx] = np.mean(baseline_true_class_scores_val_perm - permuted_true_class_scores_val)
        perm_importance_norm = minmax_norm(perm_importance_scores)
        current_dataset_timings['PermImportance'] = time.perf_counter() - t0
        print(f"Time for PermImportance: {time.perf_counter() - t0:.4f}s")

        # 20) Integrated Gradients (IG)
        print("Computing Integrated Gradients (IG)...")
        t0 = time.perf_counter()
        ig_scores_acc = np.zeros(n_feats)
        n_explain_samples_ig = min(20, X_val_bin.shape[0]) # Number of samples from validation set to explain
        n_steps_ig = 50 # Number of steps in the Rieman sum approximation, increased slightly
        epsilon_ig = 0.2 # Small value for finite difference - INCREASED SIGNIFICANTLY

        # Use a subset of X_val_bin for IG calculation to save time
        indices_ig = rng.choice(X_val_bin.shape[0], n_explain_samples_ig, replace=False)
        X_explain_ig = X_val_bin[indices_ig]
        Y_explain_ig = y_val[indices_ig]

        for i in range(n_explain_samples_ig):
            x_orig = X_explain_ig[i]
            y_true_idx = Y_explain_ig[i]
            baseline = np.zeros_like(x_orig) # Zero baseline
            
            path_attributions_sample = np.zeros(n_feats)

            for step in range(1, n_steps_ig + 1):
                alpha = float(step) / n_steps_ig
                interpolated_input = baseline + alpha * (x_orig - baseline)
                
                # Approx gradient dF/dx_j at this interpolated_input
                # F is the probability of the true class
                grads_at_interpolated_input = np.zeros(n_feats)
                for f_idx in range(n_feats):
                    perturbed_input_plus = interpolated_input.copy()
                    perturbed_input_plus[f_idx] += epsilon_ig
                    
                    # Binarize before predict_proba
                    perturbed_input_plus_bin = (np.nan_to_num(perturbed_input_plus, nan=0.0) > 0.5).astype(np.uint32)
                    
                    perturbed_input_minus = interpolated_input.copy()
                    perturbed_input_minus[f_idx] -= epsilon_ig

                    # Binarize before predict_proba
                    perturbed_input_minus_bin = (np.nan_to_num(perturbed_input_minus, nan=0.0) > 0.5).astype(np.uint32)

                    # Ensure inputs are 2D for predict_proba
                    prob_plus = predict_proba(tm, perturbed_input_plus_bin.reshape(1, -1), Y_explain_ig)[0, y_true_idx]
                    prob_minus = predict_proba(tm, perturbed_input_minus_bin.reshape(1, -1), Y_explain_ig)[0, y_true_idx]
                    
                    grads_at_interpolated_input[f_idx] = (prob_plus - prob_minus) / (2 * epsilon_ig)
                
                path_attributions_sample += grads_at_interpolated_input
                
            ig_scores_acc += (x_orig - baseline) * (path_attributions_sample / n_steps_ig)

        ig_scores = ig_scores_acc / n_explain_samples_ig
        print(f"  Raw IG scores (first 10): {ig_scores[:10]}")
        print(f"  Std dev of raw IG scores: {np.std(ig_scores)}")
        ig_norm = minmax_norm(np.abs(ig_scores)) # Take absolute for importance magnitude
        current_dataset_timings['IG'] = time.perf_counter() - t0
        print(f"Time for IG: {time.perf_counter() - t0:.4f}s")

        # ----------------------------------
        # SmoothGrad-Squared & VarGrad (based on IG logic)
        # ----------------------------------
        print("Computing SmoothGrad-Squared and VarGrad...")
        t0 = time.perf_counter()
        smoothgrad_sq_scores_acc = np.zeros(n_feats)
        vargrad_scores_acc = np.zeros(n_feats)
        
        n_noise_samples_ensemble = 10 # Number of noisy samples per input for ensemble methods
        noise_flip_prob = 0.05 # Probability to flip a bit for noise

        # Using the same subset as IG for these methods
        # X_explain_ensemble = X_explain_ig
        # Y_explain_ensemble = Y_explain_ig
        # n_explain_samples_ensemble = n_explain_samples_ig

        for i in range(n_explain_samples_ig): # Iterate over the same samples used for IG
            x_orig = X_explain_ig[i]
            y_true_idx = Y_explain_ig[i]
            
            igs_for_noisy_versions = []
            for _ in range(n_noise_samples_ensemble):
                # Generate noise by flipping bits
                noise_mask = rng.choice([0, 1], size=x_orig.shape, p=[1 - noise_flip_prob, noise_flip_prob])
                x_noisy = x_orig ^ noise_mask # XOR to flip bits
                
                # Calculate IG for this noisy sample (simplified: using x_noisy as the 'original' and zero baseline)
                # This reuses the IG logic but on a noisy input.
                # For a full SmoothGrad/VarGrad, the IG calculation itself would be on x_noisy.
                # Here, we'll approximate the "gradient" part of IG for the noisy sample.
                baseline_noisy = np.zeros_like(x_noisy)
                path_attributions_noisy_sample = np.zeros(n_feats)
                for step_ig_noisy in range(1, n_steps_ig + 1): # Using n_steps_ig for consistency
                    alpha_noisy = float(step_ig_noisy) / n_steps_ig
                    interpolated_input_noisy = baseline_noisy + alpha_noisy * (x_noisy - baseline_noisy)
                    grads_at_interpolated_noisy = np.zeros(n_feats)

                    for f_idx in range(n_feats):
                        p_input_plus_float = interpolated_input_noisy.copy()
                        p_input_plus_float[f_idx] += epsilon_ig
                        p_input_plus_bin = (np.nan_to_num(p_input_plus_float, nan=0.0) > 0.5).astype(np.uint32)

                        p_input_minus_float = interpolated_input_noisy.copy()
                        p_input_minus_float[f_idx] -= epsilon_ig
                        p_input_minus_bin = (np.nan_to_num(p_input_minus_float, nan=0.0) > 0.5).astype(np.uint32)

                        p_plus = predict_proba(tm, p_input_plus_bin.reshape(1,-1), Y_explain_ig)[0,y_true_idx]
                        p_minus = predict_proba(tm, p_input_minus_bin.reshape(1,-1), Y_explain_ig)[0,y_true_idx]
                        grads_at_interpolated_noisy[f_idx] = (p_plus - p_minus) / (2*epsilon_ig)
                    path_attributions_noisy_sample += grads_at_interpolated_noisy
                ig_noisy = (x_noisy - baseline_noisy) * (path_attributions_noisy_sample / n_steps_ig)
                igs_for_noisy_versions.append(ig_noisy)
                
            igs_for_noisy_versions_arr = np.array(igs_for_noisy_versions)
            smoothgrad_sq_scores_acc += np.mean(igs_for_noisy_versions_arr**2, axis=0)
            vargrad_scores_acc += np.var(igs_for_noisy_versions_arr, axis=0)

        smoothgrad_sq_scores_raw = smoothgrad_sq_scores_acc / n_explain_samples_ig
        vargrad_scores_raw = vargrad_scores_acc / n_explain_samples_ig
        print(f"  Raw SmoothGrad-Sq scores (first 10): {smoothgrad_sq_scores_raw[:10]}")
        print(f"  Std dev of raw SmoothGrad-Sq scores: {np.std(smoothgrad_sq_scores_raw)}")
        print(f"  Raw VarGrad scores (first 10): {vargrad_scores_raw[:10]}")
        print(f"  Std dev of raw VarGrad scores: {np.std(vargrad_scores_raw)}")
        smoothgrad_sq_norm = minmax_norm(smoothgrad_sq_scores_raw)
        vargrad_norm = minmax_norm(vargrad_scores_raw)
        current_dataset_timings['SmoothGradSq'] = time.perf_counter() - t0 # Combined time for now
        current_dataset_timings['VarGrad'] = 0 # Included in SmoothGradSq time
        print(f"Time for SmoothGradSq/VarGrad: {time.perf_counter() - t0:.4f}s")

        # -------------------------
        # 8) Collect & correlate
        # -------------------------
        scores = {
        # ———————— Filter ————————
        'MutualInfo': mi_norm,
        'Chi2':       chi2_norm,
        'Variance':   variance_norm,
        'Random' : random_norm,

        # —————— Embedded (TM-internal) ——————
        'Relevance':  relevance_n, # Original
        'Relevance-PosNeg': relevance_posneg_n, # New
        'TM-Weight':  tm_weight_norm, # Original
        'TM-Weight-PosNeg': tm_weight_posneg_norm, # New
        'CW-Sum':     cw_sum_norm, # Original
        'CW-Sum-PosNeg': cw_sum_posneg_norm, # New
        'CW-Feat':    cw_feat_n, # Original
        'CW-Feat-PosNeg': cw_feat_posneg_n, # New
        'Support-CW-Sum': support_cw_sum_norm, # Original
        'Support-CW-Sum-PosNeg': support_cw_sum_posneg_norm, # New
        'GroupLasso': group_lasso_norm, # Original
        'GroupLasso-PosNeg': group_lasso_posneg_norm, # New
        'TaylorCrit': taylor_norm, #takes too long
        'VarDropout': var_dropout_norm, #takes too long
        'AblationImpact': ablation_impact_norm,

        # —————— Wrapper ——————
        'Dropout':    drop_norm, #takes too long
        'PermImportance': perm_importance_norm,
        'SHAP':       shap_norm,
        'LIME':       lime_norm,
        'IG':         ig_norm,
        'SmoothGradSq': smoothgrad_sq_norm,
        'VarGrad':    vargrad_norm,

        # —————— Embedded (TM-internal) - Continued ——————
        'SmoothStabil': smooth_stability_norm, # Original
        'Margin':     margin_norm, # Original
        'Margin-PosNeg': margin_posneg_norm, # New
        'Entropy':    entropy_norm, # Original
        'Entropy-PosNeg': entropy_posneg_norm, # New
        'Gini':       gini_norm, # Original
        'Gini-PosNeg': gini_posneg_norm, # New
        'Stability':  stab_norm, # Original
        'Stability-PosNeg': stab_posneg_norm # New
        }


        method_names = list(scores.keys())
        arr   = np.vstack([scores[m_name] for m_name in method_names])
        corr  = np.corrcoef(arr)

        print("\nStandard deviations of normalized scores (check for zeros):")
        for i, current_method_name in enumerate(method_names):
            std_dev = np.std(arr[i, :])
            print(f"  {current_method_name:15s}: {std_dev:.4f}")
            if std_dev < 1e-6 : # Check for effectively zero std dev
                print(f"    WARNING: Method '{current_method_name}' has near-zero standard deviation after normalization. Raw scores might have been constant.")

        

        plt.figure(figsize=(10,8))
        im = plt.imshow(corr, vmin=-1, vmax=1, cmap='coolwarm')
        plt.xticks(range(len(method_names)), method_names, rotation=45, ha='right')
        plt.yticks(range(len(method_names)), method_names)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        for i in range(len(method_names)):
            for j in range(len(method_names)):
                c = corr[i,j]
                color = 'white' if abs(c)>0.5 else 'black'
                plt.text(j, i, f"{c:.2f}", ha='center', va='center', color=color, fontsize=7) # Smaller font for crowded plots
        plt.title(f'Score Correlations ({current_dataset_name})')
        plt.tight_layout()
        plt.savefig(f"score_correlations_{current_dataset_name}.png") # Save plot
        plt.close() # Close plot to free memory
        
        # -------------------------
        # 9) Top‐K comparison
        # -------------------------
        K_list = list(np.unique(np.linspace(1, min(50,n_feats), 25, dtype=int))) # Adjusted K_list for potentially more features
        trials = 10
        results = {m_name: [] for m_name in method_names}
        
        cmap = plt.get_cmap('tab20') # tab20 has 20 distinct colors
        base_cmap_colors = list(cmap.colors)
        num_total_methods = len(method_names)
        # Create a list of colors, cycling through cmap.colors if necessary
        colors_for_plot = [base_cmap_colors[i % len(base_cmap_colors)] for i in range(num_total_methods)]

        for current_method_name in method_names:
            ordering = np.argsort(scores[current_method_name])[::-1]
            for K in K_list:
                # Ensure K is not zero and selected features are unique if K is small
                sel_indices = ordering[:K]
                if K == 0 or len(sel_indices) == 0: # Handle K=0 or no features selected
                    results[current_method_name].append(0.0) # Or handle as appropriate, e.g., accuracy of a default predictor
                    continue

                accs = []
                print(f"\n-- Using ordering by {current_method_name} --")
                for _ in range(trials):
                    tm2 = TMClassifier(
                        number_of_clauses=clauses, T=T_current_dataset, s=s_current_dataset,
                        max_included_literals=max_lit, platform='CPU'
                    )
                    tm2.fit(X_train_bin[:,sel_indices], y_train, epochs=epochs)
                    accs.append(100*(tm2.predict(X_test_bin[:,sel_indices])==y_test).mean()) # Evaluate on X_test_bin for the plot
                    print(f"Trial {_+1}/{trials} for K={K} with {current_method_name} ordering")
                results[current_method_name].append(np.mean(accs))

        plt.figure(figsize=(10,6))
        for idx, current_method_name in enumerate(method_names):
            plt.plot(K_list, results[current_method_name], marker='o', label=current_method_name, color=colors_for_plot[idx])
        plt.xlabel('Number of Features (K)')
        plt.ylabel(f'Avg Test Accuracy % on {current_dataset_name}')
        plt.title(f'Top-K Feature Pruning - Performance on Test Set ({current_dataset_name}, avg of {trials} runs)')
        plt.legend(ncol=3, fontsize=8)
        plt.tight_layout()
        plt.savefig(f"top_k_performance_{current_dataset_name}.png")
        plt.close()
        # --- 10) Compute & display normalized AUC of each Top-K curve ---

        # 1) compute AUC (area under accuracy vs K)
        aucs = {}
        for current_method_name in method_names:
            # trapz over K_list; you could also do simple sum if K_list evenly spaced
            aucs[current_method_name] = trapz(results[current_method_name], K_list)

        # 2) normalize across methods
        auc_vals = np.array(list(aucs.values()))
        auc_norm = minmax_norm(auc_vals)

        # 3) print table
        print(f"\nNormalized AUC of Test Set Performance for {current_dataset_name} (higher = better overall):")
        for current_method_name, val in zip(method_names, auc_norm):
            print(f"  {current_method_name:15s}: {val:.3f}")

        # 4) bar chart
        plt.figure(figsize=(6,4))
        plt.bar(method_names, auc_norm, color=colors_for_plot)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel("Normalized AUC")
        plt.title(f"Area under Accuracy–vs–K Curves (Test Set Performance - {current_dataset_name})")
        plt.tight_layout()
        plt.savefig(f"auc_top_k_{current_dataset_name}.png")
        plt.close()

        print("\n––––––––––––––––––––––––")
        print(f"Timing summary for {current_dataset_name} (seconds):")
        for method_name, secs in current_dataset_timings.items():
            print(f"  {method_name:15s} : {secs:6.6f}s")
        print("––––––––––––––––––––––––\n")

        # Final evaluation on the true test set (already done for the plot)
        # The script's main goal is to compare FS methods.
        # The `results` dictionary now holds test accuracies for each K and method.
        all_dataset_results[current_dataset_name] = {
            "timings_seconds": current_dataset_timings,
            "feature_correlation_matrix": {"method_names": method_names, "matrix": corr.tolist()},
            "normalized_auc_top_k": dict(zip(method_names, auc_norm.tolist())),
            "top_k_results": results
        }

        # -------------------------
        # 11) Save results to JSON
        # -------------------------
        # Calculate accuracy of the TM trained on all features on the test set
        y_test_pred = tm.predict(X_test_bin)
        full_model_test_accuracy = 100 * (y_test_pred == y_test).mean()
        full_model_test_f1_ =  100 * f1_score(y_test, y_test_pred, average="macro")
        full_model_test_bal_accuracy = balanced_accuracy_score(y_test, y_test_pred)
        full_model_test_matthews_corrcoef = matthews_corrcoef(y_test, y_test_pred)

        output_data = {
            "experiment_description": {
            "dataset_name": f"{current_dataset_name}{dataset_type_info}",
                "X_train_shape": list(X_train_bin.shape),
                "y_train_shape": list(y_train.shape),
                "X_val_shape": list(X_val_bin.shape),
                "y_val_shape": list(y_val.shape),
                "X_test_shape": list(X_test_bin.shape),
                "y_test_shape": list(y_test.shape),
                "tm_parameters": {
                    "clauses": clauses,
                    "T_used": T_current_dataset,
                    "s_used": s_current_dataset,
                    "max_included_literals": max_lit,
                    "epochs_for_feature_scoring_model": epochs # Epochs used to train the TM for FS
                },
                "top_k_comparison_trials": trials
            },
            "timings_seconds": current_dataset_timings,
            "feature_correlation_matrix": {
                "method_names": method_names, # Names corresponding to rows/cols of the matrix
                "matrix": corr.tolist() # Convert numpy array to list for JSON
            },
            "normalized_auc_top_k": dict(zip(method_names, auc_norm.tolist())),
            "full_model_test_accuracy_percent": full_model_test_accuracy,
            "full_model_test_f1_score": full_model_test_f1_,
            "full_model_test_balanced_accuracy": full_model_test_bal_accuracy,
            "full_model_test_matthews_corrcoef": full_model_test_matthews_corrcoef,
            "top_k_accuracies_vs_k": results # Store the actual Top-K curves
        }

        if ground_truth_features is not None:
            output_data["experiment_description"]["ground_truth_important_features"] = ground_truth_features.tolist()
        json_filename = f"fs_experiment_results_{current_dataset_name}.json"
        
        with open(json_filename, 'w') as f:
            json.dump(output_data, f, indent=4)
        print(f"\nExperiment results for {current_dataset_name} saved to: {json_filename}")
