import time
import warnings
import json # Added for JSON output
import numpy as np
import matplotlib.pyplot as plt

# Switch to Wisconsin Breast Cancer Dataset
from sklearn.datasets import load_breast_cancer # Added import
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score

from numpy import trapz
from tmu.models.classification.vanilla_classifier import TMClassifier
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.feature_selection import chi2, VarianceThreshold # Added for new filters



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
    timings = {}

    # -------------------------
    # 1) Load Wisconsin Breast Cancer Dataset & preprocess
    # -------------------------
    print("Loading Wisconsin Breast Cancer Dataset...")
    cancer = load_breast_cancer()
    X_all_float = cancer.data
    y_all = cancer.target.astype(np.uint32)

    # Binarize features at their respective medians
    X_binarized = np.zeros_like(X_all_float, dtype=np.uint32)
    for i in range(X_all_float.shape[1]):
        median_val = np.median(X_all_float[:, i])
        X_binarized[:, i] = (X_all_float[:, i] > median_val).astype(np.uint32)

    rng = np.random.default_rng(42)

    # Split data into training/validation pool and test set
    # (e.g., 80% for train/val pool, 20% for test)
    X_train_val_pool, X_test_bin, y_train_val_pool, y_test = train_test_split(
        X_binarized, y_all,
        test_size=0.2, 
        random_state=42,
        stratify=y_all
    )

    # Split the training/validation pool into actual training and validation sets
    # (e.g., 75% of pool for train, 25% for val -> 60% train, 20% val of total)
    X_train_bin, X_val_bin, y_train, y_val = train_test_split(
        X_train_val_pool, y_train_val_pool,
        test_size=0.25, 
        random_state=42,
        stratify=y_train_val_pool
    )
    
    print(f"Dataset shapes: X_train_bin: {X_train_bin.shape}, X_val_bin: {X_val_bin.shape}, X_test_bin: {X_test_bin.shape}")

    # -------------------------
    # 2) Train TM & record weight history
    # -------------------------
    clauses = 500
    T       = 600
    s       = 3.0
    max_lit = 32
    epochs  = 10

    tm = TMClassifier(
        number_of_clauses      = clauses,
        T                      = T,
        s                      = s,
        max_included_literals  = max_lit,
        platform               = 'CPU'
    )

    net_w_history = []
    for ep in range(epochs):
        tm.fit(X_train_bin, y_train, epochs=1)
        ta = count_ta_states(tm)
        pos, neg = count_clause_weights(tm, ta)
        net_w_history.append(pos - neg)
        print("Epoch", ep+1, epochs)
    net_w_history = np.stack(net_w_history, axis=0)

    # Accuracy on validation set (used for subsequent calculations for feature scoring)
    y_pred_val = tm.predict(X_val_bin)
    full_acc_val = 100 * (y_pred_val == y_val).mean()
    print(f"Full‐feature Accuracy on Breast Cancer Validation Set: {full_acc_val:.2f}%\n")

    # -------------------------
    # 3) Base quantities
    # -------------------------
    net_w = net_w_history[-1]         # (n_classes, n_feats)
    abs_w = np.abs(net_w)
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
    # 1) TM‐weight (max abs)
    t0 = time.perf_counter()
    weight_score = abs_w.max(axis=0)
    tm_norm      = minmax_norm(weight_score)
    print(f"Time for TM-Weight: {time.perf_counter() - t0:.4f}s")
    timings['TM-Weight'] = time.perf_counter() - t0

    # 2) Class‐weighted sum (CW-Sum)
    t0 = time.perf_counter()
    cw_sum       = (class_w[:,None] * abs_w).sum(axis=0)
    cw_sum_norm  = minmax_norm(cw_sum)

    print(f"Time for CW-Sum: {time.perf_counter() - t0:.4f}s")
    # -------------------------
    # 2b) Per‐feature CW (CW-Feat)
    # -------------------------
    timings['CW-Sum'] = time.perf_counter() - t0 # Belongs to CW-Sum
    t0_cw_feat = time.perf_counter() # New timer for CW-Feat
    freq      = ta.sum(axis=(1,2))                   # (C, F)
    alpha     = freq/(freq.sum(axis=0,keepdims=True)+1e-12)
    cw_feat   = (alpha * abs_w).sum(axis=0)
    cw_feat_n = minmax_norm(cw_feat)
    print(f"Time for CW-Feat: {time.perf_counter() - t0_cw_feat:.4f}s")
    timings['CW-Feat'] = time.perf_counter() - t0_cw_feat # Corrected timing

    # 2c) Supportive Class-Weighted Sum (Prioritizes features for low-accuracy classes)
    t0_support_cw_sum = time.perf_counter()
    class_error_rate = 1.0 - class_acc # Error rate for each class
    class_error_weight = class_error_rate / (class_error_rate.sum() + 1e-12)
    support_cw_sum = (class_error_weight[:, None] * abs_w).sum(axis=0)
    support_cw_sum_norm = minmax_norm(support_cw_sum)
    print(f"Time for Support-CW-Sum: {time.perf_counter() - t0_support_cw_sum:.4f}s")
    timings['Support-CW-Sum'] = time.perf_counter() - t0_support_cw_sum
    # 3) Margin top vs runner‐up
    t0_margin = time.perf_counter() # New timer for Margin
    sorted_abs   = np.sort(abs_w, axis=0)
    margin       = sorted_abs[-1] - sorted_abs[-2]
    margin_norm  = minmax_norm(margin)

    print(f"Time for Margin: {time.perf_counter() - t0_margin:.4f}s")
    # 4) Entropy‐based (invert so low‐entropy→high score)
    timings['Margin'] = time.perf_counter() - t0_margin # Corrected timing
    t0_entropy = time.perf_counter() # New timer for Entropy
    p = abs_w / (abs_w.sum(axis=0, keepdims=True) + 1e-12)
    entropy = - (p * np.log(p + 1e-12)).sum(axis=0)
    #entropy_score = 1.0 - (entropy / np.log(n_classes))
    entropy_score = np.log(n_classes) - entropy
    entropy_norm  = minmax_norm(entropy_score)

    print(f"Time for Entropy: {time.perf_counter() - t0_entropy:.4f}s")
    # 5) Gini‐based (invert so mass concentrated→high score)
    timings['Entropy'] = time.perf_counter() - t0_entropy # Corrected timing
    t0_gini = time.perf_counter() # New timer for Gini
    gini_score = 1.0 - (1.0 - (p**2).sum(axis=0))  # = sum(p^2)
    gini_norm  = minmax_norm(gini_score)

    print(f"Time for Gini: {time.perf_counter() - t0_gini:.4f}s")
    # 6) Stability across epochs
    timings['Gini'] = time.perf_counter() - t0_gini # Corrected timing
    t0_stability = time.perf_counter() # New timer for Stability
    max_abs_hist = np.max(np.abs(net_w_history), axis=1)  # (epochs, n_feats)
    std_hist     = max_abs_hist.std(axis=0)
    mean_abs     = max_abs_hist.mean(axis=0)
    stab_score   = mean_abs / (std_hist + 1e-6)
    stab_norm    = minmax_norm(stab_score)

    print(f"Time for Stability: {time.perf_counter() - t0_stability:.4f}s")
    # -------------------------
    # 5) Other metrics
    # -------------------------
    timings['Stability'] = time.perf_counter() - t0_stability # Corrected timing
    t0_dropout = time.perf_counter() # New timer for Dropout
    # Dropout sensitivity
    drop = np.zeros(n_feats)
    for f in range(n_feats):
        Xm = X_val_bin.copy() # Use validation set
        Xm[:,f] = 0
        drop[f] = full_acc_val - 100*(tm.predict(Xm)==y_val).mean() # Use validation set
    drop_norm = minmax_norm(drop)

    print(f"Time for Dropout: {time.perf_counter() - t0_dropout:.4f}s")

    # Mutual information
    timings['Dropout'] = time.perf_counter() - t0_dropout # Corrected timing
    t0_mutual = time.perf_counter() # New timer for MutualInfo
    correct_val = (y_pred_val == y_val).astype(int) # Correctness on validation set
    mi      = np.array([mutual_info_score(X_val_bin[:,f], correct_val) for f in range(n_feats)]) # Use validation set
    mi_norm = minmax_norm(mi)

    print(f"Time for MutualInfo: {time.perf_counter() - t0_mutual:.4f}s")
    timings['MutualInfo'] = time.perf_counter() - t0_mutual # Corrected timing name
    t0_relevance = time.perf_counter() # New timer for Relevance
    norm_abs   = abs_w / (abs_w.sum(axis=1, keepdims=True) + 1e-12)
    relevance  = (class_w[:,None] * norm_abs).sum(axis=0)
    relevance_n= minmax_norm(relevance)
    print(f"Time for Relevance: {time.perf_counter() - t0_relevance:.4f}s")
    timings['Relevance'] = time.perf_counter() - t0_relevance

    # ----------------------------------
    # NEW FEATURE SCORING TECHNIQUES
    # ----------------------------------

    # 7) L1 Regularization Score (TM Adaptation)
    t0_l1_reg = time.perf_counter()
    l1_reg_score = abs_w.sum(axis=0) # Sum of absolute weights for each feature across all classes
    l1_reg_norm = minmax_norm(l1_reg_score)
    print(f"Time for L1-Reg: {time.perf_counter() - t0_l1_reg:.4f}s")
    timings['L1-Reg'] = time.perf_counter() - t0_l1_reg

    # 8) Group Lasso Score (TM Adaptation for MNIST 28x28)
    t0_group_lasso = time.perf_counter()
    img_dim = int(np.sqrt(n_feats))
    if img_dim * img_dim == n_feats: # Check if features form a square image
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
    else:
        print("Warning: Group Lasso score not computed as n_feats is not a perfect square.")
        group_lasso_norm = np.full(n_feats, 0.5) # Default if not applicable
    print(f"Time for GroupLasso: {time.perf_counter() - t0_group_lasso:.4f}s")
    timings['GroupLasso'] = time.perf_counter() - t0_group_lasso

    # 9) Taylor Criteria Score (First-Order Perturbation Impact)
    t0_taylor = time.perf_counter()
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
    print(f"Time for TaylorCrit: {time.perf_counter() - t0_taylor:.4f}s")
    timings['TaylorCrit'] = time.perf_counter() - t0_taylor

    # 10) Variational Dropout Score (TM Adaptation)
    t0_var_dropout = time.perf_counter()
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
    print(f"Time for VarDropout: {time.perf_counter() - t0_var_dropout:.4f}s")
    timings['VarDropout'] = time.perf_counter() - t0_var_dropout

    # 11) Ablation Impact Score (TM Adaptation)
    t0_ablation_impact = time.perf_counter()
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
    print(f"Time for AblationImpact: {time.perf_counter() - t0_ablation_impact:.4f}s")
    timings['AblationImpact'] = time.perf_counter() - t0_ablation_impact

    # 12) Smooth Output Stability Score (TM Adaptation)
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
    print(f"Time for SmoothStabil: {time.perf_counter() - t0_smooth_stability:.4f}s")
    timings['SmoothStabil'] = time.perf_counter() - t0_smooth_stability

    # 13) Chi-squared (Filter) - Calculated on training data
    t0_chi2 = time.perf_counter()
    # chi2 requires non-negative features, which X_train_bin is.
    chi2_scores, _ = chi2(X_train_bin, y_train)
    chi2_norm = minmax_norm(np.nan_to_num(chi2_scores)) # Handle potential NaNs if a feature is constant
    print(f"Time for Chi2: {time.perf_counter() - t0_chi2:.4f}s")
    timings['Chi2'] = time.perf_counter() - t0_chi2

    # 14) Variance Threshold (Filter) - Score is the variance - Calculated on training data
    t0_variance = time.perf_counter()
    selector_var = VarianceThreshold()
    selector_var.fit(X_train_bin)
    variance_scores = selector_var.variances_
    variance_norm = minmax_norm(variance_scores)
    print(f"Time for Variance: {time.perf_counter() - t0_variance:.4f}s")
    timings['Variance'] = time.perf_counter() - t0_variance

    # -------------------------
    # 6) SHAP
    # -------------------------
    
    # SHAP Explanation
    t0 = time.perf_counter()
    print("Computing SHAP explanation...")
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
    print(f"Time for SHAP: {time.perf_counter() - t0:.4f}s")
    shap_norm = minmax_norm(shap_scores)
    timings['SHAP'] = time.perf_counter() - t0


    # -------------------------
    # 7) LIME
    # -------------------------
    t0 = time.perf_counter()
    print("Computing LIME explanation…")
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
    print(f"Time for LIME: {time.perf_counter() - t0:.4f}s")
    timings['LIME'] = time.perf_counter() - t0

    # 15) Permutation Importance (Wrapper) - Calculated on validation data using the trained TM
    t0_perm_importance = time.perf_counter()
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
    print(f"Time for PermImportance: {time.perf_counter() - t0_perm_importance:.4f}s")
    timings['PermImportance'] = time.perf_counter() - t0_perm_importance

    # ----------------------------------
    # Integrated Gradients (IG)
    # ----------------------------------
    t0_ig = time.perf_counter()
    print("Computing Integrated Gradients (IG)...")
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
    print(f"Time for IG: {time.perf_counter() - t0_ig:.4f}s")
    ig_norm = minmax_norm(np.abs(ig_scores)) # Take absolute for importance magnitude
    timings['IG'] = time.perf_counter() - t0_ig

    # ----------------------------------
    # SmoothGrad-Squared & VarGrad (based on IG logic)
    # ----------------------------------
    t0_smooth_var_grad = time.perf_counter()
    print("Computing SmoothGrad-Squared and VarGrad...")
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
    print(f"Time for SmoothGradSq/VarGrad: {time.perf_counter() - t0_smooth_var_grad:.4f}s")
    smoothgrad_sq_norm = minmax_norm(smoothgrad_sq_scores_raw)
    vargrad_norm = minmax_norm(vargrad_scores_raw)
    timings['SmoothGradSq'] = time.perf_counter() - t0_smooth_var_grad # Combined time for now
    timings['VarGrad'] = 0 # Included in SmoothGradSq time

    # -------------------------
    # 8) Collect & correlate
    # -------------------------
    scores = {
    # ———————— Filter ————————
    'MutualInfo': mi_norm,
    'Chi2':       chi2_norm,
    'Variance':   variance_norm,

    # —————— Embedded (TM-internal) ——————
    'Relevance':  relevance_n,
    'TM-weight':  tm_norm,
    'CW-Sum':     cw_sum_norm,
    'CW-Feat':    cw_feat_n,
    'Support-CW-Sum': support_cw_sum_norm,
    'L1-Reg':     l1_reg_norm,
    'GroupLasso': group_lasso_norm,
    'TaylorCrit': taylor_norm, #takes too long
    'VarDropout': var_dropout_norm, #takes too long
    'AblationImpact': ablation_impact_norm,
    'SmoothStabil': smooth_stability_norm,
    'Margin':     margin_norm,
    'Entropy':    entropy_norm,
    'Gini':       gini_norm,
    'Stability':  stab_norm,

    # —————— Wrapper ——————
    'Dropout':    drop_norm, #takes too long
    'PermImportance': perm_importance_norm,
    'SHAP':       shap_norm,
    'LIME':       lime_norm,
    'IG':         ig_norm,
    'SmoothGradSq': smoothgrad_sq_norm,
    'VarGrad':    vargrad_norm
    }


    names = list(scores.keys())
    arr   = np.vstack([scores[n] for n in names])
    corr  = np.corrcoef(arr)

    print("\nStandard deviations of normalized scores (check for zeros):")
    for i, name in enumerate(names):
        std_dev = np.std(arr[i, :])
        print(f"  {name:15s}: {std_dev:.4f}")
        if std_dev < 1e-6 : # Check for effectively zero std dev
            print(f"    WARNING: Method '{name}' has near-zero standard deviation after normalization. Raw scores might have been constant.")

    

    plt.figure(figsize=(10,8))
    im = plt.imshow(corr, vmin=-1, vmax=1, cmap='coolwarm')
    plt.xticks(range(len(names)), names, rotation=45, ha='right')
    plt.yticks(range(len(names)), names)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    for i in range(len(names)):
        for j in range(len(names)):
            c = corr[i,j]
            color = 'white' if abs(c)>0.5 else 'black'
            plt.text(j, i, f"{c:.2f}", ha='center', va='center', color=color, fontsize=8)
    plt.title('Score Correlations (Breast Cancer)')
    plt.tight_layout()
    plt.show()
    
    # -------------------------
    # 9) Top‐K comparison
    # -------------------------
    K_list = list(np.unique(np.linspace(1, min(50,n_feats), 25, dtype=int))) # Adjusted K_list for potentially more features
    trials = 10
    results = {name: [] for name in names}
    
    cmap = plt.get_cmap('tab20') # tab20 has 20 distinct colors
    base_cmap_colors = list(cmap.colors)
    num_total_methods = len(names)
    # Create a list of colors, cycling through cmap.colors if necessary
    colors_for_plot = [base_cmap_colors[i % len(base_cmap_colors)] for i in range(num_total_methods)]

    for name in names:
        ordering = np.argsort(scores[name])[::-1]
        for K in K_list:
            sel = ordering[:K]
            accs = []
            print(f"\n-- Using ordering by {name} --")
            for _ in range(trials):
                tm2 = TMClassifier(
                    number_of_clauses=clauses, T=T, s=s,
                    max_included_literals=max_lit, platform='CPU'
                )
                tm2.fit(X_train_bin[:,sel], y_train, epochs=epochs) 
                accs.append(100*(tm2.predict(X_test_bin[:,sel])==y_test).mean()) # Evaluate on X_test_bin for the plot
                print("Trial", _+1)
            results[name].append(np.mean(accs))

    plt.figure(figsize=(10,6))
    for name in names:
        plt.plot(K_list, results[name], marker='o', label=name, color=colors_for_plot[names.index(name)])
    plt.xlabel('Number of Features (K)')
    plt.ylabel('Avg Test Accuracy %')
    plt.title(f'Top-K Feature Pruning - Performance on Test Set (Breast Cancer, avg of {trials} runs)')
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.show()
    # --- 10) Compute & display normalized AUC of each Top-K curve ---

    # 1) compute AUC (area under accuracy vs K)
    aucs = {}
    for name in names:
        # trapz over K_list; you could also do simple sum if K_list evenly spaced
        aucs[name] = trapz(results[name], K_list)

    # 2) normalize across methods
    auc_vals = np.array(list(aucs.values()))
    auc_norm = minmax_norm(auc_vals)

    # 3) print table
    print("\nNormalized AUC of Test Set Performance (higher = better overall):")
    for name, val in zip(names, auc_norm):
        print(f"  {name:10s}: {val:.3f}")

    # 4) bar chart
    plt.figure(figsize=(6,4))
    plt.bar(names, auc_norm, color=colors_for_plot)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Normalized AUC")
    plt.title("Area under Accuracy–vs–K Curves (Test Set Performance - Breast Cancer)")
    plt.tight_layout()
    plt.show()

    print("\n––––––––––––––––––––––––")
    print("Timing summary (seconds):")
    for name, secs in timings.items():
        print(f"  {name:15s} : {secs:6.6f}s")
    print("––––––––––––––––––––––––\n")

    # Final evaluation on the true test set (already done for the plot)
    # The script's main goal is to compare FS methods.
    # The `results` dictionary now holds test accuracies for each K and method.

    # -------------------------
    # 11) Save results to JSON
    # -------------------------
    # Calculate accuracy of the TM trained on all features on the test set
    full_model_test_accuracy = 100 * (tm.predict(X_test_bin) == y_test).mean()

    output_data = {
        "experiment_description": {
            "dataset_name": "Wisconsin Breast Cancer Dataset (Binarized at Median)",
            "X_train_shape": list(X_train_bin.shape),
            "y_train_shape": list(y_train.shape),
            "X_val_shape": list(X_val_bin.shape),
            "y_val_shape": list(y_val.shape),
            "X_test_shape": list(X_test_bin.shape),
            "y_test_shape": list(y_test.shape),
            "tm_parameters": {
                "clauses": clauses,
                "T": T,
                "s": s,
                "max_included_literals": max_lit,
                "epochs_for_feature_scoring_model": epochs # Epochs used to train the TM for FS
            },
            "top_k_comparison_trials": trials
        },
        "timings_seconds": timings,
        "feature_correlation_matrix": {
            "method_names": names, # Names corresponding to rows/cols of the matrix
            "matrix": corr.tolist() # Convert numpy array to list for JSON
        },
        "normalized_auc_top_k": dict(zip(names, auc_norm.tolist())),
        "full_model_test_accuracy_percent": full_model_test_accuracy
    }

    json_filename = "fs_experiment_results_breast_cancer.json"
    
    with open(json_filename, 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"\nExperiment results saved to: {json_filename}")
