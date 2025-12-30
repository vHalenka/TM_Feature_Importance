"""
Feature scoring methods for Tsetlin Machines.

This module contains all feature importance scoring methods tested in the study.
"""
import time
import numpy as np
import warnings
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import chi2, VarianceThreshold
import shap
from lime.lime_tabular import LimeTabularExplainer

from ..utils import tm_utils, serialization


def compute_tm_feature_scores(tm, X_train, X_val, y_train, y_val, ta, pos_w, neg_w, 
                              pos_w_history, neg_w_history, epochs, rng):
    """
    Compute all TM-based feature importance scores.
    
    Args:
        tm: Trained TMClassifier
        X_train, X_val: Training and validation features
        y_train, y_val: Training and validation labels
        ta: TA states from count_ta_states
        pos_w, neg_w: Positive and negative clause weights
        pos_w_history, neg_w_history: History of weights across epochs
        epochs: Number of training epochs
        rng: Random number generator
        
    Returns:
        Dictionary mapping method names to normalized feature scores
        Dictionary mapping method names to computation times
    """
    scores = {}
    timings = {}
    
    n_classes, n_feats = pos_w.shape
    abs_w = np.abs(pos_w - neg_w)
    abs_w_posneg = np.abs(pos_w + neg_w)
    
    # Per-class validation accuracies for class weighting
    y_pred_val = tm.predict(X_val)
    class_acc = np.array([
        (y_pred_val[y_val == c] == c).mean() if np.any(y_val == c) else 0.0
        for c in range(n_classes)
    ])
    class_w = class_acc / (class_acc.sum() + 1e-12)
    
    # --- TM-Weight (max abs) ---
    t0 = time.perf_counter()
    weight_score = abs_w.max(axis=0)
    scores['TM-Weight'] = serialization.minmax_norm(weight_score)
    timings['TM-Weight'] = time.perf_counter() - t0
    
    # --- TM-Weight-PosNeg ---
    t0 = time.perf_counter()
    weight_score_posneg = abs_w_posneg.max(axis=0)
    scores['TM-Weight-PosNeg'] = serialization.minmax_norm(weight_score_posneg)
    timings['TM-Weight-PosNeg'] = time.perf_counter() - t0
    
    # --- CW-Sum (Class-weighted sum) ---
    t0 = time.perf_counter()
    cw_sum = (class_w[:, None] * abs_w).sum(axis=0)
    scores['CW-Sum'] = serialization.minmax_norm(cw_sum)
    timings['CW-Sum'] = time.perf_counter() - t0
    
    # --- CW-Sum-PosNeg ---
    t0 = time.perf_counter()
    cw_sum_posneg = (class_w[:, None] * abs_w_posneg).sum(axis=0)
    scores['CW-Sum-PosNeg'] = serialization.minmax_norm(cw_sum_posneg)
    timings['CW-Sum-PosNeg'] = time.perf_counter() - t0
    
    # --- CW-Feat (Per-feature class-weighted) ---
    t0 = time.perf_counter()
    freq = ta.sum(axis=(1, 2))
    alpha = freq / (freq.sum(axis=0, keepdims=True) + 1e-12)
    cw_feat = (alpha * abs_w).sum(axis=0)
    scores['CW-Feat'] = serialization.minmax_norm(cw_feat)
    timings['CW-Feat'] = time.perf_counter() - t0
    
    # --- CW-Feat-PosNeg ---
    t0 = time.perf_counter()
    cw_feat_posneg = (alpha * abs_w_posneg).sum(axis=0)
    scores['CW-Feat-PosNeg'] = serialization.minmax_norm(cw_feat_posneg)
    timings['CW-Feat-PosNeg'] = time.perf_counter() - t0
    
    # --- Support-CW-Sum (Prioritizes low-accuracy classes) ---
    t0 = time.perf_counter()
    class_error_rate = 1.0 - class_acc
    class_error_weight = class_error_rate / (class_error_rate.sum() + 1e-12)
    support_cw_sum = (class_error_weight[:, None] * abs_w).sum(axis=0)
    scores['Support-CW-Sum'] = serialization.minmax_norm(support_cw_sum)
    timings['Support-CW-Sum'] = time.perf_counter() - t0
    
    # --- Support-CW-Sum-PosNeg ---
    t0 = time.perf_counter()
    support_cw_sum_posneg = (class_error_weight[:, None] * abs_w_posneg).sum(axis=0)
    scores['Support-CW-Sum-PosNeg'] = serialization.minmax_norm(support_cw_sum_posneg)
    timings['Support-CW-Sum-PosNeg'] = time.perf_counter() - t0
    
    # --- Margin (top vs runner-up) ---
    t0 = time.perf_counter()
    sorted_abs = np.sort(abs_w, axis=0)
    margin = sorted_abs[-1] - sorted_abs[-2] if sorted_abs.shape[0] > 1 else sorted_abs[-1]
    scores['Margin'] = serialization.minmax_norm(margin)
    timings['Margin'] = time.perf_counter() - t0
    
    # --- Margin-PosNeg ---
    t0 = time.perf_counter()
    sorted_abs_posneg = np.sort(abs_w_posneg, axis=0)
    margin_posneg = sorted_abs_posneg[-1] - sorted_abs_posneg[-2] if sorted_abs_posneg.shape[0] > 1 else sorted_abs_posneg[-1]
    scores['Margin-PosNeg'] = serialization.minmax_norm(margin_posneg)
    timings['Margin-PosNeg'] = time.perf_counter() - t0
    
    # --- Entropy-based ---
    t0 = time.perf_counter()
    p = abs_w / (abs_w.sum(axis=0, keepdims=True) + 1e-12)
    entropy = -(p * np.log(p + 1e-12)).sum(axis=0)
    entropy_score = 1.0 - (entropy / np.log(n_classes))
    scores['Entropy'] = serialization.minmax_norm(entropy_score)
    timings['Entropy'] = time.perf_counter() - t0
    
    # --- Entropy-PosNeg ---
    t0 = time.perf_counter()
    p_posneg = abs_w_posneg / (abs_w_posneg.sum(axis=0, keepdims=True) + 1e-12)
    entropy_posneg = -(p_posneg * np.log(p_posneg + 1e-12)).sum(axis=0)
    entropy_score_posneg = 1.0 - (entropy_posneg / np.log(n_classes))
    scores['Entropy-PosNeg'] = serialization.minmax_norm(entropy_score_posneg)
    timings['Entropy-PosNeg'] = time.perf_counter() - t0
    
    # --- Gini-based ---
    t0 = time.perf_counter()
    gini_score = 1.0 - (p**2).sum(axis=0)
    scores['Gini'] = serialization.minmax_norm(gini_score)
    timings['Gini'] = time.perf_counter() - t0
    
    # --- Gini-PosNeg ---
    t0 = time.perf_counter()
    gini_score_posneg = 1.0 - (p_posneg**2).sum(axis=0)
    scores['Gini-PosNeg'] = serialization.minmax_norm(gini_score_posneg)
    timings['Gini-PosNeg'] = time.perf_counter() - t0
    
    # --- Stability (across epochs) ---
    t0 = time.perf_counter()
    net_w_history = pos_w_history - neg_w_history
    max_abs_hist = np.max(np.abs(net_w_history), axis=1)
    std_hist = max_abs_hist.std(axis=0)
    mean_abs = max_abs_hist.mean(axis=0)
    stab_score = mean_abs / (std_hist + 1e-6)
    scores['Stability'] = serialization.minmax_norm(stab_score)
    timings['Stability'] = time.perf_counter() - t0
    
    # --- Stability-PosNeg ---
    t0 = time.perf_counter()
    net_w_posneg_history = pos_w_history + neg_w_history
    max_abs_hist_posneg = np.max(np.abs(net_w_posneg_history), axis=1)
    std_hist_posneg = max_abs_hist_posneg.std(axis=0)
    mean_abs_posneg = max_abs_hist_posneg.mean(axis=0)
    stab_score_posneg = mean_abs_posneg / (std_hist_posneg + 1e-6)
    scores['Stability-PosNeg'] = serialization.minmax_norm(stab_score_posneg)
    timings['Stability-PosNeg'] = time.perf_counter() - t0
    
    # --- Relevance ---
    t0 = time.perf_counter()
    norm_abs = abs_w / (abs_w.sum(axis=1, keepdims=True) + 1e-12)
    relevance = (class_w[:, None] * norm_abs).sum(axis=0)
    scores['Relevance'] = serialization.minmax_norm(relevance)
    timings['Relevance'] = time.perf_counter() - t0
    
    # --- Relevance-PosNeg ---
    t0 = time.perf_counter()
    norm_abs_posneg = abs_w_posneg / (abs_w_posneg.sum(axis=1, keepdims=True) + 1e-12)
    relevance_posneg = (class_w[:, None] * norm_abs_posneg).sum(axis=0)
    scores['Relevance-PosNeg'] = serialization.minmax_norm(relevance_posneg)
    timings['Relevance-PosNeg'] = time.perf_counter() - t0
    
    # --- GroupLasso (for image-like features) ---
    t0 = time.perf_counter()
    img_dim = int(np.sqrt(n_feats))
    if img_dim * img_dim == n_feats and n_feats > 0:
        group_size = 4
        abs_w_reshaped = abs_w.reshape(n_classes, img_dim, img_dim)
        num_groups_h = img_dim // group_size
        num_groups_w = img_dim // group_size
        group_l2_norms_per_class = np.zeros((n_classes, num_groups_h, num_groups_w))
        
        for r_group in range(num_groups_h):
            for c_group in range(num_groups_w):
                r_start, r_end = r_group * group_size, (r_group + 1) * group_size
                c_start, c_end = c_group * group_size, (c_group + 1) * group_size
                group_weights = abs_w_reshaped[:, r_start:r_end, c_start:c_end]
                l2_norm_group = np.sqrt(np.sum(group_weights**2, axis=(1, 2)))
                group_l2_norms_per_class[:, r_group, c_group] = l2_norm_group
        
        mean_group_l2_norms = group_l2_norms_per_class.mean(axis=0)
        group_lasso_feature_scores = np.zeros(n_feats)
        for r_pixel in range(img_dim):
            for c_pixel in range(img_dim):
                r_group, c_group = r_pixel // group_size, c_pixel // group_size
                if r_group < num_groups_h and c_group < num_groups_w:
                    feature_idx = r_pixel * img_dim + c_pixel
                    group_lasso_feature_scores[feature_idx] = mean_group_l2_norms[r_group, c_group]
        scores['GroupLasso'] = serialization.minmax_norm(group_lasso_feature_scores)
    else:
        scores['GroupLasso'] = np.full(n_feats, 0.5)
    timings['GroupLasso'] = time.perf_counter() - t0
    
    # --- GroupLasso-PosNeg ---
    t0 = time.perf_counter()
    if img_dim * img_dim == n_feats and n_feats > 0:
        abs_w_posneg_reshaped = abs_w_posneg.reshape(n_classes, img_dim, img_dim)
        group_l2_norms_posneg_per_class = np.zeros((n_classes, num_groups_h, num_groups_w))
        
        for r_group in range(num_groups_h):
            for c_group in range(num_groups_w):
                r_start, r_end = r_group * group_size, (r_group + 1) * group_size
                c_start, c_end = c_group * group_size, (c_group + 1) * group_size
                group_weights_posneg = abs_w_posneg_reshaped[:, r_start:r_end, c_start:c_end]
                l2_norm_group_posneg = np.sqrt(np.sum(group_weights_posneg**2, axis=(1, 2)))
                group_l2_norms_posneg_per_class[:, r_group, c_group] = l2_norm_group_posneg
        
        mean_group_l2_norms_posneg = group_l2_norms_posneg_per_class.mean(axis=0)
        group_lasso_feature_scores_posneg = np.zeros(n_feats)
        for r_pixel in range(img_dim):
            for c_pixel in range(img_dim):
                r_group, c_group = r_pixel // group_size, c_pixel // group_size
                if r_group < num_groups_h and c_group < num_groups_w:
                    feature_idx = r_pixel * img_dim + c_pixel
                    group_lasso_feature_scores_posneg[feature_idx] = mean_group_l2_norms_posneg[r_group, c_group]
        scores['GroupLasso-PosNeg'] = serialization.minmax_norm(group_lasso_feature_scores_posneg)
    else:
        scores['GroupLasso-PosNeg'] = np.full(n_feats, 0.5)
    timings['GroupLasso-PosNeg'] = time.perf_counter() - t0
    
    # --- Taylor Criteria (First-order perturbation) ---
    t0 = time.perf_counter()
    original_class_sums_val = tm.predict(X_val, return_class_sums=True)[1]
    original_scores_for_true_class_val = original_class_sums_val[np.arange(len(y_val)), y_val]
    taylor_scores = np.zeros(n_feats)
    for f_idx in range(n_feats):
        X_perturbed = X_val.copy()
        X_perturbed[:, f_idx] = 1 - X_perturbed[:, f_idx]
        perturbed_class_sums_val = tm.predict(X_perturbed, return_class_sums=True)[1]
        perturbed_scores_for_true_class_val = perturbed_class_sums_val[np.arange(len(y_val)), y_val]
        taylor_scores[f_idx] = np.mean(np.abs(original_scores_for_true_class_val - perturbed_scores_for_true_class_val))
    scores['TaylorCrit'] = serialization.minmax_norm(taylor_scores)
    timings['TaylorCrit'] = time.perf_counter() - t0
    
    # --- Variational Dropout ---
    t0 = time.perf_counter()
    N_MASKS_VAR_DROPOUT = 10
    var_dropout_scores = np.zeros(n_feats)
    original_sums_val = tm.predict(X_val, return_class_sums=True)[1]
    original_true_class_scores_val = original_sums_val[np.arange(len(y_val)), y_val]
    for f_idx in range(n_feats):
        score_changes_for_f = []
        for _ in range(N_MASKS_VAR_DROPOUT):
            X_val_temp = X_val.copy()
            mask_f = rng.integers(0, 2, size=X_val.shape[0])
            X_val_temp[:, f_idx] = X_val_temp[:, f_idx] * mask_f
            perturbed_sums_val = tm.predict(X_val_temp, return_class_sums=True)[1]
            perturbed_true_class_scores_val = perturbed_sums_val[np.arange(len(y_val)), y_val]
            score_changes_for_f.append(np.mean(np.abs(original_true_class_scores_val - perturbed_true_class_scores_val)))
        var_dropout_scores[f_idx] = np.mean(score_changes_for_f)
    scores['VarDropout'] = serialization.minmax_norm(var_dropout_scores)
    timings['VarDropout'] = time.perf_counter() - t0
    
    # --- Ablation Impact ---
    t0 = time.perf_counter()
    ablation_impact_scores = np.zeros(n_feats)
    for f_idx in range(n_feats):
        X_val_ablated_f = X_val.copy()
        X_val_ablated_f[:, f_idx] = 0
        ablated_sums_val = tm.predict(X_val_ablated_f, return_class_sums=True)[1]
        ablated_true_class_scores_val = ablated_sums_val[np.arange(len(y_val)), y_val]
        ablation_impact_scores[f_idx] = np.mean(np.abs(original_true_class_scores_val - ablated_true_class_scores_val))
    scores['AblationImpact'] = serialization.minmax_norm(ablation_impact_scores)
    timings['AblationImpact'] = time.perf_counter() - t0
    
    # --- Smooth Output Stability ---
    t0 = time.perf_counter()
    N_NOISE_SAMPLES_SMOOTH = 10
    NOISE_LEVEL = 0.05
    smooth_stability_scores = np.zeros(n_feats)
    for f_idx in range(n_feats):
        score_diffs_for_f = []
        for _ in range(N_NOISE_SAMPLES_SMOOTH):
            noise = rng.choice([0, 1], size=X_val.shape, p=[1-NOISE_LEVEL, NOISE_LEVEL])
            X_val_noisy = X_val ^ noise
            noisy_sums1 = tm.predict(X_val_noisy, return_class_sums=True)[1][np.arange(len(y_val)), y_val]
            X_val_noisy_f_flipped = X_val_noisy.copy()
            X_val_noisy_f_flipped[:, f_idx] = 1 - X_val_noisy_f_flipped[:, f_idx]
            noisy_sums2_f_flipped = tm.predict(X_val_noisy_f_flipped, return_class_sums=True)[1][np.arange(len(y_val)), y_val]
            score_diffs_for_f.append(np.mean(np.abs(noisy_sums1 - noisy_sums2_f_flipped)))
        smooth_stability_scores[f_idx] = np.mean(score_diffs_for_f)
    scores['SmoothStabil'] = serialization.minmax_norm(smooth_stability_scores)
    timings['SmoothStabil'] = time.perf_counter() - t0
    
    # --- Dropout sensitivity ---
    t0 = time.perf_counter()
    full_acc_val = 100 * (y_pred_val == y_val).mean()
    drop = np.zeros(n_feats)
    for f in range(n_feats):
        Xm = X_val.copy()
        Xm[:, f] = 0
        drop[f] = full_acc_val - 100 * (tm.predict(Xm) == y_val).mean()
    scores['Dropout'] = serialization.minmax_norm(drop)
    timings['Dropout'] = time.perf_counter() - t0
    
    # --- Filter methods ---
    
    # Mutual Information
    t0 = time.perf_counter()
    correct_val = (y_pred_val == y_val).astype(int)
    mi = np.array([mutual_info_score(X_val[:, f], correct_val) for f in range(n_feats)])
    scores['MutualInfo'] = serialization.minmax_norm(mi)
    timings['MutualInfo'] = time.perf_counter() - t0
    
    # Chi-squared
    t0 = time.perf_counter()
    chi2_scores, _ = chi2(X_train, y_train)
    scores['Chi2'] = serialization.minmax_norm(np.nan_to_num(chi2_scores))
    timings['Chi2'] = time.perf_counter() - t0
    
    # Variance
    t0 = time.perf_counter()
    selector_var = VarianceThreshold()
    try:
        selector_var.fit(X_train)
        variance_scores = selector_var.variances_
    except ValueError:
        variance_scores = np.zeros(X_train.shape[1])
    scores['Variance'] = serialization.minmax_norm(variance_scores)
    timings['Variance'] = time.perf_counter() - t0
    
    # Random baseline
    t0 = time.perf_counter()
    random_scores = rng.random(n_feats)
    scores['Random'] = serialization.minmax_norm(random_scores)
    timings['Random'] = time.perf_counter() - t0
    
    return scores, timings


def compute_explainer_scores(tm, X_train, X_val, y_train, y_val, n_feats, n_classes, rng):
    """
    Compute explainer-based feature scores (SHAP, LIME, Permutation Importance, IG, SmoothGrad, VarGrad).
    These are computationally expensive and can be skipped if needed.
    
    Args:
        tm: Trained TMClassifier
        X_train, X_val: Training and validation features
        y_train, y_val: Training and validation labels
        n_feats: Number of features
        n_classes: Number of classes
        rng: Random number generator
        
    Returns:
        Dictionary mapping method names to normalized feature scores
        Dictionary mapping method names to computation times
    """
    scores = {}
    timings = {}
    
    # Helper function for predict_proba
    def predict_proba_wrapper(model, X_input, y_ctx):
        """Wrapper for TM predict_proba."""
        try:
            tup = model.predict(X_input, return_class_sums=True)
            if isinstance(tup, (list, tuple)) and len(tup) == 2:
                preds, sums = tup
                shifted = sums - sums.min(axis=1, keepdims=True)
                total = shifted.sum(axis=1, keepdims=True)
                P = np.zeros_like(shifted, float)
                mask = total.flatten() > 1e-9
                P[mask] = shifted[mask] / total[mask]
                for i, ok in enumerate(~mask):
                    if ok:
                        continue
                    cls = int(preds[i])
                    P[i, cls] = 1.0
                return P
        except:
            pass
        preds = model.predict(X_input)
        ncl = model.number_of_classes if hasattr(model, 'number_of_classes') else len(np.unique(y_ctx))
        P = np.zeros((X_input.shape[0], ncl), float)
        for i, p in enumerate(preds):
            if 0 <= p < ncl:
                P[i, p] = 1.0
        return P
    
    # --- SHAP ---
    print("Computing SHAP explanation...")
    t0 = time.perf_counter()
    try:
        sample_idx_for_shap = 0
        sample_shap = X_val[sample_idx_for_shap].reshape(1, -1)
        background_size = min(100, X_train.shape[0])
        background_indices = rng.choice(X_train.shape[0], background_size, replace=False)
        background = X_train[background_indices]
        
        def shap_predict(X):
            if X.dtype != np.uint32 or X.min() < 0 or X.max() > 1:
                X_processed = np.nan_to_num(X, nan=0.0)
                X_binarized = (X_processed > 0.5).astype(np.uint32)
                return predict_proba_wrapper(tm, X_binarized, y_val)
            return predict_proba_wrapper(tm, X, y_val)
        
        masker = shap.maskers.Independent(X_train)
        shap_explainer = shap.KernelExplainer(shap_predict, background, masker=masker)
        shap_values = shap_explainer.shap_values(sample_shap, nsamples=min(50, X_val.shape[0]))
        predicted_class = tm.predict(sample_shap)[0]
        
        if isinstance(shap_values, list):
            shap_exp = shap_values[predicted_class] if predicted_class < len(shap_values) else shap_values[0]
        else:
            shap_exp = shap_values[predicted_class] if shap_values.shape[0] > 1 else shap_values
        shap_exp = np.atleast_3d(shap_exp)
        shap_scores = np.mean(np.abs(shap_exp), axis=2).flatten()
        scores['SHAP'] = serialization.minmax_norm(shap_scores)
    except Exception as e:
        print(f"Warning: SHAP failed - {e}")
        scores['SHAP'] = scores.get('Random', np.full(n_feats, 0.5))
    timings['SHAP'] = time.perf_counter() - t0
    
    # --- LIME ---
    print("Computing LIME explanation...")
    t0 = time.perf_counter()
    try:
        feat_names = [f"f{i}" for i in range(n_feats)]
        class_names = [str(c) for c in range(n_classes)]
        lime_expl = LimeTabularExplainer(
            training_data=X_train[:min(2000, X_train.shape[0])],
            feature_names=feat_names,
            class_names=class_names,
            mode='classification',
            discretize_continuous=False,
            verbose=False,
            random_state=42
        )
        
        def lime_pred(X):
            X_processed = np.nan_to_num(X, nan=0.0)
            X_binarized = (X_processed > 0.5).astype(np.uint32)
            return predict_proba_wrapper(tm, X_binarized, y_train)
        
        lime_imp = np.zeros(n_feats)
        num_lime_samples = min(20, X_val.shape[0])
        for i in range(num_lime_samples):
            exp = lime_expl.explain_instance(
                data_row=X_val[i],
                predict_fn=lime_pred,
                num_features=n_feats,
                num_samples=min(1000, X_train.shape[0])
            )
            for feat_str, w in exp.as_list():
                idx = int(feat_str.split()[0][1:])
                lime_imp[idx] += abs(w)
        scores['LIME'] = serialization.minmax_norm(lime_imp)
    except Exception as e:
        print(f"Warning: LIME failed - {e}")
        scores['LIME'] = scores.get('Random', np.full(n_feats, 0.5))
    timings['LIME'] = time.perf_counter() - t0
    
    # --- Permutation Importance ---
    t0 = time.perf_counter()
    perm_importance_scores = np.zeros(n_feats)
    baseline_sums_val = tm.predict(X_val, return_class_sums=True)[1]
    baseline_true_class_scores_val = baseline_sums_val[np.arange(len(y_val)), y_val]
    
    for f_idx in range(n_feats):
        X_val_permuted = X_val.copy()
        rng.shuffle(X_val_permuted[:, f_idx])
        permuted_sums_val = tm.predict(X_val_permuted, return_class_sums=True)[1]
        permuted_true_class_scores_val = permuted_sums_val[np.arange(len(y_val)), y_val]
        perm_importance_scores[f_idx] = np.mean(baseline_true_class_scores_val - permuted_true_class_scores_val)
    scores['PermImportance'] = serialization.minmax_norm(perm_importance_scores)
    timings['PermImportance'] = time.perf_counter() - t0
    
    # --- Integrated Gradients (IG) ---
    print("Computing Integrated Gradients (IG)...")
    t0 = time.perf_counter()
    ig_scores_acc = np.zeros(n_feats)
    n_explain_samples_ig = min(20, X_val.shape[0])
    n_steps_ig = 50
    epsilon_ig = 0.2
    
    indices_ig = rng.choice(X_val.shape[0], n_explain_samples_ig, replace=False)
    X_explain_ig = X_val[indices_ig]
    Y_explain_ig = y_val[indices_ig]
    
    for i in range(n_explain_samples_ig):
        x_orig = X_explain_ig[i]
        y_true_idx = Y_explain_ig[i]
        baseline = np.zeros_like(x_orig)
        path_attributions_sample = np.zeros(n_feats)
        
        for step in range(1, n_steps_ig + 1):
            alpha = float(step) / n_steps_ig
            interpolated_input = baseline + alpha * (x_orig - baseline)
            grads_at_interpolated_input = np.zeros(n_feats)
            
            for f_idx in range(n_feats):
                perturbed_input_plus = interpolated_input.copy()
                perturbed_input_plus[f_idx] += epsilon_ig
                perturbed_input_plus_bin = (np.nan_to_num(perturbed_input_plus, nan=0.0) > 0.5).astype(np.uint32)
                
                perturbed_input_minus = interpolated_input.copy()
                perturbed_input_minus[f_idx] -= epsilon_ig
                perturbed_input_minus_bin = (np.nan_to_num(perturbed_input_minus, nan=0.0) > 0.5).astype(np.uint32)
                
                prob_plus = predict_proba_wrapper(tm, perturbed_input_plus_bin.reshape(1, -1), Y_explain_ig)[0, y_true_idx]
                prob_minus = predict_proba_wrapper(tm, perturbed_input_minus_bin.reshape(1, -1), Y_explain_ig)[0, y_true_idx]
                grads_at_interpolated_input[f_idx] = (prob_plus - prob_minus) / (2 * epsilon_ig)
            
            path_attributions_sample += grads_at_interpolated_input
        
        ig_scores_acc += (x_orig - baseline) * (path_attributions_sample / n_steps_ig)
    
    ig_scores = ig_scores_acc / n_explain_samples_ig
    scores['IG'] = serialization.minmax_norm(np.abs(ig_scores))
    timings['IG'] = time.perf_counter() - t0
    
    # --- SmoothGrad-Squared & VarGrad ---
    print("Computing SmoothGrad-Squared and VarGrad...")
    t0 = time.perf_counter()
    smoothgrad_sq_scores_acc = np.zeros(n_feats)
    vargrad_scores_acc = np.zeros(n_feats)
    
    n_noise_samples_ensemble = 10
    noise_flip_prob = 0.05
    
    for i in range(n_explain_samples_ig):
        x_orig = X_explain_ig[i]
        y_true_idx = Y_explain_ig[i]
        igs_for_noisy_versions = []
        
        for _ in range(n_noise_samples_ensemble):
            noise_mask = rng.choice([0, 1], size=x_orig.shape, p=[1 - noise_flip_prob, noise_flip_prob])
            x_noisy = x_orig ^ noise_mask
            baseline_noisy = np.zeros_like(x_noisy)
            path_attributions_noisy_sample = np.zeros(n_feats)
            
            for step_ig_noisy in range(1, n_steps_ig + 1):
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
                    
                    p_plus = predict_proba_wrapper(tm, p_input_plus_bin.reshape(1, -1), Y_explain_ig)[0, y_true_idx]
                    p_minus = predict_proba_wrapper(tm, p_input_minus_bin.reshape(1, -1), Y_explain_ig)[0, y_true_idx]
                    grads_at_interpolated_noisy[f_idx] = (p_plus - p_minus) / (2 * epsilon_ig)
                
                path_attributions_noisy_sample += grads_at_interpolated_noisy
            
            ig_noisy = (x_noisy - baseline_noisy) * (path_attributions_noisy_sample / n_steps_ig)
            igs_for_noisy_versions.append(ig_noisy)
        
        igs_for_noisy_versions_arr = np.array(igs_for_noisy_versions)
        smoothgrad_sq_scores_acc += np.mean(igs_for_noisy_versions_arr**2, axis=0)
        vargrad_scores_acc += np.var(igs_for_noisy_versions_arr, axis=0)
    
    smoothgrad_sq_scores_raw = smoothgrad_sq_scores_acc / n_explain_samples_ig
    vargrad_scores_raw = vargrad_scores_acc / n_explain_samples_ig
    scores['SmoothGradSq'] = serialization.minmax_norm(smoothgrad_sq_scores_raw)
    scores['VarGrad'] = serialization.minmax_norm(vargrad_scores_raw)
    timings['SmoothGradSq'] = time.perf_counter() - t0
    timings['VarGrad'] = 0  # Included in SmoothGradSq time
    
    return scores, timings

