import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits, load_breast_cancer, load_wine, fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mutual_info_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tmu.models.classification.vanilla_classifier import TMClassifier
from numpy import trapz # For AUC calculation

import shap
from lime.lime_tabular import LimeTabularExplainer
from tqdm import tqdm
from tmu.data import MNIST
from sklearn.feature_selection import chi2, VarianceThreshold
from sklearn.feature_selection import RFE, SelectFromModel
from skrebate import ReliefF
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats import entropy

# TM hyperparameters
clauses = 500
T = 150
s = 3.0
max_lit = 20
epochs = 10

# --- Local implementations ---
def count_ta_states(tm):
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
    """Normalize array to [0,1] range."""
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn) if mx > mn else np.full_like(arr, 0.5)

def predict_proba(model, X, Y):
    """One-hot style probabilities for LIME/SHAP."""
    preds = model.predict(X)
    n_samples = X.shape[0]
    classes = np.unique(Y)
    n_classes = len(classes)
    P = np.zeros((n_samples, n_classes), dtype=float)
    for i, p in enumerate(preds):
        P[i, p] = 1.0
    return P

def l1_regularization_score(tm, ta_states=None):
    """Compute L1 regularization score for features based on absolute weights."""
    if ta_states is None:
        ta_states = count_ta_states(tm)
    pos_w, neg_w = count_clause_weights(tm, ta_states)
    # L1 score is the sum of absolute weights across all clauses and classes
    l1_score = np.sum(np.abs(pos_w) + np.abs(neg_w), axis=0)
    return minmax_norm(l1_score)

def group_lasso_score(tm, ta_states=None, group_size=8):
    """Compute Group Lasso score for features, grouping by spatial proximity.
    For image data, groups pixels in square patches."""
    if ta_states is None:
        ta_states = count_ta_states(tm)
    pos_w, neg_w = count_clause_weights(tm, ta_states)
    n_classes, n_feats = pos_w.shape
    
    # Reshape weights to image format (assuming square image)
    img_size = int(np.sqrt(n_feats))
    if img_size * img_size != n_feats:
        raise ValueError("Number of features must be a perfect square for image data")
    
    # Reshape weights to (n_classes, img_size, img_size)
    pos_w_img = pos_w.reshape(n_classes, img_size, img_size)
    neg_w_img = neg_w.reshape(n_classes, img_size, img_size)
    
    # Compute group norms (L2 norm within each group)
    n_groups = img_size // group_size
    group_scores = np.zeros((n_classes, n_groups, n_groups))
    
    for i in range(n_groups):
        for j in range(n_groups):
            # Extract group
            group_pos = pos_w_img[:, i*group_size:(i+1)*group_size, j*group_size:(j+1)*group_size]
            group_neg = neg_w_img[:, i*group_size:(i+1)*group_size, j*group_size:(j+1)*group_size]
            # Compute L2 norm of the group
            group_scores[:, i, j] = np.sqrt(np.sum(group_pos**2 + group_neg**2, axis=(1,2)))
    
    # Average group scores across classes
    mean_group_scores = np.mean(group_scores, axis=0)
    
    # Map group scores back to feature scores
    feature_scores = np.zeros(n_feats)
    for i in range(n_groups):
        for j in range(n_groups):
            start_idx = (i * group_size * img_size) + (j * group_size)
            end_idx = start_idx + group_size
            feature_scores[start_idx:end_idx] = mean_group_scores[i, j]
    
    return minmax_norm(feature_scores)

def variational_dropout_score(tm, X_val, y_val, n_samples=10):
    """Compute feature importance using variational dropout.
    Learns dropout probabilities for each feature by measuring impact on accuracy."""
    n_feats = X_val.shape[1]
    dropout_scores = np.zeros(n_feats)
    
    # Compute baseline accuracy
    baseline_acc = 100 * (tm.predict(X_val) == y_val).mean()
    
    # For each feature, compute average impact of dropout
    for f in range(n_feats):
        feature_impacts = []
        for _ in range(n_samples):
            # Create masked version of validation data
            X_masked = X_val.copy()
            # Apply dropout to feature f
            mask = np.random.binomial(1, 0.5, size=X_val.shape[0])
            X_masked[:, f] = X_masked[:, f] * mask
            
            # Compute accuracy with dropout
            acc = 100 * (tm.predict(X_masked) == y_val).mean()
            # Impact is the difference from baseline
            feature_impacts.append(baseline_acc - acc)
        
        # Average impact across samples
        dropout_scores[f] = np.mean(feature_impacts)
    
    return minmax_norm(dropout_scores)

def integrated_gradients_score(tm, X_val, y_val, n_steps=50):
    """Compute feature importance using Integrated Gradients.
    Integrates gradients along path from baseline to input."""
    n_samples, n_feats = X_val.shape
    ig_scores = np.zeros(n_feats)
    
    # Create baseline (all zeros)
    baseline = np.zeros_like(X_val)
    
    # For each sample
    for i in range(min(20, n_samples)):  # Limit to 20 samples for speed
        x = X_val[i:i+1]  # Keep batch dimension
        
        # Create path from baseline to input
        alphas = np.linspace(0, 1, n_steps)
        path = np.array([baseline[i:i+1] + alpha * (x - baseline[i:i+1]) for alpha in alphas])
        
        # Compute predictions along path
        preds = tm.predict(path.reshape(-1, n_feats))
        preds = preds.reshape(n_steps, -1)
        
        # Compute gradients (approximate using finite differences)
        grads = np.zeros((n_steps, n_feats))
        for j in range(n_steps-1):
            grads[j] = (preds[j+1] - preds[j]) / (alphas[j+1] - alphas[j])
        
        # Integrate gradients
        ig = np.trapz(grads, alphas, axis=0)
        ig_scores += np.abs(ig)
    
    # Average across samples
    ig_scores /= min(20, n_samples)
    return minmax_norm(ig_scores)

def taylor_criteria_score(tm, X_val, y_val, epsilon=1e-3):
    """Compute feature importance using first-order Taylor expansion.
    Estimates impact of feature removal on model output."""
    n_samples, n_feats = X_val.shape
    taylor_scores = np.zeros(n_feats)
    
    # For each feature
    for f in range(n_feats):
        # Create perturbed version of validation data
        X_perturbed = X_val.copy()
        X_perturbed[:, f] = X_perturbed[:, f] + epsilon
        
        # Compute predictions for original and perturbed data
        preds_orig = tm.predict(X_val)
        preds_pert = tm.predict(X_perturbed)
        
        # Compute first-order Taylor approximation of impact
        impact = np.abs(preds_pert - preds_orig).mean()
        taylor_scores[f] = impact
    
    return minmax_norm(taylor_scores)

def feature_masking_score(tm, X_val, y_val, n_iterations=5):
    """Compute feature importance using learnable feature masks.
    Iteratively learns mask values that minimize accuracy impact."""
    n_samples, n_feats = X_val.shape
    mask_scores = np.zeros(n_feats)
    
    # Initialize masks randomly
    masks = np.random.uniform(0, 1, size=(n_iterations, n_feats))
    
    # Compute baseline accuracy
    baseline_acc = 100 * (tm.predict(X_val) == y_val).mean()
    
    # For each iteration
    for i in range(n_iterations):
        # Apply current mask
        X_masked = X_val * masks[i]
        
        # Compute accuracy with masked features
        acc = 100 * (tm.predict(X_masked) == y_val).mean()
        
        # Update mask scores based on accuracy impact
        impact = baseline_acc - acc
        mask_scores += masks[i] * impact
    
    # Average across iterations
    mask_scores /= n_iterations
    return minmax_norm(mask_scores)

def information_bottleneck_score(tm, X_val, y_val, n_bins=10):
    """Compute feature importance using Information Bottleneck principle.
    Measures how much information each feature preserves about the target."""
    n_samples, n_feats = X_val.shape
    ib_scores = np.zeros(n_feats)
    
    # Get predictions for validation set
    preds = tm.predict(X_val)
    
    # For each feature
    for f in range(n_feats):
        # Discretize feature values into bins
        feature_bins = np.linspace(X_val[:, f].min(), X_val[:, f].max(), n_bins)
        feature_disc = np.digitize(X_val[:, f], feature_bins)
        
        # Compute mutual information between discretized feature and predictions
        mi = mutual_info_score(feature_disc, preds)
        
        # Compute mutual information between discretized feature and true labels
        mi_true = mutual_info_score(feature_disc, y_val)
        
        # Information bottleneck score is the ratio of preserved information
        ib_scores[f] = mi / (mi_true + 1e-10)
    
    return minmax_norm(ib_scores)

def tsetlin_mutual_info_score(tm, X_val, y_val, n_bins=10):
    """Compute feature importance using mutual information with Tsetlin-specific adaptations.
    Considers both the feature values and their impact on clause activation."""
    n_samples, n_feats = X_val.shape
    mi_scores = np.zeros(n_feats)
    
    # Discretize features into bins
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    X_discrete = discretizer.fit_transform(X_val)
    
    # For each feature
    for f in range(n_feats):
        # Compute mutual information between feature and target
        mi = mutual_info_score(X_discrete[:, f], y_val)
        
        # Get clause activations for this feature
        clause_acts = tm.get_clause_activations(X_val)
        clause_acts = clause_acts.reshape(n_samples, -1)
        
        # Compute mutual information between feature and clause activations
        mi_clause = mutual_info_score(X_discrete[:, f], clause_acts.mean(axis=1))
        
        # Combine both scores
        mi_scores[f] = 0.7 * mi + 0.3 * mi_clause
    
    return minmax_norm(mi_scores)

def tsetlin_relieff_score(tm, X_val, y_val, n_neighbors=5):
    """Compute feature importance using ReliefF algorithm adapted for Tsetlin Machines.
    Considers both feature distances and their impact on clause activation patterns."""
    n_samples, n_feats = X_val.shape
    relieff = ReliefF(n_neighbors=n_neighbors, n_features_to_keep=n_feats)
    
    # Get clause activations
    clause_acts = tm.get_clause_activations(X_val)
    clause_acts = clause_acts.reshape(n_samples, -1)
    
    # Combine original features with clause activations
    X_combined = np.hstack([X_val, clause_acts])
    
    # Fit ReliefF
    relieff.fit(X_combined, y_val)
    
    # Get feature scores (only for original features)
    scores = relieff.feature_importances_[:n_feats]
    
    return minmax_norm(scores)

def tsetlin_rfe_score(tm, X_val, y_val, step=1):
    """Compute feature importance using Recursive Feature Elimination with Tsetlin Machine.
    Iteratively removes features and measures impact on performance."""
    n_samples, n_feats = X_val.shape
    rfe_scores = np.zeros(n_feats)
    
    # Create a copy of the Tsetlin Machine for feature elimination
    tm_copy = TMClassifier(
        number_of_clauses=tm.number_of_clauses,
        T=tm.T,
        s=tm.s,
        max_included_literals=tm.max_included_literals
    )
    
    # Initialize feature mask
    feature_mask = np.ones(n_feats, dtype=bool)
    
    # Compute baseline accuracy
    baseline_acc = 100 * (tm.predict(X_val) == y_val).mean()
    
    # Iteratively eliminate features
    while np.sum(feature_mask) > 1:
        # Train on current feature subset
        tm_copy.fit(X_val[:, feature_mask], y_val, epochs=1)
        
        # Compute accuracy with current features
        acc = 100 * (tm_copy.predict(X_val[:, feature_mask]) == y_val).mean()
        
        # Update scores for remaining features
        impact = baseline_acc - acc
        rfe_scores[feature_mask] += impact
        
        # Remove least important features
        n_to_remove = min(step, np.sum(feature_mask) - 1)
        if n_to_remove <= 0:
            break
            
        # Find least important features
        remaining_scores = rfe_scores[feature_mask]
        threshold = np.sort(remaining_scores)[n_to_remove-1]
        feature_mask[feature_mask] = remaining_scores > threshold
    
    return minmax_norm(rfe_scores)

def clause_analysis_score(tm, ta_states=None):
    """Compute feature importance based on Tsetlin Machine clause analysis.
    Analyzes how features are used in clauses and their impact on classification."""
    if ta_states is None:
        ta_states = count_ta_states(tm)
    
    n_classes, _, n_clauses, n_feats = ta_states.shape
    clause_scores = np.zeros(n_feats)
    
    # For each class
    for c in range(n_classes):
        # Get clause weights
        pos_weights = np.array([tm.get_weight(the_class=c, polarity=0, clause=cl) 
                              for cl in range(n_clauses)])
        neg_weights = np.array([tm.get_weight(the_class=c, polarity=1, clause=cl) 
                              for cl in range(n_clauses)])
        
        # For each feature
        for f in range(n_feats):
            # Count how often feature is used in clauses
            pos_usage = np.sum(ta_states[c,0,:,f] != 0)
            neg_usage = np.sum(ta_states[c,1,:,f] != 0)
            
            # Weight by clause weights
            pos_score = np.sum(pos_weights * (ta_states[c,0,:,f] != 0))
            neg_score = np.sum(neg_weights * (ta_states[c,1,:,f] != 0))
            
            # Combine scores
            clause_scores[f] += abs(pos_score) + abs(neg_score)
    
    return minmax_norm(clause_scores)

def compute_feature_agreement(scores):
    """Compute agreement between different feature selection methods.
    Returns a matrix of agreement scores between methods."""
    n_methods = len(scores)
    agreement = np.zeros((n_methods, n_methods))
    
    for i in range(n_methods):
        for j in range(n_methods):
            # Get top 20% features for each method
            k = int(len(scores[list(scores.keys())[0]]) * 0.2)
            top_i = np.argsort(scores[list(scores.keys())[i]])[-k:]
            top_j = np.argsort(scores[list(scores.keys())[j]])[-k:]
            
            # Compute Jaccard similarity
            intersection = len(set(top_i) & set(top_j))
            union = len(set(top_i) | set(top_j))
            agreement[i,j] = intersection / union if union > 0 else 0
    
    return agreement

def analyze_feature_stability(scores, X_train, y_train, n_splits=5):
    """Analyze stability of feature selection across data splits.
    Returns stability scores for each method."""
    from sklearn.model_selection import KFold
    
    n_methods = len(scores)
    stability_scores = {name: [] for name in scores.keys()}
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for train_idx, val_idx in tqdm(kf.split(X_train), desc="Computing stability scores", total=n_splits):
        # Train TM on this split
        tm_split = TMClassifier(
            number_of_clauses=clauses, T=T, s=s,
            max_included_literals=32, platform='CPU'
        )
        tm_split.fit(X_train[train_idx], y_train[train_idx], epochs=epochs)
        
        # Compute scores for this split
        split_scores = {}
        for name, score_fn in tqdm([
            ('tm_importance', lambda: tm_importance_score(tm_split)),
            ('tm_frequency', lambda: tm_frequency_score(tm_split)),
            ('tm_confidence', lambda: tm_confidence_score(tm_split)),
            ('tm_coverage', lambda: tm_coverage_score(tm_split)),
            ('tm_combined', lambda: tm_combined_score(tm_split)),
            ('lime', lambda: compute_lime_scores(tm_split, X_train[train_idx], X_train[val_idx])),
            ('shap', lambda: compute_shap_scores(tm_split, X_train[train_idx], X_train[val_idx])),
            ('group_lasso', lambda: group_lasso_score(tm_split)),
            ('l1_regularization', lambda: l1_regularization_score(tm_split)),
            ('variational_dropout', lambda: variational_dropout_score(tm_split, X_train[val_idx], y_train[val_idx])),
            ('integrated_gradients', lambda: integrated_gradients_score(tm_split, X_train[val_idx], y_train[val_idx])),
            ('taylor_criteria', lambda: taylor_criteria_score(tm_split, X_train[val_idx], y_train[val_idx])),
            ('feature_masking', lambda: feature_masking_score(tm_split, X_train[val_idx], y_train[val_idx])),
            ('information_bottleneck', lambda: information_bottleneck_score(tm_split, X_train[val_idx], y_train[val_idx])),
            ('tsetlin_mutual_info', lambda: tsetlin_mutual_info_score(tm_split, X_train[val_idx], y_train[val_idx])),
            ('tsetlin_relieff', lambda: tsetlin_relieff_score(tm_split, X_train[val_idx], y_train[val_idx])),
            ('tsetlin_rfe', lambda: tsetlin_rfe_score(tm_split, X_train[val_idx], y_train[val_idx])),
            ('clause_analysis', lambda: clause_analysis_score(tm_split))
        ], desc="Computing feature scores", leave=False):
            split_scores[name] = score_fn()
        
        # Compare with original scores
        for name in scores.keys():
            if name in split_scores:
                # Compute correlation between original and split scores
                corr = np.corrcoef(scores[name], split_scores[name])[0,1]
                stability_scores[name].append(corr)
    
    # Average stability scores
    return {name: np.mean(scores) for name, scores in stability_scores.items()}

def compute_lime_scores(tm, X_train, X_val):
    """Compute LIME scores for a given TM model."""
    lime_explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=[f"feature_{i}" for i in range(X_train.shape[1])],
        class_names=[str(x) for x in np.unique(y_train)],
        mode='classification',
        discretize_continuous=False,
        verbose=False,
        random_state=42
    )
    
    def lime_predict(X):
        return predict_proba(tm, X, y_train)
    
    sample = X_val[0].reshape(1, -1)
    lime_exp = lime_explainer.explain_instance(
        data_row=sample.flatten(),
        predict_fn=lime_predict,
        num_features=min(100, X_train.shape[1]),  # Limit number of features
        num_samples=1000  # Reduced from 5000
    )
    
    lime_scores = np.zeros(X_train.shape[1])
    for feature_str, weight in lime_exp.as_list():
        try:
            idx = int(feature_str.split('_')[1])
            lime_scores[idx] = abs(weight)
        except Exception:
            continue
    return lime_scores

def compute_shap_scores(tm, X_train, X_val):
    """Compute SHAP scores for a given TM model."""
    background = X_train[np.random.choice(X_train.shape[0], min(100, X_train.shape[0]), replace=False)]
    masker = shap.maskers.Independent(X_train)
    
    def shap_predict(X):
        return predict_proba(tm, X, y_train)
    
    explainer = shap.KernelExplainer(shap_predict, background, masker=masker)
    shap_values = explainer.shap_values(X_val[0:1])
    
    if isinstance(shap_values, list):
        shap_exp = shap_values[tm.predict(X_val[0:1])[0]]
    else:
        shap_exp = shap_values
    
    shap_scores = np.mean(np.abs(shap_exp), axis=0)
    return shap_scores

def analyze_computational_complexity(timings):
    """Analyze computational complexity of each method.
    Returns normalized complexity scores."""
    # Normalize timings to [0,1] range
    times = np.array(list(timings.values()))
    complexity = minmax_norm(times)
    return dict(zip(timings.keys(), complexity))

def analyze_feature_redundancy(scores, X_val, threshold=0.8):
    """Analyze redundancy in selected features.
    Returns redundancy scores for each method."""
    redundancy_scores = {}
    
    for name, score in scores.items():
        # Get top 20% features
        k = int(len(score) * 0.2)
        top_features = np.argsort(score)[-k:]
        
        # Compute correlation matrix for selected features
        corr_matrix = np.corrcoef(X_val[:, top_features].T)
        
        # Count highly correlated feature pairs
        high_corr = np.sum(np.abs(corr_matrix) > threshold) - len(top_features)
        max_possible = len(top_features) * (len(top_features) - 1)
        
        # Normalize redundancy score
        redundancy_scores[name] = high_corr / max_possible if max_possible > 0 else 0
    
    return redundancy_scores

def analyze_class_balance_impact(scores, X_val, y_val):
    """Analyze how feature selection methods handle class imbalance.
    Returns class balance scores for each method."""
    class_balance_scores = {}
    
    for name, score in scores.items():
        # Get top 20% features
        k = int(len(score) * 0.2)
        top_features = np.argsort(score)[-k:]
        
        # Train TM on selected features
        tm_selected = TMClassifier(
            number_of_clauses=clauses, T=T, s=s,
            max_included_literals=max_lit, platform='CPU'
        )
        tm_selected.fit(X_val[:, top_features], y_val, epochs=epochs)
        
        # Compute per-class accuracy
        preds = tm_selected.predict(X_val[:, top_features])
        class_acc = np.array([
            (preds[y_val==c] == c).mean() if np.any(y_val==c) else 0.0
            for c in range(n_classes)
        ])
        
        # Score based on standard deviation of class accuracies (lower is better)
        class_balance_scores[name] = 1.0 - np.std(class_acc)
    
    return class_balance_scores

def load_dataset(dataset_name, num_train_val_samples=5000, num_test_samples=1000):
    """Load and preprocess dataset."""
    rng = np.random.default_rng(42)
    
    if dataset_name == 'mnist':
        mnist_loader = MNIST()
        mnist_data = mnist_loader.get()
        
        # Subsample from training data
        train_pool_indices = rng.choice(mnist_data["x_train"].shape[0], num_train_val_samples, replace=False)
        X_train_val_pool = mnist_data["x_train"][train_pool_indices].astype(np.uint32)
        y_train_val_pool = mnist_data["y_train"][train_pool_indices].astype(np.uint32)
        
        # Split into train and validation
        X_train_bin, X_val_bin, y_train, y_val = train_test_split(
            X_train_val_pool, y_train_val_pool,
            test_size=0.2,
            random_state=42,
            stratify=y_train_val_pool
        )
        
        # Subsample test data
        test_indices = rng.choice(mnist_data["x_test"].shape[0], num_test_samples, replace=False)
        X_test_bin = mnist_data["x_test"][test_indices].astype(np.uint32)
        y_test = mnist_data["y_test"][test_indices].astype(np.uint32)
        
    elif dataset_name == 'digits':
        digits = load_digits()
        X = digits.images.reshape(len(digits.images), -1).astype(np.uint32)
        y = digits.target.astype(np.uint32)
        X_bin = (X > np.median(X)).astype(np.uint32)
        
        # Split into train+val and test
        X_train_val, X_test_bin, y_train_val, y_test = train_test_split(
            X_bin, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Split train+val into train and validation
        X_train_bin, X_val_bin, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
        )
        
    elif dataset_name == 'breast_cancer':
        cancer = load_breast_cancer()
        X = cancer.data
        y = cancer.target
        X_bin = (X > np.median(X, axis=0)).astype(np.uint32)
        
        # Split into train+val and test
        X_train_val, X_test_bin, y_train_val, y_test = train_test_split(
            X_bin, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Split train+val into train and validation
        X_train_bin, X_val_bin, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
        )
        
    elif dataset_name == 'wine':
        wine = load_wine()
        X = wine.data
        y = wine.target
        X_bin = (X > np.median(X, axis=0)).astype(np.uint32)
        
        # Split into train+val and test
        X_train_val, X_test_bin, y_train_val, y_test = train_test_split(
            X_bin, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Split train+val into train and validation
        X_train_bin, X_val_bin, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
        )
    
    return X_train_bin, X_val_bin, X_test_bin, y_train, y_val, y_test

def analyze_dataset(dataset_name, X_train_bin, X_val_bin, X_test_bin, y_train, y_val, y_test):
    """Analyze a single dataset with all feature selection methods."""
    timings = {}
    
    # TM hyperparameters
    clauses = 50
    T = 150
    s = 3.0
    max_lit = 20
    epochs = 10
    
    # Train TM
    tm = TMClassifier(
        number_of_clauses=clauses,
        T=T,
        s=s,
        max_included_literals=max_lit,
        platform='CPU'
    )
    n_clauses = tm.number_of_clauses // 2
    
    # Record weight history and clause information
    net_w_history = []
    clause_evolution = []  # Track clause evolution
    feature_clause_usage = np.zeros((X_train_bin.shape[1], clauses))  # Track feature usage in clauses
    
    for ep in range(epochs):
        tm.fit(X_train_bin, y_train, epochs=1)
        ta = count_ta_states(tm)
        pos, neg = count_clause_weights(tm, ta)
        net_w_history.append(pos - neg)
        
        # Track clause evolution
        clause_states = []
        for i in range(n_clauses):
            clause = []
            for j in range(X_train_bin.shape[1]):
                if ta[0, 0, i, j] > 0:  # Feature is included in clause
                    clause.append(j)
            clause_states.append(clause)
        clause_evolution.append(clause_states)
        
        # Update feature-clause usage
        for i in range(n_clauses):
            for j in range(X_train_bin.shape[1]):
                if ta[0, 0, i, j] > 0:
                    feature_clause_usage[j][i] += 1
        
        print(f"Epoch {ep+1}/{epochs}")
    
    net_w_history = np.stack(net_w_history, axis=0)
    
    # Get validation accuracy
    y_pred_val = tm.predict(X_val_bin)
    full_acc_val = 100 * (y_pred_val == y_val).mean()
    print(f"Full-feature Accuracy on {dataset_name} Validation Set: {full_acc_val:.2f}%\n")
    
    # Calculate base quantities
    net_w = net_w_history[-1]
    abs_w = np.abs(net_w)
    n_classes, n_feats = net_w.shape
    
    # Calculate class weights
    class_acc = np.array([
        (y_pred_val[y_val==c] == c).mean() if np.any(y_val==c) else 0.0
        for c in range(n_classes)
    ])
    class_w = class_acc / (class_acc.sum() + 1e-12)
    
    # Calculate all feature selection scores
    scores = {}
    
    # 1. TM-based scores
    t0 = time.perf_counter()
    weight_score = abs_w.max(axis=0)
    scores['TM-weight'] = minmax_norm(weight_score)
    timings['TM-weight'] = time.perf_counter() - t0
    
    # 2. Class-weighted scores
    t0 = time.perf_counter()
    cw_sum = (class_w[:,None] * abs_w).sum(axis=0)
    scores['CW-Sum'] = minmax_norm(cw_sum)
    timings['CW-Sum'] = time.perf_counter() - t0
    
    # 3. Per-feature CW
    t0 = time.perf_counter()
    freq = ta.sum(axis=(1,2))
    alpha = freq/(freq.sum(axis=0,keepdims=True)+1e-12)
    cw_feat = (alpha * abs_w).sum(axis=0)
    scores['CW-Feat'] = minmax_norm(cw_feat)
    timings['CW-Feat'] = time.perf_counter() - t0
    
    # 4. Supportive Class-Weighted Sum
    t0 = time.perf_counter()
    class_error_rate = 1.0 - class_acc
    class_error_weight = class_error_rate / (class_error_rate.sum() + 1e-12)
    support_cw_sum = (class_error_weight[:, None] * abs_w).sum(axis=0)
    scores['Support-CW-Sum'] = minmax_norm(support_cw_sum)
    timings['Support-CW-Sum'] = time.perf_counter() - t0
    
    # 5. Margin top vs runner-up
    t0 = time.perf_counter()
    sorted_abs = np.sort(abs_w, axis=0)
    margin = sorted_abs[-1] - sorted_abs[-2]
    scores['Margin'] = minmax_norm(margin)
    timings['Margin'] = time.perf_counter() - t0
    
    # 6. Entropy-based
    t0 = time.perf_counter()
    p = abs_w / (abs_w.sum(axis=0, keepdims=True) + 1e-12)
    entropy = - (p * np.log(p + 1e-12)).sum(axis=0)
    entropy_score = np.log(n_classes) - entropy
    scores['Entropy'] = minmax_norm(entropy_score)
    timings['Entropy'] = time.perf_counter() - t0
    
    # 7. Gini-based
    t0 = time.perf_counter()
    gini_score = 1.0 - (1.0 - (p**2).sum(axis=0))
    scores['Gini'] = minmax_norm(gini_score)
    timings['Gini'] = time.perf_counter() - t0
    
    # 8. Stability across epochs
    t0 = time.perf_counter()
    max_abs_hist = np.max(np.abs(net_w_history), axis=1)
    std_hist = max_abs_hist.std(axis=0)
    mean_abs = max_abs_hist.mean(axis=0)
    stab_score = mean_abs / (std_hist + 1e-6)
    scores['Stability'] = minmax_norm(stab_score)
    timings['Stability'] = time.perf_counter() - t0
    
    # 9. Dropout sensitivity
    t0 = time.perf_counter()
    drop = np.zeros(n_feats)
    for f in range(n_feats):
        Xm = X_val_bin.copy()
        Xm[:,f] = 0
        drop[f] = full_acc_val - 100*(tm.predict(Xm)==y_val).mean()
    scores['Dropout'] = minmax_norm(drop)
    timings['Dropout'] = time.perf_counter() - t0
    
    # 10. Mutual information
    t0 = time.perf_counter()
    correct_val = (y_pred_val == y_val).astype(int)
    mi = np.array([mutual_info_score(X_val_bin[:,f], correct_val) for f in range(n_feats)])
    scores['MutualInfo'] = minmax_norm(mi)
    timings['MutualInfo'] = time.perf_counter() - t0
    
    # 11. L1 Regularization
    t0 = time.perf_counter()
    l1_reg_score = abs_w.sum(axis=0)
    scores['L1-Reg'] = minmax_norm(l1_reg_score)
    timings['L1-Reg'] = time.perf_counter() - t0
    
    # 12. Group Lasso (for image data)
    t0 = time.perf_counter()
    img_dim = int(np.sqrt(n_feats))
    if img_dim * img_dim == n_feats:
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
                l2_norm_group = np.sqrt(np.sum(group_weights**2, axis=(1,2)))
                group_l2_norms_per_class[:, r_group, c_group] = l2_norm_group
        
        mean_group_l2_norms = group_l2_norms_per_class.mean(axis=0)
        group_lasso_feature_scores = np.zeros(n_feats)
        for r_pixel in range(img_dim):
            for c_pixel in range(img_dim):
                r_group, c_group = r_pixel // group_size, c_pixel // group_size
                if r_group < num_groups_h and c_group < num_groups_w:
                    feature_idx = r_pixel * img_dim + c_pixel
                    group_lasso_feature_scores[feature_idx] = mean_group_l2_norms[r_group, c_group]
        scores['GroupLasso'] = minmax_norm(group_lasso_feature_scores)
    else:
        scores['GroupLasso'] = np.full(n_feats, 0.5)
    timings['GroupLasso'] = time.perf_counter() - t0
    
    # 13. Taylor Criteria
    t0 = time.perf_counter()
    original_class_sums_val = tm.predict(X_val_bin, return_class_sums=True)[1]
    original_scores_for_true_class_val = original_class_sums_val[np.arange(len(y_val)), y_val]
    taylor_scores = np.zeros(n_feats)
    for f_idx in range(n_feats):
        X_perturbed = X_val_bin.copy()
        X_perturbed[:, f_idx] = 1 - X_perturbed[:, f_idx]
        perturbed_class_sums_val = tm.predict(X_perturbed, return_class_sums=True)[1]
        perturbed_scores_for_true_class_val = perturbed_class_sums_val[np.arange(len(y_val)), y_val]
        taylor_scores[f_idx] = np.mean(np.abs(original_scores_for_true_class_val - perturbed_scores_for_true_class_val))
    scores['TaylorCrit'] = minmax_norm(taylor_scores)
    timings['TaylorCrit'] = time.perf_counter() - t0
    
    # 14. Variational Dropout
    t0 = time.perf_counter()
    N_MASKS_VAR_DROPOUT = 10
    var_dropout_scores = np.zeros(n_feats)
    original_sums_val_for_var_dropout = tm.predict(X_val_bin, return_class_sums=True)[1]
    original_true_class_scores_val_var_dropout = original_sums_val_for_var_dropout[np.arange(len(y_val)), y_val]
    for f_idx in range(n_feats):
        score_changes_for_f = []
        for _ in range(N_MASKS_VAR_DROPOUT):
            X_val_temp = X_val_bin.copy()
            mask_f = np.random.randint(0, 2, size=X_val_bin.shape[0])
            X_val_temp[:, f_idx] = X_val_temp[:, f_idx] * mask_f
            perturbed_sums_val = tm.predict(X_val_temp, return_class_sums=True)[1]
            perturbed_true_class_scores_val = perturbed_sums_val[np.arange(len(y_val)), y_val]
            score_changes_for_f.append(np.mean(np.abs(original_true_class_scores_val_var_dropout - perturbed_true_class_scores_val)))
        var_dropout_scores[f_idx] = np.mean(score_changes_for_f)
    scores['VarDropout'] = minmax_norm(var_dropout_scores)
    timings['VarDropout'] = time.perf_counter() - t0
    
    # 15. Chi-squared
    t0 = time.perf_counter()
    chi2_scores, _ = chi2(X_train_bin, y_train)
    scores['Chi2'] = minmax_norm(np.nan_to_num(chi2_scores))
    timings['Chi2'] = time.perf_counter() - t0
    
    # 16. Variance Threshold
    t0 = time.perf_counter()
    selector_var = VarianceThreshold()
    selector_var.fit(X_train_bin)
    variance_scores = selector_var.variances_
    scores['Variance'] = minmax_norm(variance_scores)
    timings['Variance'] = time.perf_counter() - t0
    
    # 17. SHAP
    t0 = time.perf_counter()
    print("Computing SHAP explanation...")
    sample_idx_for_shap = 0
    sample_shap = X_val_bin[sample_idx_for_shap].reshape(1, -1)
    background_size = min(100, X_train_bin.shape[0])
    background_indices = np.random.choice(X_train_bin.shape[0], background_size, replace=False)
    background = X_train_bin[background_indices]
    
    def shap_predict(X):
        return predict_proba(tm, X, y_val)
    
    masker = shap.maskers.Independent(X_train_bin)
    shap_explainer = shap.KernelExplainer(shap_predict, background, masker=masker)
    shap_values = shap_explainer.shap_values(sample_shap, nsamples=min(50, X_val_bin.shape[0]))
    predicted_class = tm.predict(sample_shap)[0]
    
    if isinstance(shap_values, list):
        shap_exp = shap_values[predicted_class] if predicted_class < len(shap_values) else shap_values[0]
    else:
        shap_exp = shap_values[predicted_class] if shap_values.shape[0] > 1 else shap_values
    
    shap_exp = np.atleast_3d(shap_exp)
    shap_scores = np.mean(np.abs(shap_exp), axis=2).flatten()
    scores['SHAP'] = minmax_norm(shap_scores)
    timings['SHAP'] = time.perf_counter() - t0
    
    # 18. LIME
    t0 = time.perf_counter()
    print("Computing LIME explanation...")
    feat_names = [f"f{i}" for i in range(n_feats)]
    class_names = [str(c) for c in range(n_classes)]
    lime_expl = LimeTabularExplainer(
        training_data=X_train_bin,
        feature_names=feat_names,
        class_names=class_names,
        mode='classification',
        discretize_continuous=False,
        verbose=False,
        random_state=42
    )
    
    def lime_pred(X):
        return predict_proba(tm, X, y_train)
    
    lime_imp = np.zeros(n_feats)
    num_lime_samples = min(20, X_val_bin.shape[0])
    for i in range(num_lime_samples):
        exp = lime_expl.explain_instance(
            data_row=X_val_bin[i],
            predict_fn=lime_pred,
            num_features=n_feats,
            num_samples=min(1000, X_train_bin.shape[0])
        )
        for feat_str, w in exp.as_list():
            idx = int(feat_str.split()[0][1:])
            lime_imp[idx] += abs(w)
    scores['LIME'] = minmax_norm(lime_imp)
    timings['LIME'] = time.perf_counter() - t0
    
    # 29. Clause-based Feature Importance (CFI)
    t0 = time.perf_counter()
    print("Computing CFI scores...")
    cfi_scores = np.zeros(n_feats)
    
    # Get final clause states
    final_ta = count_ta_states(tm)
    
    # For each feature, calculate its importance based on clause usage
    for f in range(n_feats):
        feature_importance = 0
        for c in range(n_clauses):
            if final_ta[0, 0, c, f] > 0:  # Feature is included in clause
                # Calculate clause accuracy
                clause_predictions = np.zeros(len(X_val_bin))
                for i in range(len(X_val_bin)):
                    if all(X_val_bin[i][j] == 1 for j in range(n_feats) if final_ta[0, 0, c, j] > 0):
                        clause_predictions[i] = 1
                
                clause_acc = np.mean(clause_predictions == y_val)
                feature_importance += clause_acc * abs_w[c % n_classes][f]
        
        cfi_scores[f] = feature_importance
    
    scores['CFI'] = minmax_norm(cfi_scores)
    timings['CFI'] = time.perf_counter() - t0
    
    # 30. Tsetlin Pattern Recognition (TPR)
    t0 = time.perf_counter()
    print("Computing TPR scores...")
    tpr_scores = np.zeros(n_feats)
    
    # Analyze feature co-occurrence patterns in clauses
    for f in range(n_feats):
        pattern_score = 0
        for c in range(n_clauses):
            if final_ta[0, 0, c, f] > 0:  # Feature is in clause
                # Find other features in this clause
                clause_features = [j for j in range(n_feats) if final_ta[0, 0, c, j] > 0]
                # Calculate pattern strength based on clause weight and number of features
                pattern_strength = abs_w[c % n_classes][f] / (len(clause_features) + 1)
                pattern_score += pattern_strength
        
        tpr_scores[f] = pattern_score
    
    scores['TPR'] = minmax_norm(tpr_scores)
    timings['TPR'] = time.perf_counter() - t0
    
    # 31. Dynamic Clause Evolution (DCE)
    t0 = time.perf_counter()
    print("Computing DCE scores...")
    dce_scores = np.zeros(n_feats)
    
    # Track feature importance evolution across epochs
    feature_evolution = np.zeros((n_feats, epochs))
    for ep in range(epochs):
        ta = count_ta_states(tm)
        for f in range(n_feats):
            # Calculate feature importance in this epoch
            importance = 0
            for c in range(n_clauses):
                if ta[0, 0, c, f] > 0:
                    importance += abs_w[c % n_classes][f]
            feature_evolution[f][ep] = importance
    
    # Calculate stability and trend of feature importance
    for f in range(n_feats):
        # Calculate stability (inverse of variance)
        stability = 1 / (np.var(feature_evolution[f]) + 1e-10)
        # Calculate positive trend
        trend = np.polyfit(range(epochs), feature_evolution[f], 1)[0]
        # Combine stability and trend
        dce_scores[f] = stability * (1 + trend) if trend > 0 else stability
    
    scores['DCE'] = minmax_norm(dce_scores)
    timings['DCE'] = time.perf_counter() - t0
    
    return scores, timings

def main():
    # List of datasets to analyze
    datasets = ['mnist', 'digits', 'breast_cancer', 'wine']
    
    all_results = {}
    all_timings = {}
    
    for dataset_name in datasets:
        print(f"\nAnalyzing {dataset_name}...")
        
        # Load dataset
        X_train_bin, X_val_bin, X_test_bin, y_train, y_val, y_test = load_dataset(dataset_name)
        
        # Analyze dataset
        scores, timings = analyze_dataset(
            dataset_name, X_train_bin, X_val_bin, X_test_bin,
            y_train, y_val, y_test
        )
        
        all_results[dataset_name] = scores
        all_timings[dataset_name] = timings
    
    # Plot results
    plot_results(all_results, all_timings)

def plot_results(results, timings):
    """Plot comparison of methods across datasets."""
    datasets = list(results.keys())
    methods = list(results[datasets[0]].keys())
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(datasets))
    width = 0.8 / len(methods)
    
    # Create a list to track which methods were skipped
    skipped_methods = []
    
    for i, method in enumerate(methods):
        accuracies = [results[d][method] for d in datasets]
        # Check if method was skipped for all datasets
        if all(acc == 0 for acc in accuracies):
            skipped_methods.append(method)
            continue
        plt.bar(x + i*width, accuracies, width, label=method)
    
    plt.xlabel('Dataset')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Method Performance Across Datasets')
    plt.xticks(x + width*len(methods)/2, datasets)
    
    # Add note about skipped methods
    if skipped_methods:
        plt.figtext(0.02, 0.02, f"Skipped methods: {', '.join(skipped_methods)}", 
                   fontsize=8, style='italic')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('FS_3_types/ROAD_results/method_comparison.png')
    plt.close()

if __name__ == "__main__":
    main()
