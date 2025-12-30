"""
Evaluation protocols for feature selection methods.

This module implements the 4 testing protocols:
1. Top-K: Evaluate accuracy with top K features selected
2. Deletion: Remove top K features and evaluate
3. Insertion: Add features incrementally and evaluate
4. ROAR: Remove top K features, retrain, and evaluate
5. ROAD-Mask: Mask top K features, retrain, and evaluate
"""
import numpy as np
from numpy import trapz
from tmu.models.classification.vanilla_classifier import TMClassifier
from sklearn.metrics import accuracy_score


def evaluate_top_k(scores_dict, X_train, y_train, X_test, y_test, 
                   model_params, n_trials=10, max_k=50):
    """
    Protocol 1: Top-K Performance
    Select top K features based on scores and evaluate model accuracy.
    
    Args:
        scores_dict: Dictionary mapping method names to feature scores
        X_train, y_train: Training data
        X_test, y_test: Test data
        model_params: Dictionary with TM parameters (clauses, T, s, max_included_literals, epochs)
        n_trials: Number of trials for averaging
        max_k: Maximum number of features to evaluate
        
    Returns:
        Dictionary mapping method names to lists of accuracies for each K
        List of K values evaluated
    """
    n_feats = X_train.shape[1]
    K_list = list(np.unique(np.linspace(1, min(max_k, n_feats), 25, dtype=int)))
    results = {method: [] for method in scores_dict.keys()}
    
    clauses = model_params['clauses']
    T = model_params['T']
    s = model_params['s']
    max_lit = model_params['max_included_literals']
    epochs = model_params['epochs']
    
    for method_name, scores in scores_dict.items():
        ordering = np.argsort(scores)[::-1]
        for K in K_list:
            sel_indices = ordering[:K]
            if K == 0 or len(sel_indices) == 0:
                results[method_name].append(0.0)
                continue
            
            accs = []
            for _ in range(n_trials):
                tm = TMClassifier(
                    number_of_clauses=clauses,
                    T=T,
                    s=s,
                    max_included_literals=max_lit,
                    platform='CPU'
                )
                tm.fit(X_train[:, sel_indices], y_train, epochs=epochs)
                accs.append(100 * accuracy_score(y_test, tm.predict(X_test[:, sel_indices])))
            results[method_name].append(np.mean(accs))
    
    return results, K_list


def evaluate_deletion_curve(scores_dict, X_train, y_train, X_test, y_test,
                            model_params, n_points=25):
    """
    Protocol 2: Deletion Curve
    Remove top K features (mask to zero) and evaluate without retraining.
    
    Args:
        scores_dict: Dictionary mapping method names to feature scores
        X_train, y_train: Training data
        X_test, y_test: Test data
        model_params: Dictionary with TM parameters
        n_points: Number of points to evaluate
        
    Returns:
        Dictionary mapping method names to lists of accuracies
        List of K values (number of features masked)
    """
    n_feats = X_train.shape[1]
    num_points = min(n_feats + 1, n_points)
    K_perturb_list = np.unique(
        np.concatenate(([0], np.linspace(1, n_feats, num=num_points, dtype=int)))
    )
    
    # Train full model once
    clauses = model_params['clauses']
    T = model_params['T']
    s = model_params['s']
    max_lit = model_params['max_included_literals']
    epochs = model_params['epochs']
    
    tm_full = TMClassifier(
        number_of_clauses=clauses,
        T=T,
        s=s,
        max_included_literals=max_lit,
        platform='CPU'
    )
    tm_full.fit(X_train, y_train, epochs=epochs)
    full_acc = 100 * accuracy_score(y_test, tm_full.predict(X_test))
    
    results = {}
    for method_name, scores in scores_dict.items():
        ordering = np.argsort(scores)[::-1]
        accs = []
        for K in K_perturb_list:
            X_test_masked = X_test.copy()
            if K > 0:
                top_k_features = ordering[:K]
                X_test_masked[:, top_k_features] = 0
            acc = 100 * accuracy_score(y_test, tm_full.predict(X_test_masked))
            accs.append(acc)
        results[method_name] = accs
    
    return results, K_perturb_list.tolist()


def evaluate_insertion_curve(scores_dict, X_train, y_train, X_test, y_test,
                             model_params, n_points=25):
    """
    Protocol 3: Insertion Curve
    Start with all features masked, then reveal top K features incrementally.
    
    Args:
        scores_dict: Dictionary mapping method names to feature scores
        X_train, y_train: Training data
        X_test, y_test: Test data
        model_params: Dictionary with TM parameters
        n_points: Number of points to evaluate
        
    Returns:
        Dictionary mapping method names to lists of accuracies
        List of K values (number of features revealed)
    """
    n_feats = X_train.shape[1]
    num_points = min(n_feats + 1, n_points)
    K_perturb_list = np.unique(
        np.concatenate(([0], np.linspace(1, n_feats, num=num_points, dtype=int)))
    )
    
    clauses = model_params['clauses']
    T = model_params['T']
    s = model_params['s']
    max_lit = model_params['max_included_literals']
    epochs = model_params['epochs']
    
    results = {}
    for method_name, scores in scores_dict.items():
        ordering = np.argsort(scores)[::-1]
        accs = []
        for K in K_perturb_list:
            if K == 0:
                # All features masked - use random baseline
                accs.append(100.0 / len(np.unique(y_test)))  # Random accuracy
                continue
            
            # Reveal top K features
            revealed_features = ordering[:K]
            X_train_revealed = X_train.copy()
            X_test_revealed = X_test.copy()
            
            # Mask all features first
            X_train_revealed[:, :] = 0
            X_test_revealed[:, :] = 0
            
            # Reveal top K
            X_train_revealed[:, revealed_features] = X_train[:, revealed_features]
            X_test_revealed[:, revealed_features] = X_test[:, revealed_features]
            
            # Train and evaluate
            tm = TMClassifier(
                number_of_clauses=clauses,
                T=T,
                s=s,
                max_included_literals=max_lit,
                platform='CPU'
            )
            tm.fit(X_train_revealed, y_train, epochs=epochs)
            acc = 100 * accuracy_score(y_test, tm.predict(X_test_revealed))
            accs.append(acc)
        results[method_name] = accs
    
    return results, K_perturb_list.tolist()


def evaluate_roar_curve(scores_dict, X_train, y_train, X_test, y_test,
                       model_params, n_trials=3, n_points=25):
    """
    Protocol 4: ROAR (Remove And Retrain) Curve
    Remove top K features, retrain model, and evaluate.
    
    Args:
        scores_dict: Dictionary mapping method names to feature scores
        X_train, y_train: Training data
        X_test, y_test: Test data
        model_params: Dictionary with TM parameters
        n_trials: Number of trials for averaging
        n_points: Number of points to evaluate
        
    Returns:
        Dictionary mapping method names to lists of accuracies
        List of K values (number of features removed)
    """
    n_feats = X_train.shape[1]
    num_points = min(n_feats + 1, n_points)
    K_perturb_list = np.unique(
        np.concatenate(([0], np.linspace(1, n_feats, num=num_points, dtype=int)))
    )
    
    clauses = model_params['clauses']
    T = model_params['T']
    s = model_params['s']
    max_lit = model_params['max_included_literals']
    epochs = model_params['epochs']
    
    results = {}
    for method_name, scores in scores_dict.items():
        ordering = np.argsort(scores)[::-1]
        accs = []
        for K in K_perturb_list:
            if K >= n_feats:
                # All features removed
                accs.append(100.0 / len(np.unique(y_test)))
                continue
            
            # Remove top K features
            remaining_features = ordering[K:]
            if len(remaining_features) == 0:
                accs.append(100.0 / len(np.unique(y_test)))
                continue
            
            trial_accs = []
            for _ in range(n_trials):
                tm = TMClassifier(
                    number_of_clauses=clauses,
                    T=T,
                    s=s,
                    max_included_literals=max_lit,
                    platform='CPU'
                )
                tm.fit(X_train[:, remaining_features], y_train, epochs=epochs)
                trial_accs.append(100 * accuracy_score(y_test, tm.predict(X_test[:, remaining_features])))
            accs.append(np.mean(trial_accs))
        results[method_name] = accs
    
    return results, K_perturb_list.tolist()


def evaluate_road_mask_curve(scores_dict, X_train, y_train, X_test, y_test,
                            model_params, n_trials=3, n_points=25):
    """
    Protocol 5: ROAD-Mask (Remove And retrain with Masking) Curve
    Mask top K features to zero, retrain model, and evaluate.
    
    Args:
        scores_dict: Dictionary mapping method names to feature scores
        X_train, y_train: Training data
        X_test, y_test: Test data
        model_params: Dictionary with TM parameters
        n_trials: Number of trials for averaging
        n_points: Number of points to evaluate
        
    Returns:
        Dictionary mapping method names to lists of accuracies
        List of K values (number of features masked)
    """
    n_feats = X_train.shape[1]
    num_points = min(n_feats + 1, n_points)
    K_perturb_list = np.unique(
        np.concatenate(([0], np.linspace(1, n_feats, num=num_points, dtype=int)))
    )
    
    clauses = model_params['clauses']
    T = model_params['T']
    s = model_params['s']
    max_lit = model_params['max_included_literals']
    epochs = model_params['epochs']
    
    results = {}
    for method_name, scores in scores_dict.items():
        ordering = np.argsort(scores)[::-1]
        accs = []
        for K in K_perturb_list:
            # Mask top K features
            X_train_masked = X_train.copy()
            X_test_masked = X_test.copy()
            if K > 0:
                top_k_features = ordering[:K]
                X_train_masked[:, top_k_features] = 0
                X_test_masked[:, top_k_features] = 0
            
            trial_accs = []
            for _ in range(n_trials):
                tm = TMClassifier(
                    number_of_clauses=clauses,
                    T=T,
                    s=s,
                    max_included_literals=max_lit,
                    platform='CPU'
                )
                tm.fit(X_train_masked, y_train, epochs=epochs)
                trial_accs.append(100 * accuracy_score(y_test, tm.predict(X_test_masked)))
            accs.append(np.mean(trial_accs))
        results[method_name] = accs
    
    return results, K_perturb_list.tolist()


def compute_normalized_auc(accuracies, K_list):
    """
    Compute normalized AUC (Area Under Curve) for performance curves.
    
    Args:
        accuracies: List of accuracies for different K values
        K_list: List of K values
        
    Returns:
        Normalized AUC value
    """
    auc = trapz(accuracies, K_list)
    # Normalize by range
    auc_vals = np.array(accuracies)
    if len(auc_vals) > 0 and auc_vals.max() > auc_vals.min():
        auc_norm = (auc - auc_vals.min() * (K_list[-1] - K_list[0])) / (
            (auc_vals.max() - auc_vals.min()) * (K_list[-1] - K_list[0]) + 1e-12
        )
    else:
        auc_norm = 0.5
    return auc_norm

