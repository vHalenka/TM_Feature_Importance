"""
Main experiment runner for Top-K feature selection evaluation.

This script orchestrates the complete feature selection evaluation:
1. Load and preprocess datasets
2. Train Tsetlin Machine
3. Compute feature importance scores using all methods
4. Evaluate using 4 protocols (Top-K, Deletion, Insertion, ROAR, ROAD-Mask)
5. Generate visualizations
6. Save results
"""
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, balanced_accuracy_score, matthews_corrcoef

from tmu.models.classification.vanilla_classifier import TMClassifier

from ..utils import (
    data_loading,
    synthetic_datasets,
    tm_utils,
    paths,
    serialization
)
from . import feature_scoring, evaluation_protocols


def run_single_dataset_experiment(dataset_name, tm_params_dict, config):
    """
    Run complete feature selection experiment for a single dataset.
    
    Args:
        dataset_name: Name of the dataset
        tm_params_dict: Dictionary with dataset-specific TM parameters
        config: Configuration dictionary with experiment settings
        
    Returns:
        Dictionary with all results for this dataset
    """
    print(f"\n{'='*60}")
    print(f"Processing Dataset: {dataset_name}")
    print(f"{'='*60}")
    
    # Load dataset
    ground_truth_features = None
    if dataset_name == "Increasing_Parity_Complexity":
        X_bin, y_processed, _, _, ground_truth_features = synthetic_datasets.generate_increasing_parity_dataset()
        dataset_type_info = " (Pre-binarized Synthetic)"
    elif dataset_name == "Hierarchical_Boolean_Rules":
        X_bin, y_processed, _, _, ground_truth_features = synthetic_datasets.generate_hierarchical_boolean_dataset()
        dataset_type_info = " (Pre-binarized Synthetic)"
    elif dataset_name == "Progressive_Feature_Interaction":
        X_bin, y_processed, _, _, ground_truth_features = synthetic_datasets.generate_progressive_interaction_dataset()
        dataset_type_info = " (Pre-binarized Synthetic)"
    else:
        X_raw, y_raw = data_loading.load_dataset(dataset_name)
        X_bin, y_processed = data_loading.preprocess_data(X_raw, y_raw, max_bins=config['max_bins'])
        dataset_type_info = f" (Thermometer Encoded, max_bins={config['max_bins']})"
    
    # Split data: 60% train, 20% validation, 20% test
    X_train_val_pool, X_test_bin, y_train_val_pool, y_test = train_test_split(
        X_bin, y_processed, test_size=0.2, random_state=42, stratify=y_processed
    )
    X_train_bin, X_val_bin, y_train, y_val = train_test_split(
        X_train_val_pool, y_train_val_pool, test_size=0.25, random_state=42, stratify=y_train_val_pool
    )
    
    print(f"Dataset shapes: X_train: {X_train_bin.shape}, X_val: {X_val_bin.shape}, X_test: {X_test_bin.shape}")
    
    # Get TM parameters
    dataset_params = tm_params_dict.get(dataset_name, {})
    s_current = dataset_params.get('best_s', 3.0)
    T_current = dataset_params.get('best_T', 600)
    print(f"Using TM parameters: s={s_current:.4f}, T={T_current}")
    
    # Train TM and collect weight history
    print("\nTraining Tsetlin Machine...")
    tm = TMClassifier(
        number_of_clauses=config['clauses'],
        T=T_current,
        s=s_current,
        max_included_literals=config['max_lit'],
        platform='CPU'
    )
    
    pos_w_history = []
    neg_w_history = []
    for ep in range(config['epochs']):
        tm.fit(X_train_bin, y_train, epochs=1)
        ta = tm_utils.count_ta_states(tm)
        pos, neg = tm_utils.count_clause_weights(tm, ta)
        pos_w_history.append(pos)
        neg_w_history.append(neg)
        if (ep + 1) % 5 == 0:
            y_pred_val = tm.predict(X_val_bin)
            val_acc = 100 * (y_pred_val == y_val).mean()
            print(f"  Epoch {ep+1}/{config['epochs']}, Val acc: {val_acc:.2f}%")
    
    pos_w_history = np.stack(pos_w_history, axis=0)
    neg_w_history = np.stack(neg_w_history, axis=0)
    pos_w = pos_w_history[-1]
    neg_w = neg_w_history[-1]
    ta = tm_utils.count_ta_states(tm)
    
    # Compute final validation accuracy
    y_pred_val = tm.predict(X_val_bin)
    full_acc_val = 100 * (y_pred_val == y_val).mean()
    print(f"\nFull-feature Validation Accuracy: {full_acc_val:.2f}%")
    
    # Compute feature scores
    print("\nComputing feature importance scores...")
    rng = np.random.default_rng(42)
    
    scores, timings = feature_scoring.compute_tm_feature_scores(
        tm, X_train_bin, X_val_bin, y_train, y_val,
        ta, pos_w, neg_w, pos_w_history, neg_w_history,
        config['epochs'], rng
    )
    
    # Optionally compute explainer-based scores (expensive)
    if config.get('compute_explainers', False):
        print("\nComputing explainer-based scores (SHAP, LIME, IG, etc.)...")
        explainer_scores, explainer_timings = feature_scoring.compute_explainer_scores(
            tm, X_train_bin, X_val_bin, y_train, y_val,
            X_train_bin.shape[1], len(np.unique(y_train)), rng
        )
        scores.update(explainer_scores)
        timings.update(explainer_timings)
    
    print(f"Computed {len(scores)} feature scoring methods")
    
    # Run evaluation protocols
    print("\nRunning evaluation protocols...")
    model_params = {
        'clauses': config['clauses'],
        'T': T_current,
        's': s_current,
        'max_included_literals': config['max_lit'],
        'epochs': config['epochs']
    }
    
    # Protocol 1: Top-K
    print("  Protocol 1: Top-K Performance...")
    top_k_results, K_list = evaluation_protocols.evaluate_top_k(
        scores, X_train_bin, y_train, X_test_bin, y_test,
        model_params, n_trials=config.get('n_trials', 10),
        max_k=config.get('max_k', 50)
    )
    
    # Protocol 2: Deletion Curve
    print("  Protocol 2: Deletion Curve...")
    deletion_results, K_perturb_list = evaluation_protocols.evaluate_deletion_curve(
        scores, X_train_bin, y_train, X_test_bin, y_test,
        model_params, n_points=config.get('n_points', 25)
    )
    
    # Protocol 3: Insertion Curve
    print("  Protocol 3: Insertion Curve...")
    insertion_results, _ = evaluation_protocols.evaluate_insertion_curve(
        scores, X_train_bin, y_train, X_test_bin, y_test,
        model_params, n_points=config.get('n_points', 25)
    )
    
    # Protocol 4: ROAR Curve
    print("  Protocol 4: ROAR Curve...")
    roar_results, _ = evaluation_protocols.evaluate_roar_curve(
        scores, X_train_bin, y_train, X_test_bin, y_test,
        model_params, n_trials=config.get('n_trials', 3),
        n_points=config.get('n_points', 25)
    )
    
    # Protocol 5: ROAD-Mask Curve
    print("  Protocol 5: ROAD-Mask Curve...")
    road_mask_results, _ = evaluation_protocols.evaluate_road_mask_curve(
        scores, X_train_bin, y_train, X_test_bin, y_test,
        model_params, n_trials=config.get('n_trials', 3),
        n_points=config.get('n_points', 25)
    )
    
    # Compute correlation matrix
    print("\nComputing correlation matrix...")
    method_names = list(scores.keys())
    arr = np.vstack([scores[m_name] for m_name in method_names])
    corr = np.corrcoef(arr)
    
    # Compute normalized AUC for Top-K
    print("Computing normalized AUC...")
    aucs = {}
    for method_name in method_names:
        if method_name in top_k_results and len(top_k_results[method_name]) > 0:
            aucs[method_name] = evaluation_protocols.compute_normalized_auc(
                top_k_results[method_name], K_list
            )
        else:
            aucs[method_name] = 0.0
    
    # Compute final test metrics
    y_test_pred = tm.predict(X_test_bin)
    full_model_test_accuracy = 100 * (y_test_pred == y_test).mean()
    full_model_test_f1 = 100 * f1_score(y_test, y_test_pred, average="macro")
    full_model_test_bal_accuracy = balanced_accuracy_score(y_test, y_test_pred)
    full_model_test_matthews_corrcoef = matthews_corrcoef(y_test, y_test_pred)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    generate_plots(
        dataset_name, scores, method_names, corr,
        top_k_results, K_list, aucs,
        deletion_results, insertion_results, roar_results, road_mask_results,
        K_perturb_list, config.get('local_only', False)
    )
    
    # Prepare output data
    output_data = {
        "experiment_description": {
            "dataset_name": f"{dataset_name}{dataset_type_info}",
            "X_train_shape": list(X_train_bin.shape),
            "y_train_shape": list(y_train.shape),
            "X_val_shape": list(X_val_bin.shape),
            "y_val_shape": list(y_val.shape),
            "X_test_shape": list(X_test_bin.shape),
            "y_test_shape": list(y_test.shape),
            "tm_parameters": {
                "clauses": config['clauses'],
                "T_used": T_current,
                "s_used": s_current,
                "max_included_literals": config['max_lit'],
                "epochs_for_feature_scoring_model": config['epochs']
            },
            "top_k_comparison_trials": config.get('n_trials', 10)
        },
        "timings_seconds": timings,
        "feature_correlation_matrix": {
            "method_names": method_names,
            "matrix": corr.tolist()
        },
        "normalized_auc_top_k": aucs,
        "full_model_test_accuracy_percent": full_model_test_accuracy,
        "full_model_test_f1_score": full_model_test_f1,
        "full_model_test_balanced_accuracy": full_model_test_bal_accuracy,
        "full_model_test_matthews_corrcoef": full_model_test_matthews_corrcoef,
        "top_k_accuracies_vs_k": {k: [v] if not isinstance(v, list) else v for k, v in top_k_results.items()},
        "deletion_curve_accuracies_vs_k": deletion_results,
        "insertion_curve_accuracies_vs_k": insertion_results,
        "roar_curve_accuracies_vs_k": roar_results,
        "road_mask_curve_accuracies_vs_k": road_mask_results
    }
    
    if ground_truth_features is not None:
        output_data["experiment_description"]["ground_truth_important_features"] = ground_truth_features.tolist()
    
    return output_data


def generate_plots(dataset_name, scores, method_names, corr,
                   top_k_results, K_list, aucs,
                   deletion_results, insertion_results, roar_results, road_mask_results,
                   K_perturb_list, local_only=False):
    """Generate all visualization plots for a dataset."""
    non_random_methods = [m for m in method_names if m != "Random"]
    
    # 1. Score correlations heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(method_names)))
    ax.set_yticks(range(len(method_names)))
    ax.set_xticklabels(method_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(method_names, fontsize=8)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    for i in range(len(method_names)):
        for j in range(len(method_names)):
            c = corr[i, j]
            color = 'white' if abs(c) > 0.5 else 'black'
            plt.text(j, i, f"{c:.2f}", ha='center', va='center', color=color, fontsize=7)
    plt.title(f'Score Correlations ({dataset_name})')
    plt.tight_layout()
    output_path = paths.get_correlation_plot_path(dataset_name, local_only=local_only)
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    # 2. Top-K performance plot
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap('tab20')
    base_colors = list(cmap.colors)
    colors_for_plot = [base_colors[i % len(base_colors)] for i in range(len(method_names))]
    
    for idx, method_name in enumerate(method_names):
        if method_name in top_k_results:
            ax.plot(K_list, top_k_results[method_name], marker='o', 
                   label=method_name, color=colors_for_plot[idx], markersize=3)
    ax.set_xlabel('Number of Features (K)')
    ax.set_ylabel(f'Avg Test Accuracy % on {dataset_name}')
    ax.set_title(f'Top-K Feature Pruning - Performance on Test Set ({dataset_name})')
    ax.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    output_path = paths.get_top_k_plot_path(dataset_name, local_only=local_only)
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    # 3. AUC Top-K bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    auc_vals = [aucs[m] for m in method_names]
    ax.bar(method_names, auc_vals, color=colors_for_plot)
    ax.set_xticks(range(len(method_names)))
    ax.set_xticklabels(method_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("Normalized AUC")
    ax.set_title(f"Area under Accuracy–vs–K Curves (Test Set Performance - {dataset_name})")
    plt.tight_layout()
    output_path = paths.get_auc_top_k_plot_path(dataset_name, local_only=local_only)
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    # 4-7. Pruning curves
    for curve_type, results in [
        ('deletion', deletion_results),
        ('insertion', insertion_results),
        ('roar', roar_results),
        ('road_mask', road_mask_results)
    ]:
        fig, ax = plt.subplots(figsize=(10, 6))
        for method_name in sorted(non_random_methods):
            if method_name in results and len(results[method_name]) == len(K_perturb_list):
                idx = method_names.index(method_name)
                ax.plot(K_perturb_list, results[method_name], 
                       marker='o', label=method_name, color=colors_for_plot[idx], markersize=3)
        if 'Random' in results and len(results['Random']) == len(K_perturb_list):
            ax.plot(K_perturb_list, results['Random'], 
                   marker='s', label='Random', color='gray', linestyle='--', markersize=3)
        
        titles = {
            'deletion': f'Deletion Curve ({dataset_name})',
            'insertion': f'Insertion Curve ({dataset_name})',
            'roar': f'ROAR Curve ({dataset_name})',
            'road_mask': f'ROAD-Mask Curve ({dataset_name})'
        }
        ax.set_xlabel('Number of Features (K)')
        ax.set_ylabel(f'Test Accuracy % on {dataset_name}')
        ax.set_title(titles[curve_type])
        ax.legend(ncol=3, fontsize=8)
        plt.tight_layout()
        output_path = paths.get_pruning_curve_path(curve_type, dataset_name, local_only=local_only)
        plt.savefig(output_path, dpi=150)
        plt.close()
    
    # 8. Combined pruning curves (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.ravel()
    
    for ax, (curve_type, results, title) in zip(axes, [
        ('insertion', insertion_results, 'Insertion Curve'),
        ('deletion', deletion_results, 'Deletion Curve'),
        ('roar', roar_results, 'ROAR Curve'),
        ('road_mask', road_mask_results, 'ROAD-Mask Curve')
    ]):
        for method_name in sorted(non_random_methods):
            if method_name in results and len(results[method_name]) == len(K_perturb_list):
                idx = method_names.index(method_name)
                ax.plot(K_perturb_list, results[method_name], 
                       marker='o', label=method_name, color=colors_for_plot[idx], markersize=2)
        if 'Random' in results and len(results['Random']) == len(K_perturb_list):
            ax.plot(K_perturb_list, results['Random'], 
                   marker='s', label='Random', color='gray', linestyle='--', markersize=2)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
    
    for ax in axes:
        ax.set_xlabel("Number of features K", fontsize=9)
        ax.set_ylabel("Test Accuracy %", fontsize=9)
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=7, 
              frameon=False, bbox_to_anchor=(0.5, -0.05))
    plt.suptitle(f"Feature‐pruning curves on {dataset_name}", fontsize=12, y=0.995)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    
    output_path = paths.get_pruning_curve_path('combined', dataset_name, local_only=local_only)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Generated all plots for {dataset_name}")

