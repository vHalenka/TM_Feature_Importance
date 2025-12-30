"""
Entry point script for Top-K feature selection experiments.

This script runs comprehensive feature selection analysis including:
- Multiple feature scoring methods (30+ methods)
- 4 testing protocols (Top-K, Deletion, Insertion, ROAR, ROAD-Mask)
- Visualization generation
- Results saving

Usage:
    python 3_run_ktop_experiments.py
"""
import sys
import os
import json

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import paths, serialization
from src.experiments import ktop_experiment

# Ensure output directories exist
paths.ensure_dirs()

# Configuration
CONFIG = {
    'clauses': 500,
    'epochs': 30,
    'max_lit': 32,
    'max_bins': 10,
    'n_trials': 10,  # For Top-K and ROAR/ROAD protocols
    'max_k': 50,  # Maximum K for Top-K evaluation
    'n_points': 25,  # Number of points for pruning curves
    'compute_explainers': False,  # Set to True to include SHAP, LIME, IG (expensive!)
    'local_only': False  # Set to True to save to _local_only instead of outputs/
}

# Datasets to run experiments on
DATASETS = [
    "breast_cancer", "pima", "ionosphere", "sonar",
    "heart", "wine", "glass", "vehicle", "steel",
    "iris", "digits", "spambase", "ecoli",
    "balance_scale", "banknote", "transfusion",
    # Synthetic datasets (optional)
    # "Increasing_Parity_Complexity",
    # "Hierarchical_Boolean_Rules",
    # "Progressive_Feature_Interaction",
]

if __name__ == "__main__":
    print("=" * 60)
    print("Top-K Feature Selection Experiments")
    print("=" * 60)
    print(f"Datasets: {len(DATASETS)}")
    print(f"Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    
    # Load optimized hyperparameters
    print("\nLoading optimized hyperparameters...")
    try:
        params_path = paths.get_aggregated_params_path("all_best_tm_params.json", local_only=False)
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                tm_params = json.load(f)
            print(f"✓ Loaded TM parameters from {params_path}")
            print(f"  Found parameters for {len(tm_params)} datasets")
        else:
            print(f"⚠ Warning: Parameters file not found at {params_path}")
            print("  Please run 1_optimize_hyperparameters.py first")
            print("  Using default parameters (s=3.0, T=600)")
            tm_params = {}
    except Exception as e:
        print(f"⚠ Warning: Could not load parameters: {e}")
        print("  Using default parameters (s=3.0, T=600)")
        tm_params = {}
    
    # Run experiments for each dataset
    all_results = {}
    
    for i, dataset_name in enumerate(DATASETS, 1):
        print(f"\n[{i}/{len(DATASETS)}] Processing {dataset_name}...")
        try:
            results = ktop_experiment.run_single_dataset_experiment(
                dataset_name, tm_params, CONFIG
            )
            all_results[dataset_name] = results
            
            # Save results for this dataset
            output_path = paths.get_experiment_results_path(
                dataset_name, local_only=CONFIG['local_only']
            )
            serialization.save_json(results, output_path)
            print(f"✓ Results saved to {output_path}")
            
        except Exception as e:
            print(f"✗ Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 60)
    print("Experiments Complete!")
    print("=" * 60)
    print(f"Successfully processed: {len(all_results)}/{len(DATASETS)} datasets")
    print(f"Results saved to: {paths.EXPERIMENT_RESULTS_DIR if not CONFIG['local_only'] else paths.LOCAL_RESULTS_DIR}")
    print(f"Figures saved to: {paths.FIGURES_DIR if not CONFIG['local_only'] else paths.LOCAL_FIGURES_DIR}")
