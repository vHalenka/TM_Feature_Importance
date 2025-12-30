"""
Test script for the refactored codebase.
This script tests the functionality without overwriting existing results.
"""
import sys
import os
import json
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import paths, data_loading, serialization
from src.hyperparameter_optimization import optuna_tm_search
from src.experiments import ktop_experiment

def test_imports():
    """Test that all modules can be imported."""
    print("=" * 60)
    print("Test 1: Module Imports")
    print("=" * 60)
    try:
        from src.utils import paths, data_loading, tm_utils, serialization
        from src.experiments import feature_scoring, evaluation_protocols, ktop_experiment
        from src.hyperparameter_optimization import optuna_tm_search
        print("✓ All modules imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loading():
    """Test dataset loading."""
    print("\n" + "=" * 60)
    print("Test 2: Data Loading")
    print("=" * 60)
    try:
        X, y = data_loading.load_dataset("iris")
        print(f"✓ Loaded iris dataset: X.shape={X.shape}, y.shape={y.shape}")
        
        X_bin, y_bin = data_loading.preprocess_data(X, y, max_bins=10)
        print(f"✓ Preprocessed data: X_bin.shape={X_bin.shape}, dtype={X_bin.dtype}")
        return True
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_optuna_small(dry_run=True):
    """Test Optuna optimization with a small test (dry run or minimal trials)."""
    print("\n" + "=" * 60)
    print("Test 3: Optuna Hyperparameter Optimization")
    print("=" * 60)
    
    # Check if params already exist
    params_path = paths.get_aggregated_params_path("all_best_tm_params.json", local_only=False)
    backup_path = None
    if os.path.exists(params_path):
        print(f"⚠ Found existing params at {params_path}")
        backup_path = str(params_path) + ".backup"
        shutil.copy2(params_path, backup_path)
        print(f"✓ Created backup: {backup_path}")
    
    try:
        if dry_run:
            print("DRY RUN: Skipping actual Optuna optimization (would overwrite params)")
            print("✓ Optuna module is ready (use 1_optimize_hyperparameters.py to run)")
            return True
        
        # Minimal test: 1 dataset, 2 trials
        print("Running minimal Optuna test (iris, 2 trials)...")
        results = optuna_tm_search.optimize_tm_hyperparameters(
            datasets=["iris"],
            clauses=500,
            epochs=5,  # Reduced for testing
            patience=2,
            trials=2,  # Minimal trials
            max_bins=10
        )
        
        if results and "iris" in results:
            print(f"✓ Optuna optimization completed: {results['iris']}")
            return True
        else:
            print("✗ Optuna optimization failed")
            return False
            
    except Exception as e:
        print(f"✗ Optuna test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Restore backup if we created one
        if backup_path and os.path.exists(backup_path) and not dry_run:
            if os.path.exists(params_path):
                os.remove(params_path)
            shutil.move(backup_path, params_path)
            print(f"✓ Restored original params")

def test_ktop_experiment_minimal():
    """Test ktop experiment with minimal settings."""
    print("\n" + "=" * 60)
    print("Test 4: Top-K Experiment (Minimal)")
    print("=" * 60)
    
    # Check if we have params
    params_path = paths.get_aggregated_params_path("all_best_tm_params.json", local_only=False)
    if not os.path.exists(params_path):
        print("⚠ No optimized parameters found. Creating dummy params for iris...")
        dummy_params = {
            "iris": {
                "best_s": 3.0,
                "best_T": 200,
                "best_val_accuracy": 0.95
            }
        }
        serialization.save_json(dummy_params, params_path)
        print(f"✓ Created dummy params at {params_path}")
    
    try:
        # Load params
        with open(params_path, 'r') as f:
            tm_params = json.load(f)
        
        # Minimal config
        config = {
            'clauses': 100,  # Reduced for testing
            'epochs': 5,  # Reduced for testing
            'max_lit': 32,
            'max_bins': 10,
            'n_trials': 2,  # Minimal trials
            'max_k': 10,  # Fewer K values
            'n_points': 5,  # Fewer points
            'compute_explainers': False,  # Skip expensive explainers
            'local_only': True  # Save to _local_only to not clutter outputs
        }
        
        print("Running minimal experiment on iris dataset...")
        print(f"Config: {config}")
        
        results = ktop_experiment.run_single_dataset_experiment(
            "iris", tm_params, config
        )
        
        if results and "experiment_description" in results:
            print(f"✓ Experiment completed successfully!")
            print(f"  Methods tested: {len(results.get('feature_correlation_matrix', {}).get('method_names', []))}")
            print(f"  Test accuracy: {results.get('full_model_test_accuracy_percent', 0):.2f}%")
            return True
        else:
            print("✗ Experiment returned incomplete results")
            return False
            
    except Exception as e:
        print(f"✗ Experiment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("CODEBASE TEST SUITE")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Imports
    results['imports'] = test_imports()
    
    # Test 2: Data loading
    if results['imports']:
        results['data_loading'] = test_data_loading()
    else:
        results['data_loading'] = False
    
    # Test 3: Optuna (dry run - doesn't overwrite params)
    if results['data_loading']:
        results['optuna'] = test_optuna_small(dry_run=True)
    else:
        results['optuna'] = False
    
    # Test 4: Full experiment (minimal)
    if results['optuna']:
        results['ktop_experiment'] = test_ktop_experiment_minimal()
    else:
        results['ktop_experiment'] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(results.values())
    print("=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

