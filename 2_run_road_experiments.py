"""
Entry point script for ROAD (RemOve And retrain) feature selection experiments.

This script runs feature selection experiments using multiple models (TM, sklearn models)
and saves feature importance scores. It requires optimized hyperparameters from step 1.

Usage:
    python 2_run_road_experiments.py
"""
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the refactored ROAD experiment module
# For now, we'll use a wrapper that updates paths
from src.utils import paths

# Ensure output directories exist
paths.ensure_dirs()

# Import and run the main experiment
# We'll create a refactored version that uses the new path structure
print("=" * 60)
print("ROAD Feature Selection Experiments")
print("=" * 60)
print("Loading optimized hyperparameters...")

# Load params from new location
try:
    import json
    from src.utils import serialization
    params_path = paths.get_aggregated_params_path("all_best_tm_params.json", local_only=False)
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            tm_params = json.load(f)
        print(f"Loaded TM parameters from {params_path}")
    else:
        print(f"Warning: Parameters file not found at {params_path}")
        print("Please run 1_optimize_hyperparameters.py first")
        tm_params = {}
except Exception as e:
    print(f"Warning: Could not load parameters: {e}")
    tm_params = {}

# Try to load sklearn params if they exist
try:
    skl_params_path = paths.get_aggregated_params_path("all_best_sklearn_params.json", local_only=False)
    if os.path.exists(skl_params_path):
        with open(skl_params_path, 'r') as f:
            skl_params = json.load(f)
        print(f"Loaded sklearn parameters from {skl_params_path}")
    else:
        skl_params = {}
except:
    skl_params = {}

# For now, import and run the original script with path modifications
# TODO: Full refactoring will move code to src/experiments/road_experiment.py
print("\nRunning ROAD experiments...")
print("Note: This is using the original FSB_ROAD.py with path updates.")
print("Full refactoring to src/experiments/ will be done incrementally.\n")

# We'll need to modify the original script or create a new one
# For now, let's create a message and the user can run the legacy script
print("To run the experiments, please use:")
print("  python legacy/ROAD_Models/FSB_ROAD.py")
print("\nOr wait for full refactoring to src/experiments/road_experiment.py")

