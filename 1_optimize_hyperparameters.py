"""
Entry point script for hyperparameter optimization.

This script should be run first to find optimal hyperparameters for Tsetlin Machines
across all datasets. The results are saved and used by subsequent experiment scripts.

Usage:
    python 1_optimize_hyperparameters.py
"""
from src.hyperparameter_optimization import optuna_tm_search

# Configuration
DATASETS = [
    "breast_cancer", "pima", "ionosphere", "sonar",
    "heart", "wine", "glass", "vehicle", "steel", 
    "iris", "digits", "spambase", "ecoli",
    "balance_scale", "banknote", "transfusion"
]

# You can uncomment synthetic datasets if needed:
# SYNTHETIC_DATASETS = [
#     "Increasing_Parity_Complexity",
#     "Hierarchical_Boolean_Rules",
#     "Progressive_Feature_Interaction"
# ]

CLauses = 500
Epochs = 30
Patience = 3
Trials = 100
MaxBins = 10

if __name__ == "__main__":
    print("=" * 60)
    print("Hyperparameter Optimization for Tsetlin Machines")
    print("=" * 60)
    print(f"Datasets: {len(DATASETS)}")
    print(f"Optuna trials per dataset: {Trials}")
    print(f"TM clauses: {CLauses}, epochs: {Epochs}, patience: {Patience}")
    print("=" * 60)
    
    results = optuna_tm_search.optimize_tm_hyperparameters(
        datasets=DATASETS,
        clauses=CLauses,
        epochs=Epochs,
        patience=Patience,
        trials=Trials,
        max_bins=MaxBins
    )
    
    print("\n" + "=" * 60)
    print("Hyperparameter optimization complete!")
    print("Results saved to outputs/params/")
    print("=" * 60)

