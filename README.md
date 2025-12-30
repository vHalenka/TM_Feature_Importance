# TM Feature Importance

Benchmark study comparing feature importance ranking methods through feature selection evaluation. Implements and evaluates 30+ methods (TM-based, filter, wrapper, explainability) across multiple datasets using 5 evaluation protocols (Top-K, Deletion, Insertion, ROAR, ROAD-Mask).

## Overview

This project implements and evaluates multiple feature selection approaches:
- **Traditional methods**: Mutual Information, Chi-squared, Variance
- **Tsetlin Machine-based**: Clause analysis and feature importance from TM training
- **ROAD (RemOve And retrain)**: Feature importance through iterative removal and retraining
- **Wrapper methods**: Dropout, Permutation Importance
- **Sklearn models**: BernoulliNB, DecisionTree, LogisticRegression, SVM, KNN

## Study Workflow

The study follows a structured workflow:

1. **Hyperparameter Optimization** → Find best parameters for each dataset
2. **Feature Selection Experiments** → Run comprehensive feature selection analysis
3. **Visualization & Analysis** → Generate plots and comparisons

## Project Structure

```
Feature Selection/
├── src/                          # Core source code
│   ├── utils/                   # Shared utilities
│   │   ├── data_loading.py     # Dataset loading and preprocessing
│   │   ├── synthetic_datasets.py # Synthetic dataset generators
│   │   ├── tm_utils.py         # Tsetlin Machine utilities
│   │   ├── sklearn_utils.py    # Sklearn model utilities
│   │   ├── serialization.py    # JSON serialization helpers
│   │   └── paths.py            # Centralized output path configuration
│   ├── hyperparameter_optimization/  # Step 1: HPO
│   │   └── optuna_tm_search.py # TM hyperparameter optimization
│   ├── experiments/             # Step 2: Main experiments
│   │   └── (to be refactored)
│   └── visualization/           # Step 3: Plotting & analysis
│       └── (to be refactored)
│
├── outputs/                      # All generated outputs
│   ├── figures/                 # Visualization outputs
│   │   ├── correlations/       # Score correlation heatmaps
│   │   ├── top_k/              # Top-K performance plots
│   │   ├── pruning_curves/     # Deletion/insertion/ROAR/ROAD curves
│   │   ├── heatmaps/           # Feature importance heatmaps
│   │   └── misc/               # Other plots
│   ├── params/                  # Hyperparameters
│   │   ├── best_params/        # Per-dataset optimized parameters
│   │   ├── aggregated/         # Aggregated parameter files
│   │   └── road/               # ROAD-specific results
│   ├── results/                 # Experiment results
│   │   └── experiments/        # Feature selection experiment JSONs
│   └── examples/                # Curated subset for GitHub
│       ├── figures/            # Example visualizations
│       └── params/             # Example parameter files
│
├── _local_only/                  # Bulk artifacts (not tracked by git)
│
├── legacy/                       # Legacy scripts (preserved for reference)
│   ├── FS_3_types/             # Original FS_3_types scripts
│   ├── ROAD_Models/            # Original ROAD experiment scripts
│   └── tm_feature_selection/   # Original TM feature selection
│
├── 1_optimize_hyperparameters.py  # Step 1: Run hyperparameter optimization
├── 2_run_road_experiments.py      # Step 2: Run ROAD experiments
├── 3_run_ktop_experiments.py      # Step 3: Run Top-K experiments (planned)
├── 4_generate_plots.py            # Step 4: Generate visualizations (planned)
│
└── requirements.txt
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- numpy
- scikit-learn
- optuna
- tmu (Tsetlin Machine implementation)
- matplotlib, seaborn
- shap, lime (optional, for explainability methods)

## Usage

### Step 1: Optimize Hyperparameters

First, find optimal hyperparameters for Tsetlin Machines on each dataset:

```bash
python 1_optimize_hyperparameters.py
```

This will:
- Run Optuna optimization for each dataset
- Save best parameters to `outputs/params/best_params/`
- Aggregate results to `outputs/params/aggregated/all_best_tm_params.json`

**Configuration**: Edit the script to modify:
- Datasets to optimize
- Number of Optuna trials
- TM hyperparameters (clauses, epochs, patience)

### Step 2: Run Feature Selection Experiments

Run the main feature selection experiments using optimized parameters:

```bash
python 2_run_road_experiments.py
```

**Note**: Currently uses legacy scripts. Full refactoring to `src/experiments/` is in progress.

For now, you can run the legacy scripts directly:
```bash
python legacy/ROAD_Models/FSB_ROAD.py
python legacy/FS_3_types/FS_KTop.py
```

### Step 3: Generate Visualizations

Generate plots from experiment results:

```bash
python legacy/FS_3_types/Replot_png_figures.py
```

## Datasets

The study uses multiple datasets:

**UCI/OpenML datasets:**
- Iris, Wine, Breast Cancer, Digits
- Heart Disease, Pima Indians Diabetes
- Ionosphere, Sonar, Glass, Vehicle
- Steel Plates Fault, Spambase, Ecoli
- Balance Scale, Banknote Authentication
- Blood Transfusion Service Center

**Synthetic datasets:**
- Increasing Parity Complexity
- Hierarchical Boolean Rules
- Progressive Feature Interaction

## Output Locations

All outputs are organized in the `outputs/` directory:

- **Figures**: `outputs/figures/{category}/`
- **Parameters**: `outputs/params/{type}/`
- **Results**: `outputs/results/experiments/`

## Feature Selection Methods Evaluated

**Filter Methods:**
- Mutual Information
- Chi-squared
- Variance

**Embedded Methods (TM-specific):**
- TM-Weight, TM-Weight-PosNeg
- CW-Sum, CW-Sum-PosNeg (Class-weighted)
- CW-Feat, CW-Feat-PosNeg
- Support-CW-Sum, Support-CW-Sum-PosNeg
- Margin, Margin-PosNeg
- Gini, Gini-PosNeg

**Wrapper Methods:**
- Dropout
- Permutation Importance
- SHAP (optional)
- LIME (optional)

**Sklearn Embedded:**
- Feature importances from DecisionTree, RandomForest
- Coefficients from LogisticRegression, LinearSVM

## Evaluation Protocols

1. **Top-K Performance**: Accuracy vs. number of top features selected
2. **Deletion Curve**: Performance when removing top features
3. **Insertion Curve**: Performance when adding features incrementally
4. **ROAR Curve**: Performance after removing and retraining
5. **ROAD-Mask Curve**: Performance after masking and retraining

## Results Summary

Results are saved as JSON files containing:
- Feature importance scores per method
- Evaluation metrics (accuracy, F1, balanced accuracy, MCC)
- Timing information
- Correlation matrices between methods
- Top-K performance curves

Visualizations include:
- Score correlation heatmaps
- Top-K performance comparisons
- Pruning curves (deletion/insertion/ROAR/ROAD)
- Feature importance visualizations

## Reproducibility

All experiments use fixed random seeds for reproducibility:
- Dataset splits: `random_state=42`
- Model training: `random_state=42` (where applicable)
- Optuna studies: Use default random state

Hyperparameters are saved and can be reloaded to reproduce exact model configurations.

## Contributing

When adding new experiments:
1. Use utilities from `src/utils/` for data loading, preprocessing
2. Use `src/utils/paths.py` for output paths
3. Follow the workflow: HPO → Experiments → Visualization
