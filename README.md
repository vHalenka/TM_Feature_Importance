# TM Feature Selection

This project compares different feature selection techniques on various datasets using Tsetlin Machines for evaluation. The main focus is on understanding how different feature selection methods perform when evaluated with Tsetlin Machine classifiers.

## Overview

The project implements and compares multiple feature selection approaches:

- **Traditional methods**: F-value, Mutual Information, Chi-squared, ReliefF, RFE
- **Tsetlin Machine-based**: Clause analysis and feature importance from TM training
- **ROAD (RemOve And retrain)**: Feature importance through iterative removal and retraining
- **VarGrad**: Gradient-based feature importance

## Datasets

Experiments are run on various UCI and sklearn datasets including:
- Iris
- Wine
- Breast Cancer
- Digits
- Heart Disease
- Pima Indians Diabetes
- And others

## Project Structure

- `feature_selection.py` - Basic feature selection comparison script
- `tm_feature_selection/` - Main Tsetlin Machine feature selection implementations
- `FS_3_types/` - Three types of feature selection analysis (Unique, Ranking, Irrelevant)
- `FSB_ROAD.py` - ROAD method implementation
- `FS_3_types_difference.py` - Comparison of different FS approaches

## Requirements

See `requirements.txt` for dependencies. Main packages:
- numpy
- pandas
- scikit-learn
- tmu (Tsetlin Machine implementation)
- matplotlib, seaborn for visualization

## Usage

Run the main feature selection experiments:

```bash
python feature_selection.py
```

Or use the Tsetlin Machine specific implementations:

```bash
cd tm_feature_selection
python feature_selection.py
```

## Results

The experiments generate various plots and metrics comparing:
- Top-k feature performance
- AUC curves for different feature subsets
- Score correlations between methods
- Pruning curves (insertion/deletion)
- ROAD mask curves

Note: Generated results (plots, JSON files) are excluded from the repository.

