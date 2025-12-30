"""
Centralized path configuration for output directories.
All scripts should use these paths to ensure consistent organization.
"""
import os
from pathlib import Path

# Base directory (Feature Selection/)
BASE_DIR = Path(__file__).parent.parent.parent.resolve()

# Output directories
OUTPUTS_DIR = BASE_DIR / "outputs"
LOCAL_ONLY_DIR = BASE_DIR / "_local_only"

# Figure paths
FIGURES_DIR = OUTPUTS_DIR / "figures"
CORRELATIONS_DIR = FIGURES_DIR / "correlations"
TOP_K_DIR = FIGURES_DIR / "top_k"
PRUNING_CURVES_DIR = FIGURES_DIR / "pruning_curves"
HEATMAPS_DIR = FIGURES_DIR / "heatmaps"
MISC_FIGURES_DIR = FIGURES_DIR / "misc"

# Parameter paths
PARAMS_DIR = OUTPUTS_DIR / "params"
BEST_PARAMS_DIR = PARAMS_DIR / "best_params"
AGGREGATED_PARAMS_DIR = PARAMS_DIR / "aggregated"
ROAD_PARAMS_DIR = PARAMS_DIR / "road"

# Results paths
RESULTS_DIR = OUTPUTS_DIR / "results"
EXPERIMENT_RESULTS_DIR = RESULTS_DIR / "experiments"

# Examples (for GitHub)
EXAMPLES_DIR = OUTPUTS_DIR / "examples"
EXAMPLES_FIGURES_DIR = EXAMPLES_DIR / "figures"
EXAMPLES_PARAMS_DIR = EXAMPLES_DIR / "params"

# Local only (bulk artifacts, not for GitHub)
LOCAL_FIGURES_DIR = LOCAL_ONLY_DIR / "figures"
LOCAL_PARAMS_DIR = LOCAL_ONLY_DIR / "params"
LOCAL_RESULTS_DIR = LOCAL_ONLY_DIR / "results"

# Legacy results (for backward compatibility during migration)
LEGACY_ROAD_RESULTS_DIR = BASE_DIR / "ROAD_results"


def ensure_dirs():
    """Create all output directories if they don't exist."""
    dirs = [
        CORRELATIONS_DIR,
        TOP_K_DIR,
        PRUNING_CURVES_DIR,
        HEATMAPS_DIR,
        MISC_FIGURES_DIR,
        BEST_PARAMS_DIR,
        AGGREGATED_PARAMS_DIR,
        ROAD_PARAMS_DIR,
        EXPERIMENT_RESULTS_DIR,
        EXAMPLES_FIGURES_DIR,
        EXAMPLES_PARAMS_DIR,
        LOCAL_FIGURES_DIR,
        LOCAL_PARAMS_DIR,
        LOCAL_RESULTS_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


# Helper functions for common path operations
def get_correlation_plot_path(dataset_name: str, local_only: bool = False) -> Path:
    """Get path for score correlation plot."""
    base = LOCAL_FIGURES_DIR if local_only else CORRELATIONS_DIR
    return base / f"score_correlations_{dataset_name}.png"


def get_top_k_plot_path(dataset_name: str, local_only: bool = False) -> Path:
    """Get path for top-k performance plot."""
    base = LOCAL_FIGURES_DIR if local_only else TOP_K_DIR
    return base / f"top_k_performance_{dataset_name}.png"


def get_auc_top_k_plot_path(dataset_name: str, local_only: bool = False) -> Path:
    """Get path for AUC top-k plot."""
    base = LOCAL_FIGURES_DIR if local_only else TOP_K_DIR
    return base / f"auc_top_k_{dataset_name}.png"


def get_pruning_curve_path(curve_type: str, dataset_name: str, local_only: bool = False) -> Path:
    """Get path for pruning curve (deletion, insertion, roar, road_mask, combined)."""
    base = LOCAL_FIGURES_DIR if local_only else PRUNING_CURVES_DIR
    return base / f"{curve_type}_curve_{dataset_name}.png"


def get_best_params_path(dataset_name: str, local_only: bool = False) -> Path:
    """Get path for best parameters JSON file."""
    base = LOCAL_PARAMS_DIR if local_only else BEST_PARAMS_DIR
    return base / f"best_params_{dataset_name}.json"


def get_aggregated_params_path(filename: str, local_only: bool = False) -> Path:
    """Get path for aggregated parameters file (e.g., all_best_tm_params.json)."""
    base = LOCAL_PARAMS_DIR if local_only else AGGREGATED_PARAMS_DIR
    return base / filename


def get_experiment_results_path(dataset_name: str, local_only: bool = False) -> Path:
    """Get path for experiment results JSON file."""
    base = LOCAL_RESULTS_DIR if local_only else EXPERIMENT_RESULTS_DIR
    return base / f"fs_experiment_results_{dataset_name}.json"


def get_road_results_path(dataset_name: str, model_name: str, local_only: bool = False) -> Path:
    """Get path for ROAD experiment results JSON file."""
    base = LOCAL_RESULTS_DIR if local_only else ROAD_PARAMS_DIR
    return base / f"{dataset_name}_{model_name}_results.json"

