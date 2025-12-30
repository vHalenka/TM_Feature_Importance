"""
Script to organize existing output files into the new directory structure.
Run this once to migrate existing outputs.
"""
import os
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).parent

# Output directories
LOCAL_FIGURES = BASE_DIR / "_local_only" / "figures"
LOCAL_PARAMS = BASE_DIR / "_local_only" / "params"
LOCAL_RESULTS = BASE_DIR / "_local_only" / "results"

# Create subdirectories
CORRELATIONS = LOCAL_FIGURES / "correlations"
TOP_K = LOCAL_FIGURES / "top_k"
PRUNING = LOCAL_FIGURES / "pruning_curves"
HEATMAPS = LOCAL_FIGURES / "heatmaps"
MISC = LOCAL_FIGURES / "misc"

BEST_PARAMS = LOCAL_PARAMS / "best_params"
AGGREGATED = LOCAL_PARAMS / "aggregated"
ROAD = LOCAL_PARAMS / "road"

EXPERIMENTS = LOCAL_RESULTS / "experiments"

for d in [CORRELATIONS, TOP_K, PRUNING, HEATMAPS, MISC, BEST_PARAMS, AGGREGATED, ROAD, EXPERIMENTS]:
    d.mkdir(parents=True, exist_ok=True)

# Patterns for categorization
patterns = {
    'correlations': ['score_correlations_'],
    'top_k': ['top_k_performance_', 'auc_top_k_'],
    'pruning': ['deletion_curve_', 'insertion_curve_', 'roar_curve_', 
                'road_mask_curve_', 'combined_pruning_curves_'],
    'heatmaps': ['_fs_heatmaps', '_fs_grid', '_cwsum_heatmap', '_pixel_heatmap',
                 'per_class', 'methods_comparison', 'pixel_importance'],
    'best_params': ['best_params_'],
    'aggregated': ['all_best_', 'all_feature_selection'],
    'road': ['_BernoulliNB_results', '_DecisionTree_results', '_KNN_results',
             '_LinearSVM_results', '_LogisticRegression_results', '_TsetlinMachine_results'],
    'experiments': ['fs_experiment_results_'],
}

def categorize_file(filename):
    """Determine which category a file belongs to."""
    for category, pats in patterns.items():
        if any(pat in filename for pat in pats):
            return category
    return 'misc'

# Move PNG files
print("Moving PNG files...")
png_count = 0
for png_file in BASE_DIR.glob("*.png"):
    if png_file.name.startswith('.'):
        continue
    category = categorize_file(png_file.name)
    if category == 'correlations':
        dest = CORRELATIONS / png_file.name
    elif category == 'top_k':
        dest = TOP_K / png_file.name
    elif category == 'pruning':
        dest = PRUNING / png_file.name
    elif category == 'heatmaps':
        dest = HEATMAPS / png_file.name
    else:
        dest = MISC / png_file.name
    
    shutil.move(str(png_file), str(dest))
    png_count += 1
    if png_count % 20 == 0:
        print(f"  Moved {png_count} PNG files...")

print(f"Moved {png_count} PNG files")

# Move JSON files
print("\nMoving JSON files...")
json_count = 0
for json_file in BASE_DIR.glob("*.json"):
    if json_file.name.startswith('.'):
        continue
    category = categorize_file(json_file.name)
    if category == 'best_params':
        dest = BEST_PARAMS / json_file.name
    elif category == 'aggregated':
        dest = AGGREGATED / json_file.name
    elif category == 'road':
        dest = ROAD / json_file.name
    elif category == 'experiments':
        dest = EXPERIMENTS / json_file.name
    else:
        dest = AGGREGATED / json_file.name  # Default
    
    shutil.move(str(json_file), str(dest))
    json_count += 1

print(f"Moved {json_count} JSON files")

# Move CSV files
print("\nMoving CSV files...")
csv_count = 0
for csv_file in BASE_DIR.glob("*.csv"):
    if csv_file.name.startswith('.'):
        continue
    dest = LOCAL_RESULTS / csv_file.name
    shutil.move(str(csv_file), str(dest))
    csv_count += 1

print(f"Moved {csv_count} CSV files")

# Move ROAD_results directory if it exists
if (BASE_DIR / "ROAD_results").exists():
    print("\nMoving ROAD_results directory...")
    for json_file in (BASE_DIR / "ROAD_results").glob("*.json"):
        category = categorize_file(json_file.name)
        if category == 'road':
            dest = ROAD / json_file.name
        else:
            dest = EXPERIMENTS / json_file.name
        shutil.move(str(json_file), str(dest))
    try:
        (BASE_DIR / "ROAD_results").rmdir()
        print("Removed empty ROAD_results directory")
    except:
        print("Note: ROAD_results directory not empty, leaving it")

print("\n" + "=" * 60)
print("Output organization complete!")
print("=" * 60)

