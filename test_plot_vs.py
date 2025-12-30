import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

# 1. Find all JSON files
json_files = glob.glob("fs_experiment_results_*.json")

if not json_files:
    print("No 'fs_experiment_results_*.json' files found in the current directory.")
    exit()

METHOD_GROUPS = {
    'Filter': ['MutualInfo', 'Chi2', 'Variance'],
    'Embedded': [
        'Relevance', 'Relevance-PosNeg', 'TM-Weight', 'TM-Weight-PosNeg',
        'CW-Sum', 'CW-Sum-PosNeg', 'CW-Feat', 'CW-Feat-PosNeg',
        'Support-CW-Sum', 'Support-CW-Sum-PosNeg', 'L1-Reg', 'L1-Reg-PosNeg',
        'GroupLasso', 'GroupLasso-PosNeg', 'TaylorCrit', 'VarDropout',
        'AblationImpact', 'SmoothStabil', 'Margin', 'Margin-PosNeg',
        'Entropy', 'Entropy-PosNeg', 'Gini', 'Gini-PosNeg',
        'Stability', 'Stability-PosNeg'
    ],
    'Wrapper': [
    'Dropout', 'PermImportance', 'SHAP', 'LIME',
    'IG',
    'SmoothGradSq', 'VarGrad'
    ]
}
METHOD_TO_GROUP = {method: group for group, methods_in_group in METHOD_GROUPS.items() for method in methods_in_group}


all_datasets_data = {} # To store points per dataset for normalization
dataset_eval_metrics = {} # To store various evaluation metrics per dataset

# --- Helper for normalization ---
def min_max_scale_log(values, min_out=0.01, max_out=1.0):
    min_val, max_val = np.min(values), np.max(values)
    if max_val == min_val: # Avoid division by zero if all values are the same
        return np.full_like(values, (min_out + max_out) / 2, dtype=float)
    scaled = (values - min_val) / (max_val - min_val) # 0 to 1
    return min_out + scaled * (max_out - min_out) # min_out to max_out

# 2. Extract data from each JSON file
# --- First pass: Identify JSON files meeting the accuracy criterion ---
files_to_plot = []
print("\n--- Checking Full Model Test Accuracies for Filtering ---")
for file_path in json_files:
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        dataset_name_short = os.path.basename(file_path).replace("fs_experiment_results_", "").replace(".json", "")
        full_accuracy = data.get("full_model_test_accuracy_percent")
        f1_score = data.get("full_model_test_f1_score")
        balanced_accuracy = data.get("full_model_test_balanced_accuracy")*100
        matthews_corr = data.get("full_model_test_matthews_corrcoef")
        
        s_used = None
        T_used = None
        if "experiment_description" in data and "tm_parameters" in data["experiment_description"]:
            tm_params = data["experiment_description"]["tm_parameters"]
            s_used = tm_params.get("s_used")
            T_used = tm_params.get("T_used")

        current_metrics = {
            "accuracy": full_accuracy,
            "f1_score": f1_score,
            "balanced_accuracy": balanced_accuracy,
            "matthews_corrcoef": matthews_corr,
            "s_used": s_used,
            "T_used": T_used
        }
        dataset_eval_metrics[dataset_name_short] = current_metrics

        accuracy_threshold = 0.0
        if full_accuracy is not None: # Filter based on accuracy
            if full_accuracy > accuracy_threshold:
                files_to_plot.append(file_path)
                print(f"Including: {dataset_name_short} (Accuracy: {full_accuracy:.2f}%)")
            else:
                print(f"Excluding: {dataset_name_short} (Accuracy: {full_accuracy:.2f}%) - Below {accuracy_threshold}%")
        else:
            print(f"Excluding: {dataset_name_short} - No 'full_model_test_accuracy_percent' found.")
            # dataset_eval_metrics will store None for accuracy if not found

    except Exception as e:
        print(f"Error during initial check of {file_path}: {e}")
print("---------------------------------------------------------\n")

# --- Second pass: Process only the filtered files for plotting ---
for file_path in files_to_plot: # Iterate over the filtered list
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        dataset_name_short = os.path.basename(file_path).replace("fs_experiment_results_", "").replace(".json", "")
        timings = data.get("timings_seconds", {})
        
        # Extract all relevant AUC dictionaries
        auc_top_k = data.get("normalized_auc_top_k", {})
        auc_deletion = data.get("normalized_auc_deletion", {})
        auc_insertion = data.get("normalized_auc_insertion", {})
        auc_roar = data.get("normalized_auc_roar", {})
        auc_road_mask = data.get("normalized_auc_road_mask", {})

        if dataset_name_short not in all_datasets_data:
            all_datasets_data[dataset_name_short] = []

        for method_name, time_taken in timings.items():
            # Ensure the method has at least one AUC score to be considered
            if method_name in auc_top_k or method_name in auc_deletion or \
               method_name in auc_insertion or method_name in auc_roar or \
               method_name in auc_road_mask:
                group = METHOD_TO_GROUP.get(method_name, "Other")

                all_datasets_data[dataset_name_short].append({
                    "original_time": float(time_taken) if time_taken > 0 else 1e-6,
                    "auc_top_k": auc_top_k.get(method_name), # Store specific AUCs
                    "auc_deletion": auc_deletion.get(method_name),
                    "auc_insertion": auc_insertion.get(method_name),
                    "auc_roar": auc_roar.get(method_name),
                    "auc_road_mask": auc_road_mask.get(method_name),
                    "method": method_name,
                    "dataset": dataset_name_short,
                    "group": group
                })

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from file: {file_path} - {e}")
    except KeyError as e:
        print(f"Missing key {e} in file: {file_path}")
    except Exception as e:
        print(f"An unexpected error occurred while processing {file_path}: {e}")

print("\n--- Full Model Evaluation Metrics (All Checked Files) ---")
for dataset_name, metrics in sorted(dataset_eval_metrics.items()):
    print(f"Dataset: {dataset_name:<30}")
    acc_str = f"{metrics['accuracy']:.2f}%" if isinstance(metrics['accuracy'], float) else str(metrics['accuracy'])
    f1_str = f"{metrics['f1_score']:.2f}%" if isinstance(metrics['f1_score'], float) else str(metrics['f1_score'])
    bal_acc_str = f"{metrics['balanced_accuracy']:.2f}%" if isinstance(metrics['balanced_accuracy'], float) else str(metrics['balanced_accuracy']) # Assuming it's a percentage now
    mcc_str = f"{metrics['matthews_corrcoef']:.4f}" if isinstance(metrics['matthews_corrcoef'], float) else str(metrics['matthews_corrcoef']) # MCC is typically -1 to 1
    s_str = f"{metrics['s_used']:.2f}" if isinstance(metrics['s_used'], float) else str(metrics['s_used'])
    T_str = str(metrics['T_used']) if metrics['T_used'] is not None else "N/A"
    
    print(f"  Metrics: Acc: {acc_str:<8} | F1: {f1_str:<8} | Bal.Acc: {bal_acc_str:<10} | MCC: {mcc_str:<7}")
    print(f"  TM Params: s: {s_str:<6} | T: {T_str:<5}")
print("----------------------------------\n")

# --- Normalize time per dataset and create final plot_points list ---
plot_points = []
for dataset_name, points_in_dataset in all_datasets_data.items():
    if not points_in_dataset:
        continue
    
    original_times = np.array([p['original_time'] for p in points_in_dataset])
    if len(original_times) > 0: # Ensure there are times to normalize
        normalized_times_for_dataset = min_max_scale_log(original_times)
        for i, point in enumerate(points_in_dataset):
            point['normalized_time'] = normalized_times_for_dataset[i]
            plot_points.append(point) # Add to the main list for filtering

if not plot_points:
    print("No data points to plot after initial processing and normalization.")
    exit()


def generate_auc_plot(all_plot_points, auc_key, plot_title_suffix, top_n_filter=5):
    """
    Generates a plot of Normalized Time vs. a specific AUC type.
    Filters for top N methods based on the provided auc_key.
    Black-and-white friendly:
      - method is encoded by marker SHAPE (no color),
      - dataset legend removed,
      - group ellipses use distinct hatch patterns.
    """
    # Filter points that have a valid AUC value for the current auc_key
    valid_auc_points = [p for p in all_plot_points if p.get(auc_key) is not None]
    if not valid_auc_points:
        print(f"No data points with valid AUCs for '{auc_key}' to plot.")
        return

    # --- Filter for top N performing methods per dataset for the current AUC type ---
    filtered_plot_points = []
    unique_datasets_in_data = sorted(list(set(p['dataset'] for p in valid_auc_points)))

    # User request: remove 'PermImportance' on 'Digits' and 'Steel' as it's an outlier
    valid_auc_points = [p for p in valid_auc_points if not (p['method'] == 'PermImportance' and ( p['dataset'] == 'digits' or p['dataset'] == 'steel'))]

    for dataset_name in unique_datasets_in_data:
        points_for_this_dataset = [p for p in valid_auc_points if p['dataset'] == dataset_name]
        # Sort by the current auc_key
        points_for_this_dataset.sort(key=lambda x: x[auc_key], reverse=True)
        top_n_points = points_for_this_dataset[:top_n_filter]
        filtered_plot_points.extend(top_n_points)

    if not filtered_plot_points:
        print(f"No data points left after filtering for top {top_n_filter} methods for AUC type '{auc_key}'.")
        return

    # 3. Prepare for plotting
    all_methods = sorted(list(set(p['method'] for p in filtered_plot_points)))
    all_groups = sorted(list(set(p['group'] for p in filtered_plot_points)))

    # Black-and-white: encode method by marker shape (no color)
    method_marker_cycle = ['o', 's', '^', 'D', 'v', '<', '>', 'P', 'X', '*', 'h', 'H', '8', 'd', 'p']
    if len(all_methods) > len(method_marker_cycle):
        print(f"Warning: More methods ({len(all_methods)}) than unique markers ({len(method_marker_cycle)}). Markers will repeat.")
    method_markers = {method: method_marker_cycle[i % len(method_marker_cycle)] for i, method in enumerate(all_methods)}

    # 4. Create the plot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Distinct hatch patterns for group ellipses (Filter/Embedded/Wrapper)
    group_hatches = {
        'Filter':   '///',
        'Embedded': '...',
        'Wrapper':  '\\\\',
        'Other':    None
    }

    # Scatter points: black edges, no fill; shape = method
    for point in filtered_plot_points:
        ax.scatter(
            point['normalized_time'], point[auc_key],
            marker=method_markers[point['method']],
            facecolors='none', edgecolors='black',
            s=90, linewidth=1.0
        )

    ax.set_xscale('log')
    ax.set_xlabel('Normalized Time per Dataset (log scale, 0.01 to 1.0)')
    ax.set_ylabel(f'Normalized AUC ({plot_title_suffix})')
    ax.set_title(f'Top {top_n_filter} FS Methods: Norm. Time vs. AUC ({plot_title_suffix})')
    ax.grid(True, which="both", ls="--", alpha=0.7)

    # Group ellipses with hatch fills
    N_STD_ELLIPSE = 2
    for group_name in all_groups:
        if group_name == "Other":
            continue
        group_points_x = [p['normalized_time'] for p in filtered_plot_points if p['group'] == group_name and p.get(auc_key) is not None]
        group_points_y = [p[auc_key] for p in filtered_plot_points if p['group'] == group_name and p.get(auc_key) is not None]

        if len(group_points_x) >= 2:
            mean_x, mean_y = np.mean(group_points_x), np.mean(group_points_y)
            std_x, std_y = np.std(group_points_x), np.std(group_points_y)
            ellipse_width  = N_STD_ELLIPSE * std_x if std_x > 1e-6 else 0.05
            ellipse_height = N_STD_ELLIPSE * std_y if std_y > 1e-6 else 0.05
            hatch = group_hatches.get(group_name, None)
            ellipse = Ellipse(
                xy=(mean_x, mean_y), width=ellipse_width, height=ellipse_height, angle=0,
                edgecolor='black', facecolor='white', hatch=hatch, lw=1.2,
                label=f'{group_name} group', zorder=0
            )
            ax.add_patch(ellipse)

    # Legend: methods only (shapes), no dataset legend
    legend_elements_methods = [
        Line2D([0], [0], marker=method_markers[method], linestyle='None',
               color='black', markerfacecolor='none', markeredgecolor='black',
               label=method, markersize=8)
        for method in all_methods
    ]
    legend1 = ax.legend(handles=legend_elements_methods, title="Methods",
                        loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize='small')
    ax.add_artist(legend1)

    # No datasets legend (removed)
    fig.subplots_adjust(right=0.78)
    plt.show()


# --- Generate plots for each AUC type ---
top_n_methods_to_show = 3 # Consistent filter for all plots

auc_types_to_plot = [
    #("auc_top_k", "Top-K Pruning"),
    #("auc_deletion", "Deletion Curve"),
    #("auc_insertion", "Insertion Curve"),
    #("auc_roar", "ROAR Curve"),
    ("auc_road_mask", "ROAD Curve")
]

for auc_key, title_suffix in auc_types_to_plot:
    print(f"\nGenerating plot for: {title_suffix}")
    generate_auc_plot(plot_points, auc_key, title_suffix, top_n_filter=top_n_methods_to_show)



# --- Generate Aggregate Ranking Summary + Heatmap ---
#
# For each dataset and each evaluation key, find the top‐3 methods by normalized AUC,
# then count how many times each method appears in those top‐3 slots across all datasets.

# 1) Define which AUC fields to use (keys in our `plot_points` entries)
eval_keys = [ "auc_top_k", "auc_deletion", "auc_insertion", "auc_roar", "auc_road_mask"]
# A shorter label for each evaluation, in the same order
eval_labels = ["Top-K", "Delet", "Insert", "ROAR", "ROAD"]

# 2) Initialize a nested count dictionary: aggregate_counts[method][eval_key] = how often method was top-3
all_methods = sorted({p["method"] for p in plot_points})
aggregate_counts = {m: {ek: 0 for ek in eval_keys} for m in all_methods}

# 3) For each dataset, collect per‐method AUC values and pick top-3 in each eval
for dataset_name, points in all_datasets_data.items():
    # Build a mapping method → its AUC for each eval_key
    # (some methods may be missing a particular AUC; skip those)
    per_eval_mapping = {ek: {} for ek in eval_keys}
    for entry in points:
        method = entry["method"]
        for ek in eval_keys:
            auc_val = entry.get(ek)
            if auc_val is not None:
                per_eval_mapping[ek][method] = auc_val

    # For each evaluation key, sort methods by their AUC descending and take top-3
    for ek in eval_keys:
        scores = per_eval_mapping[ek]
        if not scores:
            continue
        # Sort by auc descending
        sorted_methods = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top3 = [m for m, _ in sorted_methods[:4]]
        for m in top3:
            aggregate_counts[m][ek] += 1
        print(f"Dataset: {dataset_name}, Top-3 for {ek}: {top3}")

# 4) Prepare data matrix for heatmap: rows = methods, columns = eval_keys
method_list = all_methods
heatmap_data = np.array([[aggregate_counts[m][ek] for m in method_list] for ek in eval_keys]) # Transposed

# 5) Plot the heatmap
fig, ax = plt.subplots(figsize=(max(8, len(method_list) * 0.3), len(eval_labels) * 0.7 )) # Adjusted figsize
im = ax.imshow(heatmap_data, cmap="viridis", aspect="auto")

# Add colorbar
#cbar = ax.figure.colorbar(im, ax=ax, shrink=0.7)
#cbar.ax.set_ylabel("Count of Top-5 Appearances", rotation=-90, va="bottom")

# Configure ticks and labels
ax.set_xticks(np.arange(len(method_list)))
ax.set_yticks(np.arange(len(eval_labels)))
ax.set_xticklabels(method_list, fontsize=8)
ax.set_yticklabels(eval_labels, fontsize=10)

# Rotate x-labels for readability and prevent overlap
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Annotate each cell with the count
max_count = heatmap_data.max()
for i in range(len(eval_labels)):  # Iterate rows (eval_labels)
    for j in range(len(method_list)): # Iterate columns (method_list)
        count = heatmap_data[i, j] # Access transposed data
        color = "white" if count > (max_count / 2) else "black"
        ax.text(j, i, str(count), ha="center", va="center", color=color, fontsize=8)

ax.set_title("Number of Top-5 Appearances\nper Method and Evaluation in 12 datasets", pad=20)
fig.tight_layout()
plt.show()


# --- Generate Aggregate Ranking Summary + Heatmap for TOY DATASETS ---

# 1) Define which AUC fields to use (keys in our `plot_points` entries)
eval_keys_toy   = ["auc_top_k", "auc_deletion", "auc_insertion", "auc_roar", "auc_road_mask"]
eval_labels_toy = ["Top-K", "Delet",   "Insert",        "ROAR",     "ROAD"]

# Define toy datasets
toy_datasets = [
    "Increasing_Parity_Complexity",
    "Hierarchical_Boolean_Rules",
    "Progressive_Feature_Interaction",
]

# 2) Initialize counts
toy_plot_points    = [p for p in plot_points if p["dataset"] in toy_datasets]
all_methods_toy    = sorted({p["method"] for p in toy_plot_points})
aggregate_counts_toy = {m: {ek: 0 for ek in eval_keys_toy} for m in all_methods_toy}

# 3) For each toy dataset, take top-3 methods by AUC
for dataset_name in toy_datasets:
    pts = [p for p in plot_points if p["dataset"] == dataset_name]
    # build mapping: eval_key → { method: auc_value }
    per_eval = {ek: {} for ek in eval_keys_toy}
    for p in pts:
        for ek in eval_keys_toy:
            if p.get(ek) is not None:
                per_eval[ek][p["method"]] = p[ek]

    for ek in eval_keys_toy:
        scores = per_eval[ek]
        if not scores:
            continue
        # sort descending, take top-3
        sorted_methods = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top3 = [m for m, _ in sorted_methods[:10]]
        for m in top3:
            aggregate_counts_toy[m][ek] += 1
        print(f"Dataset: {dataset_name}, Top-3 for {ek}: {top3}")

# 4) Build data matrix (methods × evals)
method_list_toy  = all_methods_toy
heatmap_data_toy = np.array([
    [aggregate_counts_toy[m][ek] for ek in eval_keys_toy]
    for m in method_list_toy
])

# 5) Plot heatmap (evals on y, methods on x)
fig, ax = plt.subplots(
    figsize=(max(8, len(method_list_toy)*0.3), len(eval_labels_toy)*0.7)
)
im = ax.imshow(heatmap_data_toy.T, cmap="viridis", aspect="auto")

cbar = fig.colorbar(im, ax=ax, shrink=0.7)
cbar.ax.set_ylabel("Count of Top-3 Appearances", rotation=-90, va="bottom")

ax.set_xticks(np.arange(len(method_list_toy)))
ax.set_yticks(np.arange(len(eval_labels_toy)))
ax.set_xticklabels(method_list_toy, fontsize=8)
ax.set_yticklabels(eval_labels_toy, fontsize=10)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

max_ct = heatmap_data_toy.max()
for i in range(len(method_list_toy)):
    for j in range(len(eval_labels_toy)):
        ct = heatmap_data_toy[i, j]
        color = "white" if ct > (max_ct/2) else "black"
        ax.text(i, j, str(ct), ha="center", va="center", color=color, fontsize=8)

ax.set_title(
    "Number of Top-3 Appearances\nper Method and Evaluation in Toy Datasets",
    pad=20
)
fig.tight_layout()
plt.show()
