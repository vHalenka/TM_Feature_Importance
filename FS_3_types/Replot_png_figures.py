import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

# 1) Find all JSON result files
json_files = glob.glob("fs_experiment_results_*.json")
if not json_files:
    print("No 'fs_experiment_results_*.json' files found in the current directory.")
    exit()

for file_path in json_files:
    # Derive dataset name from filename
    dataset_name = os.path.basename(file_path).replace("fs_experiment_results_", "").replace(".json", "")
    with open(file_path, "r") as f:
        data = json.load(f)

    # 2) Extract experiment-level info
    exp_desc = data.get("experiment_description", {})
    # X_train_shape = [n_samples, n_feats]
    n_feats = exp_desc.get("X_train_shape", [None, None])[1]
    if n_feats is None:
        print(f"Could not find n_feats for {dataset_name}; skipping.")
        continue

    # 3) Reconstruct K-lists exactly as in the original script
    max_k = min(50, n_feats)
    K_list = np.unique(np.linspace(1, max_k, 25, dtype=int))
    num_points = min(n_feats + 1, 25)
    K_perturb_list = np.unique(
        np.concatenate(([0], np.linspace(1, n_feats, num=num_points, dtype=int)))
    )

    # 4) Extract method names (should be consistent across all keys)
    method_names = data["feature_correlation_matrix"]["method_names"]
    if "Random" not in method_names:
        print(f"'Random' not found among methods in {dataset_name}; skipping.")
        continue

    non_random_methods = [m for m in method_names if m != "Random"]

    # 5) Build a palette of 25 colors: all 20 from "tab20" + 5 more from "tab20b"
    cmap20 = plt.get_cmap("tab20").colors        # 20 colors
    cmap20b = plt.get_cmap("tab20b").colors      # another 20, but we'll only take first 5
    extra_five = list(cmap20b[:5])
    all_colors = list(cmap20) + extra_five       # total = 25 distinct RGB tuples

    # Map each “base” technique (strip off "-PosNeg") to one of those 25 colors
    base_to_color = {}
    next_color_index = 0
    for method in sorted(non_random_methods):
        base_name = method.split("-PosNeg")[0]
        if base_name not in base_to_color:
            # if we have more than 25 base names, we cycle around
            base_to_color[base_name] = all_colors[next_color_index % len(all_colors)]
            next_color_index += 1

    non_random_colors = {
        m: base_to_color[m.split("-PosNeg")[0]]
        for m in non_random_methods
    }
    random_color = "k"  # solid black for Random

    # Helper: pick line style and width (dotted if "-PosNeg", solid otherwise)
    def line_kwargs(method, on_top=False):
        is_posneg = "-PosNeg" in method
        if method == "Random":
            return {
                "color": random_color,
                "linestyle": "-",
                "linewidth": 3.0 if on_top else 2.5,
                "marker": "o",
                "zorder": 3 if on_top else 2
            }
        else:
            return {
                "color": non_random_colors[method],
                "linestyle": ":" if is_posneg else "-",
                "linewidth": 1.5,
                "marker": "o",
                "zorder": 1
            }

    # 6)  Re-plot "score_correlations"
    corr_info = data["feature_correlation_matrix"]
    corr_matrix = np.array(corr_info["matrix"])
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_matrix, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(np.arange(len(method_names)))
    ax.set_yticks(np.arange(len(method_names)))
    ax.set_xticklabels(method_names, rotation=45, ha="right")
    ax.set_yticklabels(method_names)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    for i in range(len(method_names)):
        for j in range(len(method_names)):
            val = corr_matrix[i, j]
            txt_color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=txt_color, fontsize=7)
    ax.set_title(f"Score Correlations ({dataset_name})")
    plt.tight_layout()
    plt.savefig(f"score_correlations_{dataset_name}.png")
    plt.close()

    # 7)  Re-plot "Top-K performance"
    topk_results = data["top_k_accuracies_vs_k"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for method in sorted(non_random_methods):
        accs = np.array(topk_results.get(method, []))
        if accs.size != len(K_list):
            continue
        ax.plot(K_list, accs, label=method, **line_kwargs(method))
    rand_accs = np.array(topk_results.get("Random", []))
    if rand_accs.size == len(K_list):
        ax.plot(K_list, rand_accs, label="Random", **line_kwargs("Random", on_top=True))


    fontsize = 9

    ax.set_xlabel("Number of Features (K)")
    ax.set_ylabel(f"Avg Test Accuracy % on {dataset_name}")
    ax.set_title(f"Top-K Feature Pruning Performance ({dataset_name})")
    ax.legend(ncol=3, fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(f"top_k_performance_{dataset_name}.png")
    plt.close()

    # 8)  Re-plot "AUC Top-K" bar chart
    auc_topk = data["normalized_auc_top_k"]
    values_topk = [auc_topk.get(m, 0.0) for m in method_names]
    bar_colors = [
        random_color if m == "Random" else non_random_colors[m]
        for m in method_names
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(method_names, values_topk, color=bar_colors)
    for rect, method in zip(bars, method_names):
        if method == "Random":
            rect.set_edgecolor("black")
            rect.set_linewidth(2.5)
    ax.set_xticklabels(method_names, rotation=45, ha="right")
    ax.set_ylabel("Normalized AUC")
    ax.set_title(f"AUC of Accuracy–vs–K Curves ({dataset_name})")
    plt.tight_layout()
    plt.savefig(f"auc_top_k_{dataset_name}.png")
    plt.close()

    # 9)  Re-plot "Deletion curve"
    deletion_results = data["deletion_curve_accuracies_vs_k"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for method in sorted(non_random_methods):
        accs = np.array(deletion_results.get(method, []))
        if accs.size != len(K_perturb_list):
            continue
        ax.plot(K_perturb_list, accs, label=method, **line_kwargs(method))
    rand_accs = np.array(deletion_results.get("Random", []))
    if rand_accs.size == len(K_perturb_list):
        ax.plot(K_perturb_list, rand_accs, label="Random", **line_kwargs("Random", on_top=True))

    ax.set_xlabel("Number of Most Important Features Masked (K)")
    ax.set_ylabel(f"Test Accuracy % on {dataset_name}")
    ax.set_title(f"Deletion Curve ({dataset_name})")
    ax.legend(ncol=3, fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(f"deletion_curve_{dataset_name}.png")
    plt.close()

    # 10) Re-plot "Insertion curve"
    insertion_results = data["insertion_curve_accuracies_vs_k"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for method in sorted(non_random_methods):
        accs = np.array(insertion_results.get(method, []))
        if accs.size != len(K_perturb_list):
            continue
        ax.plot(K_perturb_list, accs, label=method, **line_kwargs(method))
    rand_accs = np.array(insertion_results.get("Random", []))
    if rand_accs.size == len(K_perturb_list):
        ax.plot(K_perturb_list, rand_accs, label="Random", **line_kwargs("Random", on_top=True))

    ax.set_xlabel("Number of Most Important Features Revealed (K)")
    ax.set_ylabel(f"Test Accuracy % on {dataset_name}")
    ax.set_title(f"Insertion Curve ({dataset_name})")
    ax.legend(ncol=3, fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(f"insertion_curve_{dataset_name}.png")
    plt.close()

    # 11) Re-plot "ROAR curve"
    roar_results = data["roar_curve_accuracies_vs_k"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for method in sorted(non_random_methods):
        accs = np.array(roar_results.get(method, []))
        if accs.size != len(K_perturb_list):
            continue
        ax.plot(K_perturb_list, accs, label=method, **line_kwargs(method))
    rand_accs = np.array(roar_results.get("Random", []))
    if rand_accs.size == len(K_perturb_list):
        ax.plot(K_perturb_list, rand_accs, label="Random", **line_kwargs("Random", on_top=True))

    ax.set_xlabel("Number of Most Important Features Removed (K) before Retraining")
    ax.set_ylabel(f"Test Accuracy % on {dataset_name}")
    ax.set_title(f"ROAR Curve ({dataset_name})")
    ax.legend(ncol=3, fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(f"roar_curve_{dataset_name}.png")
    plt.close()

    # 12) Re-plot "ROAD-Mask curve"
    road_results = data["road_mask_curve_accuracies_vs_k"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for method in sorted(non_random_methods):
        accs = np.array(road_results.get(method, []))
        if accs.size != len(K_perturb_list):
            continue
        ax.plot(K_perturb_list, accs, label=method, **line_kwargs(method))
    rand_accs = np.array(road_results.get("Random", []))
    if rand_accs.size == len(K_perturb_list):
        ax.plot(K_perturb_list, rand_accs, label="Random", **line_kwargs("Random", on_top=True))

    ax.set_xlabel("Number of Most Important Features Masked (K) before Retraining")
    ax.set_ylabel(f"Test Accuracy % on {dataset_name}")
    ax.set_title(f"ROAD-Mask Curve ({dataset_name})")
    ax.legend(ncol=3, fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(f"road_mask_curve_{dataset_name}.png")
    plt.close()

    print(f"Regenerated all PNGs for dataset: {dataset_name}")

    # ————————————————————————————————————————————————————————————————
    # 13) Combined 2×2 grid of Insertion / Deletion / ROAR / ROAD
    # ————————————————————————————————————————————————————————————————
    fontsize= 14


    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.ravel()

    # A little helper to plot one protocol onto a given ax
    def plot_protocol(ax, results_dict, title):
        for method in sorted(non_random_methods):
            accs = np.array(results_dict.get(method, []))
            if accs.size != len(K_perturb_list):
                continue
            ax.plot(K_perturb_list,
                    accs,
                    label=method,
                    **line_kwargs(method))
        # Random on top
        rand_accs = np.array(results_dict.get("Random", []))
        if rand_accs.size == len(K_perturb_list):
            ax.plot(
                K_perturb_list,
                rand_accs,
                label="Random",
                **line_kwargs("Random", on_top=True)
            )
        ax.set_title(title, fontsize=fontsize+2)
        ax.grid(True)

    # Plot each of the four
    plot_protocol(axes[0], insertion_results, "Insertion Curve")
    plot_protocol(axes[1], deletion_results,  "Deletion Curve")
    plot_protocol(axes[2], roar_results,      "ROAR Curve")
    plot_protocol(axes[3], road_results,      "ROAD Curve")

    # common x/y labels
    for ax in axes:
        ax.set_xlabel("Number of features K", fontsize=fontsize)
        ax.set_ylabel("Test Accuracy %", fontsize=fontsize)

    # One shared legend below all subplots
    # Grab handles & labels from the first axes
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',
        ncol=6,
        fontsize=fontsize,
        frameon=False,
        bbox_to_anchor=(0.5, -0.17)
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.suptitle(f"Feature‐pruning curves on {dataset_name}", fontsize=fontsize+4, y=1.02)
    plt.savefig(f"combined_pruning_curves_{dataset_name}.png", bbox_inches='tight')
    plt.close()


print("Done.")
