import glob, json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram

# 1) Collect per-method ROAD-mask AUCs across all JSON results
json_files = glob.glob("fs_experiment_results_*.json")
road_key   = "normalized_auc_road_mask"

auc_vals = {}
for fp in json_files:
    with open(fp) as f:
        data = json.load(f)
    for method, val in data.get(road_key, {}).items():
        # Skip any "-PosNeg" variants
        if "-PosNeg" in method:
            continue
        if not np.isfinite(val):
            continue
        auc_vals.setdefault(method, []).append(val)

# 2) Build a matrix: rows=methods, cols=1 (the mean ROAD-mask AUC)
method_names = sorted(auc_vals.keys())
profiles = np.array([np.mean(auc_vals[m]) for m in method_names]).reshape(-1, 1)

# 3) Compute pairwise Euclidean distances and linkage
dist_vec = pdist(profiles, metric="euclidean")
Z        = linkage(dist_vec, method="average")

# 4) Bump zero distances by a tiny epsilon so log-scale works
Z[:, 2] = Z[:, 2] + 1e-5

# 5) Plot dendrogram with log y-axis
fig, ax = plt.subplots(figsize=(10, 6))
dendrogram(
    Z,
    labels=method_names,
    leaf_rotation=45,
    leaf_font_size=10,
    link_color_func=lambda k: "black",  # force every branch to be black
    ax=ax
)
ax.set_ylabel("Euclidean Distance")
ax.set_title("Hierarchical Clustering of FS Methods by Avg ROAD-mask AUC")
plt.tight_layout()
plt.show()
