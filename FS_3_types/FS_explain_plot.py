import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2, VarianceThreshold
from sklearn.metrics import mutual_info_score
from lime.lime_tabular import LimeTabularExplainer
import shap
from tmu.models.classification.vanilla_classifier import TMClassifier
import time
from tqdm import tqdm

# ─── Helpers ────────────────────────────────────────────────────────────────
def load_digits_binary(threshold=8):
    ds    = load_digits()
    X_raw = ds.data        # (n_samples,64)
    y     = ds.target.astype(np.uint32)
    return (X_raw > threshold).astype(np.uint32), y

def count_ta_states(tm):
    C    = tm.number_of_classes
    half = tm.number_of_clauses // 2
    L   = tm.clause_banks[0].number_of_features
    S    = np.zeros((C,2,half,L), int)
    for c in range(C):
        for pol in (0,1):
            for cl in range(half):
                for f in range(L):
                    S[c,pol,cl,f] = tm.get_ta_action(
                        clause=cl, ta=f, the_class=c, polarity=pol
                    )
    return S

def count_clause_weights(tm, S):
    C,_,half,L = S.shape
    pos_w = np.zeros((C,L))
    neg_w = np.zeros((C,L)) # For negative polarity
    for c in range(C):
        for cl in range(half):
            # Positive polarity clauses
            w_pos = tm.get_weight(the_class=c, polarity=0, clause=cl)
            active_pos = S[c,0,cl] != 0
            pos_w[c] += w_pos * active_pos

            # Negative polarity clauses
            w_neg = tm.get_weight(the_class=c, polarity=1, clause=cl)
            active_neg = S[c,1,cl] != 0
            neg_w[c] += w_neg * active_neg # Summing negative weights directly, will be subtracted later for net_w
    return pos_w, neg_w

def normalize_per_class(mat):
    mn = mat.min(axis=1, keepdims=True)
    mx = mat.max(axis=1, keepdims=True)
    # Avoid division by zero for constant features by adding a small epsilon
    denom = (mx - mn + 1e-12)
    # If a class's row is all zeros (or constant), set denominator to 1 to avoid NaN
    denom[denom == 1e-12] = 1.0 # If min == max, then denom is 1e-12. Make it 1.0 to get 0/1.0 = 0.
    normalized_mat = (mat - mn) / denom
    return normalized_mat

def minmax_norm_global(arr):
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        return (arr - mn) / (mx - mn)
    else:
        return np.full_like(arr, 0.5)

def predict_proba_tm(X):
    Xb = (X > 0.5).astype(np.uint32)
    sums = tm.predict(Xb, return_class_sums=True)[1]
    shifted = sums - sums.min(axis=1, keepdims=True)
    # Avoid division by zero for all-zero sums
    total_sum = shifted.sum(axis=1, keepdims=True)
    # If total_sum is zero for an instance, make probabilities uniform (1/C) or zero
    probas = np.zeros_like(shifted, dtype=float)
    valid_sums_mask = (total_sum > 1e-9).flatten()

    if np.any(valid_sums_mask):
        probas[valid_sums_mask] = shifted[valid_sums_mask] / total_sum[valid_sums_mask]

    # Fallback for instances where all class sums are zero (or equal after shifting)
    invalid_sums_mask = ~valid_sums_mask
    if np.any(invalid_sums_mask):
        n_classes_model = tm.number_of_classes if hasattr(tm, 'number_of_classes') else sums.shape[1]
        for i in np.where(invalid_sums_mask)[0]:
            # Get the predicted class if sums were all zero/equal
            # In this case, tm.predict(Xb) would return a single class prediction
            # We can set that class's probability to 1.0, others to 0.
            # This is a bit of a heuristic for LIME/SHAP to work.
            single_pred = tm.predict(Xb[i:i+1])[0]
            if 0 <= single_pred < n_classes_model:
                probas[i, single_pred] = 1.0
            else:
                probas[i, :] = 1.0 / n_classes_model # Fallback to uniform if prediction is out of bounds
    return probas


# ─── Load & train ───────────────────────────────────────────────────────────
rng = np.random.default_rng(0)
X_bin, y = load_digits_binary(threshold=8)
X_tr, X_te, y_tr, y_te = train_test_split(
    X_bin, y, stratify=y, test_size=0.2, random_state=42
)

tm = TMClassifier(
    number_of_clauses     = 500,
    T                     = 50,
    s                     = 4.7,
    max_included_literals = 16,
    platform              = "CPU"
)

# Store history for Stability calculation
pos_w_history = []
neg_w_history = []

epochs = 50 # Defined epochs to match the original structure
for ep in range(epochs):
    tm.fit(X_tr, y_tr, epochs=1) # Train for one epoch at a time
    S_current = count_ta_states(tm)
    pos_w_current, neg_w_current = count_clause_weights(tm, S_current)
    pos_w_history.append(pos_w_current)
    neg_w_history.append(neg_w_current)

print("TM acc:", (tm.predict(X_te) == y_te).mean())

C, L = tm.number_of_classes, X_bin.shape[1]

# Base quantities from the last epoch
S_final = count_ta_states(tm)
posW_final, negW_final = count_clause_weights(tm, S_final)
netW_final = posW_final - negW_final
absW_final = np.abs(netW_final)
netW_posneg_final = posW_final + negW_final
absW_posneg_final = np.abs(netW_posneg_final)

# per-class validation accuracies → class_acc (used for Relevance, CW-Sum, Support-CW-Sum etc.)
y_pred_tr = tm.predict(X_tr) # Using training set for this as per original context
class_acc = np.array([
    (y_pred_tr[y_tr==c]==c).mean() if np.any(y_tr==c) else 0
    for c in range(C)
])
class_w = class_acc / (class_acc.sum() + 1e-12)

# ─── Feature Selection Methods ──────────────────────────────────────────────

def minmax_norm(arr):
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        return (arr - mn) / (mx - mn)
    else:
        return np.full_like(arr, 0.5)

def tile_for_classes(vec, C):
    return np.tile(vec, (C, 1))

# 1) MutualInfo (using X_tr and y_tr, as no validation set in original logic)
mi_mat = np.zeros((C,L))
for c in range(C):
    yb = (y_tr==c).astype(int)
    for f in range(L):
        mi_mat[c,f] = mutual_info_score(X_tr[:,f], yb)
mi_mat = normalize_per_class(mi_mat)

# 2) Chi2
chi2_mat = np.zeros((C,L))
for c in range(C):
    yb = (y_tr==c).astype(int)
    s,_ = chi2(X_tr,yb)
    chi2_mat[c] = np.nan_to_num(s)
chi2_mat = normalize_per_class(chi2_mat)

# 3) Variance
variance_mat = np.zeros((C, L))
for c in range(C):
    X_class_samples = X_tr[y_tr == c]
    if X_class_samples.shape[0] > 0:
        variance_mat[c] = np.var(X_class_samples, axis=0)
variance_mat = normalize_per_class(variance_mat)

# 4) Random
rand_mat = normalize_per_class(rng.random((C,L)))

# --- TM-based and per-class methods ---
net_w = pos_w_history[-1] - neg_w_history[-1]  # (C, L)
abs_w = np.abs(net_w)
net_w_posneg = pos_w_history[-1] + neg_w_history[-1]
abs_w_posneg = np.abs(net_w_posneg)

# Per-class validation accuracies
class_acc = np.array([
    (y_pred_tr[y_tr==c]==c).mean() if np.any(y_tr==c) else 0.0
    for c in range(C)
])
class_w = class_acc / (class_acc.sum() + 1e-12)

# Relevance (per class, per feature, normalized per class)
relevance_mat = abs_w / (abs_w.sum(axis=1, keepdims=True) + 1e-12)
relevance_mat = normalize_per_class(relevance_mat)
# Relevance-PosNeg
norm_abs_posneg = abs_w_posneg / (abs_w_posneg.sum(axis=1, keepdims=True) + 1e-12)
relevance_posneg_mat = norm_abs_posneg[:,None] * np.ones((C, L))
relevance_posneg_mat = normalize_per_class(relevance_posneg_mat)

# TM-Weight (per class, per feature, normalized per class)
tm_weight_mat = normalize_per_class(abs_w)
# TM-Weight-PosNeg
tm_weight_posneg_mat = normalize_per_class(abs_w_posneg)

# CW-Sum: global vector (sum over classes), then tile to all classes
cw_sum_vec = (class_w[:, None] * abs_w).sum(axis=0)
cw_sum_mat = np.tile(cw_sum_vec, (C, 1))
cw_sum_mat = normalize_per_class(cw_sum_mat)
# CW-Sum-PosNeg
cw_sum_posneg_mat = np.zeros((C, L))
for c in range(C):
    cw_sum_posneg_mat[c] = class_w[c] * abs_w_posneg[c]
cw_sum_posneg_mat = normalize_per_class(cw_sum_posneg_mat)

# CW-Feat (per class)
freq = S_final.sum(axis=(1,2))
alpha = freq / (freq.sum(axis=0, keepdims=True) + 1e-12)
cw_feat_mat = alpha * abs_w
cw_feat_mat = normalize_per_class(cw_feat_mat)
# CW-Feat-PosNeg
cw_feat_posneg_mat = alpha * abs_w_posneg
cw_feat_posneg_mat = normalize_per_class(cw_feat_posneg_mat)

# Support-CW-Sum (per class)
err_rate = 1 - class_acc
err_w = err_rate / (err_rate.sum() + 1e-12) if err_rate.sum() > 0 else np.ones_like(err_rate) / C
support_mat = np.zeros((C, L))
for c in range(C):
    support_mat[c] = err_w[c] * abs_w[c]
support_mat = normalize_per_class(support_mat)
# Support-CW-Sum-PosNeg
support_posneg_mat = np.zeros((C, L))
for c in range(C):
    support_posneg_mat[c] = err_w[c] * abs_w_posneg[c]
support_posneg_mat = normalize_per_class(support_posneg_mat)

# Margin (per class, per feature)
margin_mat = np.zeros((C, L))
for c in range(C):
    for f in range(L):
        # For each class, find the margin between this class and the next best class for this feature
        class_weights = abs_w[:, f]
        this_class_weight = class_weights[c]
        other_weights = np.delete(class_weights, c)
        if len(other_weights) > 0:
            next_best = np.max(other_weights)
            margin_mat[c, f] = this_class_weight - next_best
        else:
            margin_mat[c, f] = this_class_weight
margin_mat = normalize_per_class(margin_mat)

# Entropy (per class, per feature)
entropy_mat = np.zeros((C, L))
for c in range(C):
    p = abs_w[c] / (abs_w[c].sum() + 1e-12)
    entropy_mat[c] = -p * np.log(p + 1e-12)
entropy_mat = normalize_per_class(entropy_mat)

# Gini (per class, per feature)
gini_mat = np.zeros((C, L))
for c in range(C):
    p = abs_w[c] / (abs_w[c].sum() + 1e-12)
    gini_mat[c] = 1.0 - p**2
gini_mat = normalize_per_class(gini_mat)

# Stability (per class, per feature, per epoch)
net_w_hist = np.stack([pos_w_history[e] - neg_w_history[e] for e in range(len(pos_w_history))], axis=0)  # (epochs, C, L)
stab_mat = np.zeros((C, L))
for c in range(C):
    # net_w_hist[:, c, :] shape: (epochs, L)
    std_hist = np.std(net_w_hist[:, c, :], axis=0)
    mean_abs = np.mean(np.abs(net_w_hist[:, c, :]), axis=0)
    stab_score = np.zeros_like(mean_abs)
    mask = std_hist > 1e-8
    stab_score[mask] = mean_abs[mask] / std_hist[mask]
    stab_score[~mask] = 0.5  # Neutral value if std is zero
    stab_mat[c] = minmax_norm(stab_score)
# Optionally, subtract mean across classes for each feature to highlight relative stability
stab_mat = stab_mat - stab_mat.mean(axis=0, keepdims=True)
stab_mat = normalize_per_class(stab_mat)

# --- Dropout, AblationImpact, VarDropout (per class, distinct logic) ---
print("Calculating Dropout...")
dropout_mat = np.zeros((C, L))
for c in range(C):
    idx = np.where(y_te == c)[0]
    if len(idx) == 0:
        continue
    base_acc = (tm.predict(X_te[idx]) == y_te[idx]).mean()
    for f in range(L):
        X_abl = X_te[idx].copy()
        X_abl[:, f] = 0
        acc = (tm.predict(X_abl) == y_te[idx]).mean()
        dropout_mat[c, f] = base_acc - acc
dropout_mat = normalize_per_class(dropout_mat)

print("Calculating AblationImpact...")
ablation_mat = np.zeros((C, L))
orig_sums = tm.predict(X_te, return_class_sums=True)[1]
for c in range(C):
    idx = np.where(y_te == c)[0]
    if len(idx) == 0:
        continue
    orig_true_scores = orig_sums[idx, c]
    for f in range(L):
        X_abl = X_te[idx].copy()
        X_abl[:, f] = 0
        ablated_sums = tm.predict(X_abl, return_class_sums=True)[1]
        ablated_true_scores = ablated_sums[:, c]
        ablation_mat[c, f] = np.mean(np.abs(orig_true_scores - ablated_true_scores))
ablation_mat = normalize_per_class(ablation_mat)

print("Calculating VarDropout...")
vardrop_mat = np.zeros((C, L))
N_MASKS = 10
for c in range(C):
    idx = np.where(y_te == c)[0]
    if len(idx) == 0:
        continue
    orig_true_scores = orig_sums[idx, c]
    for f in range(L):
        vals = []
        for _ in range(N_MASKS):
            X_mask = X_te[idx].copy()
            mask = rng.integers(0, 2, size=len(idx), dtype=X_te.dtype)
            X_mask[:, f] *= mask
            masked_sums = tm.predict(X_mask, return_class_sums=True)[1]
            masked_true_scores = masked_sums[:, c]
            vals.append(np.mean(np.abs(orig_true_scores - masked_true_scores)))
        vardrop_mat[c, f] = np.mean(vals)
vardrop_mat = normalize_per_class(vardrop_mat)

# IG, SmoothGradSq, VarGrad (per class, per feature, finite-difference approx)
print("Calculating IG, SmoothGradSq, VarGrad...")
ig_mat = np.zeros((C, L))
smoothgrad_mat = np.zeros((C, L))
vargrad_mat = np.zeros((C, L))
n_ig_samples = min(20, len(X_te))
n_ig_steps = 20
ig_epsilon = 0.2
n_noise_samples = 10
for c in range(C):
    idx = np.where(y_te == c)[0]
    if len(idx) == 0:
        continue
    X_explain = X_te[idx][:n_ig_samples]
    for f in range(L):
        ig_vals = []
        smooth_vals = []
        for i in range(len(X_explain)):
            x_orig = X_explain[i]
            baseline = np.zeros_like(x_orig)
            path_attr = 0.0
            for step in range(1, n_ig_steps+1):
                alpha = float(step) / n_ig_steps
                interp = baseline + alpha * (x_orig - baseline)
                # Finite difference for TM: flip bit, see effect on class sum
                inp_plus = interp.copy()
                inp_plus[f] += ig_epsilon
                inp_plus_bin = (inp_plus > 0.5).astype(np.uint32)
                inp_minus = interp.copy()
                inp_minus[f] -= ig_epsilon
                inp_minus_bin = (inp_minus > 0.5).astype(np.uint32)
                prob_plus = tm.predict(inp_plus_bin.reshape(1, -1), return_class_sums=True)[1][0, c]
                prob_minus = tm.predict(inp_minus_bin.reshape(1, -1), return_class_sums=True)[1][0, c]
                grad = (prob_plus - prob_minus) / (2 * ig_epsilon)
                path_attr += grad
            ig_val = (x_orig[f] - baseline[f]) * (path_attr / n_ig_steps)
            ig_vals.append(ig_val)
            # SmoothGradSq/VarGrad: add noise
            for _ in range(n_noise_samples):
                noise = rng.choice([0, 1], size=x_orig.shape, p=[0.95, 0.05])
                x_noisy = x_orig ^ noise
                path_attr_noisy = 0.0
                for step in range(1, n_ig_steps+1):
                    alpha = float(step) / n_ig_steps
                    interp = baseline + alpha * (x_noisy - baseline)
                    inp_plus = interp.copy()
                    inp_plus[f] += ig_epsilon
                    inp_plus_bin = (inp_plus > 0.5).astype(np.uint32)
                    inp_minus = interp.copy()
                    inp_minus[f] -= ig_epsilon
                    inp_minus_bin = (inp_minus > 0.5).astype(np.uint32)
                    prob_plus = tm.predict(inp_plus_bin.reshape(1, -1), return_class_sums=True)[1][0, c]
                    prob_minus = tm.predict(inp_minus_bin.reshape(1, -1), return_class_sums=True)[1][0, c]
                    grad = (prob_plus - prob_minus) / (2 * ig_epsilon)
                    path_attr_noisy += grad
                ig_noisy = (x_noisy[f] - baseline[f]) * (path_attr_noisy / n_ig_steps)
                smooth_vals.append(ig_noisy)
        if ig_vals:
            ig_mat[c, f] = np.mean(np.abs(ig_vals))
        if smooth_vals:
            smoothgrad_mat[c, f] = np.mean(np.square(smooth_vals))
            vargrad_mat[c, f] = np.var(smooth_vals)
ig_mat = normalize_per_class(ig_mat)
smoothgrad_mat = normalize_per_class(smoothgrad_mat)
vargrad_mat = normalize_per_class(vargrad_mat)

# 10) GroupLasso
img_dim     = 8
group_size  = 4
nGh         = img_dim // group_size
nGw         = img_dim // group_size

grouplasso_mat = np.zeros((C, L)) # Initialize before conditional logic

if img_dim * img_dim == L and L > 0: # Check if features form a square image and L > 0
    absw_img = absW_final.reshape(C, img_dim, img_dim)
    group_l2 = np.zeros((C, nGh, nGw))
    for c in range(C):
        for gh in range(nGh):
            for gw in range(nGw):
                patch = absw_img[c,
                                 gh*group_size:(gh+1)*group_size,
                                 gw*group_size:(gw+1)*group_size]
                group_l2[c,gh,gw] = np.sqrt((patch**2).sum())

    temp_grouplasso_mat = np.zeros((C, img_dim, img_dim))
    for c in range(C):
        for i in range(img_dim):
            for j in range(img_dim):
                gh = i // group_size
                gw = j // group_size
                if gh < nGh and gw < nGw: # Ensure group index is valid
                    temp_grouplasso_mat[c,i,j] = group_l2[c,gh,gw]
    grouplasso_mat = temp_grouplasso_mat.reshape(C, L)
    grouplasso_mat = normalize_per_class(grouplasso_mat)
elif L == 0:
    print(f"Warning: Group Lasso score not computed as number of features is 0.")
    # grouplasso_mat already initialized to zeros
else: # L > 0 but not a perfect square
    print(f"Warning: Group Lasso score not computed as number of features ({L}) is not a perfect square. Assigning default.")
    grouplasso_mat = np.full((C, L), 0.5) # Consistent with minmax_norm on single value, per-class for consistency

# 11) TaylorCrit (First-Order Perturbation Impact)
print("Calculating TaylorCrit...")
taylor_mat = np.zeros((C, L))
original_sums_te = tm.predict(X_te, return_class_sums=True)[1]  # (n_samples, C)

for f in tqdm(range(L), desc="TaylorCrit"):
    X_perturbed = X_te.copy()
    X_perturbed[:, f] = 1 - X_perturbed[:, f]  # Flip bits
    perturbed_sums_te = tm.predict(X_perturbed, return_class_sums=True)[1]

    # Calculate the mean absolute difference in sums for each class
    score_change_per_class = np.mean(np.abs(original_sums_te - perturbed_sums_te), axis=0)  # (C,)
    taylor_mat[:, f] = score_change_per_class

taylor_mat = normalize_per_class(taylor_mat)

# 12) DropoutSensitivity / AblationImpact (Consolidated and improved)
# Measures impact of setting a feature to its "zero" state (0 for binary data)
# on the model's accuracy. This acts as both AblationImpact and Dropout.
dropout_mat = np.zeros((C,L))
# Using a subset of test samples for efficiency, or full X_te if speed permits.
# For consistency with other per-class scores, calculate per-class accuracy drop.
test_samples_count = min(100, len(X_te)) # Limit samples for performance
test_indices = rng.choice(len(X_te), test_samples_count, replace=False)
X_te_subset = X_te[test_indices]
y_te_subset = y_te[test_indices]

base_preds_subset = tm.predict(X_te_subset)

for c_target in range(C):
    # Calculate base accuracy for the target class on the subset
    # Filter for instances actually belonging to the target class
    idx_target_class = np.where(y_te_subset == c_target)[0]
    if len(idx_target_class) == 0:
        continue # Skip if no samples for this class in subset

    base_accuracy_c = (base_preds_subset[idx_target_class] == c_target).mean()

    for f in range(L):
        X_ablated = X_te_subset.copy()
        X_ablated[:, f] = 0 # Set feature to 0 (ablate/drop)

        ablated_preds = tm.predict(X_ablated)
        ablated_accuracy_c = (ablated_preds[idx_target_class] == c_target).mean()

        # The 'importance' is the drop in accuracy
        dropout_mat[c_target, f] = base_accuracy_c - ablated_accuracy_c

dropout_mat = normalize_per_class(dropout_mat)
ablation_mat = dropout_mat # AblationImpact is conceptually the same for this context
drop_mat = dropout_mat # Dropout is conceptually the same for this context


# 13) VarDropout (Stochastic Ablation)
print("Calculating VarDropout...")
vardrop_mat = np.zeros((C, L))
n_masks = 10  # Number of random masks to average over

# Using the same subset as AblationImpact for consistency
# base_preds_subset is already calculated from before

for c_target in tqdm(range(C), desc="VarDropout Classes"):
    idx_target_class = np.where(y_te_subset == c_target)[0]
    if len(idx_target_class) == 0:
        continue

    base_accuracy_c = (base_preds_subset[idx_target_class] == c_target).mean()

    for f in range(L):
        impacts = []
        for _ in range(n_masks):
            X_masked = X_te_subset.copy()
            mask = rng.integers(0, 2, size=len(X_te_subset), dtype=X_te_subset.dtype) # Random binary mask
            X_masked[:, f] *= mask
            masked_preds = tm.predict(X_masked)
            masked_accuracy_c = (masked_preds[idx_target_class] == c_target).mean()
            impacts.append(base_accuracy_c - masked_accuracy_c)
        vardrop_mat[c_target, f] = np.mean(impacts)

vardrop_mat = normalize_per_class(vardrop_mat)


# 14) Permutation-Importance per class
# Base predictions on the full test set for comparison
base_preds_te = tm.predict(X_te)
perm_mat = np.zeros((C, L))

for c in range(C):
    idx_c = np.where(y_te == c)[0]
    if len(idx_c) == 0:
        continue
    base_accuracy_c = (base_preds_te[idx_c] == c).mean()

    for f in range(L):
        Xt_perm = X_te.copy()
        np.random.shuffle(Xt_perm[:, f]) # Shuffle the feature
        perm_preds = tm.predict(Xt_perm)
        perm_accuracy_c = (perm_preds[idx_c] == c).mean()
        perm_mat[c, f] = base_accuracy_c - perm_accuracy_c
perm_mat = normalize_per_class(perm_mat)


# 15) SmoothStabil (Embedded)
print("Calculating SmoothStabil...")
smoothstab_mat = np.zeros((C, L))
n_noise_samples = 10
noise_level = 0.05

for f in tqdm(range(L), desc="SmoothStabil"):
    score_changes_per_class = np.zeros(C)
    for _ in range(n_noise_samples):
        noise = rng.choice([0, 1], size=X_te.shape, p=[1 - noise_level, noise_level])
        X_noisy = X_te ^ noise
        original_sums_noisy = tm.predict(X_noisy, return_class_sums=True)[1]
        X_perturbed_noisy = X_noisy.copy()
        X_perturbed_noisy[:, f] = 1 - X_perturbed_noisy[:, f]
        perturbed_sums_noisy = tm.predict(X_perturbed_noisy, return_class_sums=True)[1]
        score_changes_per_class += np.mean(np.abs(original_sums_noisy - perturbed_sums_noisy), axis=0)
    smoothstab_mat[:, f] = score_changes_per_class / n_noise_samples
smoothstab_mat = normalize_per_class(smoothstab_mat)

# Margin calculation removed - using per-class margin above


# --- LIME (per class, per feature) ---
print("Calculating LIME...")
feature_names = [f"pix{r}_{c}" for r in range(8) for c in range(8)]
lime_mat = np.zeros((C, L))
lime_expl = LimeTabularExplainer(
    training_data        = X_tr,
    feature_names        = feature_names,
    class_names          = [str(c) for c in range(C)],
    mode                 = 'classification',
    discretize_continuous= False,
    random_state         = 0
)
lime_instances_per_class = 3
for c in range(C):
    idxs = np.where(y_tr==c)[0][:lime_instances_per_class]
    agg  = np.zeros(L)
    if len(idxs) == 0:
        continue
    for i in idxs:
        try:
            exp = lime_expl.explain_instance(
                data_row    = X_tr[i],
                predict_fn  = predict_proba_tm,
                labels      = [c],
                num_features= L,
                num_samples = 500
            )
            if c in exp.as_map():
                for feat_idx, weight in exp.as_map()[c]:
                    agg[feat_idx] += abs(weight)
        except Exception as e:
            print(f"Error explaining instance {i} for LIME class {c}: {e}")
            continue
    if len(idxs)>0:
        agg /= len(idxs)
    lime_mat[c] = agg
lime_mat = normalize_per_class(lime_mat)

# ─── Assemble ──────────────────────────────────────────────────────────────
methods = {
    'MutInfo':   mi_mat,
    'Chi2':         chi2_mat,
    'Varianc':     variance_mat,
    'CW-Feat':      cw_feat_mat,
    'PerImpo':perm_mat,
    'TayCrit':   taylor_mat,
    'AblImpa':ablation_mat,
    'VarDrop':   vardrop_mat,
    'IG':           ig_mat,
    'VarGrad':      vargrad_mat,
    'SmhStab': smoothstab_mat,
    'Margin':       margin_mat,
    'Entropy':      entropy_mat,
    'Gini':         gini_mat,
    'Stabili':    stab_mat,
    #'SHAP': shap_mat,
    'LIME': lime_mat,
}

# reshape to (C,8,8)
for k in methods:
    if methods[k].shape == (C,L): # Only reshape if it's a C x L matrix
        methods[k] = methods[k].reshape(C,8,8)
    else:
        print(f"Skipping reshape for {k} due to unexpected shape: {methods[k].shape}")


# ─── Plot ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    nrows=len(methods), ncols=C,
    figsize=(1.2*C, 1.2*len(methods)),
    subplot_kw={"xticks":[], "yticks":[]}
)
vmin, vmax = 0, 1
# Ensure axes is always 2D, even if nrows or ncols is 1
if len(methods) == 1:
    axes = np.expand_dims(axes, axis=0)
if C == 1:
    axes = np.expand_dims(axes, axis=1)

for i,(name,arr) in enumerate(methods.items()):
    for c in range(C):
        ax = axes[i,c]
        im = ax.imshow(arr[c], cmap="viridis", vmin=vmin, vmax=vmax)
        if i==0: ax.set_title(str(c), fontsize=16)
    axes[i,0].set_ylabel(name, rotation=0, labelpad=40, va="center", fontsize=16)

#cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.015, pad=0.01)
#cbar.set_label("Normalized Importance", rotation=270, labelpad=15)
plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig("1digits_per_class_all_fs.png", dpi=300)
plt.close()
print("→ saved digits_per_class_all_fs.png")