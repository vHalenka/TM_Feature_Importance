import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tmu.models.classification.vanilla_classifier import TMClassifier

def count_ta_states(tm):
    """Extract TA states: (n_classes, 2, n_clauses/2, n_features)."""
    n_classes = tm.number_of_classes
    n_clauses = tm.number_of_clauses // 2
    n_feats   = tm.clause_banks[0].number_of_features
    arr = np.zeros((n_classes, 2, n_clauses, n_feats), dtype=int)
    for c in range(n_classes):
        for pol in (0,1):
            for cl in range(n_clauses):
                for f in range(n_feats):
                    arr[c,pol,cl,f] = tm.get_ta_action(
                        clause=cl, ta=f, the_class=c, polarity=pol
                    )
    return arr

def count_clause_weights(tm, ta_states=None):
    """Sum clause weights per class & feature."""
    if ta_states is None:
        ta_states = count_ta_states(tm)
    n_classes, _, n_clauses, n_feats = ta_states.shape
    pos_w = np.zeros((n_classes, n_feats), dtype=float)
    neg_w = np.zeros((n_classes, n_feats), dtype=float)
    for c in range(n_classes):
        for pol in (0,1):
            for cl in range(n_clauses):
                w = tm.get_weight(the_class=c, polarity=pol, clause=cl)
                active = (ta_states[c,pol,cl] != 0)
                if pol == 0:
                    pos_w[c] += w * active
                else:
                    neg_w[c] -= w * active
    return pos_w, neg_w

def bucket_type(net_w_column):
    """Given net weights across classes, return bucket type."""
    signs       = np.sign(net_w_column)
    unique_vals = np.unique(signs)
    if len(unique_vals) == 1:
        return "Irrelevant"
    if np.count_nonzero(signs) == 1:
        return "Unique"
    return "Ranking"

def minmax_norm(x):
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn) if mx > mn else np.zeros_like(x)

if __name__ == "__main__":
    # 1) Load & binarize
    digits = load_digits(n_class=5)
    X = digits.images.reshape(len(digits.images), -1).astype(np.uint32)
    y = digits.target.astype(np.uint32)
    X = (X > np.median(X)).astype(np.uint32)

    # 2) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # 3) Train a small TM
    tm = TMClassifier(
        number_of_clauses=60,
        T=15,
        s=3.0,
        max_included_literals=20,
        platform='CPU'
    )
    tm.fit(X_train, y_train)

    # 4) Extract net weights
    ta     = count_ta_states(tm)
    pos_w, neg_w = count_clause_weights(tm, ta)
    net_w  = pos_w - neg_w      # shape (n_classes, n_feats)
    abs_w  = np.abs(net_w)
    n_classes, n_feats = net_w.shape

    # 5) Bucket into Irrelevant / Unique / Ranking
    types          = np.array([bucket_type(net_w[:,f]) for f in range(n_feats)])
    unique_idx     = np.where(types == "Unique")[0]
    ranking_idx    = np.where(types == "Ranking")[0]
    irrelevant_idx = np.where(types == "Irrelevant")[0]

    # 6) For Ranking features, compute candidate scores
    A         = abs_w[:, ranking_idx]
    sorted_abs= np.sort(A, axis=0)
    margin    = sorted_abs[-1] - sorted_abs[-2]
    p         = A / (A.sum(axis=0, keepdims=True) + 1e-12)
    gini      = (p*p).sum(axis=0)
    cw_sum    = A.sum(axis=0)

    margin_n  = minmax_norm(margin)
    gini_n    = minmax_norm(gini)
    cw_n      = minmax_norm(cw_sum)

    order_margin  = ranking_idx[np.argsort(margin_n)[::-1]]
    order_gini    = ranking_idx[np.argsort(gini_n)[::-1]]
    order_cw_r    = ranking_idx[np.argsort(cw_n)[::-1]]

    # 7) Build full CW-Sum ordering: Unique → Ranking(CW) → Irrelevant
    order_cw_full = np.concatenate([unique_idx, order_cw_r, irrelevant_idx])

    # 8) Progressive feature‐addition on partial data
    frac    = 0.5
    n_sub   = int(frac * X_train.shape[0])
    X_sub   = X_train[:n_sub]
    y_sub   = y_train[:n_sub]
    K_list  = np.unique(np.linspace(1, n_feats, 10, dtype=int))

    # prepare to collect results
    experiments     = [
        (order_margin,  "Margin"),
        (order_gini,    "Gini"),
        (order_cw_full, "CW-Sum (U→R→I)")
    ]
    results         = {name: [] for _, name in experiments}

    print(f"\nTraining on {100*frac:.0f}% of data ({n_sub} samples), progressively adding features:")
    for ordering, name in experiments:
        print(f"\n-- Using ordering by {name} --")
        for K in K_list:
            sel = ordering[:K]
            tm2 = TMClassifier(
                number_of_clauses=60,
                T=15,
                s=3.0,
                max_included_literals=20,
                platform='CPU'
            )
            tm2.fit(X_sub[:, sel], y_sub)
            acc = 100 * (tm2.predict(X_test[:, sel]) == y_test).mean()
            results[name].append(acc)
            print(f"  K={K:2d}: {acc:5.2f}%")

    # 9) Plot the curves
    plt.figure(figsize=(8,5))
    for _, name in experiments:
        plt.plot(K_list, results[name], marker='o', label=name)
    plt.xlabel("Number of Features K")
    plt.ylabel("Test Accuracy %")
    plt.title("Progressive TM on 50% data")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
