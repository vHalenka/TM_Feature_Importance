import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
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

def minmax_norm(arr):
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        return (arr - mn) / (mx - mn)
    else:
        return np.full_like(arr, 0.5)

if __name__ == "__main__":
    # 1) Load n-class digits
    digits = load_digits(n_class=5)
    X = digits.images.reshape(len(digits.images), -1).astype(np.uint32)
    y = digits.target.astype(np.uint32)

    # 2) Binarize at median
    X_bin = (X > np.median(X)).astype(np.uint32)

    # 3) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_bin, y, test_size=0.2, random_state=0
    )

    # 4) Train TM
    tm = TMClassifier(
        number_of_clauses=60,
        T=15,
        s=3.0,
        max_included_literals=20,
        platform='CPU'
    )
    tm.fit(X_train, y_train)
    y_pred = tm.predict(X_test)
    full_acc = 100*(y_pred==y_test).mean()
    print(f"Accuracy: {full_acc:.2f}%\n")

    # 5) Extract net TM-weights
    ta_states = count_ta_states(tm)
    pos_w, neg_w = count_clause_weights(tm, ta_states)
    net_w = pos_w - neg_w            # shape (n_classes, n_features)

    n_classes, n_feats = net_w.shape

    # 6) Per-class accuracies
    class_acc = np.zeros(n_classes, dtype=float)
    for c in range(n_classes):
        mask = (y_test == c)
        if mask.any():
            class_acc[c] = (y_pred[mask]==c).mean()
    # normalize into weights
    class_w = class_acc / (class_acc.sum() + 1e-12)

    # 7) Removal sensitivity
    drop = np.zeros(n_feats, dtype=float)
    for f in range(n_feats):
        Xm = X_test.copy()
        Xm[:,f] = 0
        acc = 100*(tm.predict(Xm)==y_test).mean()
        drop[f] = full_acc - acc
    drop_norm = minmax_norm(drop)

    # 8) Class-weighted relevance
    abs_w = np.abs(net_w)
    norm_abs_w = abs_w / (abs_w.sum(axis=1, keepdims=True) + 1e-12)
    relevance = (class_w[:,None] * norm_abs_w).sum(axis=0)
    relevance_n = minmax_norm(relevance)

    # 9) Mutual information vs. correctness
    correct = (y_pred == y_test).astype(int)
    mi = np.array([mutual_info_score(X_test[:,f], correct) for f in range(n_feats)])
    mi_norm = minmax_norm(mi)

    # 10) TM-weight score: collapse across classes by max absolute then normalize
    weight_score = np.max(np.abs(net_w), axis=0)
    tm_norm = minmax_norm(weight_score)

    # 11) Print table
    header = ("Feat", "WeightScore", "DropNorm", "RelWeight", "MI_norm")
    print(f"{header[0]:>4s} | {header[1]:>12s} | {header[2]:>8s} | {header[3]:>9s} | {header[4]:>7s}")
    print("-"*60)
    for f in range(n_feats):
        print(f"{f:4d} | {tm_norm[f]:12.3f} | {drop_norm[f]:8.3f} | {relevance_n[f]:9.3f} | {mi_norm[f]:7.3f}")

    # 12) Plot all four normalized scores
    features = np.arange(n_feats)
    plt.figure(figsize=(8,4))
    plt.plot(features, tm_norm,     label='TM-weight')
    plt.plot(features, drop_norm,   label='Dropout')
    plt.plot(features, relevance_n, label='Relevance')
    plt.plot(features, mi_norm,     label='MutualInfo')
    plt.xlabel('Feature Index')
    plt.ylabel('Normalized Importance')
    plt.title('Overlay of Normalized Feature Scores')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
