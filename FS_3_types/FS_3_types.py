import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tmu.models.classification.vanilla_classifier import TMClassifier

def count_ta_states(tm):
    """Extract the TA states array (n_classes, polarity=2, n_clauses/2, n_features)."""
    num_features = tm.clause_banks[0].number_of_features
    n_classes    = tm.number_of_classes
    n_clauses    = tm.number_of_clauses // 2
    ta_states    = np.zeros((n_classes, 2, n_clauses, num_features), dtype=int)
    for cls in range(n_classes):
        for pol in (0, 1):
            for cl in range(n_clauses):
                for feat in range(num_features):
                    ta_states[cls,pol,cl,feat] = tm.get_ta_action(
                        clause=cl, ta=feat, the_class=cls, polarity=pol
                    )
    return ta_states

def count_clause_weights(tm, ta_states=None):
    """Sum clause weights for each feature if its TA is active."""
    if ta_states is None:
        ta_states = count_ta_states(tm)
    num_features = tm.clause_banks[0].number_of_features
    n_classes    = tm.number_of_classes
    pos_w = np.zeros((n_classes, num_features))
    neg_w = np.zeros((n_classes, num_features))
    for cls in range(n_classes):
        for pol in (0, 1):
            for cl in range(tm.number_of_clauses // 2):
                w = tm.get_weight(the_class=cls, polarity=pol, clause=cl)
                active = ta_states[cls,pol,cl] != 0
                if pol == 0:
                    pos_w[cls] += w * active
                else:
                    neg_w[cls] += w * active
    return pos_w, neg_w

def bucketize(feature_states):
    """Map a length-3 array of {-1,0,1} to Irrelevant, Unique, or Ranking."""
    unique_vals = set(feature_states)
    # All same sign or zero
    if len(unique_vals) == 1:
        return "Irrelevant"
    # Exactly one nonzero entry
    if np.count_nonzero(feature_states) == 1:
        return "Unique"
    # Otherwise
    return "Ranking"

# --- MAIN TEST SCRIPT ---

# 1) Load a tiny 3-class dataset
digits = load_digits(n_class=3)
X = digits.images.reshape(len(digits.images), -1)
y = digits.target
y = y.astype(np.uint32)  # Ensure y is uint32
X = X.astype(np.uint32)  # Ensure X is uint32

# 2) Simple binarization at the median
X_bin = (X > np.median(X)).astype(np.uint32)

# 3) Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_bin, y, test_size=0.2, random_state=42
)

# 4) Instantiate and fit a small TM
tm = TMClassifier(
    number_of_clauses=60,
    T=15,
    s=3.0,
    max_included_literals=20,
    platform='CPU'
)
tm.fit(X_train, y_train)

# 5) Extract TA states
ta_states = count_ta_states(tm)

# 6) Compute frequency-based pos/neg counts
pos_freq = ta_states[:,0,:,:].sum(axis=1)  # shape (3, n_features)
neg_freq = ta_states[:,1,:,:].sum(axis=1)

# 7) Compute weight-based pos/neg sums
pos_w, neg_w = count_clause_weights(tm, ta_states)

# 8) Derive ternary class states: +1 if pos>neg, -1 if neg>pos, 0 if equal
def ternary_state(pos, neg):
    st = np.zeros_like(pos, dtype=int)
    st[pos>neg] =  1
    st[neg>pos] = -1
    return st

states_freq = ternary_state(pos_freq, neg_freq)  # (3, n_features)
states_w    = ternary_state(pos_w, neg_w)

# 9) Bucketize each feature and print
n_features = states_freq.shape[1]
print("Feat | FreqState | BucketFreq | WeightState | BucketWeight")
print("-----------------------------------------------------------")
for feat in range(n_features):
    sf = states_freq[:, feat]
    sw = states_w[:, feat]
    sf_str = "".join(f"{s:+d}" for s in sf)
    sw_str = "".join(f"{s:+d}" for s in sw)
    print(f"{feat:4d} | {sf_str:9s} | {bucketize(sf):11s} | {sw_str:11s} | {bucketize(sw)}")
