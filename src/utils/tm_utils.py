"""
Tsetlin Machine specific utility functions.
"""
import numpy as np


def count_ta_states(tm):
    """
    Count Tsetlin Automaton (TA) states for all clauses and features.
    
    Args:
        tm: Trained TMClassifier instance
        
    Returns:
        states: Array of shape (n_classes, 2, n_clauses, n_feats) with TA states
    """
    n_classes = tm.number_of_classes
    n_clauses = tm.number_of_clauses // 2
    n_feats = tm.clause_banks[0].number_of_features
    states = np.zeros((n_classes, 2, n_clauses, n_feats), int)
    for c in range(n_classes):
        for pol in (0, 1):
            for cl in range(n_clauses):
                for f in range(n_feats):
                    states[c, pol, cl, f] = tm.get_ta_action(
                        clause=cl, ta=f, the_class=c, polarity=pol
                    )
    return states


def count_clause_weights(tm, ta_states=None):
    """
    Compute positive and negative clause weights aggregated by feature.
    
    Args:
        tm: Trained TMClassifier instance
        ta_states: Optional precomputed TA states (from count_ta_states)
        
    Returns:
        pos: Positive weights per class and feature (n_classes, n_feats)
        neg: Negative weights per class and feature (n_classes, n_feats)
    """
    if ta_states is None:
        ta_states = count_ta_states(tm)
    n_classes, _, n_clauses, n_feats = ta_states.shape
    pos = np.zeros((n_classes, n_feats))
    neg = np.zeros((n_classes, n_feats))
    for c in range(n_classes):
        for pol in (0, 1):
            for cl in range(n_clauses):
                w = tm.get_weight(the_class=c, polarity=pol, clause=cl)
                act = ta_states[c, pol, cl] != 0
                if pol == 0:
                    pos[c] += w * act
                else:
                    neg[c] -= w * act
    return pos, neg


def tm_predict_proba(model, X, y_ctx):
    """
    Get probability predictions from a Tsetlin Machine model.
    
    Args:
        model: Trained TMClassifier instance
        X: Input features
        y_ctx: Context for determining number of classes (unused but kept for compatibility)
        
    Returns:
        P: Probability matrix (n_samples, n_classes)
    """
    try:
        tup = model.predict(X, return_class_sums=True)
        if isinstance(tup, (list, tuple)) and len(tup) == 2:
            preds, sums = tup
            shifted = sums - sums.min(axis=1, keepdims=True)
            total = shifted.sum(axis=1, keepdims=True)
            P = np.zeros_like(shifted, float)
            mask = total.flatten() > 1e-9
            P[mask] = shifted[mask] / total[mask]
            for i, ok in enumerate(~mask):
                if ok:
                    continue
                cls = int(preds[i])
                P[i, cls] = 1.0
            return P
    except:
        pass
    # Fallback to one-hot encoding
    preds = model.predict(X)
    ncl = model.number_of_classes if hasattr(model, 'number_of_classes') else len(np.unique(y_ctx))
    P = np.zeros((X.shape[0], ncl), float)
    for i, p in enumerate(preds):
        if 0 <= p < ncl:
            P[i, p] = 1.0
    return P

