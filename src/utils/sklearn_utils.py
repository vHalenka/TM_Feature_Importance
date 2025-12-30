"""
Sklearn model utility functions.
"""
import numpy as np


def sklearn_predict_proba(model, X, ncl):
    """
    Get probability predictions from a sklearn model.
    
    Args:
        model: Trained sklearn classifier
        X: Input features
        ncl: Number of classes
        
    Returns:
        P: Probability matrix (n_samples, n_classes)
    """
    if hasattr(model, 'predict_proba'):
        p = model.predict_proba(X)
        if p.shape[1] == 1 and ncl == 2:
            return np.hstack([1 - p, p])
        if p.shape[1] == ncl:
            return p
    # Fallback to one-hot encoding
    preds = model.predict(X)
    P = np.zeros((X.shape[0], ncl), float)
    for i, p in enumerate(preds):
        if 0 <= p < ncl:
            P[i, p] = 1.0
        else:
            P[i, :] = 1 / ncl
    return P


def get_sklearn_embedded_scores(model):
    """
    Extract embedded feature importance scores from a sklearn model.
    
    Args:
        model: Trained sklearn model with feature_importances_ or coef_ attribute
        
    Returns:
        Dictionary with 'Sklearn_FS' key containing normalized feature scores
    """
    from . import serialization
    
    if hasattr(model, 'feature_importances_'):
        return {'Sklearn_FS': serialization.minmax_norm(model.feature_importances_)}
    if hasattr(model, 'coef_'):
        c = np.abs(model.coef_)
        if c.ndim > 1:
            c = c.max(axis=0)
        return {'Sklearn_FS': serialization.minmax_norm(c)}
    return {}

