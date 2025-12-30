def lime_pred(X, y, model, n_samples=1000, n_features=10):
    """Compute LIME-based feature importance scores."""
    # Create a LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X,
        mode='classification',
        class_names=[str(i) for i in range(len(np.unique(y)))],
        feature_names=[f'feature_{i}' for i in range(X.shape[1])]
    )
    
    # Initialize feature importance scores
    feature_importance = np.zeros(X.shape[1])
    feature_counts = np.zeros(X.shape[1])
    
    # Sample instances from each class
    unique_classes = np.unique(y)
    samples_per_class = n_samples // len(unique_classes)
    
    for class_idx in unique_classes:
        # Get indices of samples from this class
        class_indices = np.where(y == class_idx)[0]
        
        # Sample instances from this class
        if len(class_indices) > samples_per_class:
            sampled_indices = np.random.choice(class_indices, samples_per_class, replace=False)
        else:
            sampled_indices = class_indices
        
        # Generate explanations for each sampled instance
        for idx in sampled_indices:
            exp = explainer.explain_instance(
                X[idx],
                model.predict_proba,
                num_features=n_features,
                top_labels=1
            )
            
            # Get the explanation for the predicted class
            label = exp.available_labels()[0]
            explanation = exp.as_list(label=label)
            
            # Update feature importance scores
            for feat_idx, importance in explanation:
                # Extract feature index from the feature name (e.g., 'feature_5' -> 5)
                try:
                    feat_num = int(feat_idx.split('_')[1])
                    feature_importance[feat_num] += abs(importance)
                    feature_counts[feat_num] += 1
                except (IndexError, ValueError):
                    continue
    
    # Average the importance scores
    feature_importance = np.divide(
        feature_importance,
        feature_counts,
        out=np.zeros_like(feature_importance),
        where=feature_counts != 0
    )
    
    return feature_importance 

def compute_all_scores(tm, X_train, X_val, y_train, y_val):
    """Compute all feature importance scores."""
    # Initialize scores dictionary
    scores = {}
    
    # Compute TM-based scores
    scores['tm_importance'] = tm_importance_score(tm)
    scores['tm_frequency'] = tm_frequency_score(tm)
    scores['tm_confidence'] = tm_confidence_score(tm)
    scores['tm_coverage'] = tm_coverage_score(tm)
    scores['tm_combined'] = tm_combined_score(tm)
    
    # Compute LIME scores - already returns feature indices in correct order
    lime_scores = lime_pred(X_train, y_train, tm)
    scores['lime'] = lime_scores
    
    # Compute SHAP scores - already returns feature indices in correct order
    shap_scores = shap_pred(X_train, y_train, tm)
    scores['shap'] = shap_scores
    
    # Compute Group Lasso scores
    lasso_scores = group_lasso_score(X_train, y_train)
    scores['group_lasso'] = lasso_scores
    
    return scores 