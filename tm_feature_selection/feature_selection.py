import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from pytsetlin import TsetlinMachine
import os
import json
from collections import defaultdict

def load_dataset(dataset_path):
    """Load and preprocess dataset."""
    data = pd.read_csv(dataset_path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    # Convert target to uint32 for Tsetlin Machine
    le = LabelEncoder()
    y = le.fit_transform(y).astype(np.uint32)
    
    return X, y

def chi_squared_selection(X, y, n_features):
    """Chi-squared feature selection."""
    from sklearn.feature_selection import chi2, SelectKBest
    selector = SelectKBest(chi2, k=n_features)
    X_new = selector.fit_transform(X, y)
    selected_features = selector.get_support(indices=True)
    return X_new, selected_features

def mutual_info_selection(X, y, n_features):
    """Mutual Information based feature selection."""
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    selector = SelectKBest(mutual_info_classif, k=n_features)
    X_new = selector.fit_transform(X, y)
    selected_features = selector.get_support(indices=True)
    return X_new, selected_features

def relieff_selection(X, y, n_features):
    """ReliefF feature selection."""
    from skrebate import ReliefF
    selector = ReliefF(n_features_to_select=n_features)
    X_new = selector.fit_transform(X, y)
    selected_features = selector.top_features_[:n_features]
    return X_new, selected_features

def rfe_selection(X, y, n_features):
    """Recursive Feature Elimination."""
    estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    selector = RFE(estimator, n_features_to_select=n_features)
    X_new = selector.fit_transform(X, y)
    selected_features = selector.get_support(indices=True)
    return X_new, selected_features

def clause_analysis_selection(X, y, n_features):
    """Tsetlin Machine clause analysis based selection."""
    # Train a Tsetlin Machine
    tm = TsetlinMachine(
        number_of_clauses=100,
        number_of_features=X.shape[1],
        number_of_states=100,
        s=3.0,
        threshold=15,
        number_of_classes=len(np.unique(y))
    )
    
    # Train the model
    tm.fit(X, y, epochs=100)
    
    # Analyze clause importance
    clause_importance = np.zeros(X.shape[1])
    for clause in range(tm.number_of_clauses):
        for feature in range(X.shape[1]):
            if tm.get_clause(clause, feature) != 0:
                clause_importance[feature] += 1
    
    # Select top features
    selected_features = np.argsort(clause_importance)[-n_features:]
    X_new = X[:, selected_features]
    
    return X_new, selected_features

def evaluate_selection(X, y, selected_features, method_name):
    """Evaluate feature selection method using Tsetlin Machine."""
    X_selected = X[:, selected_features]
    
    # Train Tsetlin Machine
    tm = TsetlinMachine(
        number_of_clauses=100,
        number_of_features=len(selected_features),
        number_of_states=100,
        s=3.0,
        threshold=15,
        number_of_classes=len(np.unique(y))
    )
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42
    )
    
    # Train and evaluate
    tm.fit(X_train, y_train, epochs=100)
    accuracy = tm.score(X_test, y_test)
    
    return {
        "accuracy": float(accuracy),
        "selected_features": selected_features.tolist()
    }

def main():
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Get all dataset files
    dataset_files = [f for f in os.listdir("datasets") if f.endswith(".csv")]
    
    for dataset_file in dataset_files:
        print(f"\nProcessing {dataset_file}...")
        dataset_name = dataset_file.replace(".csv", "")
        
        # Load dataset
        X, y = load_dataset(os.path.join("datasets", dataset_file))
        
        # Number of features to select (20% of total features)
        n_features = max(1, int(X.shape[1] * 0.2))
        
        # Dictionary to store results
        results = {}
        
        # Run all feature selection methods
        methods = {
            "chi_squared": chi_squared_selection,
            "mutual_info": mutual_info_selection,
            "relieff": relieff_selection,
            "rfe": rfe_selection,
            "clause_analysis": clause_analysis_selection
        }
        
        for method_name, method_func in methods.items():
            print(f"Running {method_name}...")
            try:
                X_new, selected_features = method_func(X, y, n_features)
                results[method_name] = evaluate_selection(X, y, selected_features, method_name)
            except Exception as e:
                print(f"Error in {method_name}: {str(e)}")
                results[method_name] = {"error": str(e)}
        
        # Save results
        with open(f"results/{dataset_name}_results.json", 'w') as f:
            json.dump(results, f, indent=4)

if __name__ == "__main__":
    main() 