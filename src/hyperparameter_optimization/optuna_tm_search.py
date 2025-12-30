"""
Optuna-based hyperparameter optimization for Tsetlin Machines.
"""
import json
import optuna
from sklearn.model_selection import train_test_split

from tmu.models.classification.vanilla_classifier import TMClassifier
from ..utils import data_loading, paths, serialization


def make_objective(X_train, y_train, X_val, y_val, clauses, epochs, patience):
    """Create Optuna objective function for TM hyperparameter search."""
    def objective(trial):
        s = trial.suggest_float("s", 0.9, 20.0)
        T = trial.suggest_categorical("T", [50, 200, 300, 500, 800])
        tm = TMClassifier(
            number_of_clauses=clauses,
            T=T,
            s=s,
            max_included_literals=20,
            platform='CPU'
        )
        best_val = 0.0
        wait = 0
        for ep in range(epochs):
            tm.fit(X_train, y_train, epochs=1)
            acc = (tm.predict(X_val) == y_val).mean()
            if acc > best_val + 1e-6:
                best_val = acc
                wait = 0
            else:
                wait += 1
            if wait >= patience:
                break
        return best_val
    return objective


def optimize_tm_hyperparameters(datasets, clauses=500, epochs=30, patience=3, 
                                  trials=100, max_bins=10, preserve_existing=True):
    """
    Optimize Tsetlin Machine hyperparameters for multiple datasets.
    
    Args:
        datasets: List of dataset names to optimize
        clauses: Number of clauses for TM
        epochs: Maximum training epochs
        patience: Early stopping patience
        trials: Number of Optuna trials per dataset
        max_bins: Maximum bins for thermometer encoding
        preserve_existing: If True, skip datasets that already have optimized parameters
        
    Returns:
        Dictionary mapping dataset names to best parameters
    """
    paths.ensure_dirs()
    
    # Load existing parameters if preserving
    existing_results = {}
    if preserve_existing:
        aggregated_path = paths.get_aggregated_params_path("all_best_tm_params.json", local_only=False)
        if aggregated_path.exists():
            try:
                existing_results = serialization.load_json(aggregated_path)
                print(f"Loaded existing parameters for {len(existing_results)} datasets")
            except:
                pass
    
    results = existing_results.copy() if preserve_existing else {}
    
    for name in datasets:
        # Skip if already exists and preserving
        if preserve_existing and name in results and results[name] is not None:
            print(f"\n=== Skipping {name} (already optimized) ===")
            print(f"  Existing params: s={results[name].get('best_s', 'N/A')}, T={results[name].get('best_T', 'N/A')}")
            continue
            
        print(f"\n=== Tuning {name} ===")
        try:
            X_raw, y_raw = data_loading.load_dataset(name)
            X_bin, y = data_loading.preprocess_data(X_raw, y_raw, max_bins=max_bins)
            X_train, X_val, y_train, y_val = train_test_split(
                X_bin, y, test_size=0.2, random_state=42, stratify=y
            )
            
            study = optuna.create_study(direction="maximize")
            study.optimize(
                make_objective(X_train, y_train, X_val, y_val, clauses, epochs, patience),
                n_trials=trials
            )
            
            best = study.best_trial
            result = {
                "best_s": best.params["s"],
                "best_T": best.params["T"],
                "best_val_accuracy": best.value
            }
            results[name] = result
            
            # Save individual dataset params
            output_path = paths.get_best_params_path(name, local_only=False)
            serialization.save_json(result, output_path)
            print(f"Best params for {name}: {best.value:.4f}")
            print(f"Saved to {output_path}")
        except Exception as e:
            print(f"Error optimizing {name}: {e}")
            results[name] = None
    
    # Save aggregated params
    aggregated_path = paths.get_aggregated_params_path("all_best_tm_params.json", local_only=False)
    serialization.save_json(results, aggregated_path)
    print(f"\nSaved aggregated parameters to {aggregated_path}")
    
    return results

