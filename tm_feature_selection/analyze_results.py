import os
import pandas as pd
import numpy as np
from collections import Counter
import json
import glob

def load_results(dataset_name):
    """Load results for a specific dataset."""
    results_file = f"results/{dataset_name}_results.json"
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            return json.load(f)
    return None

def analyze_top_methods(results):
    """Analyze which methods appear most frequently in top 3."""
    top_methods = []
    
    for dataset, dataset_results in results.items():
        if not dataset_results:
            continue
            
        # Get all methods and their scores
        methods_scores = []
        for method, metrics in dataset_results.items():
            if 'accuracy' in metrics:
                methods_scores.append((method, metrics['accuracy']))
        
        # Sort by accuracy and get top 3
        methods_scores.sort(key=lambda x: x[1], reverse=True)
        top_3 = [method for method, _ in methods_scores[:3]]
        top_methods.extend(top_3)
    
    # Count occurrences
    method_counts = Counter(top_methods)
    return method_counts

def main():
    # Get all dataset results
    results = {}
    for results_file in glob.glob("results/*_results.json"):
        dataset_name = os.path.basename(results_file).replace("_results.json", "")
        results[dataset_name] = load_results(dataset_name)
    
    # Analyze top methods
    method_counts = analyze_top_methods(results)
    
    # Print results
    print("\nFeature Selection Method Analysis")
    print("================================")
    print("\nMethods appearing in top 3 across all datasets:")
    for method, count in method_counts.most_common():
        print(f"{method}: {count} times")
    
    # Save analysis results
    analysis_results = {
        "method_counts": dict(method_counts),
        "total_datasets": len(results)
    }
    
    with open("results/analysis_summary.json", 'w') as f:
        json.dump(analysis_results, f, indent=4)

if __name__ == "__main__":
    main() 