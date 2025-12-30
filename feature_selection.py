import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import time

# Store results
results = {
    'Method': [],
    'Dataset': [],
    'Score': [],
    'Time': []
}

# Define feature selection methods
def select_k_best_f_classif(X, y):
    selector = SelectKBest(f_classif, k=5)
    selector.fit(X, y)
    return np.mean(selector.scores_)

def select_k_best_mutual_info(X, y):
    selector = SelectKBest(mutual_info_classif, k=5)
    selector.fit(X, y)
    return np.mean(selector.scores_)

def random_forest_importance(X, y):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    return np.mean(rf.feature_importances_)

# Define feature selection methods dictionary
feature_selection_methods = {
    'F-Value': select_k_best_f_classif,
    'Mutual Information': select_k_best_mutual_info,
    'Random Forest': random_forest_importance
}

# Load and prepare datasets
datasets = {}
for name, loader in [('Breast Cancer', load_breast_cancer),
                    ('Iris', load_iris),
                    ('Wine', load_wine)]:
    data = loader()
    X = StandardScaler().fit_transform(data.data)
    y = data.target
    datasets[name] = (X, y)

# Run feature selection methods
for method_name, method_func in feature_selection_methods.items():
    print(f"\nRunning {method_name}...")
    for dataset_name, (X, y) in datasets.items():
        try:
            start_time = time.time()
            score = method_func(X, y)
            end_time = time.time()
            
            results['Method'].append(method_name)
            results['Dataset'].append(dataset_name)
            results['Score'].append(score)
            results['Time'].append(end_time - start_time)
            
            print(f"{dataset_name}: Score = {score:.4f}, Time = {end_time - start_time:.2f}s")
        except Exception as e:
            print(f"Error with {method_name} on {dataset_name}: {str(e)}")
            results['Method'].append(method_name)
            results['Dataset'].append(dataset_name)
            results['Score'].append(None)
            results['Time'].append(None)

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save results to CSV
results_df.to_csv('feature_selection_results.csv', index=False)
print("\nResults saved to feature_selection_results.csv")

# Create visualization
plt.figure(figsize=(15, 10))

# Plot scores
plt.subplot(2, 1, 1)
sns.barplot(data=results_df, x='Method', y='Score', hue='Dataset')
plt.title('Feature Selection Method Scores')
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Plot execution times
plt.subplot(2, 1, 2)
sns.barplot(data=results_df, x='Method', y='Time', hue='Dataset')
plt.title('Feature Selection Method Execution Times')
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('feature_selection_results.png', bbox_inches='tight', dpi=300)
plt.close()

print("\nResults visualization saved to feature_selection_results.png") 