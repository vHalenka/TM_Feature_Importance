import numpy as np
import json

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

# Save results
with open('fsb_road_results.json', 'w') as f:
    serializable_results = convert_to_serializable(results)
    json.dump(serializable_results, f, indent=4) 