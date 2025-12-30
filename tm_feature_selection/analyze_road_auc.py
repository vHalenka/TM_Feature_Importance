import os
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from PIL import Image
import io

def extract_road_mask_curve(image_path):
    """Extract the ROAD mask curve data from the image."""
    # Read the image
    img = Image.open(image_path)
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Get the height and width
    height, width = img_array.shape[:2]
    
    # Extract the curve data (assuming the curve is in the center of the image)
    # We'll sample points along the height of the image
    y_values = np.linspace(0, 1, height)
    x_values = np.zeros(height)
    
    # For each y value, find the x value where the curve is
    for i, y in enumerate(y_values):
        # Convert y to image coordinates
        y_img = min(int((1 - y) * height), height - 1)  # Ensure within bounds
        # Find the first non-white pixel in this row
        row = img_array[y_img, :, :3]  # Get RGB values
        # Find where the curve is (non-white pixels)
        curve_points = np.where(np.any(row < 240, axis=1))[0]
        if len(curve_points) > 0:
            x_values[i] = curve_points[0] / width
        else:
            x_values[i] = 0
    
    return x_values, y_values

def calculate_auc(x_values, y_values):
    """Calculate the Area Under the Curve."""
    # Sort the points by x
    sort_idx = np.argsort(x_values)
    x_sorted = x_values[sort_idx]
    y_sorted = y_values[sort_idx]
    
    # Calculate AUC using trapezoidal rule
    auc = np.trapz(y_sorted, x_sorted)
    return auc

def main():
    # Look for ROAD mask curve PNGs in the parent directory
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files = [f for f in os.listdir(parent_dir) if f.startswith('road_mask_curve_') and f.endswith('.png')]
    # Expect filenames like road_mask_curve_<dataset>_<method>.png
    results = defaultdict(dict)  # dataset -> method -> auc
    for file in files:
        name = file.replace('road_mask_curve_', '').replace('.png', '')
        # Try to split dataset and method
        if '_' in name:
            parts = name.split('_')
            dataset = parts[0]
            method = '_'.join(parts[1:])
        else:
            dataset = name
            method = 'unknown'
        file_path = os.path.join(parent_dir, file)
        try:
            x_values, y_values = extract_road_mask_curve(file_path)
            auc = calculate_auc(x_values, y_values)
            results[dataset][method] = auc
        except Exception as e:
            print(f"Error processing {file}: {e}")
    # For each dataset, find top 3 methods by AUC
    method_appearance = Counter()
    print("\nROAD Mask Curve AUC Analysis (Top 3 methods per dataset)")
    print("======================================================")
    for dataset, method_auc in results.items():
        sorted_methods = sorted(method_auc.items(), key=lambda x: x[1], reverse=True)
        print(f"\n{dataset}:")
        for method, auc in sorted_methods[:3]:
            print(f"  {method}: {auc:.4f}")
            method_appearance[method] += 1
    print("\nMethods appearing in top 3 across all datasets:")
    for method, count in method_appearance.most_common():
        print(f"{method}: {count} times")

if __name__ == "__main__":
    main() 