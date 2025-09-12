import matplotlib.pyplot as plt
import numpy as np

def plot_shap_summary_bar(shap_values, feature_names, max_display=20, title="Feature Importance based on SHAP Values", new=False):
    """
    Creates a nicely formatted horizontal bar plot of SHAP values,
    ordered by their absolute magnitude.

    Parameters:
    - shap_values (np.array or list): The SHAP values for each feature.
    - feature_names (list of str): The names of the features, corresponding to shap_values.
    - max_display (int): Maximum number of features to display.
    - title (str): The title of the plot.
    """
    if len(shap_values) != len(feature_names):
        raise ValueError("Length of shap_values and feature_names must be the same.")

    # Convert to numpy arrays if they aren't already
    shap_values = np.array(shap_values)
    feature_names = np.array(feature_names)

    # 1. Get absolute SHAP values to determine importance
    abs_shap_values = np.abs(shap_values)

    # 2. Get the indices that would sort by absolute SHAP values (ascending)
    sorted_indices = np.argsort(abs_shap_values)

    # 3. Select top `max_display` features (or all if fewer)
    # These will be the ones with the largest absolute SHAP values
    # If sorted_indices is [idx_smallest_abs, ..., idx_largest_abs],
    # then sorted_indices[-max_display:] gives the indices of the top features.
    if len(sorted_indices) > max_display:
        top_indices = sorted_indices[-max_display:]
    else:
        top_indices = sorted_indices

    # 4. Get the corresponding SHAP values and feature names for the top features
    # These are still sorted by absolute value in ascending order
    top_shap_values = shap_values[top_indices]
    top_feature_names = feature_names[top_indices]

    # 5. Create colors based on the sign of the SHAP value
    colors = ['#b0c4a4' if not new else "#f0dfa2" for s in top_shap_values] # Red for negative, Blue for positive

    # 6. Create the plot
    plt.figure(figsize=(7, len(top_feature_names) * 0.4 + 1.5)) # Adjust height dynamically
    
    # Plot bars
    bars = plt.barh(range(len(top_shap_values)), top_shap_values, color=colors, align='center')

    # Add y-axis ticks and labels (feature names)
    plt.yticks(range(len(top_shap_values)), top_feature_names , fontsize=20)

    # Add labels and title
    plt.xlabel("Mean Absolute SHAP Value (Impact on Model Output)", fontsize=15)
    plt.ylabel("Feature", fontsize=18)
    plt.title(title, fontsize=18)

    # Add a vertical line at x=0 for reference
    plt.axvline(0, color='grey', linestyle='--', linewidth=0.8)

    # Add value labels on bars (optional, can be crowded for many bars)
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width + (max(top_shap_values) * 0.01) if width >= 0 else width - (abs(min(top_shap_values)) * 0.01) # Small offset
        
        # Adjust alignment based on sign
        ha = 'left' if width >= 0 else 'right'
        
        plt.text(label_x_pos, 
                 bar.get_y() + bar.get_height()/2, 
                 f'{width:.3f}', 
                 va='center', 
                 ha=ha,
                 fontsize=18)


    # Improve layout
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.grid(axis='x', linestyle=':', alpha=0.7)
    plt.tight_layout() # Adjust plot to ensure everything fits without overlapping


def to_subscript(s):
    sub_map = str.maketrans("0123456789t", "₀₁₂₃₄₅₆₇₈₉ₜ")
    return ''.join([c.translate(sub_map) if c in "0123456789t" else c for c in s])

[subscripted := [to_subscript(f) for f in ["Tt","Tt1","Wt","Wt1", "Tt*Wt", "Tt1*Wt1", "Tt-Tt1", "Wt-Wt1","Trt"]]]