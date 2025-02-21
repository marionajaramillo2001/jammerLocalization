import numpy as np
import torch
import torch.nn as nn
import os
import pickle
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# Output directory for results
output_dir = '/Users/marionajaramillocivill/Documents/GitHub/jammerLocalization/results/Execution_100_pathloss/pathloss/num_obs'

# Load the aggregated results from the pickle file
with open(os.path.join(output_dir, "aggregated_results.pkl"), "rb") as f:
    aggregate_results = pickle.load(f)

# Load the saved plot from the pickle file
plot_pickle_file = "jammer_localization_grouped_boxplot.pkl"  # Replace with your pickle file path
with open(os.path.join(output_dir, plot_pickle_file), "rb") as f:
    fig = pickle.load(f)

# Extract the axes from the figure
axes = fig.axes[0]

# Get x-axis values and label
x_axis_values = [tick.get_text() for tick in axes.get_xticklabels()]
x_label = axes.get_xlabel()

# Prepare data for plotting
plot_data = []
for value, mc_results in zip(x_axis_values, aggregate_results):
    for result in mc_results:
        plot_data.append({"Value": value, "Error": result["jam_loc_error_pl"], "Type": "PL"})
        plot_data.append({"Value": value, "Error": result["jam_loc_error_apbm"], "Type": "APBM"})

# Convert to a Pandas DataFrame for Seaborn
df = pd.DataFrame(plot_data)

# Set the color palette for the error types
palette = {"PL": "#FFC107", "APBM": "#1E88E5"}
palette_points = {"PL": "black", "APBM": "black"}

# Resize the figure for a single-column layout on letter paper (~3.5 inches width)
plt.figure(figsize=(5, 2.5))  # Adjust the width and height to maintain proportions

# Generate the grouped boxplot
sns.boxplot(
    x="Value", 
    y="Error", 
    hue="Type", 
    data=df, 
    palette=palette, 
    showmeans=True
)

# Add stripplot for individual points (optional for better visualization)
sns.stripplot(
    x="Value", 
    y="Error", 
    hue="Type", 
    data=df, 
    palette=palette_points, 
    size=3, 
    jitter=True, 
    dodge=True, 
    alpha=0.6,
    marker="o"
)

# Avoid duplicate legends from the stripplot
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[:2], labels[:2], fontsize=10)

# Add labels and grid
plt.xlabel(x_label, fontsize=10)
plt.ylabel(r"RMSE$_{\theta}$ (m)", fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adjust layout for better fit within the column
plt.tight_layout()

# Save the plot as PNG
output_path = os.path.join(output_dir, "jammer_localization_grouped_boxplot_PL_APBM_column.png")
plt.savefig(output_path, dpi=300)

# Save the figure as a pickle file
output_path_pkl = os.path.join(output_dir, "jammer_localization_grouped_boxplot_PL_APBM_column.pkl")
with open(output_path_pkl, 'wb') as f:
    pickle.dump(plt.gcf(), f)

# Show the plot
plt.show()