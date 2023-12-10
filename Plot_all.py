import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define the paths and files
paper_results_path = 'paper_results/'
data_slim_path = 'data-slim/lwc/'
prediction_files = ["decision-tree.csv", "linear-regression.csv", "xgboost.csv", "mcmc-mlp.csv"]  # Removed random-forest.csv
label_file = '20130515_lwc.csv'
std_file = 'mcmc-mlp_std.csv'  # Standard deviation file for mcmc-mlp

# Create a directory for the comparison images
output_folder = 'prediction-compare'
os.makedirs(output_folder, exist_ok=True)

# Define colors, line styles, and markers for each method
colors_and_styles = {
    "decision-tree": ('grey', '--', 'o', 0.8),  # Circle
    "linear-regression": ('orange', '--', '^', 0.8),  # Triangle
    "xgboost": ('green', '--', 'D', 0.8),  # Diamond
    "mcmc-mlp": ('red', '-', None, 1.0)  # No marker for mcmc-mlp, full opacity
}

# Function to read and process each file
def read_and_process(file, path, skip_index_column=True):
    with open(os.path.join(path, file), 'r') as f:
        timestamps = f.readline().strip().split(',')[2:]
    data = pd.read_csv(os.path.join(path, file), skiprows=1, header=None)
    if skip_index_column:
        heights = data.iloc[:, 1].values
        predictions = data.iloc[:, 2:]
    else:
        heights = data.iloc[:, 0].values
        predictions = data.iloc[:, 1:]
    return timestamps, heights, predictions

# Read label and std data
_, label_heights, label_data = read_and_process(label_file, data_slim_path, skip_index_column=False)
_, std_heights, std_data = read_and_process(std_file, paper_results_path)

# Extract timestamps from one of the prediction files
timestamps, _, _ = read_and_process(prediction_files[-1], paper_results_path)  # mcmc-mlp for timestamps

# Store data from each prediction file
all_data = {}
for file in prediction_files:
    _, heights, predictions = read_and_process(file, paper_results_path)
    all_data[file.split('.')[0]] = predictions

# Function to format timestamps for filenames
def format_timestamp(ts):
    return ts.replace('/', '-').replace(':', '-').replace(' ', '_')

# Set a fixed ylim
ylim_max = 0.1

# Plot and save comparison images
for i, timestamp in enumerate(timestamps):
    formatted_timestamp = format_timestamp(timestamp)
    plt.figure(figsize=(8, 4))  # Smaller figure size
    for method, predictions in all_data.items():
        color, style, marker, alpha = colors_and_styles[method]
        if i < predictions.shape[1]:
            plt.plot(label_heights, predictions.iloc[:, i], label=method, color=color, linestyle=style, marker=marker, alpha=alpha)
    # Plot label data (Microwave radiometer) with specific style
    if i < label_data.shape[1]:
        plt.plot(label_heights, label_data.iloc[:, i], label='Microwave radiometer', color='blue', linestyle='-', linewidth=2)
    # Plot std as transparent range for mcmc-mlp
    if i < std_data.shape[1]:
        plt.fill_between(label_heights, all_data['mcmc-mlp'].iloc[:, i] - std_data.iloc[:, i],
                         all_data['mcmc-mlp'].iloc[:, i] + std_data.iloc[:, i], color='red', alpha=0.2)
    plt.legend()
    plt.xlabel('Height (m)')
    plt.ylabel('LWC (g/m³)')
    plt.title(f'Prediction Comparison at {timestamp}')
    plt.ylim(0, ylim_max)
    plt.savefig(os.path.join(output_folder, f'{formatted_timestamp}.png'))
    plt.close()

print("All comparison images have been saved.")
