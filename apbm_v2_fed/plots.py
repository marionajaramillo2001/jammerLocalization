import numpy as np
import torch
import torch.nn as nn
import os
import pickle
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative
from statsmodels.distributions.empirical_distribution import ECDF
from matplotlib import pyplot as plt
import seaborn as sns



def plot_train_test_loss(train_losses_per_round, test_losses_per_round, pl_or_apbm_or_nn, folder=None, mc_run=None):
    """
    Plots the training losses per round and the global test loss using Plotly.

    Parameters:
    - train_losses_per_round (list): A list containing training losses for each round.
    - global_test_loss (float): The global test loss value.

    Output:
    - Displays the plot using Plotly and saves it as an HTML file.
    """
    # Create a Plotly figure
    fig = go.Figure()

    # Add trace for training loss per round
    fig.add_trace(go.Scatter(
        x=list(range(1, len(train_losses_per_round) + 1)),
        y=train_losses_per_round,
        mode='lines+markers',
        name='Training Loss per Round',
        line=dict(color='blue')
    ))
    
    # Add trace for testing loss per round
    fig.add_trace(go.Scatter(
        x=list(range(1, len(test_losses_per_round) + 1)),
        y=test_losses_per_round,
        mode='lines+markers',
        name='Test Loss per Round',
        line=dict(color='red')
    ))

    # Add trace for final test loss as a horizontal line
    fig.add_trace(go.Scatter(
        x=[1, len(test_losses_per_round)],
        y=[test_losses_per_round[-1], test_losses_per_round[-1]],
        mode='lines',
        name='Global Test Loss',
        line=dict(color='black', dash='dash')
    ))

    # Update the layout for better visualization
    fig.update_layout(
        title=f'Training Loss and Test Loss per round for the {"Pathloss Model" if pl_or_apbm_or_nn == "PL" else "APBM Model" if pl_or_apbm_or_nn == "APBM" else "NN initialization"}',
        xaxis_title='Round',
        yaxis_title='Loss',
        legend_title='Loss Type',
        template='plotly_white'
    )
    
    # Save the fig as an HTML file
    output_path = os.path.join(folder, f'train_test_loss_{pl_or_apbm_or_nn}_mc_run_{mc_run}.html')

    fig.write_html(output_path)

    
def visualize_3d_model_output(model, train_loader_splitted, test_loader, theta_init, true_jam_loc, predicted_jam_loc, t, train_or_test, pl_or_apbm_or_nn, folder=None, mc_run=None):
    """
    Visualizes a 3D surface plot of the model's output and adds test points as markers.

    Parameters:
    - model (nn.Module): The trained model to be visualized.
    - test_loader (DataLoader): DataLoader containing test data points.
    - t (int or str): Identifier for the output file (e.g., an index or timestamp).

    Output:
    - Saves an HTML file containing a 3D plot of the model's output surface.
    """
    # Extract points and measurements from the test_loader
    if train_or_test == 'train':
        num_nodes = len(train_loader_splitted)
        points_per_node, measurements_per_node = [], []
        for train_loader in train_loader_splitted:
            points_one_node, measurements_one_node = [], []
            for data, target in train_loader:
                points_one_node.append(data.numpy())
                measurements_one_node.append(target.numpy())
            points_per_node.append(np.concatenate(points_one_node, axis=0))
            measurements_per_node.append(np.concatenate(measurements_one_node, axis=0))
        points = np.concatenate(points_per_node, axis=0)
    elif train_or_test == 'test':
        points, measurements = [], []
        for data, target in test_loader:
            points.append(data.numpy())
            measurements.append(target.numpy())
        points = np.concatenate(points, axis=0)
        measurements_test = np.concatenate(measurements, axis=0)
    
    min_x, max_x = int(np.min(points[:, 0])), int(np.max(points[:, 0]))
    min_y, max_y = int(np.min(points[:, 1])), int(np.max(points[:, 1]))
    
    # Generate grid ranges for x and y
    grid_range_x = np.linspace(min_x - 1, max_x + 1, max_x - min_x + 3)  # +3 to include boundaries
    grid_range_y = np.linspace(min_y - 1, max_y + 1, max_y - min_y + 3)

    # Create a 2D grid combining all points
    grid_x, grid_y = torch.meshgrid(torch.tensor(grid_range_x), torch.tensor(grid_range_y), indexing="ij")
    grid_tensor = torch.stack([grid_x, grid_y], dim=-1)  # Shape (N, N, 2)

    # Flatten the grid for batch evaluation
    grid_points = grid_tensor.view(-1, 2)  # Shape (N^2, 2)
    
    # Compute Z values for all grid points
    with torch.no_grad():
        Z = model(grid_points.float()).view(grid_x.shape)  # Reshape back to (N, N)

    # Convert to NumPy arrays
    X = grid_x.numpy()
    Y = grid_y.numpy()
    Z = Z.numpy()

    # Create a 3D plot using Plotly
    fig = go.Figure()

    fig.add_trace(go.Surface(z=Z, x=X, y=Y, name=f'{"Pathloss" if pl_or_apbm_or_nn == "PL" else "APBM" if pl_or_apbm_or_nn == "APBM" else "NN"}'))
    # fig.add_trace(go.Surface(z=Z_nn, x=X, y=Y, colorscale='Oranges', name='NN'))

    # Update layout for better visualization
    fig.update_layout(
        title=f'3D Surface Plot: {"Pathloss Model" if pl_or_apbm_or_nn == "PL" else "APBM Model" if pl_or_apbm_or_nn == "APBM" else "NN initialization"}',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
        ),
        autosize=False,
        width=900,
        height=900,
        margin=dict(l=65, r=50, b=65, t=90)
    )
    
    # Add scatter plot of the points and measurements
    discrete_colors = qualitative.Dark24

    if train_or_test == 'train':
        for i in range(num_nodes):
            fig.add_trace(go.Scatter3d(
                x=points_per_node[i][:, 0],  
                y=points_per_node[i][:, 1],  
                z=measurements_per_node[i].flatten(), 
                mode='markers',
                marker=dict(
                    size=4,
                    color=discrete_colors[i % len(discrete_colors)],  # Cycle through colors
                ),
                name=f'Training Points Node {i+1}'  # Unique names for each node
            ))
    elif train_or_test == 'test':
        fig.add_trace(go.Scatter3d(
            x=points[:, 0],  
            y=points[:, 1],  
            z=measurements_test.flatten(), 
            mode='markers',
            marker=dict(
                size=4,
                color='black',  # Black color for test markers
                symbol='cross'  # Use 'cross' or any other symbol for differentiation
            ),
            name='Testing Points'
        ))
    
    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.01
    ))
    
    # Add vertical line for true jammer location
    fig.add_trace(go.Scatter3d(
        x=[true_jam_loc[0], true_jam_loc[0]],  
        y=[true_jam_loc[1], true_jam_loc[1]],  
        z=[Z.min(), Z.max()+10.0],  # Line from ground level to the top of the Z axis
        mode='lines',
        line=dict(color='#1E88E5', width=6, dash='dash'),
        name='True Jammer Location'
    ))

    # Add vertical line for predicted jammer location
    fig.add_trace(go.Scatter3d(
        x=[predicted_jam_loc[0], predicted_jam_loc[0]],  
        y=[predicted_jam_loc[1], predicted_jam_loc[1]],  
        z=[Z.min(), Z.max()+10.0],  # Line from ground level to the top of the Z axis
        mode='lines',
        line=dict(color='#FFC107', width=6, dash='dash'),
        name=f'Predicted Jammer Location after {"Pathloss" if pl_or_apbm_or_nn == "PL" else "APBM"} Training'
    ))
    
    
    # Add vertical line for theta_init location
    fig.add_trace(go.Scatter3d(
        x=[theta_init[0], theta_init[0]],  
        y=[theta_init[1], theta_init[1]],  
        z=[Z.min(), Z.max()+10.0],  # Line from ground level to the top of the Z axis
        mode='lines',
        line=dict(color='#004D40', width=6, dash='dash'),
        name='Predicted Jammer Location after NN Initialization'
    ))
    

    # Save the 3D plot as an HTML file
    output_path = os.path.join(folder, f'3d_surface_{train_or_test}_model_mc_run_{mc_run}.html')

    fig.write_html(output_path)
        
def plot_ECDF(mc_results, output_dir):
    """
    Plots the empirical cumulative distribution function (ECDF) of the given data.
    """

    # Extract the jam_loc_error from the Monte Carlo results
    jam_loc_errors = [result['jam_loc_error'] for result in mc_results]

    # Fit the empirical cumulative distribution function (ECDF)
    ecdf = ECDF(jam_loc_errors)

    # Generate the ECDF plot
    plt.figure(figsize=(10, 4))
    plt.plot(ecdf.x, ecdf.y, marker='o', linestyle='-', label='Empirical CDF', markersize=4)

    plt.ylim(0, 1)

    # Add labels and title
    plt.xlabel(r"RMSE$_{\theta}$ (m)", fontsize=12)
    plt.ylabel('Empirical CDF', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    # Add text with statistics
    mean_error = np.mean(jam_loc_errors)
    std_error = np.std(jam_loc_errors)
    textstr = (
        f"Mean Error: {mean_error:.6f} m\n"
        f"Std Error: {std_error:.6f} m\n"
        f"Num. MC Runs: {len(jam_loc_errors)}"
    )
    plt.text(0.05, 0.4, textstr, transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()  # Automatically adjusts the spacing to prevent label cutoff
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Save and display the plot
    output_path = os.path.join(output_dir, "jammer_localization_ecdf.png")
    plt.savefig(output_path)
    
    output_path_pkl = os.path.join(output_dir, "jammer_localization_ecdf.pkl")
    with open(output_path_pkl, 'wb') as f:
        pickle.dump(plt.gcf(), f)
        
    # Save the Monte Carlo results as a pickle file
    output_path_mc_results_pkl = os.path.join(output_dir, "mc_results.pkl")
    with open(output_path_mc_results_pkl, 'wb') as f:
        pickle.dump(mc_results, f)
    

def plot_boxplot(x_axis_values, aggregate_results, output_dir, x_label):
    """
    Plots a boxplot for the given data.
    """
    # Prepare data for plotting
    plot_data = []
    for value, mc_results in zip(x_axis_values, aggregate_results):
        for result in mc_results:
            # Extract jam_loc_error for each Monte Carlo run
            plot_data.append({"Value": value, "Error": result["jam_loc_error"]})

    # Convert to a Pandas DataFrame for Seaborn
    df = pd.DataFrame(plot_data)

    # Generate the boxplot
    plt.figure(figsize=(10, 4))
    sns.boxplot(x="Value", y="Error", data=df, palette="Set2", showmeans=True)
    sns.stripplot(x="Value", y="Error", data=df, color="black", size=4, jitter=True, alpha=0.6)

    # Add labels
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(r"RMSE$_{\theta}$ (m)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the plot as PNG
    output_path = os.path.join(output_dir, "jammer_localization_boxplot.png")
    plt.savefig(output_path)

    # Save the figure as a pickle file
    output_path_pkl = os.path.join(output_dir, "jammer_localization_boxplot.pkl")
    with open(output_path_pkl, 'wb') as f:
        pickle.dump(plt.gcf(), f)
        
    # Save the aggregated results as a pickle file
    output_path_agg_pkl = os.path.join(output_dir, "aggregated_results.pkl")
    with open(output_path_agg_pkl, 'wb') as f:
        pickle.dump(aggregate_results, f)