import numpy as np
import torch
import torch.nn as nn
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative


def pad_lists_with_nan(list_of_lists):
    """
    Pads shorter lists with np.nan to make all lists the same length.

    Parameters:
    - list_of_lists (list of lists): A list where each element is a list of values (e.g., losses).

    Output:
    - padded_lists (np.array): A 2D numpy array with lists padded with np.nan to match the length of the longest list.
    """
    max_length = max(len(lst) for lst in list_of_lists)  # Find the maximum length among the lists
    padded_lists = [lst + [np.nan] * (max_length - len(lst)) for lst in list_of_lists]  # Pad with np.nan
    return np.array(padded_lists)


def plot_train_test_loss(train_losses_per_round, test_losses_per_round, pl_or_apbm_or_nn):
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
        title=f'Training Loss and Test Loss per round for the {"Pathloss Model" if pl_or_apbm_or_nn == "pl" else "APBM Model" if pl_or_apbm_or_nn == "apbm" else "NN initialization"}',
        xaxis_title='Round',
        yaxis_title='Loss',
        legend_title='Loss Type',
        template='plotly_white'
    )

    # Ensure the output folder exists; create it if it doesn't
    output_folder = '/Users/marionajaramillocivill/Documents/GitHub/jammerLocalization/apbm/plots_output'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Define the output file path and save the plot as an HTML file
    output_file = os.path.join(output_folder, 'train_test_loss.html')
    fig.write_html(output_file)

    # Display the plot in the browser
    fig.show()
    
def plot_test_field(model, test_loader, round):
    """
    Generates and saves a 3D field visualization of the model's output.

    Parameters:
    - model (nn.Module): The trained model to be visualized.
    - t (int or str): Identifier for the output file (e.g., an index or timestamp).

    Output:
    - Saves an HTML file containing a 3D plot of the model's output surface.
    """
    model.eval()  # Set the model to evaluation mode
    x = np.linspace(0, 100, 100)  # Define a grid range for x-axis
    y = np.linspace(0, 100, 100)  # Define a grid range for y-axis
    X, Y = np.meshgrid(x, y)  # Create a mesh grid for plotting
    Z = np.zeros(X.shape)  # Initialize Z (output values) with zeros

    # Disable gradient computation for visualization
    with torch.no_grad():
        for i in range(len(X)):
            for j in range(len(Y)):
                # Predict the model's output for each (x, y) point
                Z[i, j] = model(torch.tensor([[X[i, j], Y[i, j]]]).float()).item()

    # Create a 3D plot using Plotly
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
    fig.update_layout(
        title='3D Surface Plot',
        autosize=False,
        width=500,
        height=500,
        margin=dict(l=65, r=50, b=65, t=90)
    )
    
    # Extract points and measurements from the test_loader
    points = []
    measurements = []
    for data, target in test_loader:
        points.append(data.numpy())
        measurements.append(target.numpy())

    points = np.concatenate(points, axis=0)
    measurements = np.concatenate(measurements, axis=0)

    # Add scatter plot of the points and measurements
    fig.add_trace(go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=measurements,
        mode='markers',
        marker=dict(size=4, color='red'),
        name='Test Points'
    ))

    # Save the 3D plot as an HTML file
    fig.write_html(f'figs/field_{round}.html')
    
    fig.show()
    
def visualize_3d_model_output(model, train_loader_splitted, test_loader, true_jam_loc, predicted_jam_loc, t, train_or_test, pl_or_apbm_or_nn):
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

    fig.add_trace(go.Surface(z=Z, x=X, y=Y, name=f'{"Pathloss" if pl_or_apbm_or_nn == "pl" else "APBM" if pl_or_apbm_or_nn == "apbm" else "NN"}'))
    # fig.add_trace(go.Surface(z=Z_nn, x=X, y=Y, colorscale='Oranges', name='NN'))

    # Update layout for better visualization
    fig.update_layout(
        title=f'3D Surface Plot: {"Pathloss Model" if pl_or_apbm_or_nn == "pl" else "APBM Model" if pl_or_apbm_or_nn == "apbm" else "NN initialization"}',
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
        line=dict(color='blue', width=6, dash='dash'),
        name='True Jammer Location'
    ))

    # Add vertical line for predicted jammer location
    fig.add_trace(go.Scatter3d(
        x=[predicted_jam_loc[0], predicted_jam_loc[0]],  
        y=[predicted_jam_loc[1], predicted_jam_loc[1]],  
        z=[Z.min(), Z.max()+10.0],  # Line from ground level to the top of the Z axis
        mode='lines',
        line=dict(color='orange', width=6, dash='dash'),
        name='Predicted Jammer Location'
    ))

    # Save the 3D plot as an HTML file
    output_path = 'figs'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    fig.write_html(f'{output_path}/field_{train_or_test}.html')
    fig.show()