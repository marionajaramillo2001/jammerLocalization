import numpy as np
import torch
import torch.nn as nn
import os
import pandas as pd
import plotly.graph_objects as go

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


def plot_train_test_loss(train_losses_per_epoch, global_test_loss):
    """
    Plots the training losses per epoch and the global test loss using Plotly.

    Parameters:
    - train_losses_per_epoch (list): A list containing training losses for each epoch.
    - global_test_loss (float): The global test loss value.

    Output:
    - Displays the plot using Plotly and saves it as an HTML file.
    """
    # Create a Plotly figure
    fig = go.Figure()

    # Add trace for training loss per epoch
    fig.add_trace(go.Scatter(
        x=list(range(1, len(train_losses_per_epoch) + 1)),
        y=train_losses_per_epoch,
        mode='lines+markers',
        name='Training Loss per Epoch',
        line=dict(color='blue')
    ))

    # Add trace for global test loss as a horizontal line
    fig.add_trace(go.Scatter(
        x=[1, len(train_losses_per_epoch)],
        y=[global_test_loss, global_test_loss],
        mode='lines',
        name='Global Test Loss',
        line=dict(color='red', dash='dash')
    ))

    # Update the layout for better visualization
    fig.update_layout(
        title='Training Loss per Epoch and Global Test Loss',
        xaxis_title='Epoch',
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
    
def visualize_3d_model_output(model_nn, model_pl, train_loader, test_loader, true_jam_loc, predicted_jam_loc, t):
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
    points_train, measurements_train, points_test, measurements_test = [], [], [], []
    
    for data, target in train_loader:
        points_train.append(data.numpy())
        measurements_train.append(target.numpy())

    # Concatenate points and measurements
    points_train = np.concatenate(points_train, axis=0)
    measurements_train = np.concatenate(measurements_train, axis=0)
    
    for data, target in test_loader:
        points_test.append(data.numpy())
        measurements_test.append(target.numpy())

    # Concatenate points and measurements
    points_test = np.concatenate(points_test, axis=0)
    measurements_test = np.concatenate(measurements_test, axis=0)
    
    # Concatenate training and testing points
    all_points = np.concatenate((points_train, points_test), axis=0)
    min_x, max_x = int(np.min(all_points[:, 0])), int(np.max(all_points[:, 0]))
    min_y, max_y = int(np.min(all_points[:, 1])), int(np.max(all_points[:, 1]))
    
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
        # Full model (combination of pathloss and NN contributions)
        Z = model_pl(grid_points.float()).view(grid_x.shape)  # Reshape back to (N, N)
        Z_nn = model_nn(grid_points.float()).view(grid_x.shape)  # Reshape back to (N, N)

    # Convert grid_x and grid_y to NumPy arrays
    X = grid_x.numpy()
    Y = grid_y.numpy()

    # Convert Z values to NumPy arrays
    Z = Z.numpy()

    # Create a 3D plot using Plotly
    fig = go.Figure()

    fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='Blues', name='Pathloss'))
    fig.add_trace(go.Surface(z=Z_nn, x=X, y=Y, colorscale='Oranges', name='NN'))

    # Update layout for better visualization
    fig.update_layout(
        title='3D Surface Plot: Combined, Pathloss, and NN Contributions',
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
    fig.add_trace(go.Scatter3d(
        x=points_train[:, 0],  
        y=points_train[:, 1],  
        z=measurements_train.flatten(), 
        mode='markers',
        marker=dict(size=4),
        name='Training Points'
    ))
    
    fig.add_trace(go.Scatter3d(
        x=points_test[:, 0],  
        y=points_test[:, 1],  
        z=measurements_test.flatten(), 
        mode='markers',
        marker=dict(size=4),
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
        line=dict(color='green', width=6, dash='dash'),
        name='True Jammer Location Line'
    ))

    # Add vertical line for predicted jammer location
    fig.add_trace(go.Scatter3d(
        x=[predicted_jam_loc[0], predicted_jam_loc[0]],  
        y=[predicted_jam_loc[1], predicted_jam_loc[1]],  
        z=[Z.min(), Z.max()+10.0],  # Line from ground level to the top of the Z axis
        mode='lines',
        line=dict(color='orange', width=6, dash='dash'),
        name='Predicted Jammer Location Line'
    ))

    # Save the 3D plot as an HTML file
    output_path = 'figs'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    fig.write_html(f'{output_path}/field_{t}.html')
    fig.show()