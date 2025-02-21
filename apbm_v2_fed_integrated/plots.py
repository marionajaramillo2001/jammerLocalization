import numpy as np
import torch
import torch.nn as nn
import os
import pickle
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import qualitative
from statsmodels.distributions.empirical_distribution import ECDF
from matplotlib import pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


def plot_horizontal_visualization_with_nodes_v2_matplotlib(
    model_nn, model_pl, model_apbm, train_loader_splitted, test_loader,
    true_jam_loc, nn_predicted_jam_loc, pl_predicted_jam_loc, apbm_predicted_jam_loc,
    folder=None, mc_run=None
):
    """
    Generates a horizontal layout with 4 3D subplots in a single row for Letter size visualization using Matplotlib.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    from mpl_toolkits.mplot3d import Axes3D

    # Prepare training points split by nodes
    training_points_per_node, training_measurements_per_node = [], []
    for train_loader in train_loader_splitted:
        node_points, node_measurements = [], []
        for data, target in train_loader:
            node_points.append(data.numpy())
            node_measurements.append(target.numpy())
        training_points_per_node.append(np.concatenate(node_points, axis=0))
        training_measurements_per_node.append(np.concatenate(node_measurements, axis=0))
    training_points = np.concatenate(training_points_per_node, axis=0)
    training_measurements = np.concatenate(training_measurements_per_node, axis=0)

    # Prepare testing points
    testing_points, testing_measurements = [], []
    for data, target in test_loader:
        testing_points.append(data.numpy())
        testing_measurements.append(target.numpy())
    testing_points = np.concatenate(testing_points, axis=0)
    testing_measurements = np.concatenate(testing_measurements, axis=0)

    # Define grid for 3D field visualization
    min_x, max_x = int(np.min(training_points[:, 0])), int(np.max(training_points[:, 0]))
    min_y, max_y = int(np.min(training_points[:, 1])), int(np.max(training_points[:, 1]))

    # Round min and max values to the nearest 100
    min_x = (min_x // 100) * 100  # Round down
    max_x = ((max_x + 99) // 100) * 100  # Round up
    min_y = (min_y // 100) * 100  # Round down
    max_y = ((max_y + 99) // 100) * 100  # Round up
    
    # Determine min and max values for z-axis (RSS values)
    min_z = min(np.min(training_measurements), np.min(testing_measurements))
    max_z = max(np.max(training_measurements), np.max(testing_measurements))

    # Round min and max z values to the nearest 20
    min_z = (min_z // 20) * 20  # Round down
    max_z = ((max_z + 19) // 20) * 20  # Round up

    grid_x, grid_y = np.meshgrid(
        np.linspace(min_x, max_x, 100),
        np.linspace(min_y, max_y, 100)
    )
    grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]

    # Evaluate the model on the grid
    with torch.no_grad():
        field_values_nn = model_nn(torch.tensor(grid_points, dtype=torch.float32)).numpy().reshape(grid_x.shape)
        field_values_pl = model_pl(torch.tensor(grid_points, dtype=torch.float32)).numpy().reshape(grid_x.shape)
        field_values_apbm = model_apbm(torch.tensor(grid_points, dtype=torch.float32)).numpy().reshape(grid_x.shape)

    # Plotting
    fig = plt.figure(figsize=(14, 5))  # Adjusted size for better spacing
    axes = [fig.add_subplot(1, 4, i + 1, projection='3d') for i in range(4)]
    titles = ["Training Points", "Initial NN Field", "Pathloss Field", "APBM Field"]

    # Colors for training nodes
    node_colors = plt.cm.get_cmap("tab10", len(training_points_per_node))

    # Legend elements
    legend_elements = [
        plt.Line2D([0], [0], color=node_colors(i), marker='o', linestyle='None', markersize=6, label=f"Training Node {i + 1}")
        for i in range(len(training_points_per_node))
    ]
    legend_elements += [
        plt.Line2D([0], [0], color='black', marker='o', linestyle='None', markersize=6, label="Testing Points"),
        plt.Line2D([0], [0], color='red', lw=1, linestyle='--', label="True Jammer Location"),
        plt.Line2D([0], [0], color='#004D40', lw=1, linestyle='--', label="NN Predicted Jammer Location"),
        plt.Line2D([0], [0], color='#FFC107', lw=1, linestyle='--', label="PL Predicted Jammer Location"),
        plt.Line2D([0], [0], color='#1E88E5', lw=1, linestyle='--', label="APBM Predicted Jammer Location"),
    ]

    # Data for the plots
    fields = [None, field_values_nn, field_values_pl, field_values_apbm]
    predicted_jams = [None, nn_predicted_jam_loc, pl_predicted_jam_loc, apbm_predicted_jam_loc]

    for i, ax in enumerate(axes):
        if i == 0:
            for j, node_points in enumerate(training_points_per_node):
                ax.scatter(
                    node_points[:, 0], node_points[:, 1], training_measurements_per_node[j].flatten(),
                    color=node_colors(j), label=f"Training Node {j + 1}", s=10
                )
        else:
            ax.plot_surface(grid_x, grid_y, fields[i], cmap="plasma", alpha=1.0)

        # Testing points
        if i != 0:
            ax.scatter(
                testing_points[:, 0], testing_points[:, 1], testing_measurements.flatten(),
                color="black", s=10, marker="x", alpha=1.0
            )

        # Jammer locations
        ax.plot([true_jam_loc[0]], [true_jam_loc[1]], [min_z, max_z+25], 'r--', lw=1.5)
        if predicted_jams[i] is not None:
            ax.plot(
                [predicted_jams[i][0]], [predicted_jams[i][1]], [min_z, max_z+25],
                color=["#004D40", "#FFC107", "#1E88E5"][i - 1], linestyle='--', lw=1.5
            )

        ax.set_title(titles[i], fontsize=12)
        ax.set_xlabel(r"$\theta_1$ (m)", fontsize=10, labelpad=2)
        ax.set_ylabel(r"$\theta_2$ (m)", fontsize=10, labelpad=2)
        ax.set_zlabel("RSS (dBW)", fontsize=10, labelpad=0)
        ax.set_xticks(np.arange(min_x, max_x + 1, 250))
        ax.set_yticks(np.arange(min_y, max_y + 1, 250))
        ax.set_zticks(np.linspace(min_z, max_z, 5))
        ax.tick_params(axis='both', which='major', labelsize=8, pad=2)  # Reduced tick padding
        ax.tick_params(axis='z', pad=0)  # Specifically adjust z-tick padding

        # Adjust the view angle for better visibility
        ax.view_init(elev=20, azim=120)
        ax.dist = 12

    def sync_view(event):
        """Synchronize all subplots to the view angle of the first subplot."""
        elev, azim = axes[0].elev, axes[0].azim
        for ax in axes:
            ax.view_init(elev, azim)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', sync_view)

    # Unified legend
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=5,
        fontsize=12,
        bbox_to_anchor=(0.5, 0.1)
    )

    # Adjust layout to minimize overlap
    plt.subplots_adjust(left=0.00, right=0.97, top=1.0, bottom=0.13, wspace=0.105, hspace=0.0)
    

    if folder:
        output_path = os.path.join(folder, f"3d_surface_model_mc_run_{mc_run}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
        
    plt.show()
    
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
    output_path = os.path.join(folder, f'3d_surface_{pl_or_apbm_or_nn}_{train_or_test}_model_mc_run_{mc_run}.html')

    fig.write_html(output_path)
    
def plot_horizontal_visualization_with_nodes(
    model_nn, model_pl, model_apbm, train_loader_splitted, test_loader, theta_init, true_jam_loc, predicted_jam_loc, folder=None, mc_run=None):
    """
    Generates a horizontal layout with 4 3D subplots for A4 size visualization.

    Parameters:
    - model_nn: Neural network model.
    - model: Trained model (PL or APBM).
    - train_loader_splitted: Training data loaders (split by nodes).
    - test_loader: Test data loader.
    - theta_init: Initial NN-based jammer location.
    - true_jam_loc: True jammer location.
    - predicted_jam_loc: Predicted jammer location.
    - folder: Directory to save the output plot.
    - mc_run: Identifier for Monte Carlo run.
    """

    # Prepare training points split by nodes
    training_points_per_node, training_measurements_per_node = [], []
    for train_loader in train_loader_splitted:
        node_points, node_measurements = [], []
        for data, target in train_loader:
            node_points.append(data.numpy())
            node_measurements.append(target.numpy())
        training_points_per_node.append(np.concatenate(node_points, axis=0))
        training_measurements_per_node.append(np.concatenate(node_measurements, axis=0))
    training_points = np.concatenate(training_points_per_node, axis=0)

    # Prepare testing points
    testing_points, testing_measurements = [], []
    for data, target in test_loader:
        testing_points.append(data.numpy())
        testing_measurements.append(target.numpy())
    testing_points = np.concatenate(testing_points, axis=0)
    testing_measurements = np.concatenate(testing_measurements, axis=0)

    # Define grid for 3D field visualization
    min_x, max_x = int(np.min(training_points[:, 0])), int(np.max(training_points[:, 0]))
    min_y, max_y = int(np.min(training_points[:, 1])), int(np.max(training_points[:, 1]))
    grid_x, grid_y = np.meshgrid(
        np.linspace(min_x - 10, max_x + 10, 100),
        np.linspace(min_y - 10, max_y + 10, 100)
    )
    grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]

    # Evaluate the model on the grid
    with torch.no_grad():
        field_values_nn = model_nn(torch.tensor(grid_points, dtype=torch.float32)).numpy().reshape(grid_x.shape)
        field_values_pl = model_pl(torch.tensor(grid_points, dtype=torch.float32)).numpy().reshape(grid_x.shape)
        field_values_apbm = model_apbm(torch.tensor(grid_points, dtype=torch.float32)).numpy().reshape(grid_x.shape)

    # Add scatter plot of the points and measurements
    discrete_colors = qualitative.Dark24

    # Create a figure with 4 subplots
    fig = make_subplots(
        rows=1, cols=4,
        specs=[[{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]],
        subplot_titles=[
            "Training Points",
            "Initial NN Field",
            "Pathloss Field",
            "APBM Field with Points"
        ],
        horizontal_spacing=0.01,
    )
    
    for annotation in fig.layout.annotations:
        annotation.update(font=dict(size=14))  # Change font size to 10

    # Subplot 1: Training points (per node) + Testing points subplot 3 and 4 (for legend order)
    for i, node_points in enumerate(training_points_per_node):
        fig.add_trace(go.Scatter3d(
            x=node_points[:, 0],
            y=node_points[:, 1],
            z=training_measurements_per_node[i].flatten(),
            mode='markers',
            marker=dict(
                size=4,
                color=discrete_colors[i % len(discrete_colors)],
            ),
            name=f'Training Node {i+1}'
        ), row=1, col=1)
        
    fig.add_trace(go.Scatter3d(
        x=testing_points[:, 0], y=testing_points[:, 1], z=testing_measurements.flatten(),
        mode='markers',
        marker=dict(size=4, color='black', symbol='cross'),
        name='Testing Points',
        legendgroup='Testing Points'
    ), row=1, col=3)
        
    fig.add_trace(go.Scatter3d(
        x=testing_points[:, 0], y=testing_points[:, 1], z=testing_measurements.flatten(),
        mode='markers',
        marker=dict(size=4, color='black', symbol='cross'),
        name='Testing Points',
        legendgroup='Testing Points',
        showlegend=False  # Hide repeated legend
    ), row=1, col=4)

    # Subplot 2: Initial NN field
    fig.add_trace(go.Surface(
        z=field_values_nn,
        x=grid_x,
        y=grid_y,
        colorscale='Plasma',
        opacity=0.9,  # Transparent field
        showscale=False
    ), row=1, col=2)

    fig.add_trace(go.Scatter3d(
        x=[theta_init[0], theta_init[0]],
        y=[theta_init[1], theta_init[1]],
        z=[field_values_nn.min(), field_values_nn.max() + 10.0],
        mode='lines',
        line=dict(color='#004D40', width=6, dash='solid'),
        name='NN Predicted Jammer Location',
        legendgroup='NN'  # Grouped
    ), row=1, col=2)

    fig.add_trace(go.Scatter3d(
        x=[true_jam_loc[0], true_jam_loc[0]],
        y=[true_jam_loc[1], true_jam_loc[1]],
        z=[field_values_nn.min(), field_values_nn.max() + 10.0],
        mode='lines',
        line=dict(color='#1E88E5', width=6, dash='solid'),
        name='True Jammer Location',
        legendgroup='True Jammer'  # Grouped
    ), row=1, col=2)

    # Subplot 3: PL/APBM field
    fig.add_trace(go.Surface(
        z=field_values_pl,
        x=grid_x,
        y=grid_y,
        colorscale='Plasma',
        opacity=0.9,  # Transparent field
        showscale=False
    ), row=1, col=3)

    fig.add_trace(go.Scatter3d(
        x=[predicted_jam_loc[0], predicted_jam_loc[0]],
        y=[predicted_jam_loc[1], predicted_jam_loc[1]],
        z=[field_values_pl.min(), field_values_pl.max() + 10.0],
        mode='lines',
        line=dict(color='#FFC107', width=6, dash='solid'),
        name=f'PL Predicted Jammer Location',
        legendgroup='PL/APBM'  # Grouped
    ), row=1, col=3)

    fig.add_trace(go.Scatter3d(
        x=[theta_init[0], theta_init[0]],
        y=[theta_init[1], theta_init[1]],
        z=[field_values_pl.min(), field_values_pl.max() + 10.0],
        mode='lines',
        line=dict(color='#004D40', width=6, dash='solid'),
        name='NN Predicted Jammer Location',
        legendgroup='NN',  # Grouped
        showlegend=False  # Hide repeated legend
    ), row=1, col=3)

    fig.add_trace(go.Scatter3d(
        x=[true_jam_loc[0], true_jam_loc[0]],
        y=[true_jam_loc[1], true_jam_loc[1]],
        z=[field_values_pl.min(), field_values_pl.max() + 10.0],
        mode='lines',
        line=dict(color='#1E88E5', width=6, dash='solid'),
        name='True Jammer Location',
        legendgroup='True Jammer',  # Grouped
        showlegend=False  # Hide repeated legend
    ), row=1, col=3)

    # Subplot 4: PL/APBM field with testing points
    fig.add_trace(go.Surface(
        z=field_values_apbm,
        x=grid_x,
        y=grid_y,
        colorscale='Plasma',
        opacity=0.9,  # Transparent field
        showscale=False
    ), row=1, col=4)

    fig.add_trace(go.Scatter3d(
        x=[predicted_jam_loc[0], predicted_jam_loc[0]],
        y=[predicted_jam_loc[1], predicted_jam_loc[1]],
        z=[field_values_apbm.min(), field_values_apbm.max() + 10.0],
        mode='lines',
        line=dict(color='#D81B60', width=6, dash='solid'),
        name=f'APBM Predicted Jammer Location',
        legendgroup='PL/APBM'  # Grouped
    ), row=1, col=4)

    fig.add_trace(go.Scatter3d(
        x=[theta_init[0], theta_init[0]],
        y=[theta_init[1], theta_init[1]],
        z=[field_values_apbm.min(), field_values_apbm.max() + 10.0],
        mode='lines',
        line=dict(color='#004D40', width=6, dash='solid'),
        name='NN Predicted Jammer Location',
        legendgroup='NN',  # Grouped
        showlegend=False  # Hide repeated legend
    ), row=1, col=4)

    fig.add_trace(go.Scatter3d(
        x=[true_jam_loc[0], true_jam_loc[0]],
        y=[true_jam_loc[1], true_jam_loc[1]],
        z=[field_values_apbm.min(), field_values_apbm.max() + 10.0],
        mode='lines',
        line=dict(color='#1E88E5', width=6, dash='solid'),
        name='True Jammer Location',
        legendgroup='True Jammer',  # Grouped
        showlegend=False  # Hide repeated legend
    ), row=1, col=4)

    fig.update_layout(
        height=400, 
        width=1200,
        legend=dict(
            orientation="h",  # Vertical legend
            x=0.5,  # Align right
            y=-0.2,  # Center vertically
            xanchor="center",
            yanchor="bottom",
            traceorder="normal",
            bgcolor="rgba(255, 255, 255, 0.7)",  # Semi-transparent background
            font=dict(size=12)
        ),
        margin=dict(l=20, r=20, t=20, b=50),  # Reduce padding around the plots
    )
    
    for i in range(1, 5):
        fig.update_scenes(
            xaxis=dict(showgrid=True, showticklabels=False),  # Show grid but hide tick labels
            yaxis=dict(showgrid=True, showticklabels=False),  # Show grid but hide tick labels
            zaxis=dict(showgrid=True),  # Keep z-axis grid visible
            row=1,
            col=i
        )
        
    # Save and display the plot
    output_path = os.path.join(folder, f'3d_surface_model_mc_run_{mc_run}.png')
    fig.write_image(output_path, format="png", engine="kaleido", width=1600, height=800)
    print(f"3D plot saved to {output_path}")
    fig.show()
    
def plot_horizontal_visualization_with_nodes_v2(
    model_nn, model_pl, model_apbm, train_loader_splitted, test_loader, theta_init, true_jam_loc, predicted_jam_loc, folder=None, mc_run=None):
    """
    Generates a horizontal layout with 4 3D subplots for A4 size visualization.

    Parameters:
    - model_nn: Neural network model.
    - model: Trained model (PL or APBM).
    - train_loader_splitted: Training data loaders (split by nodes).
    - test_loader: Test data loader.
    - theta_init: Initial NN-based jammer location.
    - true_jam_loc: True jammer location.
    - predicted_jam_loc: Predicted jammer location.
    - folder: Directory to save the output plot.
    - mc_run: Identifier for Monte Carlo run.
    """

    # Prepare training points split by nodes
    training_points_per_node, training_measurements_per_node = [], []
    for train_loader in train_loader_splitted:
        node_points, node_measurements = [], []
        for data, target in train_loader:
            node_points.append(data.numpy())
            node_measurements.append(target.numpy())
        training_points_per_node.append(np.concatenate(node_points, axis=0))
        training_measurements_per_node.append(np.concatenate(node_measurements, axis=0))
    training_points = np.concatenate(training_points_per_node, axis=0)

    # Prepare testing points
    testing_points, testing_measurements = [], []
    for data, target in test_loader:
        testing_points.append(data.numpy())
        testing_measurements.append(target.numpy())
    testing_points = np.concatenate(testing_points, axis=0)
    testing_measurements = np.concatenate(testing_measurements, axis=0)

    # Define grid for 3D field visualization
    min_x, max_x = int(np.min(training_points[:, 0])), int(np.max(training_points[:, 0]))
    min_y, max_y = int(np.min(training_points[:, 1])), int(np.max(training_points[:, 1]))
    grid_x, grid_y = np.meshgrid(
        np.linspace(min_x - 10, max_x + 10, 100),
        np.linspace(min_y - 10, max_y + 10, 100)
    )
    grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]

    # Evaluate the model on the grid
    with torch.no_grad():
        field_values_nn = model_nn(torch.tensor(grid_points, dtype=torch.float32)).numpy().reshape(grid_x.shape)
        field_values_pl = model_pl(torch.tensor(grid_points, dtype=torch.float32)).numpy().reshape(grid_x.shape)
        field_values_apbm = model_apbm(torch.tensor(grid_points, dtype=torch.float32)).numpy().reshape(grid_x.shape)

    # Add scatter plot of the points and measurements
    discrete_colors = qualitative.Dark24

    # Create a figure with 4 subplots
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'surface'}, {'type': 'surface'}], [{'type': 'surface'}, {'type': 'surface'}]],
        subplot_titles=[
            "Training Points",
            "Initial NN Field",
            "Pathloss Field",
            "APBM Field with Points"
        ],
        horizontal_spacing=0.0001,
        vertical_spacing=0.05,

    )
    
    for annotation in fig.layout.annotations:
        annotation.update(font=dict(size=14))  # Change font size to 10

    # Subplot 1: Training points (per node) + Testing points subplot 3 and 4 (for legend order)
    for i, node_points in enumerate(training_points_per_node):
        fig.add_trace(go.Scatter3d(
            x=node_points[:, 0],
            y=node_points[:, 1],
            z=training_measurements_per_node[i].flatten(),
            mode='markers',
            marker=dict(
                size=4,
                color=discrete_colors[i % len(discrete_colors)],
            ),
            name=f'Training Node {i+1}'
        ), row=1, col=1)
        
    fig.add_trace(go.Scatter3d(
        x=testing_points[:, 0], y=testing_points[:, 1], z=testing_measurements.flatten(),
        mode='markers',
        marker=dict(size=4, color='black', symbol='cross'),
        name='Testing Points',
        legendgroup='Testing Points'
    ), row=2, col=1)
        
    fig.add_trace(go.Scatter3d(
        x=testing_points[:, 0], y=testing_points[:, 1], z=testing_measurements.flatten(),
        mode='markers',
        marker=dict(size=4, color='black', symbol='cross'),
        name='Testing Points',
        legendgroup='Testing Points',
        showlegend=False  # Hide repeated legend
    ), row=2, col=2)

    # Subplot 2: Initial NN field
    fig.add_trace(go.Surface(
        z=field_values_nn,
        x=grid_x,
        y=grid_y,
        colorscale='Plasma',
        opacity=0.9,  # Transparent field
        showscale=False
    ), row=1, col=2)

    fig.add_trace(go.Scatter3d(
        x=[theta_init[0], theta_init[0]],
        y=[theta_init[1], theta_init[1]],
        z=[field_values_nn.min(), field_values_nn.max() + 10.0],
        mode='lines',
        line=dict(color='#004D40', width=6, dash='solid'),
        name='NN Predicted Jammer Location',
        legendgroup='NN'  # Grouped
    ), row=1, col=2)

    fig.add_trace(go.Scatter3d(
        x=[true_jam_loc[0], true_jam_loc[0]],
        y=[true_jam_loc[1], true_jam_loc[1]],
        z=[field_values_nn.min(), field_values_nn.max() + 10.0],
        mode='lines',
        line=dict(color='#1E88E5', width=6, dash='solid'),
        name='True Jammer Location',
        legendgroup='True Jammer'  # Grouped
    ), row=1, col=2)

    # Subplot 3: PL/APBM field
    fig.add_trace(go.Surface(
        z=field_values_pl,
        x=grid_x,
        y=grid_y,
        colorscale='Plasma',
        opacity=0.9,  # Transparent field
        showscale=False
    ), row=2, col=1)

    fig.add_trace(go.Scatter3d(
        x=[predicted_jam_loc[0], predicted_jam_loc[0]],
        y=[predicted_jam_loc[1], predicted_jam_loc[1]],
        z=[field_values_pl.min(), field_values_pl.max() + 10.0],
        mode='lines',
        line=dict(color='#FFC107', width=6, dash='solid'),
        name=f'PL Predicted Jammer Location',
        legendgroup='PL/APBM'  # Grouped
    ), row=2, col=1)

    fig.add_trace(go.Scatter3d(
        x=[theta_init[0], theta_init[0]],
        y=[theta_init[1], theta_init[1]],
        z=[field_values_pl.min(), field_values_pl.max() + 10.0],
        mode='lines',
        line=dict(color='#004D40', width=6, dash='solid'),
        name='NN Predicted Jammer Location',
        legendgroup='NN',  # Grouped
        showlegend=False  # Hide repeated legend
    ), row=2, col=1)

    fig.add_trace(go.Scatter3d(
        x=[true_jam_loc[0], true_jam_loc[0]],
        y=[true_jam_loc[1], true_jam_loc[1]],
        z=[field_values_pl.min(), field_values_pl.max() + 10.0],
        mode='lines',
        line=dict(color='#1E88E5', width=6, dash='solid'),
        name='True Jammer Location',
        legendgroup='True Jammer',  # Grouped
        showlegend=False  # Hide repeated legend
    ), row=2, col=1)

    # Subplot 4: PL/APBM field with testing points
    fig.add_trace(go.Surface(
        z=field_values_apbm,
        x=grid_x,
        y=grid_y,
        colorscale='Plasma',
        opacity=0.9,  # Transparent field
        showscale=False
    ), row=2, col=2)

    fig.add_trace(go.Scatter3d(
        x=[predicted_jam_loc[0], predicted_jam_loc[0]],
        y=[predicted_jam_loc[1], predicted_jam_loc[1]],
        z=[field_values_apbm.min(), field_values_apbm.max() + 10.0],
        mode='lines',
        line=dict(color='#D81B60', width=6, dash='solid'),
        name=f'APBM Predicted Jammer Location',
        legendgroup='PL/APBM'  # Grouped
    ), row=2, col=2)

    fig.add_trace(go.Scatter3d(
        x=[theta_init[0], theta_init[0]],
        y=[theta_init[1], theta_init[1]],
        z=[field_values_apbm.min(), field_values_apbm.max() + 10.0],
        mode='lines',
        line=dict(color='#004D40', width=6, dash='solid'),
        name='NN Predicted Jammer Location',
        legendgroup='NN',  # Grouped
        showlegend=False  # Hide repeated legend
    ), row=2, col=2)

    fig.add_trace(go.Scatter3d(
        x=[true_jam_loc[0], true_jam_loc[0]],
        y=[true_jam_loc[1], true_jam_loc[1]],
        z=[field_values_apbm.min(), field_values_apbm.max() + 10.0],
        mode='lines',
        line=dict(color='#1E88E5', width=6, dash='solid'),
        name='True Jammer Location',
        legendgroup='True Jammer',  # Grouped
        showlegend=False  # Hide repeated legend
    ), row=2, col=2)
    
    fig.update_layout(
        height=500,  # Adjust height for readability
        width=900,   # Adjust width to fit two columns
        margin=dict(l=10, r=10, t=40, b=40),  # Reduce margins to maximize space
        showlegend=False  # Disable the legend
    )
    
    # Adjust subplot aspect ratios for wider horizontal plots
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_scenes(
                xaxis=dict(showgrid=True, showticklabels=False),  # Gridlines without labels
                yaxis=dict(showgrid=True, showticklabels=False),  # Gridlines without labels
                zaxis=dict(showgrid=True),  # Keep z-axis gridlines
                aspectmode="manual",  # Manual aspect ratio
                aspectratio=dict(x=1.5, y=1.5, z=1.0),  # Wider horizontal plots
                row=i,
                col=j
            )

    # legend=dict(
        #     orientation="v",
        #     x=1.1,  # Position legend outside the plot
        #     y=0.5,  # Center the legend vertically
        #     xanchor="left",
        #     yanchor="middle",
        #     bgcolor="rgba(255, 255, 255, 0.7)",  # Semi-transparent background
        #     font=dict(size=12)  # Readable font size
        # ),

    # fig.update_layout(
    #     height=800, 
    #     width=1200,
    #     legend=dict(
    #         orientation="v",  # Vertical legend
    #         x=0.5,  # Align right
    #         y=-0.2,  # Center vertically
    #         xanchor="right",
    #         yanchor="middle",
    #         traceorder="normal",
    #         bgcolor="rgba(255, 255, 255, 0.7)",  # Semi-transparent background
    #         font=dict(size=12)
    #     ),
    #     margin=dict(l=20, r=20, t=20, b=50),  # Reduce padding around the plots
    # )
    
    # for i in range(1, 3):
    #     for j in range(1, 3):
    #         fig.update_scenes(
    #             xaxis=dict(showgrid=True, showticklabels=True),  # Show grid but hide tick labels
    #             yaxis=dict(showgrid=True, showticklabels=True),  # Show grid but hide tick labels
    #             zaxis=dict(showgrid=True),  # Keep z-axis grid visible
    #             aspectmode='data',
    #             row=i,
    #             col=j
    #         )
        
    # Save and display the plot
    output_path = os.path.join(folder, f'3d_surface_model_mc_run_{mc_run}.png')
    fig.write_image(output_path, format="png", engine="kaleido", width=1600, height=800)
    print(f"3D plot saved to {output_path}")
    fig.show()
    
        
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
        
        
def plot_grouped_boxplot(x_axis_values, aggregate_results, output_dir, x_label):
    """
    Plots a grouped boxplot for the given data with three error types: NN, PL, and APBM.
    """
    # Prepare data for plotting
    plot_data = []
    for value, mc_results in zip(x_axis_values, aggregate_results):
        for result in mc_results:
            plot_data.append({"Value": value, "Error": result["jam_init_loc_error"], "Type": "NN"})
            plot_data.append({"Value": value, "Error": result["jam_loc_error_pl"], "Type": "PL"})
            plot_data.append({"Value": value, "Error": result["jam_loc_error_apbm"], "Type": "APBM"})

    # Convert to a Pandas DataFrame for Seaborn
    df = pd.DataFrame(plot_data)

    # Set the color palette for the error types
    palette = {"NN": "#004D40", "PL": "#FFC107", "APBM": "#1E88E5"}

    # Generate the grouped boxplot
    plt.figure(figsize=(12, 6))
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
        palette=palette, 
        size=4, 
        jitter=True, 
        dodge=True, 
        alpha=0.6,
        marker="o"
    )

    # Avoid duplicate legends from the stripplot
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[:3], labels[:3], title="Type", fontsize=10)

    # Add labels and grid
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(r"RMSE$_{\theta}$ (m)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the plot as PNG
    output_path = os.path.join(output_dir, "jammer_localization_grouped_boxplot.png")
    plt.savefig(output_path)

    # Save the figure as a pickle file
    output_path_pkl = os.path.join(output_dir, "jammer_localization_grouped_boxplot.pkl")
    with open(output_path_pkl, 'wb') as f:
        pickle.dump(plt.gcf(), f)
        
    # Save the aggregated results as a pickle file
    output_path_agg_pkl = os.path.join(output_dir, "aggregated_results.pkl")
    with open(output_path_agg_pkl, 'wb') as f:
        pickle.dump(aggregate_results, f)

    print(f"Boxplot saved to {output_path}")