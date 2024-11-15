import numpy as np
import torch
import torch.nn as nn
import os
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

def plot_average_train_val_loss(all_train_losses_per_fold, all_val_losses_per_fold):
    """
    Plots and saves the average training and validation losses across all folds.

    Parameters:
    - all_train_losses_per_fold (list of lists): A list containing training losses for each fold.
    - all_val_losses_per_fold (list of lists): A list containing validation losses for each fold.

    Output:
    - Saves an HTML file containing the plot of average training and validation losses.
    - Displays the plot using Plotly's interactive visualization.
    """
    # Pad the lists with np.nan to handle different lengths of folds
    all_train_losses_per_fold = pad_lists_with_nan(all_train_losses_per_fold)
    all_val_losses_per_fold = pad_lists_with_nan(all_val_losses_per_fold)

    # Compute average losses across all folds, ignoring np.nan values
    avg_train_losses = np.nanmean(all_train_losses_per_fold, axis=0)
    avg_val_losses = np.nanmean(all_val_losses_per_fold, axis=0)
    epochs = list(range(1, len(avg_train_losses) + 1))  # Generate epoch numbers

    # Create a Plotly figure for visualizing the losses
    fig = go.Figure()

    # Add trace for average training loss
    fig.add_trace(go.Scatter(
        x=epochs, 
        y=avg_train_losses, 
        mode='lines+markers', 
        name='Average Training Loss',
        line=dict(color='blue')
    ))

    # Add trace for average validation loss
    fig.add_trace(go.Scatter(
        x=epochs, 
        y=avg_val_losses, 
        mode='lines+markers', 
        name='Average Validation Loss',
        line=dict(color='red')
    ))

    # Update the layout of the plot for better visualization
    fig.update_layout(
        title='Average Training vs Validation Loss Across Folds',
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
    output_file = os.path.join(output_folder, 'average_losses.html')
    fig.write_html(output_file)

    # Display the plot in the browser
    fig.show()
    
def plot_train_test_loss(train_losses_per_epoch, global_test_loss):
        """
        Plots the training losses per epoch and the global test loss.

        Parameters:
        - train_losses_per_epoch (list): A list containing training losses for each epoch.
        - global_test_loss (float): The global test loss value.

        Output:
        - Displays the plot using matplotlib's interactive visualization.
        """
        import matplotlib.pyplot as plt

        # Plot training losses per epoch
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses_per_epoch, label='Training Loss per Epoch')

        # Plot global test loss as a horizontal line
        plt.axhline(y=global_test_loss, color='r', linestyle='--', label='Global Test Loss')

        # Add labels and title
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss per Epoch and Global Test Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
    
def test_field(model, t):
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

    # Save the 3D plot as an HTML file
    fig.write_html(f'figs/field_{t}.html')