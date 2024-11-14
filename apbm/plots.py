import numpy as np
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

def plot_average_losses(all_train_losses_per_fold, all_val_losses_per_fold):
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