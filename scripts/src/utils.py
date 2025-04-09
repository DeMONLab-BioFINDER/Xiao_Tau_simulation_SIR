"""
Description:
    This module provides a collection of utility functions that are used throughout the SIR model simulation pipeline.
    The functions in this module facilitate tasks such as:
      - Clearing memory by forcing garbage collection.
      - Plotting brain surfaces and scatter plots to visualize simulation outputs.
      - Retrieving and processing brain atlas data.
      - Generating colormaps, colorbars, and configuring plot aesthetics.
      - Setting contour styles for plots.

Usage:
    Import the required functions from this module, for example:
        from utils import clear_memory, plot_brain_surf, get_atlas, find_right_hemisphere, plot_scatter_across_time, get_user_colors, set_contour

Created on Fri Dec 15 2023, at Lund, Sweden
Author: XIAO Yu
"""

import gc
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from scipy.stats import kendalltau


def clear_memory(*args):
    """
    Clear memory by deleting provided variables and forcing garbage collection.

    Args:
        *args: Any number of variables to be deleted.
    """
    for arg in args:
        del arg  # Delete the variable
    gc.collect() # Force garbage collection to free memory


def find_right_hemisphere(region_one_hemisphere):
    """
    Find the corresponding region name in the opposite hemisphere.

    Args:
        region_one_hemisphere (str): The name of a region from one hemisphere (e.g., "ctx_lh_entorhinal").

    Returns:
        right_hemisphere_region_name (str or None): The region name from the opposite hemisphere if a matching pattern is found; otherwise, None.
    """

    # Define mapping patterns to switch between left and right hemisphere naming conventions
    patterns = {'_l': '_r',
                '_lh_': '_rh_',
                '_LH_': '_RH_',
                'L-': 'R-'}
    
    # Iterate through the patterns and perform the replacement
    for left_pattern, right_pattern in patterns.items():
        if left_pattern in region_one_hemisphere:
            return region_one_hemisphere.replace(left_pattern, right_pattern)
        elif right_pattern in region_one_hemisphere:
            return region_one_hemisphere.replace(right_pattern,left_pattern)
    print("No corresponding right hemisphere found or pattern mismatch.")
    return None


def plot_scatter_across_time(tau, predictions, epicenter_name, times_list, save_name):
    """
    Generate and save scatter plots of observed tau versus simulated tau across selected time points.

    Args:
        tau (np.ndarray): Observed tau values, expected shape (n_regions,).
        predictions (np.ndarray): Simulated tau values, expected shape (n_regions, T_total).
        epicenter_name (str): Name or identifier of the epicenter.
        times_list (list): List of time points (indices) at which to generate scatter plots.
        save_name (str): Filename (without extension) for saving the generated plot as a PNG file.
    """
    num_columns = 8
    num_rows = math.ceil(len(times_list) / num_columns)

    # Create a figure with a grid of subplots
    fig, axs = plt.subplots(num_rows, num_columns, figsize=(30, num_rows * 2.5), 
                            constrained_layout=True)
    axs = axs.flat

    # Remove extra subplots that are not needed
    for ax in axs[len(times_list):]:
        ax.remove()

    def scatter_plot(ax, tau_tmp, best_pred, epicenter, time, group, save_name=None):
        # Calculate Pearson and Kendall Tau correlations
        r1 = np.corrcoef(tau_tmp, best_pred)[0,1]
        r2 = kendalltau(tau_tmp, best_pred)[0]
        text_str=f"""
        R={r1:.2f}  Tau={r2:.2f}
        Epicenter {epicenter}
        Time {time}
        """
        # Create a regression plot using seaborn.
        sns.regplot(x=tau_tmp, y=best_pred, ax=ax)
        ax.text(x=0.01, y=0.95, s=text_str, transform=ax.transAxes, #plt.gca().transAxes, 
                fontsize=8, verticalalignment='top', horizontalalignment='left')

        ax.set_xlabel("Observed tau probability")
        ax.set_ylabel("Simulated tau")
        ax.set_title('Group '+ str(group), fontsize=12)

    # Generate scatter plots for each specified time poin
    for i, ax in enumerate(axs[:len(times_list):]):
        scatter_plot(ax, tau, predictions[:,times_list[i]], epicenter_name, times_list[i], "All_mean")

    # Save the figure as a PNG file
    fig.savefig(save_name+".png", dpi=300)
    plt.close('all')
    plt.close()


def plot_line(data, output_path):
    """
    Plot a line graph of the provided data and save it to a file.

    Args:
        data (array-like or pandas.DataFrame): Data to be plotted as a line plot.
        output_path (str): File path (including filename and extension) where the plot will be saved.
    """
    # Create a line plot using seaborn's lineplot function
    sns.lineplot(data)
    # Save the generated plot to the specified output path
    plt.savefig(output_path)
    plt.close()


def scatter_pred_true(true, pred, x_ticks=None, save_name=False):
    """
    Create a scatter plot with a regression line comparing observed and simulated tau values.

    Args:
        true (array-like): Observed tau values, expected shape (n_samples,).
        pred (array-like): Simulated tau values, expected shape (n_samples,).
        x_ticks (list, optional): List of tick values for the x-axis. Default is None.
        save_name (str or bool, optional): If a string is provided, the plot is saved to this file path;
                                           if False, the plot is displayed interactively.
    """
    # Convert inputs to float type to ensure proper plotting
    true = true.astype(float)
    pred = pred.astype(float)

    # Create a DataFrame from the observed and simulated values for ease of plotting with seaborn
    data=pd.DataFrame({"true": true, "pred":pred})

    # Create a new figure with high resolution and specific size
    plt.figure(dpi=300, figsize=(5.2,5))

    # Generate a scatter plot with regression line using seaborn's regplot
    ax=sns.regplot(data=data, x="true", y="pred", scatter_kws={"s": 100, "alpha":0.8})
    ax = set_contour(ax) # Apply custom contour settings (e.g., tick and spine formatting) to the plot

    # If x_ticks is provided, set custom tick positions and labels on the x-axis
    if x_ticks:
        for a in ax.axes.flatten():
            a.set_xticks(x_ticks)  # Set the tick positions
            a.set_xticklabels([f'{tick}' for tick in x_ticks])

    # Format the y-axis labels in scientific notation using a formatter from matplotlib.ticker
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.1e}'))  # Converts to 1.0e+XX format

    # Set the x-axis and y-axis labels for the plot
    plt.xlabel("Observed")
    plt.ylabel("Simulated")

    # Save the figure to file if save_name is provided, otherwise display it interactively
    if save_name:
        plt.savefig(save_name)
        plt.close()
    else:
        plt.show()


def set_contour(ax=None, lw=1):
    """
    Configure plot aesthetics by setting tick parameters and spine widths for a given axis.

    Args:
        ax (matplotlib.axes.Axes, optional): The axis to configure. If None, uses the current axis.
        lw (int, optional): Line width for the ticks and spines.

    Returns:
        matplotlib.axes.Axes: The modified axis.
    """
    if not ax: ax = plt.gca()
    ax.tick_params(axis='both', which='both', direction='in', labelsize=20, width=lw)
    
    for spine in ax.spines.values():
        spine.set_linewidth(lw)
    
    # Remove the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return ax
