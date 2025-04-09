"""
Description:
    This module contains functions for evaluating the performance of the SIR model predictions.
    It provides functionality to compute various correlation metrics (e.g., Pearson, Kendalltau, MAE, MAD, R2)
    between predicted and observed data across time. In addition, it identifies top regions of interest (ROIs)
    based on peak correlation performance, applies evaluation metrics across time points in parallel, and generates
    visualizations of intermediate differences in the predictions.

    The main functions include:
        - compute_evaluation_metrics: Aggregates evaluation metrics across all ROIs, depend on `evaluate()` function below.
            - evaluate: Evaluates predictions across time using parallel computation, depend on `correlation_repeat()` function below.
                - correlation_repeat: Computes evaluation metrics for a given time point, depend on `apply_metric()` function below
                    - apply_metric: Applies evaluation metrics to each subject's data, depend on `compute_metric()` function below
                        - compute_metric: Computes a specific evaluation metric between true and predicted values, depend on `normalize_data()` function below
                            - normalize_data: Normalizes data using min-max scaling
        - get_top_rois: Identifies the top three ROIs based on peak correlation values.
        - plot_interm_difference: Generates and saves plots showing differences in intermediate results.

Usage:
    Import the desired functions from this module to evaluate model predictions. For example:
        from evaluation import evaluate, compute_metric, get_top_rois

Created on Fri Dec 15 2023, at Lund, Sweden
@author: XIAO Yu
"""

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from scipy.stats import pearsonr, kendalltau
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score


def compute_evaluation_metrics(tau_mean, predictions, eval_metrics, N_regions, T_total, roi_names):
    """
    Compute evaluation metrics for each ROI across all time points.

    Args:
        tau_mean (np.ndarray): Ground truth mean tau values, expected shape (N_regions,).
        predictions (dict): Dictionary with keys "simulation" and "Rmis". Each value is a dictionary mapping ROI identifiers to predicted arrays of shape (n_regions, T_total).
        eval_metrics (list): List of evaluation metric names.
        N_regions (int): Total number of regions.
        T_total (int): Total number of time steps.
        roi_names (list of str): List of ROI names, length should equal N_regions.

    Returns:
        dict: A dictionary (correlation_df) with keys "simulation" and "Rmis". Each key maps to a dictionary where keys are metric names and values are pandas DataFrames of shape (T_total, N_regions) containing the evaluation metrics.
    """
    correlation_df = {key: {metric: pd.DataFrame(index=range(T_total), columns=predictions[list(predictions.keys())[0]].keys())
                    for metric in eval_metrics}
                    for key in ["simulation", "Rmis"]}
    for epicenter in range(N_regions):
        print("\nEpicenter ROI " + str(epicenter) + ": " + roi_names[epicenter])
        correlations_epicenter = evaluate(predictions["simulation"][f"ROI_{epicenter}"], predictions["Rmis"][f"ROI_{epicenter}"],
                                    tau_mean, eval_metrics, T_total)
        for key in correlations_epicenter:
                print("Best",key, end=" ")
                for metric in eval_metrics:
                    mean_corr = np.nanmean(correlations_epicenter[key][metric], axis=1)
                    correlation_df[key][metric][f"ROI_{epicenter}"]= mean_corr
                    if metric in ["pearsonr", "kendalltau"]:
                        print(f"{metric} at time {np.argmax(mean_corr)}: {np.max(mean_corr)}", end="\t")
                    elif metric in ["mae","mad"]:
                        print(f"{metric} at time {np.argmin(mean_corr)}: {np.min(mean_corr)}", end="\t")
    
    return correlation_df


def evaluate(pred, Rmis, y, eval_metrics, T_total, n_jobs=-1):
    """
    Evaluate the predictions by computing correlation metrics across all time points.

    Args:
        pred (array-like): Predicted values for the simulation, expected shape (n_regions, T_total).
        Rmis (array-like): Predicted values for misfolded protein, expected shape (n_regions, T_total).
        y (np.ndarray): Ground truth values, expected shape (n_regions,).
        eval_metrics (list): List of evaluation metric names (e.g., ['pearsonr', 'mae']).
        T_total (int): Total number of time steps.
        n_jobs (int): Number of parallel jobs for evaluation (default is -1, which uses all cores).

    Returns:
        dict: A dictionary with keys "simulation" and "Rmis". Each is a dictionary where keys are metric names
              and values are NumPy arrays of shape (T_total, len(y)) containing correlation values over time.
    """
    # Ensure predictions are NumPy arrays of type float
    predictions = {"simulation": np.array(pred).astype(float),
                   "Rmis": np.array(Rmis).astype(float)}
    
    eval_metrics = eval_metrics if isinstance(eval_metrics, list) else [eval_metrics]
    correlations = {key: {metric: np.empty((T_total, len(y))) for metric in eval_metrics}
                    for key in ["simulation", "Rmis"]} # This will hold the correlations for each repeat
    
    # Parallel computation over all time steps
    results = Parallel(n_jobs=n_jobs)(delayed(correlation_repeat)(repeat, predictions, eval_metrics, y) for repeat in range(T_total))
    
    # Gather results into the correlations dictionary
    for repeat, result in enumerate(results):
        for key, df_result in result.items():
            for metric in eval_metrics:
                corr_all = np.array([metric_per_subject.get(metric, np.nan) for metric_per_subject in df_result]) # if metric in metrics_dict
                correlations[key][metric][repeat, :] = corr_all
    
    return correlations


def correlation_repeat(repeat, predictions, eval_metrics, y):
    """
    Compute evaluation metrics for a given time point (repeat) for each result type.

    Args:
        repeat (int): The current time step (0 <= repeat < T_total).
        predictions (dict): Dictionary with keys "simulation" and "Rmis", each an array of shape (n_regions, T_total).
        eval_metrics (list): List of evaluation metric names.
        y (pd.Series or DataFrame): Ground truth values; expected to be a pandas Series where each element corresponds to a region.

    Returns:
        dict: A dictionary with keys as result types ("simulation", "Rmis") and values as the computed correlation series 
              (one per subject/region).
    """
    corr_series = {}

    for key, preds in predictions.items():
        repeat_predictions = preds[:, repeat]
        # Apply the metrics
        corr_series[key] = y.apply(lambda row: apply_metric(row, repeat_predictions, eval_metrics), axis=1)

    return corr_series


def apply_metric(subject_row, predictions, eval_metrics):
    """
    Apply the specified evaluation metrics to a single subject's data.

    Args:
        subject_row (pd.Series): A row of ground truth values for one subject/region.
        predictions (array-like): Predicted values for the subject, expected shape (n_samples,).
        eval_metrics (list): List of evaluation metric names.

    Returns:
        dict: Dictionary mapping each metric to its computed value for the subject.
    """
    non_nan_indices = ~subject_row.isna()
    if non_nan_indices.any():
        return {metric: compute_metric(metric, subject_row[non_nan_indices], predictions[non_nan_indices]) 
                for metric in eval_metrics} #(eval_metrics if isinstance(eval_metrics, list) else [eval_metrics])}
    else:
        return {metric: np.nan for metric in eval_metrics} #if isinstance(eval_metrics, list) else [eval_metrics])}


def compute_metric(metric, y_true, y_pred):
    """
    Compute the specified evaluation metric between true and predicted values.

    Args:
        metric (str): The evaluation metric to compute. Supported metrics: 'pearsonr', 'kendalltau', 'mae', 'mad', 'r2'.
        y_true (array-like): True values, expected shape (n_samples,).
        y_pred (array-like): Predicted values, expected shape (n_samples,).

    Returns:
        float: The computed metric value.
    """
    if metric == 'pearsonr':
        return pearsonr(y_true, y_pred)[0]
    elif metric == 'kendalltau':
        return kendalltau(y_true, y_pred)[0]
    elif metric == 'mae':
        y_true_norm = normalize_data(y_true)
        y_pred_norm = normalize_data(y_pred)
        return mean_absolute_error(y_true_norm, y_pred_norm)
    elif metric == 'mad':
        y_true_norm = normalize_data(y_true)
        y_pred_norm = normalize_data(y_pred)
        return median_absolute_error(y_true_norm, y_pred_norm)
    elif metric == 'r2':
        return r2_score(y_true_norm, y_pred_norm)
    else:
        raise ValueError("Unknown metric: " + metric)
    

def normalize_data(data, method="minmax"):
    """
    Normalize the input data using the specified method.

    Args:
        data (array-like): Data to be normalized.
        method (str): Normalization method. Default is "minmax" which scales data to [0,1].

    Returns:
        array-like: Normalized data.
    """
    if method=="minmax":
        normalized_data = (data - data.min()) / (data.max() - data.min())
    else:
        print("NOT normalizing!!!")
        normalized_data = data
    return normalized_data


def get_top_rois(correlation_df, roi_names, key, metric):
    """
    Identify the top three ROIs based on the peak correlation values.

    Args:
        correlation_df (dict): A dictionary with keys as result types (e.g., "simulation", "Rmis") and values as pandas DataFrames.
                                Each DataFrame has shape (T_total, n_ROIs) containing correlation values over time.
        roi_names (list of str): List of ROI names, length should be equal to n_ROIs.
        key (str): Result type key (e.g., "simulation" or "Rmis").
        metric (str): The evaluation metric used, e.g., "pearsonr", "kendalltau", "mae", "mad".

    Returns:
        dict: A dictionary with keys 'ROI' (indices of top ROIs) and 'name' (names of the top ROIs).
    """
    # Find the top three ROIs based on the peak correlation
    df_corr = correlation_df[key][metric]
    top_rois_ind = df_corr.max().nlargest(3).index if metric in ["pearsonr", "kendalltau"] else df_corr.min().nsmallest(3).index
    # Convert the ROI indices to ROI names
    roi_names = [' '.join(s.split('_')[-2:]).title() for s in roi_names]
    top_rois_name = [roi_names[int(roi[3:])] for roi in top_rois_ind]
    print(f'Top three ROIs for {key} result: {top_rois_name},\t index {top_rois_ind}')

    # Print the peak performance time and value for each top ROI
    for roi, name in zip(top_rois_ind,top_rois_name):
        if metric in ["pearsonr", "kendalltau"]:
            peak_time = df_corr[roi].idxmax()
            peak_value = df_corr[roi].max()
        else:
            peak_time = df_corr[roi].idxmin()
            peak_value = df_corr[roi].min()
        print(f'{name}: Peak performance at time {peak_time} with a correlation of {peak_value:.4f}')
                
    return {"ROI": top_rois_ind, "name": top_rois_name}


def plot_interm_difference(results_tmp, output_path, save_name):
    """
    Generate and save plots of intermediate differences for selected ROIs and time points.

    Args:
        results_tmp (dict): A dictionary of intermediate result arrays. Expected shapes:\n"
                             "    - If 2D: (n_timepoints, ) for each ROI.\n"
                             "    - If 3D: (n, n, T) where T is the number of time points.\n"
        output_path (str): Directory where the plot will be saved.
        save_name (str): Base filename to use for saving the plot.

    Returns:
        None
    """
    # Define specific ROIs to highlight with colors
    selected_epicenter = {3: "red", # entorinal
                         32: "purple", # hippocampus
                         20: "yellow", # posteriorcingulate
                         12: "blue", # middletemporal
                         21: "green"} #precentral
    selected_time = [0,1,2,3,4,10,50,75,100,200,
                     300,400] #,500,750,1000,2000,3000,4000,5000,7500,
                     #10000,15000,20000,25000,29999]

    for name, result in results_tmp.items():
        if result is not None and not isinstance(result, list):
            if result.ndim == 2:
                fig, axs = plt.subplots(2, 2, figsize=(10, 10))

                axs[0,0].set_title('first 50 time points')
                axs[0,1].set_title('50 to 100')
                axs[1,0].set_title('100 to 20000')
                axs[1,1].set_title('Later than 20000 timepoints')
                for i, row in enumerate(result):
                    if i not in selected_epicenter.keys():
                        axs[0,0].plot(row[:50], color='grey')
                        axs[0,1].plot(row[50:100], color='grey')
                        axs[1,0].plot(row[100:200], color='grey') # 20000
                        axs[1,1].plot(row[200:], color='grey')
                for i in selected_epicenter:
                    axs[0,0].plot(result[i,:50], color=selected_epicenter[i])
                    axs[0,1].plot(result[i,50:100], color=selected_epicenter[i])
                    axs[1,0].plot(result[i,100:200], color=selected_epicenter[i])
                    axs[1,1].plot(result[i,200:], color=selected_epicenter[i])
    
            elif result.ndim == 3:
                fig, axs = plt.subplots(5, 5, figsize=(15, 15)) 
                for t, ax in zip(selected_time, axs.flatten()):
                    im = ax.imshow(result[:,:,t])
                    ax.set_title('Time '+str(t))
                    fig.colorbar(im, ax=ax)
                plt.tight_layout()

            plt.savefig(os.path.join(output_path, name+"_"+save_name+".png"))
            plt.close()