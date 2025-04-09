"""
Description:
    This module provides functions to extract, evaluate, and summarize simulation results 
    from SIR model runs, used by `run_summary_one_conn.py` and `run_summary_all_conns.py`. 

Key Functions:
    - infer_from_result_path(args):
        Infers the input directory and data type from the result path and updates the args Namespace.
    - find_directories_containing_string(base_path, search_string):
        Finds directories within the base_path whose names contain the specified search string.
    - extract_all_conns(conn_list, factors_dict, var_lists, args):
          Iterates through connectivity measures and regional factors to extract and summarize model performance.
        - load_result(result_folder, check_data_flag):
            Loads simulation result data from a specified folder.
        - match_and_update(label_conn, data_to_update):
            Matches region labels between connectivity matrices and simulation outputs.
        - find_best_time(preds, y, metric="pearsonr"):
            Identifies the best time point based on an evaluation metric.
        - get_best_metric(pred, y):
            Computes evaluation metrics (Pearson, MSE, Explained Variance) for the best time point.

Usage:
    Import this module and call functions. For example:
    from summary import infer_from_result_path
    
Created on Fri Dec 15 2023, at Lund, Sweden
@author: XIAO Yu
"""

import os
import re
import glob
import pickle
import numpy as np
import pandas as pd

from scipy.stats import pearsonr
from sklearn.metrics import explained_variance_score, mean_squared_error


def infer_from_result_path(args):
    """
    Infer the input directory and data type from the result path and update the args Namespace.

    Args:
        args (Namespace): A Namespace object containing at least the attribute 'result_path' (str).
    
    Returns:
        Namespace: The updated args object with 'input_dir' (str) and 'data_type' (str) set.
    """
    # Derive the input directory: take the part of result_path before "results/" and append "data/"
    args.input_dir = os.path.join(args.result_path.split("results/")[0], "data/")
    # Extract the simulated protein: assumed to be the first element in the result_path after "results/"
    args.simulated_protein = args.result_path.split("results/")[1].split("/")[0]
    # Extract the protein type: assumed to be the second element in the result_path after "results/"
    args.protein_type = args.result_path.split("results/")[1].split("/")[1]

    return args


def find_directories_containing_string(base_path, search_string):
    """
    Find directories within the base_path whose names contain the specified search string,
    and, depending on the content of the search string, either return a list of matching directories
    or a dictionary mapping extracted keys to directory paths.

    Args:
        base_path (str): The directory to search for result folders.
        search_string (str): The substring used to filter folder names. For example, "SC_".

    Returns:
        dict or list:
            - If the search string itself contains any of the substrings in ["synthesis-", "spread-", "misfold-", "clearance-"],
              returns a list of full directory paths matching the search string.
            - Otherwise, returns a dictionary where each key is derived from the folder name (using regex if applicable)
              and the value is the full directory path.
    """
    # Define the substrings to check for in the search string.
    var_list = ["synthesis-", "spread-", "misfold-", "clearance-"]

    # Use glob to find all directories within the base_path (non-recursive search).
    directories = glob.glob(os.path.join(base_path, '*'), recursive=True)
    # Filter directories that are actual directories and whose basename contains the search_string.
    matching_directories = [d for d in directories if os.path.isdir(d) and search_string in os.path.basename(d)]
    
    # If the search string itself contains any of the substrings in var_list, return the list directly
    # Used in `extract_all_models()` function
    if any(sub in search_string for sub in var_list):
        return matching_directories

    # Otherwise, process the matching directories to extract keys and return as a dictionary
    # Used in  `extract_one_conn()`` function
    result_dict = {}
    for dir in matching_directories:
        # Extract the folder name from the full directory path
        folder_name = dir.split("/")[-1]
        # If the folder name contains any substring from var_list, extract key using regex
        if any(sub in folder_name for sub in var_list):
            match = re.match(r'([^-]+)-([^_]+)', folder_name)
            if match:
                key = match.group(1) + match.group(2)
            else:
                key = folder_name  # Fallback: use folder_name if regex does not match
        else:
            key = "Baseline"       
        result_dict[key] = dir
    
    # Sort the dictionary by keys and return
    result_dict = {key: result_dict[key] for key in sorted(result_dict)}
    return result_dict


def extract_all_conns(conn_list, factors_dict, var_lists, args):
    """
    Extract and aggregate evaluation metrics from simulation models across different connectivity measures,
    regional factors and simulation mechanism variable.

    Args:
        conn_list (list of str): List of connectivity measure names.
        factors_dict (dict): Dictionary mapping each regional factor type group to a list of factor names.
        var_lists (list of str): List of simulation mechanism variables (e.g., ["syn", "spread", "mis", "clear"]).
        args (Namespace): Command-line arguments. Expected to include attributes such as:
            - input_dir (str): Directory path for input data files.
            - input_data_name (str): Filename for the input simulation data.
            - subtype (int): Subtype index; if >= 0, a specific tau subtype is loaded.
            - subtype_file (str): Filename for the tau subtype CSV file.
            - result_path (str): Directory where simulation results are stored.
            - connectivity_file (str): Filename for the connectivity matrix.
            - eval_metric (str): Evaluation metric to use (e.g., "pearsonr").
            - epicenter (str): Identifier for the region of interest.
    
    Returns:
        tuple: (df_metrics, missing_results)
            - df_metrics (pd.DataFrame): DataFrame summarizing metrics with columns:
              ['Connectivity', 'Regional_factor', 'factor_name', 'factoras', 'R', 'MSE', 'EV', 'Time', 'Hyperparam'].
            - missing_results (list): List of strings representing missing results.
    """
    # Initialize dictionaries to store metrics and predictions, and an empty DataFrame for summarizing metrics.
    metrics_across_time_dict, pred_scaled_dict = {}, {}
    df_metrics = pd.DataFrame(columns=['Connectivity', 'Regional_factor', 'factor_name', 'factoras',
                                    'R', 'MSE', 'EV', 'Time', 'Hyperparam'])
    
    # Load input data based on the type specified in result_path
    print("load input data")
    data = pickle.load(open(args.input_dir+args.input_data_name,'rb'))
    
    # Get region names and calculate the mean tau across subjects
    y_region_names = data["conn"]["name"] # Expected shape: (n_regions,)
    y_mean = data[args.simulated_protein][args.protein_type].values.astype(float).reshape(-1) # Expected shape: (n_regions,)

    # If a specific subtype is provided, load the corresponding observed tau values from a CSV file
    if args.subtype >= 0:
        print("Load observed tau from", args.subtype_file, ", subtype", args.subtype)
        df_sub = pd.read_csv(args.input_dir + args.subtype_file,index_col=[0])
        y_mean = df_sub.loc[args.subtype].values.astype(float).reshape(-1)
        print("Subtype tau subtype_file:\n", y_mean)
        save_path = args.result_path+"/Subtype"+str(args.subtype)+"/"
        if not os.path.exists(save_path): os.makedirs(save_path, exist_ok=True)
    else:
        save_path = args.result_path
    print("!!!SAVE PATH!!!!:", save_path)

    # Load the connectivity matrix and its region labels
    print("load connectivity matrix")
    conn_matrix_all = pickle.load(open(os.path.join(args.input_dir, args.connectivity_file), 'rb'))
    pred_region_names = conn_matrix_all["labels"]

    # Match region names between the connectivity matrix and simulation data
    index_y_to_conn = match_and_update(pred_region_names, y_region_names)
    y_mean_matched = y_mean[index_y_to_conn]

    missing_results = []
    # Iterate over each connectivity measure provided in conn_list
    for conn in conn_list:
        metrics_across_time_dict[conn], pred_scaled_dict[conn] = {}, {}
        # Iterate over each regional factor group in factors_dict
        for factor in factors_dict:
            metrics_across_time_dict[conn][factor], pred_scaled_dict[conn][factor] = {}, {}
            # Iterate over each factor within the current regional factor type group
            for f in factors_dict[factor]:
                metrics_across_time_dict[conn][factor][f], pred_scaled_dict[conn][factor][f] = {}, {}
                var_list = ["Baseline"] if f == "Baseline" else var_lists # for "Baseline", use a variable list containing only "Baseline"; otherwise, use the full var_lists
                # Iterate over each simulation mechanism variable (e.g., synthesis, spread, clearance, misfold)
                for var in var_list:
                    result_name = conn if factor=="Baseline" else conn + "_" + var + "-" + f
                    print("*"*10,"\nExtracting", result_name, ":", conn, factor, f, var,"\n","*"*10)
                    # Search for directories in args.result_path that contain the result name with '_hypertune'
                    result_folder = find_directories_containing_string(args.result_path,result_name)
                    for result_f in result_folder:
                        # Load the result from the folder; use args.epicenter to select the relevant ROI used as epicenter in SIR model
                        result = load_result(result_f, args.epicenter) if result_f else None
                        if result: break
                    # If a result is found, extract hyperparameters and evaluation metrics
                    if result:
                        hyperparam = result["max_combination"]
                        time, metric_across_time = find_best_time(result['max_pattern'], y_mean_matched, args.eval_metric)
                        r, mse, ev, pred_scaled = evaluate_the_best_time(result['max_pattern'][:, time], y_mean_matched)
                    else:
                        # If no result is found, assign NaN values and log the missing result
                        time, r, mse, ev, metric_across_time, pred_scaled, hyperparam = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
                        missing_results.append(conn + "_" + var + "-" + f)
                    
                    # Save the extracted metrics and predictions in the corresponding dictionaries
                    metrics_across_time_dict[conn][factor][f][var] = metric_across_time
                    pred_scaled_dict[conn][factor][f][var] = pred_scaled
                    if factor=="Baseline":
                        factor_val, f_val, var_val = np.nan, np.nan, np.nan 
                    else:
                        factor_val = factor
                        f_val = f
                        var_val = var
                    # Append the metrics for the current result to the summary DataFrame
                    df_metrics = pd.concat([df_metrics, pd.DataFrame([{'Connectivity': conn, 'Regional_factor': factor_val,
                                                                      'factor_name': f_val, 'factoras': var_val,
                                                                      'R': r, 'MSE': mse, 'EV': ev,
                                                                      'Time': time, 'Hyperparam': hyperparam}])],
                                                                      ignore_index=True)
        # Save the extracted metrics and predictions for the current connectivity measure
        pickle.dump(metrics_across_time_dict, open(save_path+"ALL_metrics_across_simulation_time_"+conn+".pkl", 'wb'))
        pickle.dump(pred_scaled_dict, open(save_path+"ALL_prediction_best_scaled_"+conn+".pkl", 'wb'))
        df_metrics.to_csv(save_path+"ALL_metrics_"+conn+".csv")
        # Reset the dictionaries and DataFrame for the next connectivity measure
        metrics_across_time_dict, pred_scaled_dict = {}, {}
        df_metrics = pd.DataFrame(columns=['Connectivity', 'Regional_factor', 'factor_name', 'factoras',
                                                'R', 'MSE', 'EV', 'Time', 'Hyperparam'])
    
    return df_metrics, missing_results


def load_result(result_folder, epicenter, check_data_flag="simulation"):
    """
    Load simulation result data from a specified result folder.

    Args:
        result_folder (str): The directory containing the result files.
        check_data_flag (str): Flag to indicate which data to check (default is "simulation").

    Returns:
        results (dict or None): The loaded result data (expected to be a dictionary). If the file does not exist, returns None.
    """
    data_full_path = result_folder+"/hyperparameters_model_intermediate_outputs_"+check_data_flag+".pkl"
    if not os.path.exists(data_full_path):
        return None
    print("load",data_full_path)
    result = pickle.load(open(data_full_path, 'rb'))
    result = result[epicenter]
    
    return result

def match_and_update(label_conn, data_to_update):
    """
    Match region labels between the connectivity matrix and the simulation output data.

    Args:
        label_conn (pd.Series): A pandas Series containing region labels from the connectivity matrix.
                                Expected shape: (n_regions,).
        data_to_update (list): A list of region labels from the simulation data, length should be n_regions.

    Returns:
        matched_indices (list): A list of matched indices that align the simulation data with the connectivity matrix.
    """
    if isinstance(data_to_update, list):
        # If the labels do not match, find indices where simulation data labels are present in the connectivity labels
        if not np.array_equal(label_conn, pd.Series(data_to_update)):
            matched_indices = [i for i in range(len(data_to_update)) if data_to_update[i] in label_conn]
            return matched_indices
        else:
            return list(range(len(data_to_update)))
        

def find_best_time(preds, y, metric):
    """
    Identify the best time point (index) for model predictions based on the specified evaluation metric.

    Args:
        preds (np.ndarray): Predicted values, expected shape (n_regions, T_total).
        y (np.ndarray): Ground truth values, expected shape (n_regions,).
        metric (str): The evaluation metric to use ("pearsonr" or "mse").

    Returns:
        tuple: (eval_max_index, eval_values)
            - eval_max_index (int): The time index at which the best metric value is observed.
            - eval_values (list): List of metric values computed for each time point.
    """
    eval_values = []
    if metric == "pearsonr":
        # Compute Pearson correlation for each time point
        eval_values = [pearsonr(y, preds[:,i])[0] for i in range(preds.shape[1])]
        eval_max = np.max(eval_values)
        eval_max_index = eval_values.index(eval_max)

    elif metric == "mse":
        # Normalize ground truth and predictions and compute MSE for each time point
        y_scale = (y-np.nanmin(y))/(np.nanmax(y)-np.nanmin(y))
        for i in range(preds.shape[1]): 
            pred_scale = (preds[:, i]-np.nanmin(preds[:, i]))/(np.nanmax(preds[:, i])-np.nanmin(preds[:, i]))
            not_nan_mask = ~np.isnan(y_scale) & ~np.isnan(pred_scale)
            mse = mean_squared_error(y_scale[not_nan_mask], pred_scale[not_nan_mask])
            eval_values.append(mse)
        eval_max = np.nanmin(eval_values) # For MSE, lower is better
        eval_max_index = eval_values.index(eval_max)

    return eval_max_index, eval_values


def evaluate_the_best_time(pred, y):
    """
    Compute the evaluation metrics (R, MSE, EV) for the selected best prediction and ground truth.

    This function normalizes the predicted data and rescales it to match the range of the ground truth.
    It then calculates the Pearson correlation, Mean Squared Error (MSE), and Explained Variance (EV) score.

    Args:
        pred (np.ndarray): Predicted values at a specific time point, expected shape (n_regions,).
        y (np.ndarray): Ground truth values, expected shape (n_regions,).

    Returns:
        tuple: (r, mse, ev, pred_rescaled)
            - r (float): Pearson correlation coefficient.
            - mse (float): Mean Squared Error.
            - ev (float): Explained Variance score.
            - pred_rescaled (np.ndarray): The rescaled predicted values, with shape (n_regions,).
    """
    # Normalize prediction between 0 and 1
    pred_scaled = (pred - np.nanmin(pred)) / (np.nanmax(pred) - np.nanmin(pred))
    # Rescale normalized predictions to the range of y
    pred_rescaled = pred_scaled * (np.nanmax(y) - np.nanmin(y)) + np.nanmin(y)

    r = pearsonr(pred_rescaled, y)[0]
    mse = mean_squared_error(pred_rescaled, y)
    ev = explained_variance_score(pred_rescaled, y)

    return r, mse, ev, pred_rescaled
