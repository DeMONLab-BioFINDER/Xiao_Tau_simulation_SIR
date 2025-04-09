"""
Description:
    This script summarizes simulation results for a single connectivity measure from the SIR model.

Key Functionalities:
    - Infer the input directory and data type from the result path.
    - Compute evaluation metrics (e.g., Pearson correlation, MSE) over simulation time.
    - Aggregate best predictions and metrics into summary DataFrames.
    - Save the aggregated results (metrics, predictions, and time-series data) to CSV files for further analysis.

Usage:
    Run the script from the command line with the following required arguments:
        --result_path: Full path to the results directory.
        --connectivity: Name of the connectivity measure to summarize.
        --epicenter: Epicenter identifier used in the simulation (e.g., "ctx_lh_entorhinal").
        --input_data_name: Filename of the main input data file.
    
    Example:
        python run_summary_one_conn.py --result_path "/path/to/results" --connectivity "SC" --epicenter "ctx_lh_entorhinal" --input_data_name "Input_SIR_example.pt"

Dependencies:
    - Standard libraries: os, numpy, pandas, argparse
    - Third-party libraries: torch, scipy.stats, scikit-learn
    - Custom modules: summary (infer_from_result_path, find_directories_containing_string)

Created on Fri Dec 15 2023, at Lund, Sweden
@author: XIAO Yu

"""

import os
import numpy as np
import pandas as pd
import pickle
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

from summary import infer_from_result_path, find_directories_containing_string

def summarize_one_conn(args):
    """
    Summarize simulation results for a single connectivity measure from the SIR model.
    
    This function:
      - Finds result folders for the specified connectivity measure using find_sc_weight_folders().
      - Loads simulation results from each folder.
      - Computes evaluation metrics (e.g., Pearson correlation) over time.
      - Aggregates best predictions and metrics into DataFrames.
      - Saves the results to CSV files in the result_path directory.

    Args:
        args (Namespace): Configuration and parameter settings with attributes:
            - data_type (str): Type of simulation data to use (e.g., "TPP").
            - result_path (str): Path to the folder containing simulation results.
            - input_dir (str): Path to the input data directory.
            - input_data_name (str): Filename for the main input data.
            - epicenter: Identifier for the epicenter (used when loading results).
            - Other attributes as needed.

    Returns:
        tuple: (df_metrics, df_max_r_list, df_results)
            - df_metrics (pd.DataFrame): Summary of metrics for each model.
            - df_max_r_list (pd.DataFrame): Time-series of maximum correlation values for each model.
            - df_results (pd.DataFrame): Best predictions and their scaled versions.
    """
    # Infer the input directory and data type
    args = infer_from_result_path(args)

    # Find result folders that match the connectivity measure naming pattern
    result_dict = find_directories_containing_string(args.result_path, args.connectivity)
    print("\n".join(f"{key}: {value}" for key, value in result_dict.items()))
    
    # Load the "data_all" variable from input data
    # (Assumes data_all is a torch file containing a dictionary with keys for different data types)
    data_all = pickle.load(open(os.path.join(args.input_dir, args.input_data_name),'rb'))
    tau_mean = data_all[args.simulated_protein][args.protein_type]
    
    # Initialize DataFrames to store results and metrics
    df_results = pd.DataFrame(data={args.protein_type: tau_mean.values}, index=tau_mean.index)
    df_metrics = pd.DataFrame(index=["Time", "Hyperparameters", "R", "p", "MSE"])
    df_max_r_list = pd.DataFrame(index=range(30000))
    
    tau_mean = tau_mean.values.reshape(-1).astype(float)
    # Loop over each model (connectivity measure) found in the result folders
    for model, path in result_dict.items():
        print("\n", model, "\n", path)
        # Load simulation result data from the folder
        if "hypertune/" in path:
            results = pickle.load(open(os.path.join(path, "hyperparameters_model_intermediate_outputs_simulation.pkl"),'rb'))
            print("Best hyperparam:", results[args.epicenter]['max_combination'])
            # Retrieve the prediction pattern across time
            pred_across_time = results[args.epicenter]['max_pattern']
        else:
            matching_files = [f for f in os.listdir(path) if f.startswith("simulated_atrophy_all_")]
            results = pickle.load(open(os.path.join(path, matching_files[0]),'rb'))
            pred_across_time = results["simulation"][args.epicenter]
        # Calculate Pearson correlation for each time point
        df_max_r_list[model] = [pearsonr(tau_mean, pred_across_time[:, i])[0] for i in range(pred_across_time.shape[1])]
        
        # Compute metrics at the best time point
        best_time = np.nanargmax(df_max_r_list[model])
        print("best time:", best_time)
        pred = results[args.epicenter]['pred_best']
        scaled_pred = ((pred - np.nanmin(pred)) / (np.nanmax(pred) - np.nanmin(pred))) * (np.nanmax(tau_mean) - np.nanmin(tau_mean)) + np.nanmin(tau_mean)
        
        df_results[model] = pred
        df_results[model + "_scaled"] = scaled_pred
        df_metrics[model] = [best_time, results[args.epicenter]["max_combination"],
                             pearsonr(tau_mean, pred)[0], pearsonr(tau_mean, pred)[1],
                             mean_squared_error(tau_mean, scaled_pred)]
    
    # Save the aggregated metrics and predictions to CSV files
    df_max_r_list.to_csv(os.path.join(args.result_path, "Fig2_R_across_time.csv"))
    df_results.to_csv(os.path.join(args.result_path, "Fig2_Best_predictions.csv"))
    df_metrics = df_metrics.T
    df_metrics.to_csv(os.path.join(args.result_path, "Fig2_Metrics.csv"))
    
    print(df_results)
    
    return df_metrics, df_max_r_list, df_results


def main():
    """
    Main function for summarizing single connectivity results from SIR model simulations.

    This function sets up necessary arguments and calls the summarize_single_connectivity_results() function.
    It is designed to be run directly from the command line.

    Args:
        None (command-line arguments are parsed internally)

    Returns:
        None
    """
    import argparse
    parser = argparse.ArgumentParser(description="Summarize SIR simulation results for a single connectivity measure")
    # Required arguments
    parser.add_argument('--result_path', type=str, required=True, help='Full path to the results directory')
    parser.add_argument('--connectivity', type=str, required=True, help='Name of connectivity to be summarized')
    parser.add_argument('--epicenter', type=str, required=True, default="ctx_lh_entorhinal", help="Epicenter identifier used in the simulation")
    parser.add_argument('--input_data_name', type=str, default="Input_SIR_example.pkl", required=True, help="Filename of the main input data file")
    
    args = parser.parse_args()
    
    # Call the summarization function and get the results DataFrames
    summarize_one_conn(args)

if __name__ == "__main__":
    main()
