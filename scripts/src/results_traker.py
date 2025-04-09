"""
This module provides classes for tracking, updating, and saving simulation results and evaluation metrics 
for the SIR model. It includes two main classes:

    - ResultsTracker: Tracks simulation results and evaluation metrics for each region of interest (ROI).
    - TuningResultsTracker: Handles hyperparameter tuning by tracking simulation performance across different 
      hyperparameter combinations.

Usage:
    Instantiate one of the classes with the appropriate simulation data and command-line arguments. Use methods 
    like update_results, summary, save_results (for ResultsTracker) or update, get_best_params, print_and_plot_best 
    (for TuningResultsTracker) to process and save results.

Created on Fri Dec 15 2023, at Lund, Sweden
@author: XIAO Yu
"""

import os
import gzip
import time
import pickle
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error

from src.evaluation import get_top_rois, compute_metric, plot_interm_difference
from src.utils import plot_scatter_across_time, plot_line, scatter_pred_true

class ResultsTracker:
    """
    A class to track, update, and save simulation results and evaluation metrics.

    Attributes:
        tau (np.ndarray): The input tau data from initialization, shape (n_regions, n_subjects).
        T_total (int): Total time steps for simulation.
        eval_metrics (list of str): List of evaluation metrics to compute, e.g., ['r2', 'pearsonr', 'mae', 'mad'].
        evaluation_flag (bool): Whether to perform evaluations.
        return_interm_results (bool): Whether to save intermediate results.
        return_flag (bool): Whether to return results directly.
        output_path (str): Directory to save the results.
        output_model_name (str): Formatted model name used in output filenames.
        simulated_data (dict): Dictionary to store simulation results.
            - 'simulation' (dict): Contains simulated atrophy/pathology data, shape (n_regions, T_total).
            - 'Rmis' (dict): Contains misfolded protein data, shape (n_regions, T_total).
        correlation_all (dict): Dictionary to store all correlation data.
            - Key: Metric name (e.g., 'pearsonr'), Value: Dict of ROI correlations, shape (T_total, n_ROIs).
        correlation_df (dict): DataFrame to store mean correlations.
            - Key: Metric name (e.g., 'r2'), Value: Pandas DataFrame, shape (T_total, n_ROIs).
    """
    def __init__(self, tau, args):
        """
        Initialize the ResultsTracker object with simulation parameters and setup storage.

        Args:
            tau (np.ndarray): Input tau data from initialization, shape (n_regions,).
            args (Namespace): Command-line arguments with model settings.
        """
        self.tau = tau
        self.T_total = args.T_total
        self.eval_metrics = args.eval_metrics
        self.evaluation_flag = args.evaluation_flag
        self.return_interm_results = args.return_interm_results
        self.return_flag = args.return_flag
        self.output_path = args.output_path
        self.output_model_name = f"p_stay_{args.p_stay}_trans_rate_{args.trans_rate}_v_{args.v}_spr_time_{args.spr_time}_k1_{args.k1}_T_total_{args.T_total}_dt_{args.dt}"

        # Initialize storage structures
        result_keys = ["{i}" for i in args.epicenter_list]
        self.simulated_data = {key: {} for key in ["simulation", "Rmis"]}
        self.correlation_all = {key: {metric: {} 
                                      for metric in self.eval_metrics}
                                      for key in ["simulation", "Rmis"]}
        self.correlation_df = {key: {metric: pd.DataFrame(index=range(self.T_total), columns=result_keys)
                                     for metric in self.eval_metrics}
                                     for key in ["simulation", "Rmis"]}
        
    def update_results(self, repeat_epicenter, correlations, results, results_tmp, P_all):
        """
        Update the results with new simulation and correlation data.

        Args:
            repeat_epicenter (int): Current epicenter being processed.
            correlations (dict): Calculated correlation metrics. Shape: (T_total, n_ROIs).
            results (dict): Simulation results data.
                - simulated_atrophy (np.ndarray): Shape (n_regions, T_total).
                - Rmis_all (np.ndarray): Shape (n_regions, T_total).
            results_tmp (dict): Intermediate results (if any).
            P_all (np.ndarray): Array of probabilities during simulation, shape (n_regions, T_total).
        """
        self.simulated_data["simulation"][repeat_epicenter], self.simulated_data["Rmis"][repeat_epicenter] = results["simulated_atrophy"], results["Rmis_all"]
        if not self.return_flag: self.save_result_per_epicenter(results, results_tmp, P_all, repeat_epicenter)
        
        if self.evaluation_flag:
            for key in correlations:
                print("\nBest",key, end=" ")
                for metric in self.eval_metrics:
                    mean_corr = np.nanmean(correlations[key][metric], axis=1)
                    self.correlation_all[key][metric][repeat_epicenter] = correlations[key][metric].T
                    self.correlation_df[key][metric][repeat_epicenter]= mean_corr
                    if metric in ["pearsonr", "kendalltau"]:
                        print(f"{metric} at time {np.argmax(mean_corr)}: {np.max(mean_corr)}", end="\t")
                    elif metric in ["mae","mad"]:
                        print(f"{metric} at time {np.argmin(mean_corr)}: {np.min(mean_corr)}", end="\t")

    def summary(self):
        """
        Print summary of the results and save final outputs if required.
        Uses self.correlation_df to calculate and print top ROIs.
        """
        if not self.return_flag: self.save_results()
        if self.evaluation_flag:
            for key in self.correlation_df:
                for metric in self.eval_metrics:
                    get_top_rois(self.correlation_df, self.data["conn"]["name"], key, metric)

    def save_result_per_epicenter(self, results, results_tmp, P_all, repeat_epicenter):
        """
        Save intermediate results for each epicenter to `args.output_path`.

        Args:
            results (dict): Final simulation results.
                - simulated_atrophy (np.ndarray): Shape (n_regions, T_total).
                - Rmis_all (np.ndarray): Shape (n_regions, T_total).
            results_tmp (dict): Intermediate results, could contain various intermediate arrays.
            P_all (np.ndarray): Array of probabilities of misfolding, shape (n_regions, T_total).
            repeat_epicenter (int): Index of the current epicenter.
        """
        print("Saving epicenter result...\nto: {}".format(self.output_path))
        pickle.dump(results, open(os.path.join(self.output_path,"results_ROI_"+str(repeat_epicenter)+"_"+self.output_model_name+".pkl"),'wb'))
        if self.return_interm_results:
            pickle.dump(results_tmp, open(os.path.join(self.output_path,"results_intermediate_ROI"+str(repeat_epicenter)+"_"+self.output_model_name+".pkl"),'wb'))
            pickle.dump(P_all, open(os.path.join(self.output_path,"P_all_ROI"+str(repeat_epicenter)+"_"+self.output_model_name+".pkl"), 'wb'))
    
    def save_results(self):
        """
        Save the final simulation and evaluation results to `args.output_path`.

        Files saved:
            - simulated_atrophy_all_<model_name>.pt: Final simulation data.
            - correlation_all_<model_name>.pt: Correlation data for each metric.
            - correlation_mean_df_<model_name>.pt: Mean correlation values as DataFrame.
        """
        print("Saving final results...\nto: {}".format(self.output_path))
        pickle.dump(self.simulated_data, open(os.path.join(self.output_path,"simulated_atrophy_all_"+self.output_model_name+".pkl"),'wb'))
        if self.evaluation_flag:
            pickle.dump(self.correlation_all, open(os.path.join(self.output_path,"correlation_all_"+self.output_model_name+".pkl"),'wb'))
            pickle.dump(self.correlation_df, open(os.path.join(self.output_path,"correlation_mean_df_"+self.output_model_name+".pkl"),'wb'))


class TuningResultsTracker:
    """
    A class to track, update, and save the results during hyperparameter tuning.

    Attributes:
        checked_epicenter (str): The epicenter region being evaluated.
        mean_scatter_times_list (str): List of time points to create scatter plots, formatted as "start,interval".
        n_jobs (int): Number of parallel jobs to use for evaluation.
        T_total (int): Total number of time steps for simulation.
        output_path (str): Directory where results are saved.
        param_combinations (list of tuples): List of hyperparameter combinations to evaluate.
        null_model (int or None): Index of the null model, if applicable.
        y (np.ndarray or None): Ground truth tau values, shape (n_regions, ).
        r_dict (dict): Dictionary storing correlation metrics and their maximum values.
        best_combination (dict): Dictionary to store the best hyperparameter combination.
        simulated_data (dict): Dictionary to store simulated data.
    """
    def __init__(self, args, param_combinations):
        """
        Initialize the TuningResultsTracker object with given arguments and hyperparameter combinations.

        Args:
            args (Namespace): Parsed arguments containing model settings.
            param_combinations (list of tuples): List of hyperparameter combinations to evaluate.
        """
        self.checked_epicenter = [x for x in args.epicenter_list.split(",")][0] if isinstance(args.epicenter_list, str) else args.epicenter_list[0]
        self.mean_scatter_times_list = args.mean_scatter_times_list
        self.n_jobs = args.n_jobs
        self.T_total = args.T_total
        self.output_path = args.output_path
        self.param_combinations = param_combinations
        self.null_model = args.null_model_i

        # Initialize storage structures
        self.y = None
        self.r_dict = {"simulation":{self.checked_epicenter: {"combinations":[],"max_r_across_time":[-1000]}},
                       "Rmis":{self.checked_epicenter: {"combinations":[],"max_r_across_time":[-1000]}},
                       "max":{}}
        self.best_combination = {"simulation":{self.checked_epicenter:{}}, "Rmis":{self.checked_epicenter:{}}}
        self.simulated_data = {"simulation":[], "Rmis":[]}

    def update(self, combination, combination_i, simulated_data, tau_mean, results_tmp):
        """
        Update results with new simulation data and calculate correlations.

        Args:
            combination (tuple): Current hyperparameter combination.
            combination_i (int): Index of the combination.
            simulated_data (dict): Dictionary of simulation results.
            tau_mean (np.ndarray): Mean tau values, shape (n_regions, ).
            results_tmp (dict): Intermediate simulation results.
        """
        if self.y is None:  self.y = tau_mean
        for result_type in simulated_data:
            print("evaluate for",result_type)
            self.simulated_data[result_type].append(simulated_data[result_type][str(self.checked_epicenter)])
            self.get_r_across_time(simulated_data[result_type][str(self.checked_epicenter)],
                                   result_type, combination, results_tmp) 
            print("max r:", self.r_dict[result_type][self.checked_epicenter]["max_r_across_time"][-1])

            if self.mean_scatter_times_list is not None: self.scatter_plot_across_time(combination, simulated_data, result_type)
    
    def summary(self):
        """
        Print and summarize the best hyperparameter combinations for each result type.
        """
        for result_type in self.simulated_data:
            print("*"*5,result_type,"*"*5)
            max_r_list, ind_max_param, pred_best = self.get_best_params(result_type)

            if self.null_model is None:
                best_model_name = f"_p_stay_{self.param_combinations[ind_max_param][0]}_trans_rate_{self.param_combinations[ind_max_param][1]}_v_{self.param_combinations[ind_max_param][2]}"
                self.print_and_plot_best(result_type, pred_best, max_r_list, best_model_name)

    def save(self):
        """
        Save the best hyperparameter combinations and simulated data to `self.output_path`.
        """
        start = time.time()
        np.save(os.path.join(self.output_path,'Y_observed.npy'), self.y)
        for result_type in self.simulated_data:
            print("*"*5,result_type,"*"*5)
            with gzip.open(os.path.join(self.output_path,'Simulated_data_'+result_type+'.npy'), 'wb') as f:
                np.save(f, np.array(self.simulated_data[result_type], dtype=np.float16))
            # Save performance metrics for the current result type
            with open(os.path.join(self.output_path, "hyperparameters_model_performance_" + result_type + ".pkl"), "wb") as f:
                pickle.dump(self.r_dict[result_type], f, protocol=4)
            # Save intermediate outputs for the current result type
            with open(os.path.join(self.output_path, "hyperparameters_model_intermediate_outputs_" + result_type + ".pkl"), "wb") as f:
                pickle.dump(self.best_combination[result_type], f, protocol=4)
        end = time.time()
        print(f"saving time: {end - start}/60 mins")
    
    def get_r_across_time(self, predictions, result_type, combination, results_tmp):
        """
        Calculate the Pearson correlation across time between predicted and observed data.

        Args:
            predictions (np.ndarray): Predicted values for each time point, shape (n_regions, T_total).
            result_type (str): Type of result ('simulation' or 'Rmis').
            combination (tuple): Current hyperparameter combination.
            results_tmp (dict): Intermediate results.
        """
        # Parallel computation of Pearson correlation across all time points
        self.r_dict[result_type][self.checked_epicenter][combination] = Parallel(n_jobs=self.n_jobs)(
                delayed(compute_metric)("pearsonr", self.y, predictions[:,i]) for i in range(predictions.shape[1]))
        
        # Track the combination and update the best combination if necessary
        self.r_dict[result_type][self.checked_epicenter]["combinations"].append(combination)
        r_max_tmp = np.nanmax(self.r_dict[result_type][self.checked_epicenter][combination])
        if r_max_tmp >= np.nanmax(self.r_dict[result_type][self.checked_epicenter]["max_r_across_time"]): #len(r_dict_tmp[epicenter]["max"])==0 or 
            self.best_combination[result_type][self.checked_epicenter]["max_pattern"] = predictions
            self.best_combination[result_type][self.checked_epicenter]["max_combination"] = combination
            self.best_combination[result_type][self.checked_epicenter]["max_results_tmp"] = results_tmp
            print("updated max combination:", combination, r_max_tmp, "original:",self.r_dict[result_type][self.checked_epicenter]["max_r_across_time"][-1])
        self.r_dict[result_type][self.checked_epicenter]["max_r_across_time"].append(r_max_tmp)

    def get_best_params(self, result_type):
        """
        Identify and return the best hyperparameter combination based on maximum correlation.

        Args:
            result_type (str): Type of result ('simulation' or 'Rmis').

        Returns:
            tuple: (max_r_list, ind_max_param, pred_best)
        """
        self.r_dict[result_type][self.checked_epicenter]["max_r_across_time"] = self.r_dict[result_type][self.checked_epicenter]["max_r_across_time"][1:] # remove the first 0
        ind_max_param = np.nanargmax(self.r_dict[result_type][self.checked_epicenter]["max_r_across_time"]) #, axis=0)
        max_r_list = self.r_dict[result_type][self.checked_epicenter][self.param_combinations[ind_max_param]]
        max_time = np.nanargmax(max_r_list)
        max_r = np.nanmax(max_r_list)
        pred_best = self.best_combination[result_type][self.checked_epicenter]["max_pattern"][:,max_time]
        self.best_combination[result_type][self.checked_epicenter]["pred_best"] = pred_best
        self.r_dict["max"][result_type] = max_r

        print("max_r_across_time:",len(self.r_dict[result_type][self.checked_epicenter]["max_r_across_time"]))
        print("max combination is:", self.param_combinations[ind_max_param], max_r)

        return max_r_list, ind_max_param, pred_best

    def print_and_plot_best(self, result_type, pred, max_r_list, best_model_name):
        """
        Normalize the predicted values, calculate MSE, and plot the best results.

        Args:
            result_type (str): Type of result ('simulation' or 'Rmis').
            pred (np.ndarray): Best predicted values.
            max_r_list (list): List of maximum correlation values.
            best_model_name (str): Formatted string for saving plots.
        """
        pred_norm = (pred - np.nanmin(pred)) / (np.nanmax(pred) - np.nanmin(pred))
        pred_scaled = pred_norm * (np.nanmax(self.y) - np.nanmin(self.y)) + np.nanmin(self.y)
        mse = mean_squared_error((self.y-min(self.y))/(max(self.y)-min(self.y)), pred_scaled)
        
        save_png_name = os.path.join(self.output_path,'R_'+result_type+best_model_name+'.png')
        plot_line(max_r_list, save_png_name)
        scatter_pred_true(self.y, pred, save_name=save_png_name.replace("/R_","/Scatter_"))
        plot_interm_difference(self.best_combination[result_type][self.checked_epicenter]["max_results_tmp"],
                               self.output_path, result_type+best_model_name)
        print("MSE:", mse)

    def scatter_plot_across_time(self, combination, simulated_data, result_type):
        """
        Generate scatter plots of predicted vs. observed data across selected time points.

        Args:
            combination (tuple): Current hyperparameter combination.
            simulated_data (dict): Dictionary containing simulated results for each ROI.
            result_type (str): Type of result ('simulation' or 'Rmis').
        """
        # Create the output directory for scatter plots
        scatter_plot_path = os.path.join(self.output_path, self.checked_epicenter) 
        os.makedirs(scatter_plot_path, exist_ok=True)

        # Define time points for scatter plots
        checked_times = [1]+list(range(int(self.mean_scatter_times_list.split(",")[0]),
                                       self.T_total,int(self.mean_scatter_times_list.split(",")[1]))) +[self.T_total-1]
        model_name = f"p_stay_{combination[0]}_trans_rate_{combination[1]}_v{combination[2]}_T_total_{self.T_total}" ### !!! Need to change if input hyperparam changes
        plot_scatter_across_time(self.y, simulated_data[result_type][str(self.checked_epicenter)],
                                 self.checked_epicenter, checked_times,
                                  scatter_plot_path+"/"+model_name+"_"+result_type)



