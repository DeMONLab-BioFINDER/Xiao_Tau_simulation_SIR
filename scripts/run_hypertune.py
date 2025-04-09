"""
run_hypertune.py

Description:
    This module serves as the entry point for hyperparameter tuning of the SIR model simulation, relies on `run_model()` function in run.py.
    It sets up the simulation environment by parsing command-line arguments specifically configured for hypertuning,
    defines a grid of hyperparameters, and executes the simulation for each hyperparameter combination.
    The performance of each combination is tracked using the TuningResultsTracker, and the best parameters
    are identified and saved. This module supports running the tuning process for both standard and null models.
    
    Key Functions:
        - hypertune(args): Iterates over all hyperparameter combinations, runs the simulation for each,
                           updates the tuning results, and saves the best-performing configuration.
        - main(): Sets up input arguments, logging, and triggers the hyperparameter tuning process.

Dependencies:
    - Standard Libraries: os, time, itertools, datetime
    - Custom Modules: run, src/log_redirector, src/params, src/results_traker, src/utils

Usage:
    Execute this module directly to start the hyperparameter tuning process:
        python run_hypertune.py
    or
        python run_hypertune.py --model_name My_SIR
        (other parameters listed in params.py could also be added)

Created on Fri Dec 15 2023, at Lund, Sweden
@author: XIAO Yu
"""

import os
import time
import itertools
from datetime import datetime

from run import run_model
from src.log_redirector import setup_logging
from src.params import parse_arguments
from src.results_traker import TuningResultsTracker
from src.utils import clear_memory



def hypertune(args):
    """
    Perform hyperparameter tuning for the model.

    Args:
        args (Namespace): Command-line arguments parsed from input. See the list in parse_arguments() of params.py.

    Returns:
        tuple: Simulated data and the maximum result from the hyperparameter search if the model name contains 'null_model'.
    """
    # Define the hyperparameter grid for tuning
    #########################################  suggested range, !CHANGE ONLY IF NEEDED!
    param_grid = {
        'p_stay': [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9],
        'trans_rate': [0.1, 0.5, 0.9, 1, 1.5, 1.9, 2, 2.5, 2.9, 3],
        'v': [0.1, 0.3, 0.5, 0.7, 0.9, 1]
    }
    ######################################### 
    # Generate all combinations of hyperparameters
    param_combinations = list(itertools.product(*param_grid.values()))
    
    # Initialize a tracker for tuning results
    hyper_results = TuningResultsTracker(args, param_combinations)
    start_model_time = time.time()
    
    # Iterate over all hyperparameter combinations
    for i, combination in enumerate(param_combinations):
        print('hyperparameters:', param_grid.keys(), combination)
        # Dynamically update arguments based on current combination
        [setattr(args, name, value) for name, value in zip(param_grid.keys(), combination)]

        # Run the model with updated hyperparameters
        simulated_data, tau_mean, results_tmp = run_model(args)

        # Store the results and clear memory
        hyper_results.update(combination, i, simulated_data, tau_mean, results_tmp)
        clear_memory(simulated_data, tau_mean, results_tmp)
    
    # End of tuning process
    end_model_time = time.time()
    hyper_results.summary()
    if "null_model" not in args.model_name: hyper_results.save()

    end_time = time.time()
    print("Program ends at {}".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S")))
    print("Total Model time: {} hrs".format((end_time - start_model_time)/3600))
    print("Best param finding time: {} seconds".format(end_time - end_model_time))

    if "null_model" in args.model_name:
        return hyper_results.simulated_data, hyper_results.r_dict["max"]

def main():
    """
    Main function to set up directories and run hyperparameter tuning.
    """
    # Setting up input arguments
    print("Basic arguments setting")    
    args = parse_arguments(hypertune=True)
    args.return_flag = True # Whether to return results in `run_model(args)` function

    # Setting up logging
    setup_logging(log_filename=os.path.join(args.output_path,'model.log'))
    print("Input arguments: {}".format(args))
    
    # Run hyperparameter tuning
    hypertune(args)


if __name__ == "__main__":
    main()

