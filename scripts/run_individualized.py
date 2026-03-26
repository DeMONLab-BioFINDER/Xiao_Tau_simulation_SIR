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
from src.results_traker import IndividualResultsTracker
from src.utils import clear_memory



def individual_simulation(args):
    """
    Perform hyperparameter tuning for the model.

    Args:
        args (Namespace): Command-line arguments parsed from input. See the list in parse_arguments() of params.py.

    Returns:
        tuple: Simulated data and the maximum result from the hyperparameter search if the model name contains 'null_model'.
    """
    # Define the subject ID list for individualized simulation
    #########################################  load from user-provided .txt file
    with open(args.input_path + "/Subject_IDs.txt") as f:
        subject_list = [line.strip() for line in f if line.strip()]
    ######################################### 
    results_subjs = IndividualResultsTracker(args)
    start_model_time = time.time()
    
    # Iterate over all hyperparameter combinations
    for subj in subject_list:
        print('runing for subject:',subj)
        args.subject_id = subj

        # Run the model with updated subject id
        sim_results, tau_subj, results_tmp = run_model(args)

        # Store the results and clear memory
        results_subjs.update(subj, sim_results, tau_subj, results_tmp)
        clear_memory(sim_results, tau_subj, results_tmp)
    
    # End of tuning process
    results_subjs.summary()
    results_subjs.save()

    end_time = time.time()
    print("Program ends at {}".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S")))
    print("Total Model time: {} hrs".format((end_time - start_model_time)/3600))


def main():
    """
    Main function to set up directories and run hyperparameter tuning.
    """
    # Setting up input arguments
    print("Basic arguments setting")    
    args = parse_arguments(hypertune=False) # hypertune=False for individualized simulation, hypertune=True for hyperparameter tuning
    args.return_flag = True # Whether to return results in `run_model(args)` function

    # Setting up logging
    setup_logging(log_filename=os.path.join(args.output_path,'model.log'))
    print("Input arguments: {}".format(args))
    
    # Run hyperparameter tuning
    individual_simulation(args)


if __name__ == "__main__":
    main()

