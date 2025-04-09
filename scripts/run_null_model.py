"""
Description:
    This module is designed to run null model simulations for the SIR model.
    In a null model, the simulation is run over a range of specified null model iterations,
    where each iteration is distinguished by an index parsed from the model_name argument.
    The module performs hyperparameter tuning for each null model iteration using the 
    hypertune() function, saves the simulated data and correlation values for each iteration,
    and then aggregates and saves the final correlation metrics.

Key Functions:
    - run_null_models(args): Iterates over the specified range of null model iterations,
                             runs the hyperparameter tuning for each, and saves the results.
    - main(): Sets up the input arguments and logging, then triggers the null model simulations.

Dependencies:
    - Standard libraries: re, os, gzip, time, numpy, datetime
    - Third-party libraries: torch
    - Custom modules: run_hypertune, src/params, src/results_traker, src/utils, src/log_redirector

Usage:
    Execute this script directly to run null model simulations:
        python run_null_model.py
    or
        python run_null_model.py --model_name My_SIR
        (other parameters listed in params.py could also be added)

Created on Fri Dec 15 2023, at Lund, Sweden
@author: XIAO Yu
"""

import re
import os
import gzip
import time
import numpy as np
from datetime import datetime

from run_hypertune import hypertune
from src.params import parse_arguments
from src.log_redirector import setup_logging


def run_null_models(args):
    """
    Run null model simulations by iterating over a specified range of null model indices.

    The function extracts the start and end indices from the args.model_name string using a regular expression.
    For each null model iteration, it updates the args.null_model_i, runs the hypertuning process,
    saves the simulated data for each result type, and aggregates the maximum correlation values.

    Args:
        args (Namespace): Parsed command-line arguments. Expected to include at least:
            - model_name (str): Should contain a substring in the format "null_modelX-Y", where X and Y are integers.
            - output_path (str): Directory path where results will be saved.
            - Additional hyperparameter and simulation settings required by hypertune().
    
    Returns:
        None if not a null model run. If args.model_name includes 'null_model', it returns a tuple:
            (simulated_data, r_all) where:
                - simulated_data (dict): Simulated results for each result type.
                - r_all (dict): Dictionary containing maximum correlation values for each result type over all iterations.
    """
    start_model_time = time.time()

    # Extract start and end null model indices from the model_name string (e.g., "null_model1-5")
    start_end = re.search(r'null_model(\d+)-(\d+)', args.model_name).groups()
    start, end = int(start_end[0]), int(start_end[1])
    # Initialize dictionary to store maximum correlation values for each result type over iterations
    r_all = {"simulation": np.zeros(end-start),"Rmis":np.zeros(end-start)}
    
    # Iterate over the specified range of null model iterations
    for i in range(start, end):
        args.null_model_i = i
        print("*"*20,"\nNULL MODEL", args.null_model_i,"\n","*"*20)
        
        # Run hyperparameter tuning for the current null model iteration
        simulated_data, r_max = hypertune(args)

        # Store the maximum correlation values and save simulated data for each result type
        for result_type in r_all:
            r_all[result_type][i-start] = r_max[result_type]
            with gzip.open(os.path.join(args.output_path, "Null_model_results_"+result_type+"_"+str(i)+".npy"), 'wb') as f:
                np.save(f, np.array(simulated_data[result_type], dtype=np.float16))
    
    # Save the aggregated maximum correlation values for each result type
    for result_type in r_all:
        with gzip.open(os.path.join(args.output_path,"R_values_"+result_type+"_"+str(i)+".npy"), 'wb') as f:
            np.save(f, np.array(r_all[result_type], dtype=np.float16))
    
    end_time = time.time()
    print("Program ends at {}".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S")))
    print("Total NULL Model time: {} hrs".format((end_time - start_model_time)/3600))


def main():
    """
    Entry point of the script. Sets up directories and arguments, then runs the model.
    """
    # Parse arguments
    print("Basic arguments setting")
    args = parse_arguments(hypertune=True)
    args.return_flag = True # Whether to return results in `run_model(args)` function

    # Setup logging
    setup_logging(log_filename=os.path.join(args.output_path,'model.log'))
    print("Input arguments: {}".format(args))
    
    # Run null model simulations
    run_null_models(args)


if __name__ == "__main__":
    main()

