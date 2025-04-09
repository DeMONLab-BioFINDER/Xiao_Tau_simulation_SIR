"""
Description:
    This module serves as the main entry point for running the SIR model simulation.
    It sets up the simulation environment by parsing command-line arguments, initializing
    required variables and data structures, and executing the tau propagation and atrophy simulation.
    
    The module performs the following steps:
        1. Parse command-line arguments and load external settings.
        2. Initialize simulation data and configuration via the init_run module.
        3. Run the simulation model across specified regions of interest (ROIs) using the simulated_atrophy module.
        4. Optionally evaluate the simulation results if evaluation is enabled.
        5. Save and summarize simulation outputs, including simulated data and correlation metrics.
    
    Key Functions:
        - run_model(args): Executes the simulation for tau propagation and atrophy across ROIs.
        - main(): Orchestrates the setup (arguments, logging) and triggers the simulation execution.
    
Dependencies:
    - Standard libraries: os, time, datetime
    - Third-party libraries: torch
    - Custom modules: log_redirector, simulated_atrophy, init_run, params, results_traker, evaluation, utils

Usage:
    Run this script directly from the command line:
        python run.py
    or
        python run.py --model_name My_SIR
        (other parameters listed in params.py could also be added)

Created on Fri Dec 15 2023, at Lund, Sweden
@author: XIAO Yu
"""

import os
import time
import pickle
from datetime import datetime

import src.simulated_atrophy as sim
from src.params import parse_arguments
from src.init_run import initialize_run
from src.results_traker import ResultsTracker
from src.evaluation import evaluate
from src.utils import clear_memory, find_right_hemisphere
from src.log_redirector import setup_logging

def run_model(args):
    """
    Run the model for simulating tau propagation and atrophy.

    Args:
        args (Namespace): Command-line arguments parsed from input.

    Returns:
        Tuple: Simulated data, tau mean, intermediate results (if return_flag is True)
    """
    start_time = time.time()

    # Load initial data and variables
    args, init_vars = initialize_run(args)

    # Start SIR model simulation
    start_model_time = time.time()
    # Initialize results tracker for tau simulation
    sim_results = ResultsTracker(init_vars["tau"], args)

    # Iterate through each epicenter
    for repeat_epicenter in args.epicenter_list:
        # Determine left and right hemisphere ROIs
        if isinstance(repeat_epicenter, str): 
            repeat_epicenter_mirror = find_right_hemisphere(repeat_epicenter)
            if isinstance(init_vars['name'], list):
                repeat_epicenter_lr = [init_vars['name'].index(x) 
                                            for x in [repeat_epicenter, repeat_epicenter_mirror] if x]
            else:
                repeat_epicenter_lr =  [init_vars['name'].index[init_vars['name']==x][0]
                                            for x in [repeat_epicenter, repeat_epicenter_mirror] if x]
        
        # Print region of interest (ROI) information
        if len(repeat_epicenter_lr) == 1:
            print(f"ROI {str(repeat_epicenter_lr[0])}: {init_vars['name'][repeat_epicenter_lr[0]]}")
        else:
            print(f"ROI {str(repeat_epicenter_lr[0])} and {str(repeat_epicenter_lr[1])}: {init_vars['name'][repeat_epicenter_lr[0]]},  {init_vars['name'][repeat_epicenter_lr[1]]}")
	    
        # Check for partial results
        print("Simulating ...")
        if not args.load_results:
            results_partial = None
        else:
            print("loading partial result from ", args.load_results)
            results_partial = pickle.load(open(os.path.join(args.load_results,"results_ROI"+str(repeat_epicenter)+".pt"), 'rb'))
        
        # Run the atrophy simulation
        results, results_tmp, P_all = sim.simulate_atrophy(init_vars["conn"], repeat_epicenter_lr, init_vars["roi_size"],
                                                           SC_len=args.SC_len, FC=args.FC, p_stay=args.p_stay,
                                                           v=args.v, trans_rate=args.trans_rate,
                                                           k1=args.k1, init_number=args.init_number,
                                                           T_total=args.T_total, dt=args.dt, spr_time=args.spr_time,
                                                           spread_var=args.spread_var, synthesis_var=args.synthesis_var, misfold_var=args.misfold_var,
                                                           clearance_nor_var=args.clearance_nor_var, clearance_mis_var=args.clearance_mis_var,
                                                           return_interm_results=args.return_interm_results, interm_variabels=args.interm_variabels,
                                                           results_partial=results_partial, Rnor0=init_vars["Rnor0"], no_norm_spread=args.no_norm_spread)
        # Evaluate the simulation results if required
        if args.evaluation_flag:
            print("Evaluating ...")
            correlations = evaluate(results["simulated_atrophy"], results["Rmis_all"], init_vars["tau"], args.eval_metrics, args.T_total, n_jobs=args.n_jobs)
            pickle.dump(correlations, open(os.path.join(args.output_path,"correlations_ROI"+str(repeat_epicenter)+".pkl"),'wb'))
        else:
            correlations = None
        
        # Update simulation results and clear memory
        sim_results.update_results(repeat_epicenter, correlations, results, results_tmp, P_all)
        clear_memory(repeat_epicenter, correlations, results, results_tmp, P_all)

    # Print simulation summary
    end_model_time = time.time()
    sim_results.summary()
    end_time = time.time()
    print("Program ends at {}".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S")))
    print("Total run time: {} hrs".format((end_time - start_time)/3600))
    print("Model run time: {} seconds".format(end_model_time - start_model_time))
    print("Output saving time: {} seconds".format(end_time - end_model_time))

    if args.return_flag:
        results_tmp["index_tau_to_conn"] = init_vars["index_tau_to_conn"]
        return sim_results.simulated_data, init_vars["tau"], results_tmp


def main():
    """
    Entry point of the script. Sets up directories and arguments, then runs the model.
    """
    # Parse arguments
    print("Basic arguments setting")
    args = parse_arguments()
    args.return_flag = False # Whether to return results in `run_model(args)` function

    # Setup logging
    setup_logging(log_filename=os.path.join(args.output_path,'model.log'))
    print("Input arguments: {}".format(args))
    
    # Run the model
    run_model(args)


if __name__ == "__main__":
    main()
    
