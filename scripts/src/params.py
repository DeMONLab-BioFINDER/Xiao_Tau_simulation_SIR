"""
Description:
    This module contains functions for parsing command-line arguments for the SIR model simulation.
    The main function provided by this module is `parse_arguments()`, which sets up and returns an 
    argparse.Namespace containing all the essential parameters for running the simulation. These 
    parameters include model settings, input file paths, simulation hyperparameters, evaluation settings, 
    and additional simulation options.

    The remaining functions in this module are affiliated with `parse_arguments()` and are used to:
        - Determine the project and script paths dynamically (`get_proj_path()`).
        - Read essential parameter settings from an external text file (`read_essential_params()`).
        - Create the output directory based on provided arguments (`make_output_dir()`).

Usage:
    Import the `parse_arguments()` function to obtain a Namespace object containing the simulation 
    configuration. For example:
        from params import parse_arguments
        args = parse_arguments(hypertune=False)

Created on Fri Dec 15 2023, at Lund, Sweden
@author: XIAO Yu
"""

import os
import argparse
from datetime import datetime


def parse_arguments(hypertune=False):
    """
    Parse command-line arguments for the SIR model.

    Args:
        proj_path (str): The project directory path.
        hypertune (bool): Whether to include hyperparameter tuning arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='SIR model.')

    # Load essential settings from external file if available
    script_dir, proj_path = get_proj_path()
    essential_params = read_essential_params(os.path.join(script_dir, 'User_input_settings.txt'))
    
    #  Model name and output settings
    # ===============================
    # Essential settings - Please change!
    parser.add_argument('--model_name', type=str, default=essential_params.get("model_name", "TEST_SIR_baseline_hypertune30000"),
                        help='Name of the model to be used as the result folder name. If you are running null models, please MAKE SURE to include "null_model" in the `model_name`')
    parser.add_argument('--simulated_protein', type=str, default=essential_params.get("simulated_protein", "tau"), help='Name of simulated protein for result folder')
    parser.add_argument('--protein_type', type=str, default=essential_params.get("protein_type", "Presence"), help='Simulated protein type. dataframe column name of the observed tau in `input_data_name` file. Choices: ["Presence", "Load", "Subtype1", "Subtype2", ...]')
    # Optional
    parser.add_argument('--output_date_time', type=str, default=essential_params.get("output_date_time"), help='Custom output folder name; defaults to current date and time') # output_date_time = "20240128_154456"
    
    # Input data files
    # ================
    # !Essential! - Please change!
    parser.add_argument('--input_data_name', type=str, default=essential_params.get("input_data_name", "Input_SIR_example.pkl"), help='Input data file name')
    # Optional: all other connectivities
    parser.add_argument('--regional_variable_file', type=str, default=essential_params.get("regional_variable_file", "regional_vulnerability_example.csv"), help='Input path of regional vulnerability for `spread_var`, `synthesis_var`, `misfold_var`, or `clearance_var`. MUST CHANGE together with rates.')
    parser.add_argument('--connectivity_file', type=str, default=essential_params.get("connectivity_file", "Connectomes_all.pkl"), help='File containing all possible connectivity matrices that could be used as `SC_len`')
    parser.add_argument('--SC', type=str, default=essential_params.get("SC", None), help='connectivity name in `connectivity_file`. If this exists, will use this to replace data["conn"] (loaded from `input_data_name`), and use as `SC_len` of simualtion')

    # Simulation essential input parameters
    # =====================================
    # !Essential! - Please change!
    parser.add_argument('--epicenter_list', type=str, default=essential_params.get("epicenter_list", "ctx_lh_entorhinal"), help='List of epicenter names / indices, seperated by commas. e.g., "ctx_lh_entorhinal,ctx_lh_precuneus" or "0,1"')
    # Optional: regional vulnerability mechanisms
    parser.add_argument('--spread_var', type=str, default=essential_params.get("spread_var", None), help='Spread rate variable name for misfolded protein')
    parser.add_argument('--synthesis_var', type=str, default=essential_params.get("synthesis_var", None), help='Synthesis rate variable name for normal protein')
    parser.add_argument('--misfold_var', type=str, default=essential_params.get("misfold_var", None), help='Misfolding rate variable name for normal protein')
    parser.add_argument('--clearance_var', type=str, default=essential_params.get("clearance_var", None), help='Clearance rate for both normal and misfolded proteins, will be renamed to clearance_nor_var & clearance_mis_var in run.py')
    
    # Simulation optional parameters
    # =============================
    parser.add_argument('--FC', type=str, default=essential_params.get("FC", None), help='Connectivity or regional information for merging with SC_den. e.g., merge default anatomical connectivity (SC_den) with functional connectivity derived from fMRI (FC)')
    parser.add_argument('--k', type=float, default=essential_params.get("k", 0), help='Weight of functional connectivity (FC), when merging with SC_den')
    parser.add_argument('--clearance_nor_var', type=str, default=essential_params.get("clearance_nor_var", None), help='Clearance rate for normal protein')
    parser.add_argument('--clearance_mis_var', type=str, default=essential_params.get("clearance_mis_var", None), help='Clearance rate for misfolded protein')

    # Simulation hyperparameters
    # ==========================
    parser.add_argument('--p_stay', type=float, default=essential_params.get("p_stay", 0.5), help='Probability of staying in the same region per unit time')
    parser.add_argument('--v', type=int, default=essential_params.get("v", 1), help='Protein spread speed from paths to regions')
    parser.add_argument('--trans_rate', type=float, default=essential_params.get("trans_rate", 1), help='Baseline infectivity control, a scalar value')
    parser.add_argument('--k1', type=float, default=essential_params.get("k1", 0.5), help='Weight ratio between misfolded protein accumulation and deafferentation for atrophy/tangle accrual')
    parser.add_argument('--init_number', type=int, default=essential_params.get("init_number", 1), help='Initial quantity of misfolded protein in epicenter')
    parser.add_argument('--T_total', type=int, default=essential_params.get("T_total", 30000), help='Total time steps for simulation')
    parser.add_argument('--dt', type=float, default=essential_params.get("dt", 0.1), help='Time step size, not equal to real world time')

    ##########################################
    # All the following arguments CAN NOT be specified using `User_input_settings.txt` file
    ##########################################
    # Evaluation settings
    # ===================
    parser.add_argument('--eval_metrics', type=list, default=['r2','pearsonr','mae','mad'], help='Metrics for model evaluation')
    parser.add_argument('--evaluation_flag', type=bool, default=False, help='Perform evaluation after simulation')

    # Additional simulation settings
    # ==============================
    parser.add_argument('--bootstrap', type=int, default=None, help='Number of data bootstraps for hyperparameter selection')
    parser.add_argument('--null_model_i', type=int, default=None, help='i th iteration of the null model')
    parser.add_argument('--n_jobs', type=int, default=3, help='Number of parallel jobs for evaluation (n_jobs=-1 (use all available computational cores))')
    # Input & Output
    parser.add_argument('--same_ROI_size', type=str, default=None, help='Uniform ROI size ("mean") or specific size number (int)')
    parser.add_argument('--return_interm_results', type=bool, default=False, help='Save and return intermediate variables')
    parser.add_argument('--interm_variabels', type=str, default="movOut_mis,movDrt_mis,N_misfolded", help='Intermediate variables names to be saved') #Rnor_after_spread,Rmis_after_spread,Rnor_cleared,Rmis_cleared,misProb,N_misfolded,
    parser.add_argument('--mean_scatter_times_list', type=str, default=None, help='Whether to check the mean scatter plot of mean tau (in order to choose hyperparameters)') 
    # Try different simulation mechanisms
    parser.add_argument('--load_results', type=str, default=None, help='Path to partial results for continuation in _mis_spread() function')
    parser.add_argument('--Rnor0', type=str, default=None, help='Whether input Rnor0 and skip _norm_spread. Chioices: ["output", "MAPT", <filepath>]')
    parser.add_argument('--file_as_Rnor0', type=str, default="mirrored_MAPT_APOE_selected.csv", help='Whether input Rnor0 and skip _norm_spread. Chioices: ["output", "MAPT", <filepath>]')
    parser.add_argument('--no_norm_spread', type=bool, default=False, help='Disable normal protein spread during misfolded spread')
    parser.add_argument('--spr_time', type=float, default=0, help='Time step when spread rate is added to the model')

    # Parse arguments and set up the output directory
    args, unknown = parser.parse_known_args()
    args = make_output_dir(args,proj_path, script_dir, hypertune)

    return args

def make_output_dir(args, proj_path, script_path, hypertune):
    """
    Create the output directory based on model arguments.

    Args:
        args (Namespace): Parsed arguments.
        proj_path (str): Project path.
        hypertune (bool): Indicates if hypertuning is being performed.

    Returns:
        Namespace: Updated arguments with output path.
    """
    # Generate output date and time
    output_date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("Program starts at {}".format(datetime.strptime(output_date_time, "%Y%m%d_%H%M%S").strftime("%d-%m-%Y %H:%M:%S")))
    if not args.output_date_time: args.output_date_time=output_date_time

    # Define paths
    args.proj_path = proj_path
    args.script_path = script_path
    args.input_path = os.path.join(proj_path, "data")
    
    # Determine folder structure based on hypertuning and epicenter
    tune_folder = "hypertune" if hypertune else ""
    epi = "precuneus" if "precuneus" in args.epicenter_list else ("entorhinal" if "entorhinal" in args.epicenter_list else "")
    if "null_model" in args.model_name: tune_folder = "Null_models"

    # Add connectivity and regional vulnerability to model_name
    if args.SC and args.SC not in args.model_name: args.model_name += "_" + args.SC
    for var in ["spread_var", "synthesis_var", "misfold_var", "clearance_nor_var", "clearance_mis_var"]:
        if isinstance(getattr(args, var), str):
            regional_name = var.split("_")[0] + "-" + getattr(args, var)
            if regional_name not in args.model_name:
                args.model_name += "_" + regional_name

    # Construct output path
    args.output_path = os.path.join(proj_path, "results", args.simulated_protein, args.data_type, tune_folder, epi, args.model_name + "_" + args.output_date_time)

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    print(f"Output directory created at: {args.output_path}")

    return args

def read_essential_params(file_path):
    """
    Read essential parameters from a .txt file.

    Args:
        file_path (str): Path to the .txt file containing essential parameters.

    Returns:
        dict: Dictionary with parameter names and values.
    """
    params = {}
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    params[key.strip()] = value.strip().strip('"')
        print("Successfully loaded essential parameters from:", file_path)
    except Exception as e:
        print("Error reading essential parameters file:", str(e))
    
    output_date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("Program starts at {}".format(datetime.strptime(output_date_time, "%Y%m%d_%H%M%S").strftime("%d-%m-%Y %H:%M:%S")))
    if "output_date_time" not in params: params["output_date_time"] = output_date_time
    return params

def get_proj_path():
    """
    Dynamically determine the project path based on the current script's location.
    
    Returns:
        list: A list containing the script directory and the project path.
    """
    # Print a message indicating that the project path is being set up
    print("Setting project path based on the current script directory")
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir = os.path.abspath(os.path.join(current_file_dir, os.pardir))

    # Set the project path to the grandparent folder (uncomment if needed)
    proj_path = os.path.abspath(os.path.join(script_dir, os.pardir)) # set to grandparent folder
    print("Project path set to:", proj_path)
    return [script_dir, proj_path]
