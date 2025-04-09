"""
Description:
    This module summarizes and extracts results from SIR model simulations. The module
    also identifies any missing simulation outputs and writes them to a text file.

Key Functionalities:
    - Summarize the simulation results by aggregating evaluation metrics from various models.
    - Extract top-performing model details based on a specific connectivity index.
    - Identify and log any missing simulation results.
    - Generate summary CSV files and save aggregated metrics to disk.

Usage:
    Run the script from the command line with the following required arguments:
        --result_path: Full path to the results directory (e.g., /Users/xiaoyu/Desktop/Lund/projects/SIR/results/tau/0-1Norm/hypertune/).
        --index_conn: Index of the connectivity measure to extract (e.g., 0).
        --epicenter: Epicenter identifier used in the SIR model (e.g., "ctx_lh_entorhinal").
        --input_data_name: Filename of the main input data file (e.g., Input_SIR_example.pt).
        --connectivity_file: Filename for the connectivity matrix file (e.g., Connectomes_all.pt).
    And the following optional arguments:
        --eval_metric: Evaluation metric to use (default: "pearsonr"; choices: "pearsonr", "mse").
        --subtype: Subtype index; if greater than or equal to 0, observed tau values are loaded for the specified subtype (default: -1).
        --subtype_file: CSV file containing tau values for each subtype (default: "Tau_subtypes_example.csv").
    
    Example:
        python run_summary_all_conns.py --result_path "/path/to/results" --index_conn 2 --epicenter "ctx_lh_entorhinal" \
            --input_data_name "Input_SIR_example.pt" --connectivity_file "Connectomes_all.pt"

!!!!!Caution!!!!!:
    Before running the script, please review and update the CONNECTIVITY MEASURE LIST (`conn_list`) and the REGIONAL FACTORS 
    DICTIONARY (`factors_dict`) in the source code if necessary, to ensure they align with your specific simulation model settings.

Dependencies:
    - Standard libraries: os, sys, argparse, warnings
    - Third-party libraries: numpy, pandas, matplotlib (pyplot)
    - Custom modules: summary (infer_from_result_path, extract_all_conns)

Created on Fri Dec 15 2023, at Lund, Sweden
@author: XIAO Yu
"""

import argparse
import warnings

from summary import infer_from_result_path, extract_all_conns

def summary_all_conns(args):
    """
    Summarize simulation results for SIR models based on connectivity measures and subtypes.

    This function sets up the necessary paths using the project directory, defines the list of connectivity
    measures and associated regional factors, and then calls the extract_all_models() function to retrieve
    and aggregate evaluation metrics from simulation outputs. If any models are missing, they are logged
    and written to a text file.

    Args:
        args (Namespace): Parsed command-line arguments containing:
            - result_path (str): Relative path to the results directory.
            - index_conn (int): Index of the connectivity measure to extract.
            - epicenter (int or str): Identifier for the epicenter used in the SIR model.
            - eval_metric (str): Evaluation metric to use (e.g., "pearsonr" or "mse").
            - subtype (int): Subtype index for selecting observed tau values.
            - input_dir (str): Directory path for input data.
            - input_data_name (str): Filename for the input simulation data.
            - connectivity_file (str): Filename for the connectivity matrix.
            - subtype_file (str): Filename for the CSV file with tau subtypes.
    """
    ######################################### !Change the conn_list and factor_dict based on your own model!
    # Define a list of connectivity measures 
    conn_list = ["sc", "gene_coexpression", "neurotransmission_similarity"]

    # Define a dictionary mapping regional factor groups to specific factors (except for "Baseline")
    factors_dict={"Baseline":["Baseline"], # <----------------------------------------------------- !!! DON'T change this !!!!!
                  "AD-related": ["MAPT", "APOE"] #, "Abeta", "Cluster1", "Cluster2", "Cluster3", "Cluster4", "Cluster5", "Cluster6", "Cluster7"],
                  #"Receptors":["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT", "A4B2", "CB1",
                  #             "D1", "D2", "DAT", "GABAa", "H3", "M1", "mGluR5", "MOR", "NET", "NMDA", "VAChT"],
                  #"Microstructure":["genepc1", "myelinmap", "thickness"],
                  #"metabolism":["cbf", "cbv", "cmr02", "cmrglc"],
                  #"Cortical expansion":["devexp", "evoexp", "scalingnih", "scalingpnc", "scalinghcp", "scaling"],
                  #"Celltype":["Astro", "Endo", "Micro", "Neuro-Ex", "Neuro-In", "Oligo", "OPC"],
                  }
    ######################################### !Change the conn_list and factor_dict based on your own model!

    # Define a list of simulation mechanism variable names (synthesis, spread, misfold, clearance)
    var_lists=["syn", "spread", "mis", "clear"]

    # Ignore FutureWarnings to keep the output clean
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Infer the input directory and data type
    args = infer_from_result_path(args)
    
    # Call extract_all_conns using the connectivity measure specified by index_conn in the args
    _, missing_results = extract_all_conns([conn_list[args.index_conn]], factors_dict, var_lists, args)

    # If any simulation results are missing, log and save them to a text file
    if len(missing_results) > 0:
        print("MODELS MISSING!!!")
        print(missing_results)
        with open(args.result_path + "/missing_results.txt", 'w') as file:
            for item in missing_results:
                file.write(f"{item}\n")
    print("Extraction done!!!!")

def main():
    """
    Main function to parse command-line arguments and summarize simulation results.

    This function sets up the command-line interface for the summarization process. It expects
    arguments for the result path, connectivity index, epicenter, evaluation metric, subtype, input data name,
    connectivity file, and subtype file. After parsing these arguments, it calls summarize_results() to
    process and summarize the simulation outputs.
    """
    parser = argparse.ArgumentParser(description="Summarize SIR results of all models")
    # Required arguments
    parser.add_argument('--result_path', type=str, required=True, help='Full path to the results directory')
    parser.add_argument('--index_conn', type=int, help='Index of the connectivity measure to extract. Index in range(conn_list), see `conn_list` in the main function `summary_all_conns(args)` in the current file for details')
    parser.add_argument('--epicenter', type=str, required=True, default="ctx_lh_entorhinal", help='Epicenter used in the SIR model')
    parser.add_argument('--input_data_name', type=str, default="Input_SIR_example.pkl", required=True, help="Filename of the main input data file")
    parser.add_argument('--connectivity_file', type=str, default="Connectomes_all.pkl", required=True, help='File containing all possible connectivity matrices that could be used as `SC_len`')
    # Optional arguments
    parser.add_argument('--eval_metric', type=str, default="pearsonr", choices=["pearsonr","mse"], help='Evaludation metric  to use')
    parser.add_argument('--subtype', type=int, default=-1, help='Subtype index; if >= 0, observed tau values are loaded for the specified subtype')
    parser.add_argument('--subtype_file', type=str, default="Tau_subtypes_example.csv", help='CSV file containing tau values for each subtype (shape: (n_regions, subtypes))')
    
    args = parser.parse_args()
    print("summarization start")
    summary_all_conns(args)

if __name__ == "__main__":
    main()

