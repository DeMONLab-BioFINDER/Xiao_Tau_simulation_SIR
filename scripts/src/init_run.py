"""
Description:
    This module is responsible for initializing the simulation environment for the SIR model.
    It loads all necessary input data (e.g., tau values, connectivity matrices, regional parameters)
    and adjusts the input arguments (args) based on the loaded data for each simulation run. The module
    sets up essential variables such as connectivity matrices, region names, tau values, ROI sizes, and
    other parameters required for running the SIR simulation. It also handles adjustments based on 
    subtypes, null models, and the availability of regional gene data.

Key Functionalities:
    - initialize_run(args): Loads simulation data, processes connectivity and tau values, adjusts ROI sizes,
      and sets up indices to match simulation outputs with connectivity matrices.
        - match_and_update(label_conn, data_to_update): Matches region labels between the connectivity matrix and
        the simulation output data, returning either matched indices or updated data accordingly. Called by initialize_run()

Usage:
    This module is intended to be used by the main simulation runner (run.py) to initialize the simulation
    environment and update the input arguments accordingly. The primary function, initialize_run(), is called by run.py.

Created on Fri Dec 15 2023, at Lund, Sweden
@author: XIAO Yu
"""
import os
import pickle
import numpy as np
import pandas as pd

def initialize_run(args):
    """
    Load and initialize simulation data and update input arguments for the SIR model run.

    This function loads the input data (e.g., tau values, connectivity matrices) from disk, processes
    the data to extract necessary simulation parameters, and updates the args Namespace with additional
    attributes needed for the simulation. These include connectivity matrices, region names, ROI sizes,
    indices to match tau with connectivity, and other regional variables.

    Returns:
        tuple: (args, initialized_variables)
            - args (Namespace): Updated command-line arguments with additional attributes such as:
                  - N_regions (int): Number of brain regions.
                  - SC_len: Connectivity length matrix, if available.
                  - input_path, proj_path, etc.
            - initialized_variables (dict): Dictionary containing initialized simulation variables:
                  - "tau_all": Tau data (after any filtering), expected shape (n_subjects, n_regions).
                  - "conn": Connectivity matrix data.
                  - "name": Region names (pd.Series) matching the connectivity matrix.
                  - "roi_size": Array of ROI sizes, shape (n_regions,).
                  - "tau_mean": Mean tau values, shape (n_regions,).
                  - Other indices such as "index_tau_to_conn" etc.
    """
    # Initialize a dictionary to store various simulation variables
    initialized_variables = { "tau": None, "roi_size": None, "index_tau_to_conn": None, "Rnor0": None}

    data, data_individual = load_main_data(args)

    load_tau(args, data, initialized_variables, data_individual)

    load_connectivity(args, data, initialized_variables, data_individual)

    setup_epicenters(args)

    load_roi_size(args, data, initialized_variables, data_individual)

    load_regional_variables(args, initialized_variables, data_individual)

    additional_simulation_settings(args, initialized_variables)

    # Print summary information about the data
    # ==============================
    print("Number of regions: {}".format(args.N_regions))
    print("SC_len: ", args.SC_len.shape, args.SC_len)
    print("Connectivity:",initialized_variables["conn"].shape, initialized_variables["conn"])
    print("Epicenter list: ", type(args.epicenter_list),type(args.epicenter_list[0]), args.epicenter_list)

    return args, initialized_variables


def load_main_data(args):
    print("Loading group data...\nfrom: {}".format(os.path.join(args.input_path, args.input_data_name)))
    data = pickle.load(open(os.path.join(args.input_path, args.input_data_name), 'rb'))

    if args.subject_id is not None:
        print("Loading individualized data...\nfrom: {}".format(os.path.join(args.input_path, args.individual_data_file)))
        data_individual = pickle.load(open(os.path.join(args.input_path, args.individual_data_file), 'rb'))
    else:
        data_individual = None

    return data, data_individual
    
    
def load_tau(args, data, initialized_variables, data_individual=None):
    """
    Load tau values (group-level or individual-level).
    convert values to numeric; expected shape: (, n_regions)
    """
    if args.subject_id is not None and args.simulated_protein in data_individual: ### individualized data
        print(f"loading individualized tau for subject {args.subject_id}")
        initialized_variables["tau"] = (data_individual["tau"][args.subject_id].astype(float).values.reshape(-1))
    else:
        if args.simulated_protein in data:
            initialized_variables["tau"] = data[args.simulated_protein][args.data_type].values.astype(float).reshape(-1)
        else:
            raise ValueError("No simulated data name found in the input file.")

def setup_epicenters(args):
    if args.epicenter_list is None:
        print("Set epicenter list to all the ROIs")
        args.epicenter_list = list(range(int(args.N_regions/2)))
    else:
        args.epicenter_list = ([x for x in args.epicenter_list.split(',')] if isinstance(args.epicenter_list, str) else args.epicenter_list)

def load_connectivity(args, data, initialized_variables, data_individual=None):
    """
    Load connectivity matrix and region names.
    """
    ### group-level
    initialized_variables["conn"] = data['conn']['conn']
    initialized_variables["name"] = pd.Series(data["conn"]["name"])
    args.N_regions = len(initialized_variables["conn"])
    # Set SC_len if available from the data file
    if 'SC_len' in data['conn'].keys():
        args.SC_len = data['conn']['SC_len']
    else:
        print("No SC_len provided, need to be cautious of `v` value!")
        args.SC_len = None

    load_alternative_connectivity(args, initialized_variables)

    ### individualized connectivity, overwrite group-level if subject_id provided. 
    if args.subject_id is not None:
        print(f"loading individualized connectivity for subject {args.subject_id}")
        initialized_variables["conn"] = data_individual["conn"][args.subject_id]
        initialized_variables["name"] = pd.Series(data_individual["name"]) # assume the name is the same for all subjects
        args.N_regions = initialized_variables["conn"].shape[0]

        if "SC_len" in data_individual: args.SC_len = data_individual["SC_len"][args.subject_id] ### overwrite group-level SC_len if provided

def load_alternative_connectivity(args, initialized_variables):
    """
    Load alternative connectivity matrix if specified in args.
    Group level only.
    """
    # If an alternative connectivity matrix (SC) is specified in args, load and update the connectivity data
    if args.SC is not None:
        print("load", args.SC, "from", args.connectivity_file)
        conn_matrix = pickle.load(open(os.path.join(args.input_path,args.connectivity_file), 'rb'))
        if args.null_model_i is not None:
            print("loading null model matrix",args.null_model_i)
            initialized_variables["conn"] = conn_matrix[args.SC][int(args.null_model_i)]
        else:
            initialized_variables["conn"] = conn_matrix[args.SC]
        if len(conn_matrix["labels"]) != len(initialized_variables["name"]): # match name and index
            initialized_variables["index_tau_to_conn"] = match_and_update(conn_matrix["labels"], initialized_variables["name"])
            initialized_variables["name"] = conn_matrix["labels"]
            initialized_variables["tau"] = initialized_variables["tau"][initialized_variables["index_tau_to_conn"]] # already 1d, .iloc[:,initialized_variables["index_tau_to_conn"]]
        print("matched tau:", initialized_variables["tau"].shape)
        if args.SC_len is not None: 
            if args.SC in ["sc","SC"] or "structural" in args.SC:
                args.SC_len = args.SC_len[np.ix_(initialized_variables["index_tau_to_conn"], initialized_variables["index_tau_to_conn"])]
            else:
                args.SC_len = None
    elif args.null_model_i is not None:
        raise ValueError("Null model index provided but no null connectivity is specified.")

def load_roi_size(args, data, initialized_variables, data_individual=None):
    # If ROI_size is provided as a dictionary, extract its values; otherwise, process as an array-like structure
    if isinstance(data['conn']['ROI_size'], dict):
        roi_values = list(data['conn']['ROI_size'].values())
    else: # ndarray or dataframe or series
        raise ValueError("ROI_size should be a dictionary")
    
    if args.subject_id is not None: # replace with individualized ROI_size if provided
        print(f"loading individualized roi_size for subject {args.subject_id}")
        roi_values = data_individual["roi_size"][args.subject_id]

    # If no uniform ROI size is specified, use the provided ROI sizes; otherwise, set ROI size to a constant value
    if args.same_ROI_size is None:
        initialized_variables["roi_size"] = np.array(roi_values).reshape(-1,)
    else:
        number = int(np.mean(roi_values)) if args.same_ROI_size == 'mean' else int(args.same_ROI_size)
        print("setting ROI_size to the same value:", number)
        initialized_variables["roi_size"]= np.full(data['conn']['conn'].shape[0], number)

    if initialized_variables["index_tau_to_conn"] is not None: # if tau and conn not match
        initialized_variables["roi_size"] = initialized_variables["roi_size"][initialized_variables["index_tau_to_conn"]]
   
def load_regional_variables(args, initialized_variables, data_individual=None):

    if any(isinstance(getattr(args, var), str) for var in ["spread_var", "synthesis_var", "misfold_var", "clearance_nor_var", "clearance_mis_var", "FC"]):

        df_genes = pd.read_csv(os.path.join(args.input_path, args.regional_variable_file),index_col=[0])
        df_genes = match_and_update(initialized_variables["name"], df_genes)

        if isinstance(args.clearance_var, str):
            print("settting the same clearance_nor_var & clearance_nor_var")
            args.clearance_nor_var = args.clearance_var
            args.clearance_mis_var = args.clearance_var
        for var in ["spread_var", "synthesis_var", "misfold_var", "clearance_nor_var", "clearance_mis_var", "FC"]:
            variable = getattr(args, var)
            if isinstance(variable, str): # if the xxx_var is not None
                values = df_genes[variable].values
                setattr(args, var, values)
                print(f"{var.split('_')[0]} rate {variable} from {args.regional_variable_file}: {type(getattr(args, var))} {getattr(args, var)}")
            elif isinstance(variable, np.ndarray):
                print(f"Add {var.split('_')[0]} rate (input in the previous run)")
        
            if args.subject_id is not None: # replace with individualized regional variables if provided
                if isinstance(variable, str):
                    print(f"loading individualized {var} for subject {args.subject_id}")
                    values = data_individual[variable][args.subject_id].values
                    setattr(args, var, values)

def additional_simulation_settings(args, initialized_variables):   
    # Process intermediate variable names if needed
    if args.return_interm_results and isinstance(args.interm_variabels, str):
        args.interm_variabels = args.interm_variabels.split(",")

    # Initialize Rnor0 based on the given specification
    if args.Rnor0 is not None:
        print("Initialize Rnor0 for _norm_spread:", args.Rnor0, end="\t")
        if args.Rnor0 == "output":
            initialized_variables["Rnor0"] = initialized_variables["tau"]
        elif args.Rnor0 == "MAPT":
            df_MAPT = pd.read_csv(os.path.join(args.input_path, args.file_as_Rnor0),index_col=[0])
            df_MAPT = match_and_update(initialized_variables["name"], df_MAPT)
            initialized_variables["Rnor0"] = df_MAPT["MAPT"].values
        else:
            initialized_variables["Rnor0"] = pickle.load(open(args.Rnor0,'rb'))
        initialized_variables["Rnor0"] = initialized_variables["Rnor0"].reshape(-1,1)
        args.no_norm_spread = True
        print(initialized_variables["Rnor0"].shape)    
    elif args.no_norm_spread == True: print("Stop spread of normal protein in _mis_spread process")


def match_and_update(label_conn, data_to_update):
    """
    Match region labels between the connectivity matrix and simulation data, and update accordingly.

    Args:
        label_conn (list): A pandas Series containing region labels from the connectivity matrix.
                                Expected shape: (n_regions,).
        data_to_update (list, pd.Series, or pd.DataFrame): Region labels from the simulation data.
                                If a list, its length should equal the number of regions.

    Returns:
        list or pd.DataFrame/Series: If data_to_update is a list, returns a list of indices that match the connectivity labels.
                                     If data_to_update is a DataFrame/Series, returns the updated data corresponding to the matching indices.
    """

    if isinstance(data_to_update, list):
        print(f"connectivity matrix and OUTPUT data (tau) not match, matching...")
        print(len(label_conn), "vs", len(data_to_update))
        matched_indices = [i for i in range(len(data_to_update)) if data_to_update[i] in label_conn]
        print(f"matched OUTPUT (tau) index:", len(matched_indices), matched_indices)
        return matched_indices
    
    elif isinstance(data_to_update, pd.Series) or isinstance(data_to_update, pd.DataFrame):
        if not np.array_equal(label_conn, data_to_update.index):
            print(f"connectivity matrix and REGIONAL information not match, matching...")
            print(len(label_conn), "vs", data_to_update.shape[0])
            matched_indices = [i for i in range(data_to_update.shape[0]) if data_to_update.index[i] in label_conn]
            data = data_to_update.iloc[matched_indices, :]
            print(f"matched REGIONAL information:", data_to_update.shape)
            return data
        else:
            return data_to_update
