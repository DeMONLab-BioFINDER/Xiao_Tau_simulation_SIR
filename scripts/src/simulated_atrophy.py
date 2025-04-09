# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 09:35:42 2021
@author: Vincent Bazinet.

Description:
    This module contains the core functions for simulating atrophy in brain networks using a S.I.R. spreading model.
    It implements the simulation of tau propagation and atrophy based on structural connectivity, incorporating mechanisms 
    such as protein synthesis, misfolding, clearance, and the spread of both normal and misfolded proteins. This code 
    represents the Python adaptation of the original SIRsimulator developed by Ying-Qiu Zheng (https://github.com/yingqiuz/SIR_simulator),
    and has been further modified for use in neurodegenerative disease studies.

Usage:
    Key functions include:
    - simulate_atrophy(): Orchestrate the overall simulation process for atrophy generation.
        - _normal_spread(): Simulate the spread of normal proteins across the network.
        - _mis_spread(): Simulate the spread of misfolded proteins, incorporating synthesis, misfolding, and clearance.
        - _atrophy(): Estimate atrophy maps from the distribution of proteins.
    - calculate_rate(): Compute regional rates using z-score normalization and the normal CDF.
    - initialize_tmp_results(): Initialize storage structures for intermediate simulation results.
      
References:
    Zheng, Y. Q., Zhang, Y., Yau, Y., Zeighami, Y., Larcher, K., Misic, B., & Dagher, A. (2019).
        Local vulnerability and global connectivity jointly shape neurodegenerative disease propagation.
        PLoS Biology, 17(11), e3000495.

    Shafiei, G., Bazinet, V., Dadar, M., Manera, A. L., Collins, D. L., Dagher, A., ... & Ducharme, S. (2023).
        Network structure and transcriptomic vulnerability shape atrophy in frontotemporal dementia.
        Brain, 146(1), 321-336.

Revisions:
    Revised by XIAO Yu on Fri Dec 20 2023, at Lund, Sweden.
    Additions include the functions calculate_rate() and initialize_tmp_results(), along with extended support for 
    regional parameters such as spread_rate, misfold_rate, clearance_rate_nor, and clearance_rate_mis.
"""

import numpy as np
from scipy.stats import norm, zscore


def calculate_rate(variable, N_regions, name, default=None):
    """
    Calculate a rate based on a regional variable using z-score normalization and the normal CDF.
    
    This function returns the cumulative distribution function (CDF) of the z-scored
    regional variable if it is provided and non-zero. If the variable is None or all zeros,
    it returns either a vector of zeros (processed through norm.cdf) or a vector of ones,
    depending on whether a default value is provided.

    Args:
        variable (array-like or None): The regional variable data. Expected shape: (N_regions,).
        N_regions (int): The number of regions.
        name (str): Name of the variable (used for printing/debugging purposes).
        default (any, optional): If provided, use a vector of ones instead of zeros when variable is None or empty.
    
    Returns:
        np.ndarray: A 1D array of rates computed using the normal CDF, with shape (N_regions,).
    """
    if variable is None or not np.any(variable):
        if default is None:
            return norm.cdf(np.zeros(N_regions))
        else:
            return np.ones(N_regions)
    else:
        return norm.cdf(zscore(variable))
    

def initialize_tmp_results(N_regions, T_total, vars_interm, return_interm_results):
    """
    Initialize dictionaries to store intermediate results and probabilities.

    Depending on whether intermediate results should be returned, this function creates
    arrays to store various intermediate variables and probability matrices during simulation.
    
    Args:
        N_regions (int): Number of regions.
        T_total (int): Total number of time steps for simulation.
        vars_interm (list of str): List of names for intermediate variables to be saved.
        return_interm_results (bool): Flag indicating whether to save and return intermediate results.
    
    Returns:
        tuple: A tuple (results_tmp, P_all) where:
            - results_tmp (dict): A dictionary mapping variable names to NumPy arrays initialized to zeros.
              For "movOut_mis" and "movDrt_mis", arrays have shape (N_regions, N_regions, T_total).
              For other variables, arrays have shape (N_regions, T_total).
            - P_all (dict): A dictionary with keys "Pnor_all" and "Pmis_all", each an array of shape (N_regions, N_regions, T_total),
              used to store probability values.
    """
    results_tmp, P_all = {}, {}
    if return_interm_results:
        print("Saving intermediate results:",vars_interm)
        for var in vars_interm:
            if var in ["movOut_mis", "movDrt_mis"]:
                results_tmp[var] = np.zeros((N_regions, N_regions, T_total))
            elif var == "P_all":
                P_all["Pnor_all"] = np.zeros((N_regions, N_regions, T_total))
                P_all["Pmis_all"] = np.zeros((N_regions, N_regions, T_total))
            else:
                results_tmp[var] = np.zeros((N_regions, T_total))

    return results_tmp, P_all

def _normal_spread(SC_den, syn_control, SC_len=None, v=1, dt=0.1, p_stay=0.5,
                   synthesis_var=None, clearance_nor_var=None, FC=None, k=0,
                   Rnor0=None):
    '''
    Function to simulate the spread of normal proteins in a brain network.
    Part 1 of SIRsimulator. SIRsimulator being the original code written by
    Ying-Qiu Zheng in Matlab (https://github.com/yingqiuz/SIR_simulator) for
    her PLoS Biology paper [SN1]

    Parameters
    ----------
    SC_den: (n, n) ndarray
        Structural connectivity matrix (strength)
    SC_len: (n, n) ndarray
        Structural connectivity matrix (len)
    syn_control: (n,) ndarray
        Parameters specifying in how many voxels proteins can be synthesized
        for each brain regions (region size, i.e., ROIsize)
    v: float
        Speed of the atrophy process. Default: 1
    dt: float
        Size of each time step. Default: 0.1
    p_stay: float
        The probability of staying in the same region per unit time.
        Default: 0.5
    GBA: (n,) ndarray
        GBA gene expression (clearance of misfolded protein). If None, then
        GBA expression is uniformly distributed across brain regions.
        Default: None
    SNCA: (n,) ndarray
        SNCA gene expression (synthesis of misfolded protein)/ If None, then
        SNCA expression is uniformly distributed across brain regions.
        Default: None
    FC: (n, n) ndarray
        Functional connectivity. Default: None
    k: float
        weight of functional connectivity.  Default: 0

    Returns
    -------
    Rnor: (n,) ndarray
         The population of normal agents in regions before pathogenic
         spreading.
    Pnor: (n,) ndarray
        The population of normal agents in edges before pathogenic spreading.

    References
    ----------
    .. [SN1] Zheng, Y. Q., Zhang, Y., Yau, Y., Zeighami, Y., Larcher, K.,
       Misic, B., & Dagher, A. (2019). Local vulnerability and global
       connectivity jointly shape neurodegenerative disease propagation.
       PLoS biology, 17(11), e3000495.
    '''

    # Compute basic information
    N_regions = len(SC_den)

    # make sure the diag are zero
    np.fill_diagonal(SC_den, 0)
    if SC_len is not None:
        np.fill_diagonal(SC_len, 0)

    # Create a Fake FC matrix if FC is none
    if FC is not None:
        if FC.ndim==2 and FC.shape[0] == FC.shape[1]:
            np.fill_diagonal(FC, 0)
        elif FC.ndim==1 or (FC.ndim==2 and (FC.shape[0]==1 or FC.shape[1]==1)):
            FC = FC.reshape(-1,1)
            print("set FC to regional information")
    else:
        FC = np.zeros((N_regions, N_regions))

    # set probabilities of moving from region i to edge (i,j))
    weights = SC_den * np.exp(k * FC) #np.exp(0.1, FC) -> small contribution
    weight_str = weights.sum(axis=0)
    weights = (1 - p_stay) * weights + p_stay * np.diag(weight_str)
    weights = weights / weight_str[:, np.newaxis]
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

    # convert gene expression scores to probabilities
    synthesis_rate = calculate_rate(synthesis_var, N_regions, "synthesis_rate")
    clearance_rate_nor = calculate_rate(clearance_nor_var, N_regions, "clearance_rate_nor")

    # Rnor, Pnor store results of single simulation at each time
    Rnor = np.zeros((N_regions, 1))  # number of normal agent in regions
    Pnor = np.zeros((N_regions, N_regions))  # number of normal agent in paths
    if Rnor0 is not None:
        Rnor = Rnor0
        Pnor = np.zeros_like(SC_den, dtype=float)
        # Iterate over SC_den to fill Pnor
        for i in range(SC_den.shape[0]):
            for j in range(SC_den.shape[1]):
                if SC_den[i, j] != 0 and i != j:  # Check if regions are connected
                    Pnor[i, j] = (Rnor[i] + Rnor[j]) / 2  # Calculate the mean of the corresponding regions
        print("skip normal protein spread in _norm_spread() & _mis_spread(), set Rnor0 as input pattern, Pnor0 as mean input of connected regions")
        return Pnor, Rnor
    
    # normal alpha-syn growth
    # fill the network with normal proteins
    iter_max = 1000000000
    #print("synthesis_rate * syn_control) * dt:", synthesis_rate, "\n",syn_control, "\n", dt)
    for t in range(iter_max):
        # moving process
        # regions towards paths
        # movDrt stores the number of proteins towards each region. i.e.
        # element in kth row lth col denotes the number of proteins in region k
        # moving towards l
        movDrt = np.repeat(Rnor, N_regions, axis=1) * weights * dt # * spread_rate[:, np.newaxis]
        np.fill_diagonal(movDrt, 0)

        # paths towards regions
        # update moving
        with np.errstate(divide='ignore', invalid='ignore'):
            movOut = Pnor * v
            if SC_len is not None:
                movOut = (Pnor * v) / SC_len
                movOut[SC_len == 0] = 0

        Pnor = Pnor - movOut * dt + movDrt
        np.fill_diagonal(Pnor, 0)

        Rtmp = Rnor
        Rnor = Rnor + movOut.sum(axis=0)[:, np.newaxis] * dt - movDrt.sum(axis=1)[:, np.newaxis]  # noqa

        # growth process
        Rnor_cleared = Rnor * (1 - np.exp(-clearance_rate_nor * dt))[:, np.newaxis]
        Rnor_synthesized = ((synthesis_rate * syn_control) * dt)[:, np.newaxis]
        Rnor = Rnor - Rnor_cleared + Rnor_synthesized

        if np.all(abs(Rnor - Rtmp) < 1e-7 * Rtmp):
            break
    print("\nnormal_spread t:", t)
    return Pnor, Rnor


def _mis_spread(SC_den, seed, syn_control, ROIsize, Rnor, Pnor, SC_len=None, v=1,
                dt=0.1, p_stay=0.5, trans_rate=1, init_number=1, T_total=1000, spr_time=0,
                spread_var=None, synthesis_var=None, misfold_var=None,
                clearance_nor_var=None, clearance_mis_var=None, FC=None, k=0,
                return_interm_results=False, interm_variabels=None, results_partial=None,
                no_norm_spread=False):
    '''
    Function to simulate the spread of misfolded proteins in a brain network.
    Part 2 of SIRsimulator. SIRsimulator being the original code written by
    Ying-Qiu Zheng in Matlab (https://github.com/yingqiuz/SIR_simulator) for
    her PLoS Biology paper [SN1]

    Parameters
    ----------
    SC_den: (n, n) ndarray
        Structural connectivity matrix (strength)
    SC_len: (n, n) ndarray
        Structural connectivity matrix (len)
    seed: int
        ID of the node to be used as a seed for the atrophy process
    syn_control: (n,) ndarray
        Parameters specifying in how many voxels proteins can be synthesized
        for each brain regions (region size, i.e., ROIsize)
    ROIsize: (n,) ndarray:
        Size of each ROIs in the parcellation
    Rnor: (n,) ndarray
         The population of normal agents in regions before pathogenic
         spreading.
    Pnor: (n,) ndarray
        The population of normal agents in edges before pathogenic spreading.
    v: float
        Speed of the atrophy process
    dt: float
        Size of each time step
    p_stay: float
        The probability of staying in the same region per unit time
    trans_rate: float
        A scalar value controlling the baseline infectivity
    init_number: int
        Number of injected misfolded protein
    T_total: int
        Total time steps of the function
    GBA: (n,) ndarray
        GBA gene expression (clearance of misfolded protein)
    SNCA: (n,) ndarray
        SNCA gene expression (synthesis of misfolded protein)
    return_interm_results: Boolean
        Whether the function should return the intermediate results. This could be
        memory-consuming. Default: False
    FC: (n, n) ndarray
        Functional connectivity
    k: float
        weight of functional connectivity

    Returns
    -------
    Rnor_all: (n_regions, T_total) ndarray
        Trajectory matrix of the distribution of normal proteins across brain
        regions for each individual time points.
    Rmis_all: (n_regions, T_total) ndarray
        Trajectory matrix of the distribution of misfolded proteins across
        brain regions for each individual time points.
    Pnor_all: (n_regions, n_regions, T_total) ndarray
        Trajectory matrices of the distribution of normal proteins across
        network paths (edges) for each individual time points.
    Pmis_all: (n_regions, n_regions, T_total) ndarray
        Trajectory matrices of the distribution of misfolded proteins across
        network paths (edges) for each individual time points.

    References
    ----------
    .. [SN1] Zheng, Y. Q., Zhang, Y., Yau, Y., Zeighami, Y., Larcher, K.,
       Misic, B., & Dagher, A. (2019). Local vulnerability and global
       connectivity jointly shape neurodegenerative disease propagation.
       PLoS biology, 17(11), e3000495.
    '''

    # Compute basic information
    N_regions = len(SC_den) # SC_len

    # make sure the diag is zero
    np.fill_diagonal(SC_den, 0)
    if SC_len is not None:
        np.fill_diagonal(SC_len, 0)

    # Create a Fake FC matrix if FC is none
    if FC is not None:
        if FC.ndim==2 and FC.shape[0] == FC.shape[1]:
            np.fill_diagonal(FC, 0)
        elif FC.ndim==1 or (FC.ndim==2 and (FC.shape[0]==1 or FC.shape[1]==1)):
            FC = FC.reshape(-1,1)
            print("set FC to regional information")
    else:
        FC = np.zeros((N_regions, N_regions))

    # set probabilities of moving from region i to edge (i,j))
    weights = SC_den * np.exp(k * FC)
    weight_str = weights.sum(axis=0)
    weights = (1 - p_stay) * weights + p_stay * np.diag(weight_str)
    weights = weights / weight_str[:, np.newaxis]
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

    # convert gene expression scores to probabilities
    spread_rate = calculate_rate(spread_var, N_regions, "spread_rate", 1)
    synthesis_rate = calculate_rate(synthesis_var, N_regions, "synthesis_rate")
    misfold_rate = calculate_rate(misfold_var, N_regions, "misfold_rate", 1)
    clearance_rate_nor = calculate_rate(clearance_nor_var, N_regions, "clearance_rate_nor")
    clearance_rate_mis = calculate_rate(clearance_mis_var, N_regions, "clearance_rate_mis")

    # store the number of normal/misfoled alpha-syn at each time step
    Rnor_all = np.zeros((N_regions, T_total))
    Rmis_all = np.zeros((N_regions, T_total))
    results_tmp, P_all = {},{}

    if results_partial:
        T_start = results_partial["Rnor_all"].shape[1]
        print("load results_partial, starting from", T_start)
        Rnor_all[:,:T_start], Rmis_all[:,:T_start] = results_partial["Rnor_all"], results_partial["Rmis_all"]

    results_tmp, P_all = initialize_tmp_results(N_regions, T_total, interm_variabels, return_interm_results)
    if results_partial:
        P_all["Pnor_all"][:,:,:T_start], P_all["Pmis_all"][:,:,:T_start] = results_partial["Pnor_all"], results_partial["Pmis_all"]

    # Rnor, Rmis, Pnor, Pmis store results of single simulation at each time
    Rmis = np.zeros((N_regions, 1))  # nb of misfolded agent in regions
    Pmis = np.zeros((N_regions, N_regions))  # nb of misfolded agent in paths
    if results_partial:
        Rmis, Rnor = results_partial["Rmis_all"][:, -1].reshape((-1,1)), results_partial["Rnor_all"][:, -1].reshape((-1,1))
        Pmis, Pnor = results_partial["Pmis_all"][:, :, -1], results_partial["Pnor_all"][:, :, -1]
    else:
        # inject misfolded alpha-syn
        Rmis[seed] = init_number
        T_start = 0
    if no_norm_spread:
        print("SKIP normal protein spread in _mis_spread()")
    # misfolded protein spreading process
    for t in range(T_start, T_total):
        #### moving process
        # normal proteins : region -->> paths
        movDrt_nor = np.repeat(Rnor, N_regions, axis=1) * weights * dt # * spread_rate[:, np.newaxis]
        np.fill_diagonal(movDrt_nor, 0)
        # normal proteins : path -->> regions
        with np.errstate(invalid='ignore'):
            movOut_nor = Pnor * v
            if SC_len is not None: 
                movOut_nor = (Pnor * v) / SC_len
                movOut_nor[SC_len == 0] = 0

        # misfolded proteins: region -->> paths   
        #row-wise multiplication, interaction of Rmis in specific region (row i) with Abeta in other regions (column j) --> likelyhood of Rmis in row i spreading to region j (Abeta)
        movDrt_mis = np.repeat(Rmis, N_regions, axis=1) * weights * dt # !!! added spread_rate to control the spread of misfolded protein into other regions
        np.fill_diagonal(movDrt_mis, 0)
        # misfolded proteins: paths -->> regions
        with np.errstate(invalid='ignore'):
            movOut_mis = Pmis * v 
            if SC_len is not None: 
                movOut_mis = (Pmis * v) / SC_len
                movOut_mis[SC_len == 0] = 0
        
        #### update regions and paths after moving process
        if no_norm_spread == False:
            Pnor = Pnor - movOut_nor * dt + movDrt_nor
            np.fill_diagonal(Pnor, 0)
            Rnor = Rnor + movOut_nor.sum(axis=0)[:, np.newaxis] * dt - movDrt_nor.sum(axis=1)[:, np.newaxis]  # noqa 

        if t>= spr_time and not np.all(spread_rate==1): # if spread_time and spread_rate exists
            if t==spr_time: print("!!!Add spread rate at time",t,"!!!")
            Pmis = Pmis - movOut_mis * spread_rate[:, np.newaxis] * dt + movDrt_mis # here should also add spread rate
            Rmis = Rmis + movOut_mis.sum(axis=0)[:, np.newaxis] * spread_rate[:, np.newaxis] * dt - movDrt_mis.sum(axis=1)[:, np.newaxis]  # noqa ## ??? add spread rate here? aftermovOut_mis
        else:
            Pmis = Pmis - movOut_mis * dt + movDrt_mis # here should also add spread rate
            Rmis = Rmis + movOut_mis.sum(axis=0)[:, np.newaxis] * dt - movDrt_mis.sum(axis=1)[:, np.newaxis]
        np.fill_diagonal(Pmis, 0)

        if return_interm_results:
            if "movOut_mis" in interm_variabels:
                results_tmp["movOut_mis"][:, :, t] = movOut_mis
            if "movDrt_mis" in interm_variabels:
                results_tmp["movDrt_mis"][:, :, t] = movDrt_mis
            if "Rmis_after_spread" in interm_variabels:
                results_tmp["Rmis_after_spread"][:, t] = Rmis.reshape(-1)
            if "Rnor_after_spread" in interm_variabels:
                results_tmp["Rnor_after_spread"][:, t] = Rnor.reshape(-1)

        #### synthesis, misfolding, clearance
        Rnor_cleared = Rnor * (1 - np.exp(-clearance_rate_nor * dt))[:, np.newaxis]
        Rnor_synthesized = ((synthesis_rate * syn_control) * dt)[:, np.newaxis]
        Rmis_cleared = Rmis * (1 - np.exp(-clearance_rate_mis * dt))[:, np.newaxis] # 0-0.095 (if clearance_rate_mis=1)
        # the probability of getting misfolded 
        gamma0 = trans_rate / ROIsize # ? larger ROIsize, less likely getting misfolded 
        misProb = 1 - np.exp(-Rmis * gamma0[:, np.newaxis] * dt) # 0 - 0.095 (if trans_rate=1, ROIsize=1, Rmis-max=1)
        # Number of newly infected 
        N_misfolded = Rnor * (np.exp(-clearance_rate_nor)[:, np.newaxis]) * misProb * misfold_rate[:, np.newaxis] #!!!! added misfold_rate to control the probability of getting misfolded in different regions
        # !!!!! changed to clearance_rate_nor????
        # Update
        Rnor = Rnor - Rnor_cleared - N_misfolded + Rnor_synthesized
        Rmis = Rmis - Rmis_cleared + N_misfolded
        Rnor_all[:, t] = np.squeeze(Rnor)
        Rmis_all[:, t] = np.squeeze(Rmis)

        if return_interm_results:
            if "movOut_mis" in interm_variabels:
                results_tmp["Rnor_cleared"][:, t] = np.squeeze(Rnor_cleared)
            if "movOut_mis" in interm_variabels:
                results_tmp["Rmis_cleared"][:, t] = np.squeeze(Rmis_cleared)
            if "movOut_mis" in interm_variabels:
                results_tmp["misProb"][:, t] = np.squeeze(misProb)
            if "movOut_mis" in interm_variabels:
                results_tmp["N_misfolded"][:, t] = np.squeeze(N_misfolded)
            if "P_all" in interm_variabels:
                P_all["Pnor_all"][:, :, t] = Pnor
                P_all["Pmis_all"][:, :, t] = Pmis
                


    if return_interm_results:
        results_tmp["Rnor_synthesized"] = Rnor_synthesized
    
    return Rnor_all, Rmis_all, results_tmp, P_all

    #else:
    #    return Rnor_all, Rmis_all


def _atrophy(SC_den, Rnor_all, Rmis_all, dt=0.1, k1=0.5, k=0, FC=None):
    '''
    Function to estimate the atrophy map from the distribution of normal and
    misfolded proteins in the brain. This function is inspired by code
    originally written in Matlab by Ying-Qiu Zheng
    (https://github.com/yingqiuz/SIR_simulator) for her PLoS Biology
    paper [SN1]

    Parameters
    ----------
    SC_den: (n_regions, n_regions) ndarray
        Structural connectivity matrix (strength).
    Rnor_all: (n_regions, T_total) ndarray
        Trajectory matrix of the distribution of normal protein across brain
        regions for each individual time points.
    Rmis_all: (n_regions, T_total) ndarray
        Trajectory matrix of the distribution of misfolded protein across brain
        regions for each individual time points.
    dt: float
        Size of each time step
    k1: float
        Ratio between weight of atrophy accrual due to accumulation of
        misfolded agends vs. weight of atrophy accrual due to deafferation.
        Must be between 0 and 1
    k: float
        weight of functional connectivity
    FC: (n, n) ndarray
        Functional connectivity

    Returns
    -------
    simulated_atrophy : (n_regions, T_total) ndarray
        Trajectory matrix of the simulated atrophy in individual brain regions.

    References
    ----------
    .. [SN1] Zheng, Y. Q., Zhang, Y., Yau, Y., Zeighami, Y., Larcher, K.,
       Misic, B., & Dagher, A. (2019). Local vulnerability and global
       connectivity jointly shape neurodegenerative disease propagation.
       PLoS biology, 17(11), e3000495.
    '''

    # Compute basic information
    N_regions = len(SC_den)

    # Create empty matrix if FC is none
    if FC is not None:
        if FC.ndim==2 and FC.shape[0] == FC.shape[1]:
            np.fill_diagonal(FC, 0)
        elif FC.ndim==1 or (FC.ndim==2 and (FC.shape[0]==1 or FC.shape[1]==1)):
            FC = FC.reshape(-1,1)
            print("set FC to regional information")
    else:
        FC = np.zeros((N_regions, N_regions))

    ratio = Rmis_all / (Rnor_all + Rmis_all)
    ratio[ratio == np.inf] = 0  # remove possible inf

    # atrophy growth
    k2 = 1 - k1
    weights = SC_den * np.exp(k * FC)
    weights = weights / weights.sum(axis=0)[:, np.newaxis]
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

    # neuronal loss caused by lack of input from neighbouring regions
    ratio_cum = np.matmul(weights,
                          (1-np.exp(-ratio * dt)))

    # one time step back
    ratio_cum = np.c_[np.zeros((N_regions, 1)), ratio_cum[:, :-1]]
    ratio_cum = k2 * ratio_cum + k1 * (1-np.exp(-ratio * dt))

    simulated_atrophy = np.cumsum(ratio_cum, axis=1)

    return simulated_atrophy


def simulate_atrophy(SC_den, seed, roi_sizes, SC_len=None, T_total=1000, dt=0.1,
                     p_stay=0.5, v=1, trans_rate=1, init_number=1, spr_time=0, spread_var=None, 
                     synthesis_var=None, misfold_var=None, clearance_nor_var=None, 
                     clearance_mis_var=None, k1=0.5, k=0, FC=None,
                     return_interm_results=False, interm_variabels=None, results_partial=None,
                     Rnor0=None, no_norm_spread=False):
    '''
    Function to simulate atrophy on a specified network, using a single
    region as a seed of the process.

    Parameters
    ----------
    SC_den: (n, n) ndarray
        Structural connectivity matrix (strength)
    SC_len: (n, n) ndarray
        Structural connectivity matrix (len)
    seed: int
        ID of the node to be used as a seed for the atrophy process
    roi_sizes: (n,) ndarray:
        Size of each ROIs in the parcellation
    T_total: int
        Total time steps of the function
    dt: float
        Size of each time step
    p_stay: float
        The probability of staying in the same region per unit time
    v: float
        Speed of the atrophy process
    trans_rate: float
        A scalar value controlling the baseline infectivity
    init_number: int
        Number of injected misfolded protein
    GBA: (n,) ndarray
        GBA gene expression (clearance of misfolded protein)
    SNCA: (n,) ndarray
        SNCA gene expression (synthesis of misfolded protein)
    k1: float
        Ratio between weight of atrophy accrual due to accumulation of
        misfolded agends vs. weight of atrophy accrual due to deafferation.
        Must be between 0 and 1
    FC: (n, n) ndarray
        Functional connectivity
    k: float
        weight of functional connectivity

    Returns
    -------
    simulated_atrophy: (n_regions, T_total) ndarray
        Trajectory matrix of the simulated atrophy in individual brain regions.
    '''

    # set-up syn_control
    syn_control = roi_sizes
    SC_den = np.nan_to_num(SC_den, nan=0)
    SC_len = np.nan_to_num(SC_len, nan=0)

    # Simulated spread of normal proteins
    Pnor0, Rnor0 = _normal_spread(SC_den,
                                  syn_control,
				                  SC_len=SC_len,
                                  dt=dt,
                                  p_stay=p_stay,
                                  synthesis_var=synthesis_var,
                                  clearance_nor_var=clearance_nor_var,
                                  k=k,
                                  FC=FC,
                                  Rnor0=Rnor0)

    # Simulated spread of misfolded atrophy
    Rnor_all, Rmis_all, results_tmp, P_all = _mis_spread(SC_den,
                                                         seed,
                                                         syn_control,
                                                         roi_sizes,
                                                         Rnor0.copy(),
                                                         Pnor0.copy(),
							                             SC_len=SC_len,
                                                         v=v,
                                                         dt=dt,
                                                         p_stay=p_stay,
                                                         trans_rate=trans_rate,
                                                         init_number=init_number,
                                                         T_total=T_total,
                                                         spr_time = spr_time,
                                                         spread_var=spread_var,
                                                         synthesis_var=synthesis_var, 
                                                         misfold_var=misfold_var,
                                                         clearance_nor_var=clearance_nor_var,
                                                         clearance_mis_var=clearance_mis_var,
                                                         k=k,
                                                         FC=FC,
                                                         return_interm_results=return_interm_results,
                                                         interm_variabels=interm_variabels,
                                                         results_partial=results_partial,
                                                         no_norm_spread=no_norm_spread)

    # Estimate atrophy
    simulated_atrophy = _atrophy(SC_den,
                                 Rnor_all,
                                 Rmis_all,
                                 dt=dt,
                                 k1=k1,
                                 k=k,
                                 FC=FC)

    results = {"Pnor0": Pnor0, "Rnor0": Rnor0, 
               "Rnor_all": Rnor_all, "Rmis_all": Rmis_all,
               "simulated_atrophy": simulated_atrophy}
    
    return results, results_tmp, P_all
