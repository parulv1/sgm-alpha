""" Computing and sorting eigenmodes for alpha and beta band spatial correlations"""
from ..forward import network_transfer_macrostable as nt
from ..utils import functions
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from scipy.io import loadmat

def run_local_coupling_forward_Xk(brain, params, freqs, PSD, SC, rois_with_MEG, band):

    """Network Transfer Function for spectral graph model.

    Args:
        brain (Brain): specific brain to calculate NTF
        parameters (dict): parameters for ntf. We shall keep this separate from Brain
        for now, as we want to change and update according to fitting.
        frequency (float): frequency at which to calculate NTF
        PSD: PSD of a subject to compute spatial correlation 
        SC: Number of eigenvectors to include
        rois_with_MEG: rois for which MEG spectra is available
        band: "alpha" or "beta"

    Returns:
        spcorr2 (numpy asarray): spatial correlation of summed eigenmodes
        eigvec_sorted (numpy asarray): sorted eigenmodes
        summed_PSD (numpy asarray): PSD summed for the frequency band of interest
        eig_ind (numpy asarray):  index of sorted eigenmodes
    """

    if band == "alpha":
        freqband = np.where((freqs>=8) & (freqs<=12))[0]
    if band == "beta":
        freqband = np.where((freqs>=13) & (freqs<=25))[0]


    eigvec_ns = np.zeros((len(rois_with_MEG),len(freqband)))

    for i in range(len(freqband)):
        w = 2 * np.pi * freqs[freqband[i]]
        eigenvectors_ns, _, _, _ = nt.network_transfer_local_alpha(
            brain, params, w, np.array([]), 1, 0
        )

        eigvec_ns[:,i] = eigenvectors_ns[rois_with_MEG]

    eigvec_ns_summed = np.sum(eigvec_ns,axis = 1)

    eigvec_summed = eigvec_ns_summed/np.linalg.norm(eigvec_ns_summed)

    summed_PSD = np.sum(PSD[:,freqband], axis = 1)

    summed_PSD = summed_PSD/np.linalg.norm(summed_PSD)

    C = loadmat('/data/rajlab1/shared_data/datasets/DK_adjacency/DK_CortexAdjacency.mat')

    C = C['Adja_cortex']

    C = C/np.linalg.norm(C)

    Cc2 = C + np.eye(len(rois_with_MEG))
    
    func1 = np.matmul(Cc2, summed_PSD)
    cost_func =  np.matmul(np.transpose(eigvec_summed),func1)
    
    return cost_func

def run_local_coupling_forward_Xk_db(brain, params, freqs, PSD, SC, rois_with_MEG, band):

    """Network Transfer Function for spectral graph model.

    Args:
        brain (Brain): specific brain to calculate NTF
        parameters (dict): parameters for ntf. We shall keep this separate from Brain
        for now, as we want to change and update according to fitting.
        frequency (float): frequency at which to calculate NTF
        PSD: PSD of a subject to compute spatial correlation 
        SC: Number of eigenvectors to include
        rois_with_MEG: rois for which MEG spectra is available
        band: "alpha" or "beta"

    Returns:
        spcorr2 (numpy asarray): spatial correlation of summed eigenmodes
        eigvec_sorted (numpy asarray): sorted eigenmodes
        summed_PSD (numpy asarray): PSD summed for the frequency band of interest
        eig_ind (numpy asarray):  index of sorted eigenmodes
    """

    if band == "alpha":
        freqband = np.where((freqs>=8) & (freqs<=12))[0]
    if band == "beta":
        freqband = np.where((freqs>=13) & (freqs<=25))[0]

    eigvec_ns = np.zeros((len(rois_with_MEG),len(freqband)))

    for i in range(len(freqband)):
        w = 2 * np.pi * freqs[freqband[i]]
        eigenvectors_ns, _, _, _ = nt.network_transfer_local_alpha(
            brain, params, w, np.array([]), 1, 0
        )

        eigvec_ns[:,i] = 20*np.log10(eigenvectors_ns)[rois_with_MEG]


    eigvec_ns_summed = np.sum(eigvec_ns,axis = 1)

    eigvec_summed = eigvec_ns_summed/np.linalg.norm(eigvec_ns_summed)

    summed_PSD = np.sum(10*np.log10(PSD)[:,freqband], axis = 1)

    summed_PSD = summed_PSD/np.linalg.norm(summed_PSD)

    C = loadmat('/data/rajlab1/shared_data/datasets/DK_adjacency/DK_CortexAdjacency.mat')

    C = C['Adja_cortex']

    C = C/np.linalg.norm(C)

    Cc2 = C + np.eye(len(rois_with_MEG))
    
    func1 = np.matmul(Cc2, summed_PSD)
    cost_func =  np.matmul(np.transpose(eigvec_summed),func1)
    
    return cost_func

