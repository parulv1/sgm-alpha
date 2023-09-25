""" Computing and sorting eigenmodes for alpha and beta band spatial correlations"""
# from ..forward import network_transfer_spatialcorrelation_notsorted_org as nt_ns
from ..forward import network_transfer_macrostable as nt
# from ..forward import network_transfer_pet as nt
# from ..forward import network_transfer_noise as nt_noise
import numpy as np
from scipy.spatial import distance

# def run_local_coupling_forward_Xk(brain, params, freqs, PSD, SC, rois_with_MEG, band, pet_tau, pet_Ab):
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

    summed_PSD_binarized = np.where(summed_PSD>np.mean(summed_PSD), 1, 0)

    eigvec_summed_binarized = np.where(eigvec_summed>np.mean(eigvec_summed), 1, 0)

    
    cost_func = distance.dice(summed_PSD_binarized, eigvec_summed_binarized)
    
    return cost_func
    # return summed_PSD, eigvec_summed