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

    C = brain.reducedConnectome
    
    C = C/np.linalg.norm(C)
    
    rowdegree = np.transpose(np.sum(C, axis=1))
    coldegree = np.sum(C, axis=0)
    
    degree = rowdegree + coldegree
    
    eps = min([i for i in degree if i > 0])
    w_spat = 10
    nroi = len(rois_with_MEG)
    L22 = np.divide(1, np.sqrt(np.multiply(rowdegree, coldegree)) + eps)
    Cc2 = np.matmul(np.diag(L22), C)[0:68,0:68] + w_spat*np.eye(nroi)
    
    Cc2 = Cc2/np.linalg.norm(Cc2)
    
#     Extra lines to match previous spatial R
    rowdegree = np.transpose(np.sum(Cc2, axis=1))
    coldegree = np.sum(Cc2, axis=0)
    
    degree = rowdegree + coldegree
    
    eps = min([i for i in degree if i > 0])
    L23 = np.divide(1, np.sqrt(np.multiply(rowdegree, coldegree)) + eps)
    Cc3 = np.matmul(np.diag(L23), Cc2)
    
    Cc3 = Cc3/np.linalg.norm(Cc3)  
    
    Cc3 = Cc3[0:len(rois_with_MEG),0:len(rois_with_MEG)]
    
    cost_func = distance.mahalanobis(summed_PSD, eigvec_summed, Cc3)
    
    return cost_func
    # return summed_PSD, eigvec_summed

def run_local_coupling_forward_Xk_distance(brain, params, freqs, PSD, SC, rois_with_MEG, band):
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

    C = brain.distance_matrix
    
    C = C/np.linalg.norm(C)
    
    Cc3 = C[0:len(rois_with_MEG),0:len(rois_with_MEG)]
    
    cost_func = distance.mahalanobis(summed_PSD, eigvec_summed, Cc3)
    
    return cost_func