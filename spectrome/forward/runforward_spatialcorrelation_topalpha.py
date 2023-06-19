""" Computing and sorting eigenmodes for alpha and beta band spatial correlations"""
from ..forward import network_transfer_macrostable as nt
from ..utils import functions
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

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

#     eigvec_ns = np.zeros((len(rois_with_MEG),SC,len(freqband)))
    eigvec_ns = np.zeros((len(rois_with_MEG),len(freqband)))

    for i in range(len(freqband)):
        w = 2 * np.pi * freqs[freqband[i]]
        eigenvectors_ns, _, _, _ = nt.network_transfer_local_alpha(
            brain, params, w, np.array([]), 1, 0
        )
#         eigvec_ns[:,:,i] = eigenvectors_ns
        eigvec_ns[:,i] = eigenvectors_ns[rois_with_MEG]
#         eigenvectors_ns, _, _, _ = nt_noise.network_transfer_local_alpha(
#             brain, params, w
#         )
#         eigvec_ns[:,i] = eigenvectors_ns[rois_with_MEG]



#     eigvec_ns_summed = np.sum(eigvec_ns[:,:,:],axis = 2)
    eigvec_ns_summed = np.sum(eigvec_ns,axis = 1)
#     eigvec_summed = np.sum(eigvec_ns_summed, axis = 1)
    eigvec_summed = eigvec_ns_summed/np.linalg.norm(eigvec_ns_summed)

    summed_PSD = np.sum(PSD[:,freqband], axis = 1)

    summed_PSD = summed_PSD/np.linalg.norm(summed_PSD)

    
#     spcorr = pearsonr(summed_PSD, eigvec_summed)[0]
#     w_spat = 10.0

#     C = brain.reducedConnectome
#     rowdegree = np.transpose(np.sum(C, axis=1))
#     coldegree = np.sum(C, axis=0)
#     qind = rowdegree + coldegree < 0.2 * np.mean(rowdegree + coldegree)
#     rowdegree[qind] = np.inf
#     coldegree[qind] = np.inf
#     L2 = np.divide(1, np.sqrt(np.multiply(rowdegree, coldegree)) + np.spacing(1))
#     Cc = np.matmul(np.diag(L2), C)
    
#     nroi = len(rois_with_MEG)
    
#     # C2 = Cc + w_spat*np.eye(86)
#     C2 = Cc + w_spat*np.eye(nroi)
#     # C2 = Cc + w_spat*np.eye(82) #when including receptor density
#     rowdegree = np.transpose(np.sum(C2, axis=1))
#     coldegree = np.sum(C2, axis=0)
#     qind = rowdegree + coldegree < 0.2 * np.mean(rowdegree + coldegree)
#     rowdegree[qind] = np.inf
#     coldegree[qind] = np.inf
#     L22 = np.divide(1, np.sqrt(np.multiply(rowdegree, coldegree)) + np.spacing(1))
#     Cc2 = np.matmul(np.diag(L22), C2)    
    
    
#     # func1 = np.matmul(Cc2[0:68,0:68], summed_PSD)
#     func1 = np.matmul(Cc2, summed_PSD)
    
    # func2 = np.matmul(Cc2[0:68,0:68], eigvec_summed)
    
    # cost_func = pearsonr(np.gradient(func1),np.gradient(func2))[0]
    
    
    # cost_func =  np.matmul(np.transpose(eigvec_summed),func1)
    
    cost_func = pearsonr(np.matmul(brain.distance_matrix,eigvec_summed),np.matmul(brain.distance_matrix,summed_PSD))[0]
    
    
    return cost_func
    # return summed_PSD, eigvec_summed