"""Module for computing basic quantities from a spectral graph model: the forward model
Makes the calculation for a single frequency only. """

import numpy as np
from scipy.io import loadmat

def network_transfer_local_alpha(brain, parameters, w, stimulus_roi, w_var, w_means):
    """Network Transfer Function for spectral graph model.

    Args:
        brain (Brain): specific brain to calculate NTF
        parameters (dict): parameters for ntf. We shall keep this separate from Brain
        for now, as we want to change and update according to fitting.
        frequency (float): frequency at which to calculate NTF

    Returns:
        model_out (numpy asarray):  Each region's frequency response for
        the given frequency (w)
        frequency_response (numpy asarray):
        ev (numpy asarray): Eigen values
        Vv (numpy asarray): Eigen vectors

    """
    C = brain.reducedConnectome
    D = brain.distance_matrix

    tau_e = parameters["tau_e"]
    tau_i = parameters["tau_i"]
    speed = parameters["speed"]
    gei = parameters[
        "gei"
    ]  # excitatory-inhibitory synaptic conductance as ratio of E-E syn
    gii = parameters[
        "gii"
    ]  # inhibitory-inhibitory synaptic conductance as ratio of E-E syn
    tauC = parameters["tauC"]
    alpha = parameters["alpha"]
#     gee = parameters["gee"]
    
    gee = 1
    
    # Defining some other parameters used:
    zero_thr = 0.05

#     Try removing thalamus from connectome
    # C = np.delete(C,[69, 78],0)
    # C = np.delete(C,[69, 78],1)
    # D = np.delete(D,[69, 78],0)
    # D = np.delete(D,[69, 78],1)
#     Add a sumsum before doing all of this
#     C = C/C.sum()
    # define sum of degrees for rows and columns for laplacian normalization
    rowdegree = np.transpose(np.sum(C, axis=1))
    coldegree = np.sum(C, axis=0)
#     Ashish suggests removing all of these
    qind = rowdegree + coldegree < 0.2 * np.mean(rowdegree + coldegree)
    rowdegree[qind] = np.inf
    coldegree[qind] = np.inf

    nroi = C.shape[0]

    K = nroi

    Tau = 0.001 * D / speed
    Cc = C * np.exp(-1j * Tau * w)

    # Eigen Decomposition of Complex Laplacian Here
    L1 = np.identity(nroi)
#     Try this normalization
    # Lr = np.divide(1, np.sqrt(rowdegree) + np.spacing(1))
    # Lc = np.divide(1, np.sqrt(coldegree) + np.spacing(1))
    # L = L1 - alpha * np.matmul(np.diag(Lr), np.matmul(Cc, np.diag(Lc)))
    
    L2 = np.divide(1, np.sqrt(np.multiply(rowdegree, coldegree)) + np.spacing(1))
    L = L1 - alpha * np.matmul(np.diag(L2), Cc)

    d, v = np.linalg.eig(L)  
    eig_ind = np.argsort(np.abs(d))  # sorting in ascending order and absolute value
    eig_vec = v[:, eig_ind]  # re-indexing eigen vectors according to sorted index
    eig_val = d[eig_ind]  # re-indexing eigen values with same sorted index

    eigenvalues = np.transpose(eig_val)
    eigenvectors = eig_vec[:, 0:K]
    
    # mica_micro_intensity = np.squeeze(loadmat('/data/rajlab1/shared_data/datasets/MICA/micro_intensity_mean.mat')['micro_intensity_mean'])
    # mica_micro_intensity = np.squeeze(loadmat('/data/rajlab1/shared_data/datasets/MICA/DK_MICA_qT1_mean_normalized.mat')['qT1_mean'])
    mica_micro_intensity = np.squeeze(loadmat('/data/rajlab1/shared_data/datasets/MICA/micro_intensity_mean_BN_subcort.mat')['micro_intensity_mean_subcort'])


#     # Cortical model
    FG = np.divide(1 / tauC ** 2, (1j * w + 1 / tauC) ** 2)
    
    Htotal_micro = np.zeros((nroi,1),dtype="complex")

    Fe = np.divide(1 / tau_e ** 2, (1j * w + 1 / tau_e) ** 2)
    Fi = np.divide(1 / tau_i ** 2, (1j * w + 1 / tau_i) ** 2)

    Hed = (1 + (Fe * Fi * gei)/(tau_e * (1j * w + Fi * gii/tau_i)))/(1j * w + Fe * gee/tau_e + (Fe * Fi * gei)**2/(tau_e * tau_i * (1j * w + Fi * gii / tau_i)))

    Hid = (1 - (Fe * Fi * gei)/(tau_i * (1j * w + Fe * gee/tau_e)))/(1j * w + Fi * gii/tau_i + (Fe * Fi * gei)**2/(tau_e * tau_i * (1j * w + Fe * gee / tau_e)))

    Htotal = Hed + Hid
    
    # for i in range(18):
    #     Htotal_micro[68+i] = Htotal

    for i in range(nroi):
        tau_e = parameters["tau_e"]*mica_micro_intensity[i]
        tau_i = parameters["tau_i"]*mica_micro_intensity[i]

        Fe = np.divide(1 / tau_e ** 2, (1j * w + 1 / tau_e) ** 2)
        Fi = np.divide(1 / tau_i ** 2, (1j * w + 1 / tau_i) ** 2)

        Hed = (1 + (Fe * Fi * gei)/(tau_e * (1j * w + Fi * gii/tau_i)))/(1j * w + Fe * gee/tau_e + (Fe * Fi * gei)**2/(tau_e * tau_i * (1j * w + Fi * gii / tau_i)))

        Hid = (1 - (Fe * Fi * gei)/(tau_i * (1j * w + Fe * gee/tau_e)))/(1j * w + Fi * gii/tau_i + (Fe * Fi * gei)**2/(tau_e * tau_i * (1j * w + Fe * gee / tau_e)))

        Htotal = Hed + Hid
        
        Htotal_micro[i] = Htotal

#         Hard coding visual stimulus input for now. Will change it.

    
    # visual_stimulus_roi = np.array([3, 37])
    # visual_stimulus_roi = np.array([69, 78])
    
    
    # Htotal_micro_vis = np.multiply(Htotal_micro,Pw)
    # Htotal_micro_vis = Htotal_micro

    q1 = (1j * w + 1 / tauC * FG * eigenvalues)
    qthr = zero_thr * np.abs(q1[:]).max()
    magq1 = np.maximum(np.abs(q1), qthr)
    angq1 = np.angle(q1)
    q1 = np.multiply(magq1, np.exp(1j * angq1))
    frequency_response2 = np.divide(1, q1)
    
    frequency_response = np.diag(frequency_response2)
    
#     p_means = np.zeros((nroi, nroi),dtype="complex")
    
#     if len(visual_stimulus_roi)==1:
#         p_means[visual_stimulus_roi,visual_stimulus_roi] = np.abs(Htotal_micro_vis[visual_stimulus_roi])**2
#     if len(visual_stimulus_roi)>1:
#         for i in range(len(visual_stimulus_roi)):
#             p_means[visual_stimulus_roi[i],visual_stimulus_roi[i]] = np.abs(Htotal_micro_vis[visual_stimulus_roi[i]])**2
#         for i in range(len(visual_stimulus_roi)-1):
#             p_means[visual_stimulus_roi[i],visual_stimulus_roi[i+1]] = Htotal_micro_vis[visual_stimulus_roi[i]]*np.conjugate(Htotal_micro_vis[visual_stimulus_roi[i+1]])
#             p_means[visual_stimulus_roi[i+1],visual_stimulus_roi[i]] = Htotal_micro_vis[visual_stimulus_roi[i+1]]*np.conjugate(Htotal_micro_vis[visual_stimulus_roi[i]])
    
    # w0 = 2*np.pi*10
    # four_cos = w0/((1e-6+1j*w)**2 + w0**2)
    
    p_means_vec = np.zeros((nroi,1),dtype="complex")
    
    # if np.any(stimulus_roi) == True:
    #     for i in range(len(stimulus_roi)):
    #         p_means_vec[stimulus_roi[i]] = Htotal_micro[stimulus_roi[i]]*four_cos
        
    p_means = np.matmul(p_means_vec,np.matrix.getH(p_means_vec))
    
    # p_var = np.diag(np.abs(Htotal_micro)**2)
    p_var = np.matmul(Htotal_micro,np.matrix.getH(Htotal_micro))
    
    p_tot = w_var*p_var + w_means*p_means
    
    p_l_FC = np.matmul(frequency_response,np.matrix.getH(eigenvectors))
    p_l_FC = np.matmul(eigenvectors,p_l_FC)
    
    p_all_FC = np.matmul(p_tot,np.matrix.getH(p_l_FC))
    p_all_FC = np.matmul(p_l_FC,p_all_FC)
    
    model_out = np.sqrt(np.abs(np.diag(p_all_FC)))
    
#     model_out2 = 0
    

#     for k in range(K):
#         model_out2 += (frequency_response2[k]) * np.matmul(np.outer(eigenvectors[:, k], np.conjugate(eigenvectors[:, k])),Htotal_micro) 
    

#     model_out3 = np.abs(model_out2)
#     model_out3 = model_out3.flatten()

    # model_out3 = np.abs(np.matmul(p_l_FC,Htotal_micro).flatten())
    
    # return model_out, p_all_FC, frequency_response2, eigenvalues, eigenvectors
    return model_out, frequency_response2, eigenvalues, eigenvectors

