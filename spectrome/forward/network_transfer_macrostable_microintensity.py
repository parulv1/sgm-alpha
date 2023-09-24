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

#     Add a sumsum before doing all of this
#     C = C/C.sum()
    C = C/np.linalg.norm(C)
    # define sum of degrees for rows and columns for laplacian normalization
    rowdegree = np.transpose(np.sum(C, axis=1))
    coldegree = np.sum(C, axis=0)
    
    degree = rowdegree + coldegree
    eps = min([i for i in degree if i > 0])
    
    
#     Ashish suggests removing all of these
    # qind = rowdegree + coldegree < 0.2 * np.mean(rowdegree + coldegree)
    # rowdegree[qind] = np.inf
    # coldegree[qind] = np.inf

    nroi = C.shape[0]

    K = nroi
    
#     Cutting connection to thalamus
    # C[69,:] = 0
    # C[:,69] = 0
    # C[78,:] = 0
    # C[:,78] = 0

    Tau = 0.001 * D / speed
    Cc = C * np.exp(-1j * Tau * w)

    # Eigen Decomposition of Complex Laplacian Here
    L1 = np.identity(nroi)
#     Try this normalization
    # Lr = np.divide(1, np.sqrt(rowdegree) + np.spacing(1))
    # Lc = np.divide(1, np.sqrt(coldegree) + np.spacing(1))
    # L = L1 - alpha * np.matmul(np.diag(Lr), np.matmul(Cc, np.diag(Lc)))
    
    L2 = np.divide(1, np.sqrt(np.multiply(rowdegree, coldegree)) + eps)
    L = L1 - alpha * np.matmul(np.diag(L2), Cc)

    d, v = np.linalg.eig(L)  
    eig_ind = np.argsort(np.abs(d))  # sorting in ascending order and absolute value
    eig_vec = v[:, eig_ind]  # re-indexing eigen vectors according to sorted index
    eig_val = d[eig_ind]  # re-indexing eigen values with same sorted index

    eigenvalues = np.transpose(eig_val)
    eigenvectors = eig_vec[:, 0:K]

    # eigenvectors_inv = np.linalg.inv(eigenvectors)
    
    mica_micro_intensity = np.squeeze(loadmat('/data/rajlab1/shared_data/datasets/MICA/micro_intensity_mean.mat')['micro_intensity_mean'])

#     # Cortical model
    FG = np.divide(1 / tauC ** 2, (1j * w + 1 / tauC) ** 2)
    
    # Htotal_micro = np.zeros((86,1),dtype="complex")
    Htotal_micro = np.zeros((82,1),dtype="complex")

    Fe = np.divide(1 / tau_e ** 2, (1j * w + 1 / tau_e) ** 2)
    Fi = np.divide(1 / tau_i ** 2, (1j * w + 1 / tau_i) ** 2)

    Hed = (1 + (Fe * Fi * gei)/(tau_e * (1j * w + Fi * gii/tau_i)))/(1j * w + Fe * gee/tau_e + (Fe * Fi * gei)**2/(tau_e * tau_i * (1j * w + Fi * gii / tau_i)))

    Hid = (1 - (Fe * Fi * gei)/(tau_i * (1j * w + Fe * gee/tau_e)))/(1j * w + Fi * gii/tau_i + (Fe * Fi * gei)**2/(tau_e * tau_i * (1j * w + Fe * gee / tau_e)))

    Htotal = Hed + Hid
    
    # for i in range(18):
    for i in range(14):
        Htotal_micro[68+i] = Htotal

    for i in range(68):
        tau_e = parameters["tau_e"]*mica_micro_intensity[i]
        tau_i = parameters["tau_i"]*mica_micro_intensity[i]

        Fe = np.divide(1 / tau_e ** 2, (1j * w + 1 / tau_e) ** 2)
        Fi = np.divide(1 / tau_i ** 2, (1j * w + 1 / tau_i) ** 2)
        
        # tau_e = parameters["tau_e"]
        # tau_i = parameters["tau_i"]

        Hed = (1 + (Fe * Fi * gei)/(tau_e * (1j * w + Fi * gii/tau_i)))/(1j * w + Fe * gee/tau_e + (Fe * Fi * gei)**2/(tau_e * tau_i * (1j * w + Fi * gii / tau_i)))

        Hid = (1 - (Fe * Fi * gei)/(tau_i * (1j * w + Fe * gee/tau_e)))/(1j * w + Fi * gii/tau_i + (Fe * Fi * gei)**2/(tau_e * tau_i * (1j * w + Fe * gee / tau_e)))

        Htotal = Hed + Hid
        
        Htotal_micro[i] = Htotal


    #     q1 = (1j * w + 1 / tau_e * Fe * eigenvalues)
    q1 = (1j * w + 1 / tauC * FG * eigenvalues)
    qthr = zero_thr * np.abs(q1[:]).max()
    magq1 = np.maximum(np.abs(q1), qthr)
    angq1 = np.angle(q1)
    q1 = np.multiply(magq1, np.exp(1j * angq1))
    frequency_response = np.divide(Htotal, q1)
    frequency_response2 = np.divide(1, q1)
    
    model_out = 0
    

    for k in range(K):
        model_out += (frequency_response2[k]) * np.matmul(np.outer(eigenvectors[:, k], np.conjugate(eigenvectors[:, k])),Htotal_micro) 
        # model_out += (frequency_response2[k]) * np.matmul(np.outer(eigenvectors[:, k], eigenvectors_inv[k, :]),Htotal_micro) 
    

    model_out2 = np.abs(model_out)
    model_out2 = model_out2.flatten()

    return model_out2, frequency_response2, eigenvalues, eigenvectors

