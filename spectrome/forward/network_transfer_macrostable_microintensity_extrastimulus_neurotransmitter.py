"""Module for computing basic quantities from a spectral graph model: the forward model
Makes the calculation for a single frequency only. """

import numpy as np
from scipy.io import loadmat
import xarray as xr
import pandas as pd

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
    
    mica_micro_intensity = np.squeeze(loadmat('/protected/data/rajlab1/shared_data/datasets/MICA/micro_intensity_mean.mat')['micro_intensity_mean'])
    # Load excitatory profile
    ex_template_xr = xr.open_dataarray('/protected/data/rajlab1/shared_data/datasets/neurotransmitters/ex_template.nc')
    ex_template = ex_template_xr.values
    # Load inhibtory profile
    inh_template_xr = xr.open_dataarray('/protected/data/rajlab1/shared_data/datasets/neurotransmitters/inh_template.nc')
    inh_template = inh_template_xr.values
    
    a = 0.9 
    b = 1.1
    
    mine = np.amin(ex_template)
    maxe = np.amax(ex_template);
    
    ex_template_scaled = np.zeros(ex_template.shape)

    for i in range(len(ex_template)):
        ex_template_scaled[i] = (b-a)*(ex_template[i]-mine)/(maxe-mine) + a
    
    mine = np.amin(inh_template)
    maxe = np.amax(inh_template);
    
    inh_template_scaled = np.zeros(inh_template.shape)

    for i in range(len(inh_template)):
        inh_template_scaled[i] = (b-a)*(inh_template[i]-mine)/(maxe-mine) + a

#     Lobes

#     lobes = pd.read_excel("/data/rajlab1/shared_data/datasets/RSN/DK_lobes.xlsx",header=None)
    
#     inh_template_lobe = np.zeros((len(inh_template),1))
    
#     for i in range(len(inh_template)):
#         if lobes[1][i] == "Frontal" or lobes[1][i] == "Temporal":
#             inh_template_lobe[i] = np.amax(inh_template)
#         else:
#             inh_template_lobe[i] = inh_template[i]
                
#     inh_template = inh_template_lobe

    
#     # Cortical model
    FG = np.divide(1 / tauC ** 2, (1j * w + 1 / tauC) ** 2)
    
    Htotal_micro = np.zeros((82,1),dtype="complex")

    Fe = np.divide(1 / tau_e ** 2, (1j * w + 1 / tau_e) ** 2)
    Fi = np.divide(1 / tau_i ** 2, (1j * w + 1 / tau_i) ** 2)

    
    for i in range(14):
        # gee = ex_template[68+i]
        gii = parameters["gii"]*inh_template_scaled[68+i]
        gei = parameters["gei"]*np.sqrt(ex_template_scaled[68+i]*inh_template_scaled[68+i])
        Hed = (1 + (Fe * Fi * gei)/(tau_e * (1j * w + Fi * gii/tau_i)))/(1j * w + Fe * gee/tau_e + (Fe * Fi * gei)**2/(tau_e * tau_i * (1j * w + Fi * gii / tau_i)))

        Hid = (1 - (Fe * Fi * gei)/(tau_i * (1j * w + Fe * gee/tau_e)))/(1j * w + Fi * gii/tau_i + (Fe * Fi * gei)**2/(tau_e * tau_i * (1j * w + Fe * gee / tau_e)))
        
        Htotal = Hed + Hid
        Htotal_micro[68+i] = Htotal

    for i in range(68):
        # gee = ex_template[i]
        gii = parameters["gii"]*inh_template_scaled[i]
        gei = parameters["gei"]*np.sqrt(ex_template_scaled[i]*inh_template_scaled[i])
        tau_e = parameters["tau_e"]*mica_micro_intensity[i]
        tau_i = parameters["tau_i"]*mica_micro_intensity[i]

        Fe = np.divide(1 / tau_e ** 2, (1j * w + 1 / tau_e) ** 2)
        Fi = np.divide(1 / tau_i ** 2, (1j * w + 1 / tau_i) ** 2)

        Hed = (1 + (Fe * Fi * gei)/(tau_e * (1j * w + Fi * gii/tau_i)))/(1j * w + Fe * gee/tau_e + (Fe * Fi * gei)**2/(tau_e * tau_i * (1j * w + Fi * gii / tau_i)))

        Hid = (1 - (Fe * Fi * gei)/(tau_i * (1j * w + Fe * gee/tau_e)))/(1j * w + Fi * gii/tau_i + (Fe * Fi * gei)**2/(tau_e * tau_i * (1j * w + Fe * gee / tau_e)))

        Htotal = Hed + Hid
        
        Htotal_micro[i] = Htotal


    q1 = (1j * w + 1 / tauC * FG * eigenvalues)
    qthr = zero_thr * np.abs(q1[:]).max()
    magq1 = np.maximum(np.abs(q1), qthr)
    angq1 = np.angle(q1)
    q1 = np.multiply(magq1, np.exp(1j * angq1))
    frequency_response2 = np.divide(1, q1)
    
    frequency_response = np.diag(frequency_response2)
    
        
    p_means_vec = np.zeros((len(Htotal_micro),1),dtype="complex")
    
    if np.any(stimulus_roi) == True:
        for i in range(len(stimulus_roi)):
            p_means_vec[stimulus_roi[i]] = Htotal_micro[stimulus_roi[i]]
        
    p_means = np.matmul(p_means_vec,np.matrix.getH(p_means_vec))
    
    # p_var = np.diag(np.abs(Htotal_micro)**2)
    p_var = np.matmul(Htotal_micro,np.matrix.getH(Htotal_micro))
    
    p_tot = w_var*p_var + w_means*p_means
    
    p_l_FC = np.matmul(frequency_response,np.matrix.getH(eigenvectors))
    p_l_FC = np.matmul(eigenvectors,p_l_FC)
    
    p_all_FC = np.matmul(p_tot,np.matrix.getH(p_l_FC))
    p_all_FC = np.matmul(p_l_FC,p_all_FC)
    
    model_out = np.sqrt(np.abs(np.diag(p_all_FC)))
    
    return model_out, frequency_response2, eigenvalues, eigenvectors

