import numpy as np
from ..forward import runforward
from ..forward import runforward_spatialcorrelation_mahalanobis
# from ..forward import runforward_spatialcorrelation_old
# from ..forward import runforward_spatialcorrelation_topalpha
from scipy.signal import firls
from scipy.stats import pearsonr
from ..utils import functions, path
from ..brain import Brain


# def global_corr(x, brain, F_ind, rois_with_MEG, fvec, lpf):
# Following def with reordered spectra
# F_ind should already be in dB
def global_corr(x, brain, F_ind_db, F_ind, rois_with_MEG, fvec):
    
    # re-assigne optimized parameters:
    brain.ntf_params['tau_e'] = x[0]/1000
    brain.ntf_params['tau_i'] = x[1]/1000
    brain.ntf_params['alpha'] = x[2]
    brain.ntf_params['speed'] = x[3]
    brain.ntf_params['gei'] = x[4]
    brain.ntf_params['gii'] = x[5]
    brain.ntf_params['tauC'] = x[6]/1000
    # brain.ntf_params['tau_e'] = x[0]/1000
    # brain.ntf_params['tau_i'] = x[1]/1000
    # brain.ntf_params['alpha'] = 0
    # brain.ntf_params['speed'] = x[2]
    # brain.ntf_params['gei'] = x[3]
    # brain.ntf_params['gii'] = x[4]
    # brain.ntf_params['tauC'] = x[5]/1000
    # simulate model spectra:
    freq_mdl, _, _, _ = runforward.run_local_coupling_forward(brain, brain.ntf_params, fvec)
    freq_mdl = freq_mdl[rois_with_MEG,:]

    # smooth out spectra
    freq_out = np.zeros(freq_mdl.shape)
    for p in np.arange(0,len(freq_mdl)):
        freq_out[p,:] = functions.mag2db(np.abs(freq_mdl[p,:]))

                                         
    corrs = np.zeros(len(freq_out))
    for c in np.arange(0, len(freq_out)):
        corrs[c] = pearsonr(F_ind_db[c,:], freq_out[c,:])[0]

    ri_corr = np.mean(corrs)

#     sp_corr, _, _ = runforward_spatialcorrelation.run_local_coupling_forward_Xk(brain, brain.ntf_params, fvec, F_ind, 86, rois_with_MEG, "alpha")

    weighted_corr = runforward_spatialcorrelation_mahalanobis.run_local_coupling_forward_Xk_distance(brain, brain.ntf_params, fvec, F_ind, 86, rois_with_MEG, "alpha")

    return ri_corr, weighted_corr