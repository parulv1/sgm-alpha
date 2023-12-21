import numpy as np
from ..forward import runforward
# from ..forward import runforward_spatialcorrelation_mahalanobis
# from ..forward import runforward_spatialcorrelation_old
# from ..forward import runforward_spatialcorrelation_topalpha
# from ..forward import runforward_spatialcorrelation_adjacency 
from ..forward import runforward_spatialcorrelation_pearson
from scipy.signal import firls
from scipy.stats import pearsonr
from ..utils import functions, path
from ..brain import Brain

from statsmodels.stats import weightstats


# def global_corr(x, brain, F_ind, rois_with_MEG, fvec, lpf):
# Following def with reordered spectra
# F_ind should already be in dB
def global_corr(x, brain, F_ind_db, F_ind, rois_with_MEG, fvec):
    
    # re-assigne optimized parameters:s
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

                                         
    w1 = np.ones((len(fvec)))
    w1[0:10] = 2
                                         
    corrs = np.zeros(len(freq_out))
    for c in np.arange(0, len(freq_out)):
        d1 = weightstats.DescrStatsW(F_ind_db[c,:],weights=w1)
        d2 = weightstats.DescrStatsW(freq_out[c,:],weights=w1)
        
        cov_matrix = np.cov(d1.data, d2.data, aweights=d1.weights)

        corr_coefficient = cov_matrix[0, 1] / np.sqrt(cov_matrix[0, 0] * cov_matrix[1, 1])
        corrs[c] = corr_coefficient
        # corrs[c] = pearsonr(F_ind_db[c,:], freq_out[c,:])[0]

    ri_corr = np.mean(corrs)

#     sp_corr, _, _ = runforward_spatialcorrelation.run_local_coupling_forward_Xk(brain, brain.ntf_params, fvec, F_ind, 86, rois_with_MEG, "alpha")
    weighted_corr = runforward_spatialcorrelation_pearson.run_local_coupling_forward_Xk(brain, brain.ntf_params, fvec, F_ind, 86, rois_with_MEG, "alpha")

    return -ri_corr - 0.3*weighted_corr
    # return -ri_corr - (1- weighted_corr)