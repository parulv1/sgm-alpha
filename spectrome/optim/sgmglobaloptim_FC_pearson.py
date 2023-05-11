import numpy as np
from ..forward import runforward
from ..forward import est_FC
from scipy.stats import pearsonr
from ..utils import functions


# def global_corr(x, brain, F_ind, rois_with_MEG, fvec, lpf):
# Following def with reordered spectra
# F_ind should already be in dB
def global_corr(x, brain, F_ind_db, meg_FC, rois_with_MEG, fvec):
    
    # re-assigne optimized parameters:s
    brain.ntf_params['tau_e'] = x[0]/1000
    brain.ntf_params['tau_i'] = x[1]/1000
    brain.ntf_params['alpha'] = x[2]
    brain.ntf_params['speed'] = x[3]
    brain.ntf_params['gei'] = x[4]
    brain.ntf_params['gii'] = x[5]
    brain.ntf_params['tauC'] = x[6]/1000

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

    model_FC = est_FC.build_fc_freq(brain, brain.ntf_params, rois_with_MEG)

    
    fc_corr = [pearsonr(vec_meg,vec_model)[0] for vec_meg, vec_model in zip(functions.divideFc(meg_FC), functions.divideFc(model_FC))]
    
    # fc_corr = pearsonr(meg_FC.flatten(),model_FC.flatten())[0]

    return ri_corr, np.mean(fc_corr)