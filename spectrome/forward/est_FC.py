""" 
Reference: "Spectral graph theory of brain oscillations--revisited and improved"

The code computes the frequency domain functional network (normalized) at a given frequency. 
To note, the code does not include the P(\omega) noise matrix, will be added in a future version.
"""

import numpy as np
from spectrome.forward import network_transfer_macrostable_microintensity_extrastimulus as nt
# import mne
# import mne_connectivity
# from mne_connectivity import envelope_correlation


# For Matlab struct use dict, e.g. p.x --> p['x']

def build_fc_freq(brain, params, rois_with_MEG):

    """
    Input:
    
    brain: brain model
    params: brain parameters
    freqrange: a struct containing the frequency range (bandwidth) of interest, in Hz, with 
    ranges alpha, beta, delta, theta, gamma.

    Output:

    absFC, the normalized absolute value of the estimated complex FC at the given frequency computed as the mean 
    of the range given in freqrange.
    
    realFC: real part of the estimated FC, normalized

    imgFC: imaginary part of the estimated FC, normalized
    """
    # numAxis = 2 # Axis along which to compute time series

    # Check the dimensionality of mod_fq_resp below, size KxK where K is network number of nodes
    # 3D, see earlier
    # mod_fq_resp, _, _, _ = runforward.run_local_coupling_forward( brain, params, freqs );

    mean_freq = 2 * np.pi * 10.5
    _, estFC, _, _, _ = nt.network_transfer_local_alpha( brain , params, mean_freq, np.array([]), 1, 0)
    
    
    # Absolute value of FC
    
    absFC = np.abs( estFC[0:len(rois_with_MEG),0:len(rois_with_MEG)] )
    diagFC = np.diag( absFC )
    diagFC = 1./np.sqrt( diagFC )
    D = np.diag( diagFC )
    absFC = np.matmul( D , absFC )
    absFC = np.matmul( absFC , np.matrix.getH(D) )
    absFC = absFC - np.diag(np.diag( absFC ))
    
    return absFC