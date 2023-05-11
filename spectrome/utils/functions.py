import numpy as np
import math


def mag2db(y):
    """Convert magnitude response to decibels for a simple array.

    Args:
        y (numpy array): Power spectrum, raw magnitude response.

    Returns:
        dby (numpy array): Power spectrum in dB

    """
    dby = 20 * np.log10(y)
    return dby

def divideFc(fc):
    """Divide the FC into three blocks and vector them
    """
    size_fc = len(fc)
    hs_fc = int(size_fc/2)
    fc_block1 = fc[:hs_fc, :hs_fc]
    fc_block2 = fc[hs_fc:, hs_fc:];
    fc_block3 = fc[:hs_fc, hs_fc:];
    vec1 = fc_block1[np.triu_indices(hs_fc, k = 1)].flatten()
    vec2 = fc_block2[np.triu_indices(hs_fc, k = 1)].flatten()
    vec3 = fc_block3.flatten()
    return vec1, vec2, vec3