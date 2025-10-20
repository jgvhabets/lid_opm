"""
functions to apply within preprocessing steps (preprocessing.py)
"""

# import public libraries
import numpy as np
from itertools import product, compress
from scipy.signal import butter, filtfilt, iirnotch
from mne.filter import filter_data, notch_filter, resample



def resample_aux_array(temp_auxdat, aux_chnames, auxtimes,
                       aux_sfreq, FACTOR,):
    
    auxdat_resampled = []

    for i_ch, chname in enumerate(aux_chnames):
        if 'acc' in chname or 'emg' in chname:
            auxdat_resampled.append(resample(temp_auxdat[:, i_ch], down=FACTOR))
    
    new_auxdat = np.array(auxdat_resampled).T

    new_auxtimes = auxtimes[0] + np.arange(0, temp_auxdat.shape[0]) * 1/aux_sfreq
    new_aux_chnames = aux_chnames[1:]  # aligned_time is not in auxdat anymore

    return new_auxdat, new_aux_chnames, new_auxtimes


def get_signal_vector_magn(triax_sig):

    if triax_sig.shape[0] != 3: triax_sig = triax_sig.T
    assert triax_sig.shape[0] == 3, 'no 3-axial signal fiven for SVM'

    svm = np.sqrt(
        triax_sig[0, :] ** 2 +
        triax_sig[1, :] ** 2 +
        triax_sig[2, :] ** 2
    )

    return svm


def get_emg_envelop(sig, sfreq, low_bpass=20, high_bpass=250, env_lowpass=4):

    # apply notch filters
    for f in [50, 100, 150, 200,]:
        sig = apply_filter(sig, low_f=f, sfreq=sfreq, type='notch')

    # bandpass filter to remove artefactsd and isolate oscillations
    sig = apply_filter(sig, low_f=low_bpass, high_f=high_bpass,
                              order=4, sfreq=sfreq, type='band')
    
    # rectify signal
    sig = np.abs(sig)

    # get slow oscillations of interest, get "envelop"
    sig = apply_filter(sig, order=2, low_f=env_lowpass,
                               sfreq=sfreq, type='low',)

    return sig


def get_emg_tkeo(sig, sfreq, smooth_winlen=20,):
    """
    Teager-Kaiser Energy Operator (TKEO):
    more precise for EMG onset detection
    """
    # apply notch filters
    for f in [50, 100, 150, 200,]:
        sig = apply_filter(sig, low_f=f, sfreq=sfreq, type='notch')

    tkeo_sig = sig[1:-1] ** 2 - (sig[0:-2] * sig[2:])  # length reduces by 2
    temp = np.zeros_like(sig)  # pad to original length
    temp[1:-1] = tkeo_sig
    tkeo_sig = temp
    tkeo_sig = np.abs(tkeo_sig)  # rectify
    # smoothen
    kernel = np.ones(smooth_winlen) / max(1, smooth_winlen)
    tkeo_sig = np.sqrt(np.convolve(tkeo_sig ** 2, kernel, mode='same'))

    return tkeo_sig
    

def apply_filter(sig, low_f, sfreq, order=None, high_f=None,  type='band',
                 Q=30,):
    """
    for notch filter, freq = low_f

    q is for notch, larger Q, narrower notch
    """
    
    assert type in ['band', 'low', 'notch'], 'wrong type butter filter'

    if not order:
        if type == 'band': order = 4
        else: order = 2


    if type == 'band':
        b, a = butter(order, [low_f, high_f], fs=sfreq, btype='band')  # in older scipy version: high_bpass/(sfreq/2), without fs argument

    elif type == 'low':
        b, a = butter(order, low_f, fs=sfreq, btype='low')  # in older scipy version: high_bpass/(sfreq/2), without fs argument

    elif type == 'notch':
        b, a = iirnotch(low_f, Q=Q, fs=sfreq,)

    sig = filtfilt(b, a, sig)

    return sig