"""
functions to apply within preprocessing steps (preprocessing.py)
"""

# import public libraries
import os
import numpy as np
from itertools import product, compress
from scipy.signal import butter, filtfilt, iirnotch, hilbert
from mne.filter import filter_data, notch_filter, resample
from sklearn.preprocessing import StandardScaler
from mne import pick_types, Epochs
from mne.io import BaseRaw

from utils.load_utils import get_onedrive_path

def apply_acc_preprocessing_in_Raw(
    aux_raw,
    bandpass_low: float = 0.5, bandpass_high: float = 20,
    signal_vector_magn: bool = False, zscore: bool = True,
):
    """
    applies acc preprocessing steps to mne Raw object, in-place.
    doesnt return anything, modifies Raw object directly.
    Steps include:
    - bandpass filter (default 0.5-20Hz)
    - z-score scaling (default True)
    """
    misc_picks = pick_types(aux_raw.info, misc=True)   # ACC
    
    if not bandpass_low and not bandpass_high:
        pass
    else:
        aux_raw.filter(l_freq=bandpass_low, h_freq=bandpass_high, picks=misc_picks)
    
    if signal_vector_magn:
        acc_data = aux_raw.get_data(picks=misc_picks)
        # combine x,y,z each side into one signal, adjust info file from 3 to 1 (or add svm)
        # acc_data = 
        aux_raw._data[misc_picks] = acc_data


    if zscore:
        acc_data = aux_raw.get_data(picks=misc_picks)
        scaler = StandardScaler()
        acc_data_zscored = scaler.fit_transform(acc_data.T).T
        aux_raw._data[misc_picks] = acc_data_zscored



def apply_emg_preprocessing_in_Raw(
    aux_raw,
    bandpass_low: float = 20, bandpass_high: float = 200,
    zscore: bool = False, notch_freqs: list = [50, 100, 150, 200],
    emg_signal_type : str = 'raw',
):
    """
    applies emg preprocessing steps to mne Raw object, in-place.
    doesnt return anything, modifies Raw object directly.
    Steps include:
    - bandpass filter (default 20-200Hz)
    - z-score scaling (default True)
    """
    # apply only on EMG channels, not on acc channels (if both are present)
    emg_picks = pick_types(aux_raw.info, emg=True)

    # notch filter
    if type(notch_freqs) != list or len(notch_freqs) == 0:
        pass
    else:
        aux_raw.notch_filter(freqs=notch_freqs, picks=emg_picks)

    # bandpass filter
    if not bandpass_low and not bandpass_high:
        pass
    else:
        aux_raw.filter(l_freq=bandpass_low, h_freq=bandpass_high, picks=emg_picks)
    
    # convert into hilbert envelope by rectifying, or get TKEO, depending on settings
    assert emg_signal_type in ['hilbert_env', 'tkeo', 'raw'], 'EMG_SIGNAL_TYPE must be either "hilbert_env", "tkeo" or "raw"'
    
    if emg_signal_type == 'hilbert_env':
        emg_data = aux_raw.get_data(picks=emg_picks)
        analytic = hilbert(emg_data, axis=1)
        emg_env = np.abs(analytic)
        aux_raw._data[emg_picks] = emg_env
    elif emg_signal_type == 'tkeo':
        emg_data = aux_raw.get_data(picks=emg_picks)
        tkeo_emg = calculate_tkeo(emg_data)
        aux_raw._data[emg_picks] = tkeo_emg
    else:
        pass
    
    if zscore:
        emg_data = aux_raw.get_data(picks=emg_picks)
        scaler = StandardScaler()
        emg_data_zscored = scaler.fit_transform(emg_data.T).T
        aux_raw._data[emg_picks] = emg_data_zscored



def calculate_tkeo(emg_data):
    """
    converts 2d emg data into 2d tkeo-emg data, by applying TKEO
    and rectifying. TKEO function: x[n]^2 - x[n-1]*x[n+1], and
    conceptually captures the energy of the signal,
    more precise for EMG onset detection.
    """
    tkeo = np.zeros_like(emg_data)
    
    smooth_kernel_size = 20  # adjust as needed

    for i in range(emg_data.shape[0]):
        sig = emg_data[i, :]
        tkeo_sig = sig[1:-1] ** 2 - (sig[0:-2] * sig[2:])  # length reduces by 2
        # smoothen after TKEO, with moving average kernel
        kernel = np.ones(smooth_kernel_size) / smooth_kernel_size
        tkeo_sig = np.sqrt(np.convolve(tkeo_sig ** 2, kernel, mode='same'))

        temp = np.zeros_like(sig)  # pad to original length
        temp[1:-1] = tkeo_sig
        tkeo[i, :] = np.abs(temp)  # rectify
        
    return tkeo


def resample_aux_array(temp_auxdat, aux_chnames, auxtimes,
                       aux_sfreq, FACTOR,):
    
    auxdat_resampled = []

    for i_ch, chname in enumerate(aux_chnames):
        if 'acc' in chname or 'emg' in chname:
            auxdat_resampled.append(resample(temp_auxdat[:, i_ch], down=FACTOR))
    
    new_auxdat = np.array(auxdat_resampled).T

    new_auxtimes = auxtimes[0] + np.arange(0, new_auxdat.shape[0]) * 1/aux_sfreq
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


def save_cleaned_data(
    meg_data=None, aux_data=None, SUB=None, SES=None, ACQ=None, TASK=None,
    config_version=None,
):
    """
    after all preprocessing steps, save the cleaned raw and epoch objects
    """

    # check path existence, otherwise create
    deriv_dir = os.path.join(
        get_onedrive_path('cleaned_data'),
        f"cleaned_data_preproc_{config_version}",
        f"sub-{SUB}",
    )
    os.makedirs(deriv_dir, exist_ok=True)

    if meg_data is not None: meg_present = True
    if aux_data is not None: aux_present = True

    # define whether data is Raw or Epochs for correct saving
    if meg_present and isinstance(meg_data, BaseRaw):
        data_type = 'raw'
    elif aux_present and isinstance(aux_data, BaseRaw):
        data_type = 'raw'
    elif meg_present and isinstance(meg_data, Epochs):
        data_type = 'epochs'
    elif aux_present and isinstance(aux_data, Epochs):
        data_type = 'epochs'
    else:
        raise ValueError('meg_data should be either mne.io.Raw or mne.Epochs')
    
    if meg_present:
        filename_meg = f"MEG_cleaned_{data_type}_sub-{SUB}_ses-{SES}_acq-{ACQ}_task-{TASK}.fif"
    if aux_present:
        filename_aux = f"AUX_cleaned_{data_type}_sub-{SUB}_ses-{SES}_acq-{ACQ}_task-{TASK}.fif"
    
    if meg_present:
        meg_data.save(os.path.join(deriv_dir, filename_meg), overwrite=True)
    if aux_present:
        aux_data.save(os.path.join(deriv_dir, filename_aux), overwrite=True)
    