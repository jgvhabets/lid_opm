"""
Signal pre-processing functions for OPM and MEG, ACC
"""


import numpy as np
from dataclasses import dataclass
from itertools import product, compress
from scipy.signal import butter, filtfilt, iirnotch


from source_raw_conversion.load_lsl import convert_source_lsl_to_raw

@dataclass()
class rawData_singleRec:
    """
    contains aligned raw data, t-0 is start of
    OPM recording; for one single recording (task)

    TODO: signal processing/state detection try to merge emg per extr
    """
    sub: str
    task: str
    acq: str
    INCL_AUX: bool = True
    INCL_OPM: bool = False

    def __post_init__(self,):

        if self.INCL_AUX:
            # load aux
            (
                self.auxdat, self.aux_chnames,
                self.aux_sfreq, self.tasktimings
            ) = convert_source_lsl_to_raw(self.sub, self.task, self.acq)

            self.times = self.auxdat[:, 0]
            
            if self.task != 'rest':
                self.task_epochs = add_task_epoch_idx(self=self)
            
            self.rel_aux_sigs = []  # storing naming of relevant attributes

            ### preprocess acc
            # bandpass all acc channels
            for i_ch in np.where(['acc' in c for c in self.aux_chnames])[0]:
                sig = self.auxdat[:, i_ch].copy()
                sig = apply_butter_filter(sig, sfreq=self.aux_sfreq, type='band',
                                          low_f=2, high_f=20,)
                self.auxdat[:, i_ch] = sig

            # get sign vector magn for every extremity
            for side, extr in product(['left', 'right'], ['hand', 'foot']):
                triacc = [all(['acc' in c, side in c, extr in c])
                          for c in self.aux_chnames]
                triacc = self.auxdat[:, triacc]
                # calc svm
                svm = get_signal_vector_magn(triacc)

                setattr(self, f'acc_svm_{side}{extr}', svm)
                self.rel_aux_sigs.append(f'acc_svm_{side}{extr}')


            ### preprocess EMG
            self.emg_chnames = [c for c in self.aux_chnames if 'emg' in c.lower()]
            for i_ch in np.where(['emg' in c for c in self.aux_chnames])[0]:
                name = self.aux_chnames[i_ch]
                setattr(
                    self,
                    name.replace('emg', 'emg_env'),
                    get_emg_envelop(self.auxdat[:, i_ch], sfreq=self.aux_sfreq)
                )
                self.rel_aux_sigs.append(name.replace('emg', 'emg_env'))
                setattr(
                    self,
                    name.replace('emg', 'emg_tkeo'),
                    get_emg_tkeo(self.auxdat[:, i_ch], sfreq=self.aux_sfreq,)
                )
                self.rel_aux_sigs.append(name.replace('emg', 'emg_tkeo'))
            
            ### ZSCORE RELEVANT FEATURES
            for ft in self.rel_aux_sigs:
                sig = getattr(self, ft)
                sig = (sig - np.mean(sig)) / np.std(sig)
                setattr(self, ft, sig)
            



def add_task_epoch_idx(self,):
    """
    takes one second before start and one second after end
    """

    INCL_EDGE = 1  # second extra prior and post

    task_epochs = {}

    for task, timings in self.tasktimings.items():

        task_epochs[task] = []

        for t0, t1 in zip(timings['start'], timings['end']):

            i0 = np.argmin(abs(self.times - (t0 - INCL_EDGE)))
            i1 = np.argmin(abs(self.times - (t1 + INCL_EDGE)))

            if i0 == i1: continue  # happened when task was incorrectly started before real start
            if (i1 - i0) < (.8 + INCL_EDGE + INCL_EDGE) * self.aux_sfreq:
                print(f'skipped epoch with sample length {i1-i0}')
                continue  # too short
            
            task_epochs[task].append((i0, i1))

    return task_epochs



def get_signal_vector_magn(triax_sig):

    if triax_sig.shape[0] != 3: triax_sig = triax_sig.T
    assert triax_sig.shape[0] == 3, 'no 3-axial signal fiven for SVM'

    svm = np.sqrt(
        triax_sig[0, :] ** 2 +
        triax_sig[1, :] ** 2 +
        triax_sig[2, :] ** 2
    )

    return svm


def get_emg_envelop(sig, sfreq, low_bpass=20, high_bpass=450, env_lowpass=4):

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