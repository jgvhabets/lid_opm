"""
Signal pre-processing functions for OPM and MEG, ACC
"""


import numpy as np
from dataclasses import dataclass, field
from itertools import product, compress
from scipy.signal import butter, filtfilt, iirnotch
import datetime as dt 

from mne.filter import filter_data, notch_filter, resample
from mne.preprocessing import ICA, compute_proj_hfc
from mne import create_info
from mne.io import RawArray

import source_raw_conversion.load_source_opm as source_opm
from source_raw_conversion.load_lsl import convert_source_lsl_to_raw
from utils import load_utils

@dataclass()
class rawData_singleRec:
    """
    contains aligned raw data, t-0 is start of
    OPM recording; for one single recording (task)

    Inputs:
    - sub: "XX", ie "03"
    - task: either "rest" or "task"
    - acq: "predopa", or "dopaXX"
    - INCL_AUX: include acc and emg, defaults true
    - INCL_OPM: include OPM MEG, defaults false
    - OPM_AXES_INCL: defaults to 'Z'
    - OPM_PREPROC: default dict: {'resample': True, 'bandpass': False, 'notch': False}


    TODO: signal processing/state detection try to merge emg per extr
    """
    sub: str
    task: str
    acq: str
    INCL_AUX: bool = True
    INCL_OPM: bool = False
    OPM_AXES_INCL: list = field(default_factory=lambda: ['Z',])
    OPM_PREPROC: dict = field(default_factory=lambda: {
        'resample': True, 'bandpass': False, 'notch': False
    })
    ZSCORE_ACC: bool = False
    ZSCORE_EMG: bool = False
    CONFIG_VERSION: str = 'v1'

    def __post_init__(self,):
        # load config for sub and general
        self.sub_config = load_utils.load_subject_config(subject_id=self.sub,)
        self.preproc_config = load_utils.load_preproc_config(version=self.CONFIG_VERSION,)

        if self.INCL_AUX:
            # load aux
            # dont make aux_dat a class attribute to prevent storing of double data
            (
                temp_auxdat, self.aux_chnames,
                self.aux_sfreq, self.tasktimings
            ) = convert_source_lsl_to_raw(self.sub, self.task, self.acq)
            self.auxtimes = temp_auxdat[:, 0]
            # self.rel_aux_sigs = []  # storing naming of relevant attributes


            print(f'0000 SFREQ: {self.aux_sfreq}, TIMINGS-sec: {self.tasktimings["go_left"]["start"][:3]}')
            
            # # add indices from task stimulations, including 1-sec prior and 1-sec post!!
            # self.aux_task_epochs = add_task_epoch_idx(
            #     self=self, times_to_use=self.auxtimes, sfreq=self.aux_sfreq
            # )
            # # check resulting indices and timings
            # i_test = self.aux_task_epochs["go_left"][0][0]
            # print(f'1111 SFREQ: {self.aux_sfreq}, TIMINGS: {self.auxtimes[:3]}, E TIME: @ {i_test}: {self.auxtimes[i_test]}')

            ### preprocess acc
            auxdat_resampled = []
            resample_factor = self.aux_sfreq / self.preproc_config["TARGET_SFREQ"]
        
            for i_ch, chname in enumerate(self.aux_chnames):
                if 'acc' in chname or 'emg' in chname:
                    auxdat_resampled.append(resample(temp_auxdat[:, i_ch], down=resample_factor))
            temp_auxdat = np.array(auxdat_resampled).T
            self.aux_sfreq = self.preproc_config["TARGET_SFREQ"]
            self.auxtimes = self.auxtimes[0] + np.arange(0, temp_auxdat.shape[0]) * 1/self.aux_sfreq
            self.aux_chnames = self.aux_chnames[1:]  # aligned_time is not included anymore

            # # check resulting indices and timings
            # i_test = self.aux_task_epochs["go_left"][0][0]
            # print(f'2222: SFREQ: {self.aux_sfreq}, TIMINGS: {self.auxtimes[:3]}, E TIME: @ {i_test}: {self.auxtimes[i_test]}')
            
            # bandpass all acc channels
            for i_ch in np.where(['acc' in c for c in self.aux_chnames])[0]:
                sig = temp_auxdat[:, i_ch].copy()
                sig = apply_filter(sig, sfreq=self.aux_sfreq, type='band',
                                          low_f=2, high_f=20,)
                temp_auxdat[:, i_ch] = sig

            # get sign vector magn for every extremity
            # TODO: consider to use only dominant-acc-axis, without absolute values
            acc_chnames = []
            acc_data = []
            for side, extr in product(['left', 'right'], ['hand', 'foot']):
                triacc = [all(['acc' in c, side in c, extr in c])
                          for c in self.aux_chnames]
                triacc = temp_auxdat[:, triacc]
                # calc svm
                acc_data.append(get_signal_vector_magn(triacc))
                acc_chnames.append(f'acc_svm_{side}{extr}')
            acc_data = np.array(acc_data)
            
            # standardise all emg and acc channels
            if self.ZSCORE_EMG:
                for i in np.arange(len(acc_chnames)):
                    acc_data[i, :] = (acc_data[i, :] - np.mean(acc_data[i, :])) / np.std(acc_data[i, :])
            
            # add EMG signals as rawmne array
            acc_info = create_info(ch_names=acc_chnames, sfreq=self.aux_sfreq, ch_types='misc')
            setattr(self, 'ACC', RawArray(acc_data, acc_info))


            ### preprocess EMG
            emg_chnames, emg_data = [], []
            for i_ch in np.where(['emg' in c for c in self.aux_chnames])[0]:
                name = self.aux_chnames[i_ch]
                # get envelop
                emg_chnames.append(name.replace('emg', 'emg_env'))
                emg_data.append(get_emg_envelop(temp_auxdat[:, i_ch], sfreq=self.aux_sfreq))
                
                # get TKEO
                # emg_chnames.append(name.replace('emg', 'emg_tkeo'))
                # emg_data.append(get_emg_tkeo(temp_auxdat[:, i_ch], sfreq=self.aux_sfreq,))
            
            emg_data = np.array(emg_data)
            
            # standardise all emg and acc channels
            if self.ZSCORE_ACC:
                # currently z-scoring within task-recording, not over total recording
                for i in np.arange(len(emg_chnames)):
                    emg_data[i, :] = (emg_data[i, :] - np.mean(emg_data[i, :])) / np.std(emg_data[i, :])

            
            # add EMG signals as rawmne array
            emg_info = create_info(ch_names=emg_chnames, sfreq=self.aux_sfreq, ch_types='emg')
            setattr(self, 'EMG', RawArray(emg_data, emg_info))
            # average arm EMG's
            take_mean_emgArms(self,)  # replaces RawArray with mean inst of 2 sep arm emgs
        
            
            ### ADD EPOCHING AFTER SIGNAL PREPROCESSING
            # add indices from task stimulations, including 1-sec prior and 1-sec post!!
            self.aux_task_epochs = add_task_epoch_idx(
                self=self, times_to_use=self.auxtimes, sfreq=self.aux_sfreq
            )
            # # check resulting indices and timings
            # i_test = self.aux_task_epochs["go_left"][0][0]
            # print(f'3333: SFREQ: {self.aux_sfreq}, TIMINGS: {self.auxtimes[:3]}, E TIME: @ {i_test}: {self.auxtimes[i_test]}')


        #####
        if self.INCL_OPM:
            
            # load data per opm axis
            for ax in self.OPM_AXES_INCL:
                axdata, axtimes = source_opm.select_and_store_axis_data(
                    AX=ax, ACQ=self.acq, TASK=self.task,
                    sub_config=self.sub_config, LOAD=True,
                )
                rawmne = source_opm.load_raw_opm_into_mne(
                    meg_data=axdata, AX=ax, sub_config=self.sub_config,
                )
                # currently one time axis for all opm axes
                self.opmrec_times = axtimes
                setattr(self, f'OPM_{ax}', rawmne)

                # preprocess opm axis
                preprocess_opm(self, axis=ax,)

                self.opm_task_epochs = add_task_epoch_idx(
                    self=self,
                    times_to_use=getattr(self, f'OPM_{ax}').times,
                    sfreq=getattr(self, f'OPM_{ax}').info['sfreq']
                )
                # create array to mark event epochs
                self.event_codes, self.event_arr = get_mne_event_array(self)




def preprocess_opm(self, axis,):
    """
    
    Input:
    - self
    - axis: x, y, z
    - opm_rectimes: timestamps in seconds since start of opm recording,
    on this time line antneuro and opm data are synced and aligned
    
    """
    meg_info = getattr(self, f"OPM_{axis}").info

    if self.OPM_PREPROC['resample']:
        # resample
        goal_fs = self.preproc_config["TARGET_SFREQ"]
        if meg_info['sfreq'] <= goal_fs:
            print(f'original sfreq {meg_info.info["sfreq"]} vs sfreq {goal_fs}')
        else:
            print(f'resample original sampling rate {meg_info["sfreq"]} to {goal_fs}')
            setattr(
                self,
                f"OPM_{axis}",
                getattr(self, f"OPM_{axis}").resample(goal_fs, verbose=False)
            )
        # align opmrectimes
        nsamples = len(getattr(self, f"OPM_{axis}").times)
        self.opmrec_times = self.opmrec_times[0] + np.arange(0, nsamples) * 1/self.aux_sfreq

    
    if self.OPM_PREPROC['bandpass']:
        temp_dat = getattr(self, f"OPM_{axis}").copy()
        # Bandpass filter (1-100 Hz); use .filter() to remain Raw Mne Object
        temp_dat = temp_dat.filter(
            l_freq=self.preproc_config['BANDPASS_LOW'],
            h_freq=self.preproc_config['BANDPASS_HIGH'], 
            method='fir', verbose=False,
        )  # sfreq=meg_dat.info['sfreq'], is given within Raw Object
        setattr(self, f"OPM_{axis}", temp_dat)

    if self.OPM_PREPROC['notch']:
        # Apply notch filters (50 Hz and harmonics)
        sfreq = meg_info['sfreq']
        temp_dat = getattr(self, f"OPM_{axis}").get_data()
        for freq in self.preproc_config['NOTCH_FREQS']:
            temp_dat = notch_filter(
                temp_dat, 
                Fs=sfreq,  
                freqs=freq,
                verbose=False
            )
        getattr(self, f"OPM_{axis}")._data = temp_dat

    if self.OPM_PREPROC['hfc']:
        # Apply homogeneous field correction
        proj_hfc = compute_proj_hfc(meg_info,
                                    order=self.preproc_config['HFC_ORDER'])
        # temp_dat = meg_dat.copy()
        getattr(self, f"OPM_{axis}").add_proj(proj_hfc)
        getattr(self, f"OPM_{axis}").apply_proj()




def get_mne_event_array(self,):

    event_codes = {key: i+1 for i, key in enumerate(self.opm_task_epochs)}

    event_lists = []  # to creat array afterwards [start_index, 0, event_code]
    for e_key, e_idx in self.opm_task_epochs.items():
        for e_i in e_idx:
            event_lists.append([e_i[0], 0, event_codes[e_key]])

    event_arr = np.array(event_lists)

    return event_codes, event_arr


def add_task_epoch_idx(self, times_to_use, sfreq, INCL_EDGE = 0):
    """
    takes one second before start and one second after end
    INCL_EDGE = 1  # second extra prior and post

    """


    task_epochs = {}

    # take extracted timings from behavioral task, for rest
    # take every three-second window as an epoch
    if self.task == 'rest':
        task_dict = {'rest': {'start': [], 'end': []}}
        # take every 3 sec window
        for sec in np.arange(np.round(times_to_use[0]) + .5 + INCL_EDGE,  # add 1 for edge, half for rounding up
                             np.round(times_to_use[-1] - INCL_EDGE),
                             1+ 2):  # create distance between imaginary rest epochs
            task_dict['rest']['start'].append(sec)
            task_dict['rest']['end'].append(sec + 1)
    else:
        task_dict = self.tasktimings

    for task, timings in task_dict.items():

        task_epochs[task] = []

        for t0, t1 in zip(timings['start'], timings['end']):

            i0 = np.argmin(abs(times_to_use - (t0 - INCL_EDGE)))
            i1 = np.argmin(abs(times_to_use - (t1 + INCL_EDGE)))

            if i0 == i1: continue  # happened when task was incorrectly started before real start
            if (i1 - i0) < (.8 + INCL_EDGE + INCL_EDGE) * sfreq:
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


def take_mean_emgArms(self,):
    tempRaw = self.EMG.copy()

    for side in ['left', 'right']:

        # apply function to selected channels
        sel_chs = [ch for ch in self.EMG.info['ch_names']
                if side in ch and ('delt' in ch or 'arm' in ch)]
        tempEMG = self.EMG.copy().pick(sel_chs)
        dat = np.mean(tempEMG.get_data(), axis=0)
        tempEMG._data[0, :] = dat
        # rename the single channel
        n = sel_chs[0].split('_')
        tempEMG.rename_channels({sel_chs[0]: f'{n[0]}_{n[1]}_{side}armmean'})

        # drop old channels and add averaged one
        tempRaw.drop_channels(sel_chs)
        tempEMG.drop_channels(sel_chs[1])
        tempRaw.add_channels([tempEMG])

    setattr(self, 'EMG', tempRaw)