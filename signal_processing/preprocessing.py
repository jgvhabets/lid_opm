"""
Signal pre-processing functions for OPM and MEG, ACC
"""

import os
import json
import numpy as np
from dataclasses import dataclass, field
from itertools import product, compress
from scipy.signal import butter, filtfilt, iirnotch
import datetime as dt 

from mne.filter import filter_data, notch_filter, resample
from mne.preprocessing import ICA, compute_proj_hfc
from mne import create_info
from mne.io import RawArray

# import custom
import source_raw_conversion.load_PTB_source_opm as PTB_source_opm
from source_raw_conversion.load_lsl import convert_source_lsl_to_raw
from utils import load_utils
from signal_processing import preproc_functions as prepr_funcs
import signal_processing.epoching as epoching
from source_raw_conversion.load_fieldline_source_opm import get_fieldline_in_mne

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
    ses: str = '01'
    HEALTHY_CONTROL: bool = False
    INCL_AUX: bool = True
    INCL_OPM: bool = False
    OPM_AXES_INCL: list = field(default_factory=lambda: ['Z',])
    OPM_PREPROC: dict = field(default_factory=lambda: {
        'resample': True, 'bandpass': False, 'notch': False
    })
    ZSCORE_ACC: bool = False
    ZSCORE_EMG: bool = False
    COMBINE_ARM_EMG: bool = False
    CONFIG_VERSION: str = 'v1'

    def __post_init__(self,):
        # load config for sub and general
        self.sub_config = load_utils.load_subject_config(subject_id=self.sub,)
        self.preproc_config = load_utils.load_preproc_config(version=self.CONFIG_VERSION,)

        if not self.HEALTHY_CONTROL:
            self.source_path = os.path.join(
                load_utils.get_onedrive_path('source_data'),
                f'sub-{self.sub}'
            )
            self.REC_LOC = self.sub_config['rec_location']
        
        else:
            self.source_path = os.path.join(
                load_utils.get_onedrive_path('source_data'),
                f'sub-{self.sub}',
                f'sub-{self.ses}'
            )
            self.REC_LOC = self.sub_config['rec_location'][f'ses-{self.ses}']
        
        

        if self.INCL_AUX:
            ### load aux, w/o making aux_dat a class attribute to prevent storing of double data
            (
                temp_auxdat, self.aux_chnames,
                self.aux_sfreq, self.tasktimings
            ) = convert_source_lsl_to_raw(self.sub, self.task, self.acq,
                                          source_path=self.source_path)
            self.auxtimes = temp_auxdat[:, 0]

            # add means and stddevs to zscore
            if self.ZSCORE_ACC or self.ZSCORE_EMG:
                self.aux_zscore_values = get_aux_zscore_values(self=self)
            
            # # add indices from task stimulations, only start indices, corresponding to auxtimes
            # self.aux_task_epochs = add_task_epoch_idx(
            #     self=self, times_to_use=self.auxtimes, sfreq=self.aux_sfreq
            # )

            # RESAMPLE aux (both acc and emg)
            resample_factor = self.aux_sfreq / self.preproc_config["TARGET_SFREQ"]
            if resample_factor != 1:
                self.aux_sfreq = self.preproc_config["TARGET_SFREQ"]
                (temp_auxdat,
                 self.aux_chnames,
                 self.auxtimes) = prepr_funcs.resample_aux_array(
                    temp_auxdat=temp_auxdat,
                    aux_chnames=self.aux_chnames,
                    auxtimes=self.auxtimes,
                    aux_sfreq=self.aux_sfreq,
                    FACTOR=resample_factor,
                )


            ### PREPROCESS ACC

            # BANDPASS all acc channels
            for i_ch in np.where(['acc' in c for c in self.aux_chnames])[0]:
                sig = temp_auxdat[:, i_ch].copy()
                sig = prepr_funcs.apply_filter(sig, sfreq=self.aux_sfreq, type='band',
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
                acc_data.append(prepr_funcs.get_signal_vector_magn(triacc))
                acc_chnames.append(f'acc_svm_{side}{extr}')
            acc_data = np.array(acc_data)
            
            # STANDARDISE all acc channels
            if self.ZSCORE_ACC:
                for i, ch in enumerate(acc_chnames):
                    m = self.aux_zscore_values[ch]['mean']
                    sd = self.aux_zscore_values[ch]['sd']
                    acc_data[i, :] = (acc_data[i, :] - m) / sd
            
            # add ACC signals as rawmne array, MNE-Object-Times cannot be customized, therefore object-times start from zero
            # use for all AUX self.axutimes for the synchronized MEG time
            acc_info = create_info(ch_names=acc_chnames, sfreq=self.aux_sfreq, ch_types='misc')
            setattr(self, 'ACC', RawArray(acc_data, acc_info))


            ### PREPROCESS EMG

            # process EMG signals
            emg_chnames, emg_data = [], []
            for i_ch in np.where(['emg' in c for c in self.aux_chnames])[0]:
                name = self.aux_chnames[i_ch]
                # get envelop
                emg_chnames.append(name.replace('emg', 'emg_env'))
                emg_data.append(prepr_funcs.get_emg_envelop(temp_auxdat[:, i_ch], sfreq=self.aux_sfreq))
                
                # get TKEO
                # emg_chnames.append(name.replace('emg', 'emg_tkeo'))
                # emg_data.append(prepr_funcs.get_emg_tkeo(temp_auxdat[:, i_ch], sfreq=self.aux_sfreq,))
            
            emg_data = np.array(emg_data)
            
            # standardise all emg and acc channels
            if self.ZSCORE_EMG:
                for i, ch in enumerate(emg_chnames):
                    m = self.aux_zscore_values[ch]['mean']
                    sd = self.aux_zscore_values[ch]['sd']
                    emg_data[i, :] = (emg_data[i, :] - m) / sd
            
            # add EMG signals as rawmne array
            emg_info = create_info(ch_names=emg_chnames, sfreq=self.aux_sfreq, ch_types='emg')
            setattr(self, 'EMG', RawArray(emg_data, emg_info))
            
            if self.COMBINE_ARM_EMG:
                # average arm EMG's
                take_mean_emgArms(self,)  # replaces RawArray with mean inst of 2 sep arm emgs
        

            ### ADD EPOCHING AFTER SIGNAL PREPROCESSING
            # add indices from task stimulations, including 1-sec prior and 1-sec post!!
            self.aux_task_epochs = epoching.add_task_epoch_idx(
                self=self, times_to_use=self.auxtimes, sfreq=self.aux_sfreq
            )
            # create array to mark event epochs
            self.aux_event_codes, self.aux_event_arr = epoching.get_mne_event_array(self, dType='AUX')


        #####
        if self.INCL_OPM:
            
            # load data per opm axis
            for ax in self.OPM_AXES_INCL:

                if self.REC_LOC == 'PTB':
                    axdata, axtimes = PTB_source_opm.select_and_store_axis_data(
                        AX=ax, ACQ=self.acq, TASK=self.task,
                        sub_config=self.sub_config, LOAD=True,
                    )
                    rawmne = PTB_source_opm.load_raw_opm_into_mne(
                        meg_data=axdata, AX=ax, sub_config=self.sub_config,
                    )

                elif self.REC_LOC == 'CCM':
                    # TODO FIX
                    rawmne = get_fieldline_in_mne(self.sub, self.ses, self.acq)

                # currently one time axis for all opm axes
                self.opmrec_times = axtimes
                setattr(self, f'OPM_{ax}', rawmne)

                # preprocess opm axis
                preprocess_opm(self, axis=ax,)

                self.opm_task_epochs = epoching.add_task_epoch_idx(
                    self=self,
                    times_to_use=self.opmrec_times,
                    sfreq=getattr(self, f'OPM_{ax}').info['sfreq']
                )
                # create array to mark event epochs
                (
                    self.opm_event_codes,
                    self.opm_event_arr
                ) = epoching.get_mne_event_array(self, dType='OPM')




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
            print(f'original OPM: sfreq {meg_info.info["sfreq"]} vs sfreq {goal_fs}')
        else:
            print(f'resample original OPM-sampling rate {meg_info["sfreq"]} to {goal_fs}')
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




def get_aux_zscore_values(self,):
    fname = f'sub{self.sub}_zscore_aux_values.json'
    path = os.path.join(load_utils.get_onedrive_path('raw_data'),
                        f'sub-{self.sub}', 'emgacc')
    
    if fname in os.listdir(path):

        with open(os.path.join(path, fname), 'r') as f:
            tempdict = json.load(f)
    
    else:

        tempdict = get_aux_zscore_variables(SUB=self.SUB, RETURN=True,)
        
    return tempdict


def get_aux_zscore_variabels(SUB, RETURN=True,):
    """
    calculate and stores means and stddevs for one subject
    based on all recordings, in order to zscore data while
    importing based on the full recordings

    needs to be checked whether execution within rawData_singleRec()
    works (doesnt give circular import or such errors)
    """

    sub_config = load_utils.load_subject_config(subject_id=SUB,)
    sub_meta_info = load_utils.get_sub_rec_metainfo(config_sub=sub_config)

    temp_arrs = None

    for _, REC in enumerate(sub_meta_info['rec_name']):
        print(f'\t(sub-{SUB}, load{REC}')

        try:
            TASK, ACQ = REC.split('_')
        except ValueError:
            print(f'skipped {REC}')
            continue

        tempRaw = rawData_singleRec(
            SUB, TASK, ACQ, INCL_AUX=True, INCL_OPM=False,
            ZSCORE_ACC=False, ZSCORE_EMG=False,
        )

        if not temp_arrs:
            temp_arrs = {}
            for ch in tempRaw.ACC.ch_names: temp_arrs[ch] = []
            for ch in tempRaw.EMG.ch_names: temp_arrs[ch] = []
        
        for src in ['ACC', 'EMG']:
            temp_dat = getattr(tempRaw, src).copy()
            for ch in temp_dat.ch_names:
                temp_arrs[ch].extend(temp_dat.copy().pick(ch).get_data().ravel())

    store_dict = {}
    for ch, values in temp_arrs.items():
        store_dict[ch] = {'mean': np.nanmean(values), 'sd': np.nanstd(values)}

    fname = f'sub{SUB}_zscore_aux_values.json'
    fpath = os.path.join(
        load_utils.get_onedrive_path('raw_data'),
        f'sub-{SUB}', 'emgacc', fname
    )

    with open(fpath, 'w') as f:
        json.dump(store_dict, f)

    print(f'\tAUX-zscores ({fname}) succesfully stored as ')

    if RETURN: return store_dict

