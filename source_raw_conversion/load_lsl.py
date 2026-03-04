"""
Contains functions to load source data stored with LSL,
typically containing AntNeuro acc + emg, AntNeuro triggers,
and behavioral markers from the PyGame Stream
"""

import os
import pyxdf
import numpy as np
import json
import datetime as dt
from itertools import compress
from mne import (
    create_info,
    Annotations,
    Epochs,
    events_from_annotations, 
    pick_types
)
from mne.io import RawArray

from signal_processing.preproc_functions import (
    apply_acc_preprocessing_in_Raw,
    apply_emg_preprocessing_in_Raw
)
from utils.load_utils import (
    get_onedrive_path,
    load_subject_config,
    get_sub_rec_metainfo
)
import source_raw_conversion.time_syncing as sync
from source_raw_conversion.load_fieldline_source_opm import get_fieldline_in_mne
from plotting.sync_checking import plot_check_trigger_distances_AN_FL


def load_AN_to_mne(
    SUB, SES, TASK, ACQ, HEALTHY=False,
    INCL_ANNOTATIONS=False, CROP_AROUND_TRIGGERS=False, CROP_MARGIN_SEC=10,
    RETURN_AS_EPOCHS=False, EVENT_T_PRE=-1.0, EVENT_T_POST=2.0,
    APPLY_ACC_PREPROCESSING: bool = False, AUX_RESAMPLE_SFREQ: int = 512,
    Z_SCORE_ACC: bool = True,
    APPLY_EMG_PREPROCESSING: bool = False,
    EMG_SIGNAL_TYPE: str = 'hilbert_env', Z_SCORE_EMG: bool = False,
):
    """
    Loads LSL data into mne, with option to resample and crop.
    
    resampling is done if target_sfreq is set
    CROP_RETURN_TRIGGERS defaults False, if True start and ending of recording
    iscropped based on present triggers.
    CROP_MARGIN is taken around triggers if > 0.

    if cropping is true, trigger_times/types are returned bcs of change in
    time-axis due to cropping

    EMG_SIGNAL_TYPE: 'hilbert_env' or 'tkeo', defines which emg preprocessing step is applied if
    """
    # load config and meta info
    sub_config = load_subject_config(subject_id=SUB,)

    if not HEALTHY and not SUB.startswith('9'):
        sub_meta_info = get_sub_rec_metainfo(config_sub=sub_config)
    else:
        sub_meta_info = None


    # load lsl .xdf-data and define streams
    lsl_data_dict = get_source_streams(
        SUB=SUB,
        SES=SES,
        ACQ=ACQ,
        TASK=TASK,
        RETURN_STEAMS_HEADER=False
    )  # return 'data', 'markers', 'pygame' streams in dict

    aux_sfreq = int(float(lsl_data_dict['data']['info']['nominal_srate'][0]))
    an_channels_dict = sync.get_lsl_channel_dict(lsl_data_dict['data'])
    ch_aux_sel = [chdict['type'][0] == 'aux' for chdict in an_channels_dict]
    assert sum (ch_aux_sel) == len(sub_config["antneuro_chs"]), (
        "AntNeuro-data contains different number of AUX channels vs "
        "given AntNeuro channels in subject CONFIG-json: "
        f"{sum(ch_aux_sel)} vs {len(sub_config['antneuro_chs'])}"
    )

    # select only aux data, and convert to numpy array
    aux_data = lsl_data_dict['data']['time_series'][:, ch_aux_sel]
    # aux_times = lsl_data_dict['data']['time_stamps'] - lsl_data_dict['data']['time_stamps'][0]
    aux_chnames = list(sub_config["antneuro_chs"].values())
    aux_chtypes = ['emg' if 'emg' in chname.lower() else 'misc' for chname in aux_chnames]

    aux_info = create_info(
        ch_names=aux_chnames,
        sfreq=aux_sfreq,
        ch_types=aux_chtypes
    )
    aux_raw = RawArray(aux_data.T, aux_info)
    if AUX_RESAMPLE_SFREQ and AUX_RESAMPLE_SFREQ != aux_sfreq:
        aux_raw.resample(sfreq=AUX_RESAMPLE_SFREQ,)


    if INCL_ANNOTATIONS:
        # get AN trial start times, zero-center at first trial start
        an_trigger_times = sync.get_antneuro_arduino_times(lsldat=lsl_data_dict['data'])
        # an_trigger_times = an_trigger_times - an_trigger_times[0]
        an_trigger_markers = sync.get_an_trigger_markers(lsl_data_dict['pygame'])

        annotations_task = Annotations(
            onset=an_trigger_times,
            duration=[0.25] * len(an_trigger_times),
            description=an_trigger_markers,
        )
        # annotations_event = rawHFC.annotations 
        aux_raw.set_annotations(annotations_task)

    if CROP_AROUND_TRIGGERS:
        an_trigger_times = sync.get_antneuro_arduino_times(lsldat=lsl_data_dict['data'])
        
        aux_raw = aux_raw.crop(
            tmin=an_trigger_times[0],
            tmax=an_trigger_times[-1] + CROP_MARGIN_SEC
        )
        # # adjust triggertimes, zeroed to start first trigger accodingly
        # an_trigger_times = np.array(an_trigger_times) - an_trigger_times[0]

    if APPLY_ACC_PREPROCESSING:
        apply_acc_preprocessing_in_Raw(aux_raw, zscore=Z_SCORE_ACC,)
    
    if APPLY_EMG_PREPROCESSING:
        apply_emg_preprocessing_in_Raw(
            aux_raw, emg_signal_type=EMG_SIGNAL_TYPE, zscore=Z_SCORE_EMG,
        )

    if RETURN_AS_EPOCHS:
        
        aux_events, aux_event_id = events_from_annotations(aux_raw)
        aux_raw = Epochs(aux_raw, aux_events, aux_event_id,
                         tmin=EVENT_T_PRE, tmax=EVENT_T_POST,
                         baseline=None, preload=True)

    return aux_raw


    



def get_source_streams(
    SUB, SES, ACQ, TASK,
    RETURN_STEAMS_HEADER: bool = False,
    RETURN_ONLY_AN: bool = False,
    RETURN_ONLY_PYGAME: bool = False,
):
    """
    Input:
    - SUB
    - SES
    - ACQ
    - TASK

    Returns:
    - streams
    - fileheader
    """

    lsl_source_path = os.path.join(
        get_onedrive_path('source_data'),
        f'sub-{SUB}',
        f'ses-{SES}',
        'lsl'
    )
    # gets folder with defined task and acquisition
    try:
        sel_folder = [f for f in os.listdir(lsl_source_path)
                      if ACQ.lower() in f.lower() and TASK.lower() in f.lower()][0]
    except IndexError:
        raise ValueError(f'IndexError catched, no folder for sub-{SUB, ACQ, TASK}')
    
    if sel_folder.endswith('xdf'):
        sel_path = os.path.join(lsl_source_path, sel_folder)
    
    elif 'eeg' in os.listdir(os.path.join(lsl_source_path, sel_folder)):
    
        sel_path = os.path.join(lsl_source_path, sel_folder, 'eeg')  # convert to path
        sel_fname = [f for f in os.listdir(sel_path)][0]
        sel_path = os.path.join(sel_path, sel_fname)

    # load defined LSL file
    streams, fileheader = pyxdf.load_xdf(sel_path)

    if RETURN_STEAMS_HEADER:
        return streams, fileheader
    else:
        lsldat, lslmrk, lslpyg = define_streams(streams)

    if RETURN_ONLY_AN:
        return lsldat
    elif RETURN_ONLY_PYGAME:
        return lslpyg
    else:
        return {'data': lsldat, 'pygame': lslpyg}


def define_streams(streams):
    """
    Input:
    - streams

    Returns:
    - AN_DATA stream
    - AN_MRK stream
    - Pygame stream (if found), otherwise None
    """

    AN_DATA, AN_MRK, PYGAME = None, None, None

    print(f"### Found {len(streams)} streams")

    for i, stream in enumerate(streams):
        # print(f"\n--- Stream {i} ---")
        # print("Name:", stream['info']['name'][0])
        # print("Type:", stream['info']['type'][0])
        # print("Channel count:", stream['info']['channel_count'][0])
        
        # select streams
        if stream['info']['type'][0] == 'Markers':
            if 'TRG' in stream['info']['name'][0]:
                AN_MRK = stream
                print(f'\tinclude as AntNeuro Marker:', stream['info']['name'][0])
            
            elif 'gonogo' in stream['info']['name'][0].lower():
                PYGAME = stream
                print(f'\tinclude as PyGame:', stream['info']['name'][0])
    
            else:
                print('unknown MARKERS stream', stream['info']['name'][0])

        elif stream['info']['type'][0] == 'EEG':
            AN_DATA = stream
            print(f'\tinclude as AntNeuro DATA:', stream['info']['name'][0])
    
        else:
            print('unknown stream', stream['info']['type'][0], stream['info']['name'][0])
    
    if not PYGAME: print('\tno pygame stream found')
    
    return AN_DATA, AN_MRK, PYGAME


def convert_source_lsl_to_raw(SUB, TASK, ACQ, SES, source_path, HEALTHY=False,
                              REC_LOC=False,):

    # load config and meta info
    sub_config = load_subject_config(subject_id=SUB,)
    if not HEALTHY:
        sub_meta_info = get_sub_rec_metainfo(config_sub=sub_config)
    else:
        sub_meta_info = None

    # load and define source lsl streams
    streams, fileheader = get_source_streams(SUB, ACQ, TASK, source_path)
    lsldat, lslmrk, lslpyg = define_streams(streams)

    # compare timing opm and lsl
    if REC_LOC == 'PTB':
        (meg_trigger_diffs,
         meg_time_trigger0) = sync.get_meg_trigger_diffs(SUB=SUB, ACQ=ACQ, TASK=TASK,
                                                        RETURN_MEGTIME_TRIGGER0=True)
        meg_trigger_times = [meg_time_trigger0 + tdiff for tdiff in meg_trigger_diffs]
        (sync_diffs,
         lsl_t_trigger0,
         lsl_clock_t0) = sync.compare_triggers(
            mrk_stream=lslmrk, lsl_header=fileheader,
            meg_trigger_diffs=meg_trigger_diffs
        )
        meg_trigger_types = None

    elif REC_LOC == 'CCM':
        # get trigger timings and behavioral-types from fieldline data
        raw = get_fieldline_in_mne(SUB=SUB, SES=SES, ACQ=ACQ, TASK=TASK)
        (meg_trigger_times, meg_trigger_types) = sync.find_arduino_triggers(raw_mne_opm=raw)
        meg_time_trigger0 = meg_trigger_times[0]
        # get trigger timings in antneuro/lsl data
        AN_trig_times = sync.get_antneuro_arduino_times(lsldat)  # gives relativ time since lsl recording start
        lsl_t_trigger0 = AN_trig_times[0] + lsldat['time_stamps'][0]
        # get start time of lsl recording
        if "datetime" in fileheader['info']:
            lsl_clock_t0 = fileheader['info']['datetime'][0]
            lsl_clock_t0 = dt.datetime.strptime(lsl_clock_t0, "%Y-%m-%dT%H:%M:%S%z")
            print(f'found lsl starttime: {lsl_clock_t0}')

        # # plot comparison of trigger intervals
        # plot_check_trigger_distances_AN_FL(
        #     SUB, SES, ACQ, TASK, AN_trig_times=AN_trig_times,
        #     FL_trigger_times=meg_trigger_times,
        # )


        

    ### LSL AntNeuro Data (CUT internally on second and last trigger)
    print(f'\n\tLSL TRIGGER 0: {lsl_t_trigger0}')
    (auxdat, aux_chnames, aux_sfreq) = convert_antneuro_stream(
        lsldat, lsl_t_trigger0, meg_trigger_times,
        SUB, ACQ, TASK, sub_meta_info, sub_config,
        lsl_clock_t0=lsl_clock_t0,
        HEALTHY=HEALTHY,
    )


    ### LSL Pygame Data
    if TASK != 'rest':
        pyg_timings = extract_game_markers(
            pyg_stream=lslpyg,
            lsl_t_trigger0=lsldat['time_stamps'][0],
            meg_time_trigger0=meg_time_trigger0,
            SUB=SUB, ACQ=ACQ, TASK=TASK,
        )
    else:
        pyg_timings = None

    # print(auxdat)

    return auxdat, aux_chnames, aux_sfreq, pyg_timings



def convert_antneuro_stream(lsldat, lsl_t_trigger0, meg_trigger_times,
                            SUB, ACQ, TASK, sub_meta_info, sub_config,
                            lsl_clock_t0, HEALTHY: bool = False,):
    
    # try to load ant-neur values
    fpath = os.path.join(get_onedrive_path('raw_data'), f'sub-{SUB}', 'emgacc')
    if not os.path.exists(fpath): os.makedirs(fpath)
    fname = f'rawAligned_sub-{SUB}_{TASK}_{ACQ}.npy'
    jsonname = f'chInfo_sub-{SUB}_{TASK}_{ACQ}.json'
    if fname in os.listdir(fpath):
        auxdat = np.load(os.path.join(fpath, fname))
    
        if jsonname in os.listdir(fpath):
            with open(os.path.join(fpath, jsonname), 'r') as f:
                auxinfo = json.load(f)
            
            chnames = auxinfo['chnames']
            sfreq = auxinfo['sfreq']
            print('LOADED AUXDAT:', fname)

        else:
            raise ValueError(f'no auxInfo for {SUB}_{TASK}_{ACQ}')
    
        return auxdat, chnames, sfreq
    
    meg_t_trigger0 = meg_trigger_times[0]

    # convert lsl times into meg-time alligning
    an_times_inmeg, clocktime_megt0 = sync.convert_lsltimes_to_megtimes_sec(
        lsltimestamps=lsldat['time_stamps'],
        lsl_t_trigger0=lsl_t_trigger0,
        meg_time_trigger0=meg_t_trigger0,
        lsl_clock_t0=lsl_clock_t0
    )

    # get data from lsl data stream
    aux_sfreq = int(float(lsldat['info']['nominal_srate'][0]))
    an_channeldicts_list = lsldat['info']['desc'][0]['channels'][0]['channel']
    ch_aux_sel = [chdict['type'][0] == 'aux' for chdict in an_channeldicts_list]
    aux_chnames = list(sub_config["antneuro_chs"].values())

    assert sum (ch_aux_sel) == len(sub_config["antneuro_chs"]), (
        "AntNeuro contains different number of AUX channels "
        "then given AntNeuro channels in CONFIG"
    )

    auxdat = np.array(lsldat['time_series'][:, ch_aux_sel])

    if not HEALTHY:
        auxdat, auxtimes = sync.cut_data_to_task_timing(
            tempdata=auxdat,
            temptimes=an_times_inmeg,
            sub_meta=sub_meta_info,
            TASK=TASK,
            ACQ=ACQ,
            SFREQ=aux_sfreq,
        )
    else:
        auxtimes = an_times_inmeg

    print('auxtimes', auxtimes, 'HEALTHY', HEALTHY)

    # merge aligned time and data
    auxdat = np.concatenate([auxtimes[:, np.newaxis], auxdat], axis=1)
    aux_chnames = ['aligned_time',] + aux_chnames

    ### cut on triggers (between second and last, TODO: replace with START and END triggers)
    print('TODO: ANTNEURO CUTTING/CROPPING')
    # idx_start = np.where(auxtimes == meg_trigger_times[1])[0][0]
    # idx_end = np.where(auxtimes == meg_trigger_times[-1])[0][0]
    # print(f'CUT LSL DATA from {meg_trigger_times[1]} to {meg_trigger_times[-1]}'
    #       f' on indices: {idx_start, idx_end}, orig shape: {auxdat.shape}')
    # auxdat = auxdat[idx_start:idx_end, :]

    ## storing
    auxdat = np.save(os.path.join(fpath, fname), auxdat)
    auxinfo = {'chnames': aux_chnames, 'sfreq': aux_sfreq, 'clocktime_t0': str(clocktime_megt0)}
    with open(os.path.join(fpath, jsonname), 'w') as f:
        json.dump(auxinfo, f)
        

    return auxdat, aux_chnames, aux_sfreq


def extract_game_markers(SUB, ACQ, TASK, pyg_stream=None,
                         lsl_t_trigger0=None, meg_time_trigger0=None,):
    """
    Input:
    - pyg_data: LSL stream with (n_samples)
    - pyg_times: (n_samples), expressed in opm-meg alligned time

    Returns:
    - 
    """
    ### load data if available
    fpath = os.path.join(get_onedrive_path('raw_data'), f'sub-{SUB}')
    fname = f'trialtimings_sub-{SUB}_{TASK}_{ACQ}.json'
    
    if fname in os.listdir(fpath):
        with open(os.path.join(fpath, fname), 'r') as f:
            game_timings = json.load(f)
        print(f'\nPygame timings loaded from rawdata: {fname} in {fpath}')

        return game_timings

    ### create dict w timings if not existing yet
    
    # convert lsl times to opm times
    pyg_times, _ = sync.convert_lsltimes_to_megtimes_sec(
        lsltimestamps=pyg_stream['time_stamps'],
        lsl_t_trigger0=lsl_t_trigger0,
        meg_time_trigger0=meg_time_trigger0,
    )
    
    
    game_timings = {
        "go_left": {"start": [], "end": [], "trial_n": []},
        "go_right": {"start": [], "end": [], "trial_n": []},
        "nogo_left": {"start": [], "end": [], "trial_n": []},
        "nogo_right": {"start": [], "end": [], "trial_n": []},
        "abort_left": {"start": [], "abort": [], "end": [], "trial_n": []},
        "abort_right": {"start": [], "abort": [], "end": [], "trial_n": []}
    }

    ACTIVE_TYPE = None  # default start

    for m, t in zip(pyg_stream['time_series'], pyg_times):
        
        m = m[0]

        # take trial start
        if m.startswith('TRIAL_START'):
            _, _, n_trial, trialtype = m.split('_')
            ACTIVE_TYPE = trialtype

        # take moment of task-arrow appearing on screen
        elif m.startswith('STIM_ONSET'):
            if ACTIVE_TYPE in ['go', 'nogo']:
                _, _, trialtype, side = m.split('_')
                game_timings[f'{trialtype}_{side}']['start'].append(t)
                game_timings[f'{trialtype}_{side}']['trial_n'].append(n_trial)
            
            elif ACTIVE_TYPE == 'abort':
                _, _, trialtype, aborttype, side = m.split('_')
                if aborttype == 'go':
                    game_timings[f'{trialtype}_{side}']['start'].append(t)
                    game_timings[f'{trialtype}_{side}']['trial_n'].append(n_trial)
                elif aborttype == 'nogo':
                    game_timings[f'{trialtype}_{side}']['abort'].append(t)
        
        # take trial start
        elif m.startswith('TRIAL_END'):
            _, _, n_trial = m.split('_')
            try:
                game_timings[f'{trialtype}_{side}']['end'].append(t)
            except KeyError:
                print(f'\n##### ERROR in game markers, type: {trialtype} or side: {side} not defined')
                print(f'happened for marker: {m}; skipped for now')
                continue
            # reset variables
            ACTIVE_TYPE, side, trialtype = None, None, None
    
    # store created dict
    with open(os.path.join(fpath, fname), 'w') as f:
        json.dump(game_timings, f)

    print(f'\nPygame timings stored to rawdata: {fname}')


    return game_timings
                