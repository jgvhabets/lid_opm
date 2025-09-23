"""
Contains functions to load source data stored with LSL,
typically containing AntNeuro acc + emg, AntNeuro triggers,
and behavioral markers from the PyGame Stream
"""

import os
import pyxdf
import numpy as np
import json

from utils.load_utils import (
    get_onedrive_path,
    load_subject_config,
    get_sub_rec_metainfo
)
import source_raw_conversion.time_syncing as sync

def get_source_streams(SUB, ACQ, TASK):
    """
    Input:
    - SUB
    - ACQ
    - TASK

    Returns:
    - streams
    - fileheader
    """

    lsl_source_path = os.path.join(get_onedrive_path('source_data'),
                                   f'sub-{SUB}', 'lsl')
    # gets folder with defined task and acquisition
    try:
        sel_folder = [f for f in os.listdir(lsl_source_path)
                      if ACQ in f.lower() and TASK in f.lower()][0]
    except IndexError:
        raise ValueError(f'IndexError catched, no folder for sub-{SUB, ACQ, TASK}')
    
    sel_path = os.path.join(lsl_source_path, sel_folder, 'eeg')  # convert to path
    sel_fname = sel_f = [f for f in os.listdir(sel_path)][0]

    # load defined LSL file
    streams, fileheader = pyxdf.load_xdf(os.path.join(sel_path, sel_fname, ))

    return streams, fileheader


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


def convert_source_lsl_to_raw(SUB, TASK, ACQ):

    # load config and meta info
    sub_config = load_subject_config(subject_id=SUB,)
    sub_meta_info = get_sub_rec_metainfo(config_sub=sub_config)

    # load and define source lsl streams
    streams, fileheader = get_source_streams(SUB, ACQ, TASK)
    lsldat, lslmrk, lslpyg = define_streams(streams)

    # compare timing opm and lsl
    (meg_trigger_diffs,
     meg_time_trigger0) = sync.get_meg_trigger_diffs(SUB=SUB, ACQ=ACQ, TASK=TASK,
                                                RETURN_MEGTIME_TRIGGER0=True)
    (sync_diffs,
     lsl_t_trigger0,
     lsl_clock_t0) = sync.compare_triggers(
        mrk_stream=lslmrk, lsl_header=fileheader,
        meg_trigger_diffs=meg_trigger_diffs
    )
    # TODO: create l-dopa time based on lsl_clock_t0


    ### LSL AntNeuro Data
    auxdat, auxtimes, aux_chnames, aux_sfreq = convert_antneuro_stream(
        lsldat, lsl_t_trigger0, meg_time_trigger0,
        ACQ, TASK, sub_meta_info, sub_config
    )
    # TODO: storing of trimmed, alligned raw data


    ### LSL Pygame Data
    # convert times
    pyg_times = sync.convert_lsltimes_to_megtimes_sec(
        lsltimestamps=lslpyg['time_stamps'],
        lsl_t_trigger0=lsl_t_trigger0,
        meg_time_trigger0=meg_time_trigger0,
    )
    pyg_timings = extract_game_markers(
        pyg_data=lslpyg['time_series'],
        pyg_times=pyg_times,
        SUB=SUB, ACQ=ACQ, TASK=TASK,
    )


    return auxdat, auxtimes, aux_chnames, aux_sfreq, pyg_timings



def convert_antneuro_stream(lsldat, lsl_t_trigger0, meg_t_trigger0,
                            ACQ, TASK, sub_meta_info, sub_config,):
    # convert lsl times into meg-time alligning
    an_times_inmeg = sync.convert_lsltimes_to_megtimes_sec(
        lsltimestamps=lsldat['time_stamps'],
        lsl_t_trigger0=lsl_t_trigger0,
        meg_time_trigger0=meg_t_trigger0,
    )
    # get data from lsl data stream
    AN_SFREQ = int(float(lsldat['info']['nominal_srate'][0]))
    an_channeldicts_list = lsldat['info']['desc'][0]['channels'][0]['channel']
    ch_aux_sel = [chdict['type'][0] == 'aux' for chdict in an_channeldicts_list]
    aux_chnames = list(sub_config["antneuro_chs"].values())

    assert sum (ch_aux_sel) == len(sub_config["antneuro_chs"]), (
        "AntNeuro contains different number of AUX channels "
        "then given AntNeuro channels in CONFIG"
    )

    auxdat = np.array(lsldat['time_series'][:, ch_aux_sel])

    auxdat, auxtimes = sync.cut_data_to_task_timing(
        tempdata=auxdat,
        temptimes=an_times_inmeg,
        sub_meta=sub_meta_info,
        TASK=TASK,
        ACQ=ACQ,
        SFREQ=AN_SFREQ,
    )

    return auxdat, auxtimes, aux_chnames, AN_SFREQ


def extract_game_markers(pyg_data, pyg_times,
                         SUB, ACQ, TASK,):
    """
    Input:
    - pyg_data: LSL stream with (n_samples)
    - pyg_times: (n_samples), expressed in opm-meg alligned time

    Returns:
    - 
    """
    fpath = os.path.join(get_onedrive_path('raw_data'), f'sub-{SUB}')
    fname = f'trialtimings_sub-{SUB}_{TASK}_{ACQ}.json'
    
    if fname in os.listdir(fpath):
        with open(os.path.join(fpath, fname), 'r') as f:
            game_timings = json.load(f)
        print(f'\nPygame timings loaded from rawdata: {fname}')

        return game_timings

    # create dict w timings if not existing yet
    game_timings = {
        "go_left": {"start": [], "end": [], "trial_n": []},
        "go_right": {"start": [], "end": [], "trial_n": []},
        "nogo_left": {"start": [], "end": [], "trial_n": []},
        "nogo_right": {"start": [], "end": [], "trial_n": []},
        "abort_left": {"start": [], "abort": [], "end": [], "trial_n": []},
        "abort_right": {"start": [], "abort": [], "end": [], "trial_n": []}
    }

    ACTIVE_TYPE = None  # default start

    for m, t in zip(pyg_data, pyg_times):
        
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
            game_timings[f'{trialtype}_{side}']['end'].append(t)
            # reset variables
            ACTIVE_TYPE, side, trialtype = None, None, None
    
    # store created dict
    with open(os.path.join(fpath, fname), 'w') as f:
        json.dump(game_timings, f)

    print(f'\nPygame timings stored to rawdata: {fname}')


    return game_timings
                