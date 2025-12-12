"""
Functions to check sync of OPM vs AntNeuro data,
and to allign all timeseries to 1) opm-time, and
2) ldopa-time
"""

import datetime as dt
import os
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

from utils.load_utils import get_onedrive_path


TRIGGER_SCHEME = {
    'start_pulse': 0.1,
    'go': 0.05,
    'nogo': .15,
    'abort': .3
}  # TODO: import from json used for sending triggers during task


def find_arduino_triggers(raw_mne_opm, PLOT_CHECK: bool = False):
    """
    only works in fieldline data so far bcs antneuro doesnot capture
    the duration of the pulses, only the onset
    """
    # find trigger start and end times
    idx_trig = np.where([t == 'stim' for t in raw_mne_opm.get_channel_types()])[0][0]
    trigger = raw_mne_opm.get_data()[idx_trig]
    diff_trigger = np.diff(trigger)

    pos_peaks, pos_peaks_props = find_peaks(diff_trigger, height=1, distance=10,)
    neg_peaks, neg_peaks_props = find_peaks(-1 * diff_trigger, height=1, distance=10,)
    
    trig_times_starts = raw_mne_opm.times[pos_peaks]
    trig_times_ends = raw_mne_opm.times[neg_peaks]

    if PLOT_CHECK:
        plt.plot(raw_mne_opm.times, trigger)

        plt.scatter(trig_times_starts, [3.5] * len(pos_peaks), color='purple',)
        plt.scatter(trig_times_ends, [3.5] * len(neg_peaks), color='orange',)

        plt.show()


    # attribute triggers with task-types based on schedule

    trigger_times, trigger_types = [], []  # to store

    TRIG_ACTIVE = False  # start with, activate when first trigger duration (always .1) is found


    for t1, t2 in zip(trig_times_starts, trig_times_ends):
        # calculate duration between start and end
        dur = round(t2 - t1, 2)
        # if START-duration is found, and no trigger is ongoing, activate trigger and save start time
        if dur == TRIGGER_SCHEME['start_pulse'] and not TRIG_ACTIVE:
            TRIG_ACTIVE = True
            temp_t = t1
            continue
        # if trigger is activated
        if TRIG_ACTIVE:
            # compare which 2nd duration is matching
            for trig_key, trig_dur in list(TRIGGER_SCHEME.items()):
                # double check whether a trigger duration was already matched and trigger deactivated
                if TRIG_ACTIVE:
                    # if trigger gets matched: add time and type, and deactivate trigger-bool
                    if dur == trig_dur:
                        trigger_types.append(trig_key)
                        trigger_times.append(temp_t)
                        TRIG_ACTIVE = False
                        temp_t = None
    
    return trigger_times, trigger_types


def get_antneuro_arduino_times(lsldat):
    """
    find starting times for indices in antneuro data,
    recorded via lsl
    """

    lsl_rec_timestamps = lsldat['time_stamps'] - lsldat['time_stamps'][0]

    an_channeldicts_list = lsldat['info']['desc'][0]['channels'][0]['channel']

    AN_ch_trig_sel = [chdict['type'][0] == 'trigger' for chdict in an_channeldicts_list]
    AN_trig_dat = np.array(lsldat['time_series'][:, AN_ch_trig_sel])

    AN_trig_idx = np.where(AN_trig_dat > .5)[0][::2]
    AN_trig_times = lsl_rec_timestamps[AN_trig_idx]

    ### for potential internal check
    # plt.plot(lsl_rec_timestamps, AN_trig_dat)

    # plt.scatter(AN_trig_times, [1] * len(AN_trig_idx))

    # plt.show()

    return AN_trig_times


def get_meg_trigger_diffs(SUB, TASK, ACQ, RETURN_MEGTIME_TRIGGER0=True,):
    # load meg trigger times (in meg time) from raw data
    fpath = os.path.join(get_onedrive_path('raw_data'),
                        f'sub-{SUB}', 'opm',
                        f'sub-{SUB}_{TASK}_{ACQ}_opm_triggertimes.npy')
    meg_triggers = np.load(fpath)
    meg_trigger_diffs = [
        dt.timedelta(seconds= t - meg_triggers[0]) for t in meg_triggers
    ]

    if RETURN_MEGTIME_TRIGGER0:
        # get meg time of first trigger
        meg_t_trigger0 = meg_triggers[0]

        return meg_trigger_diffs, meg_t_trigger0
    
    else:

        return meg_triggers


def get_AN_trigger_diffs(mrk_times, fileheader):
    """
    create time differences compared to first marker signal
    # todo:
    ## include dopatime based on MED intake (via clock time)
    

    Returns:
    - differences to first-marker, for all markers in file
    - lsl_t_trigger0: original LSL marker time for first trigger
    - clock_start: relevant for ldopa time
    """

    if mrk_times[0] > 1e9:  # looks like absolute Unix time (seconds since 1970)
        # mrk_times = [dt.datetime.fromtimestamp(t) for t in mrk_times]
        clock_tstart = None

    else:  # relative to boot, need offset
        # File header sometimes contains "clock_offset" or "first_timestamp"
        if "datetime" in fileheader['info']:
            clock_tstart = fileheader['info']['datetime'][0]
            clock_tstart = dt.datetime.strptime(clock_tstart, "%Y-%m-%dT%H:%M:%S%z")
            print(f'found lsl starttime: {clock_tstart}')
        
        else:
            clock_tstart = None

    # convert times and define diffs
    rel_times = [dt.datetime.fromtimestamp(t - mrk_times[0])
                 for t in mrk_times]  # gets times relative to first marker
    lsl_t_trigger0 = mrk_times[0]
    timediffs = [t - rel_times[0] for t in rel_times]  # converts rel times to timedelta

    return timediffs, lsl_t_trigger0, clock_tstart


def compare_triggers(mrk_stream, lsl_header, meg_trigger_diffs):
    
    # get AN trigger times from stream (first trigger time, and time diffs)
    (an_trigger_diffs,
     lsl_t_trigger0,
     lsl_clock_t0) = get_AN_trigger_diffs(
        mrk_times=mrk_stream['time_stamps'], fileheader=lsl_header,
    )

    # get differences in trigger timing meg vs antneuro
    sync_diffs = [abs(an_trigger_diffs[i] - meg_trigger_diffs[i])
                  for i in np.arange(len(an_trigger_diffs))]
    
    print(f'### Time differences (in seconds) of sync triggers'
          f' AntNeuro vs OPM: {[t.total_seconds() for t in sync_diffs]}')

    return sync_diffs, lsl_t_trigger0, lsl_clock_t0


def convert_lsltimes_to_megtimes_sec(
    lsltimestamps, lsl_t_trigger0, meg_time_trigger0,
    lsl_clock_t0=None,
):
    """
    Returns:
    - converted lsl timestamp, relative to t=0 -> opm-start
    - clocktime of opm-start t=0
    """
    # get timestamps for antneuro data timestamps
    lsldat_dt_times  = [
        dt.datetime.fromtimestamp(t) for t in lsltimestamps
    ]
    # get relative timestamps in seconds, t = 0 sec starting from start lsl recording
    lsldat_rel_times_sec = [
        (t - lsldat_dt_times[0])
        for t in lsldat_dt_times
    ]
    # subtract n-seconds of first trigger in lsl timestamps (sets t=0 to first trigger)
    t_trig0 = dt.datetime.fromtimestamp(lsl_t_trigger0)  # timestamp of lsl time trigger0
    lsl_start_trigger0_gap = t_trig0 - lsldat_dt_times[0]  # as timedelta to lsl-starttime
    lsldat_rel_times_sec = [(t - lsl_start_trigger0_gap).total_seconds()
                            for t in lsldat_rel_times_sec]  # lsl-timestamps rel to trigger0 lsl in seconds

    # add n-seconds of first trigger in opm timestamps (sets t=0 to start opm recording)
    lsl_tstamps_in_megtime_sec = np.array(
        [t + meg_time_trigger0 for t in lsldat_rel_times_sec]
    )

    ### add clocktime (coming from LSL) of opm t=0 (for later dopa_time)
    if lsl_clock_t0:
        clocktime_trigger0 = lsl_clock_t0 + lsl_start_trigger0_gap
        clocktime_opm_t0 = clocktime_trigger0 - dt.timedelta(seconds=meg_time_trigger0)
    else:
        clocktime_opm_t0 = None
    
    return lsl_tstamps_in_megtime_sec, clocktime_opm_t0


def cut_data_to_task_timing(
    tempdata, temptimes, sub_meta,
    TASK: str, ACQ: str, SFREQ: int,
    ASSUME_TSTART_OPMt0: bool = False,
):
    """
    cut data based on task beginning and end
    get real-task-timings

    Input:
    - tempdata: 2d array [samples, channels]
    - temptimes: 1d array [samples,], time in seconds
    - sub_meta: meta info file
    - TASK, ACQ, SFREQ ...
    - ASSUME_TSTART_OPMt0: given times have 0 at
        start of OPM data, no index/time search necessary
    """
    rec_sel = [TASK in n.lower() and ACQ in n.lower()
               for n in sub_meta['rec_name']]
    rec_start_end = sub_meta[rec_sel][['real_task_start', 'real_task_end']].values.ravel()

    if ASSUME_TSTART_OPMt0:
        i_start = (rec_start_end[0].minute * 60 + rec_start_end[0].second) * SFREQ
        i_end = (rec_start_end[1].minute * 60 + rec_start_end[1].second) * SFREQ

    else:
        sec_start = rec_start_end[0].minute * 60 + rec_start_end[0].second
        sec_end = rec_start_end[1].minute * 60 + rec_start_end[1].second
        # find indices
        i_start = np.argmin(abs(temptimes - sec_start))
        i_end = np.argmin(abs(temptimes - sec_end))
        
    tempdata = tempdata[i_start:i_end, :]
    temptimes = temptimes[i_start:i_end]

    return tempdata, temptimes