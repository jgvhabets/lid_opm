"""
Functions to check sync of OPM vs AntNeuro data,
and to allign all timeseries to 1) opm-time, and
2) ldopa-time
"""

import datetime as dt
import os
import numpy as np

from utils.load_utils import get_onedrive_path




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
):
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
    
    return lsl_tstamps_in_megtime_sec


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