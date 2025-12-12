"""
supportive functions to find and create epochs/segments
in raw mne-Objects
"""

import numpy as np
from mne import Epochs
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def get_mne_event_array(self, dType: str):

    EPOCH_ATTR = f'{dType.lower()}_task_epochs'
    event_codes = {key: i+1 for i, key in enumerate(getattr(self, EPOCH_ATTR))}

    event_lists = []  # to creat array afterwards [start_index, 0, event_code]
    for e_key, e_idx in getattr(self, EPOCH_ATTR).items():
        for e_i in e_idx:
            event_lists.append([e_i, 0, event_codes[e_key]])

    event_arr = np.array(event_lists)

    return event_codes, event_arr


def add_task_epoch_idx(self, times_to_use, sfreq, INCL_EDGE = 0,
                       REST_EPOC_LEN: int = 3,):
    """
    takes one second before start and one second after end
    INCL_EDGE = 1  # second extra prior and post

    # adjust 14.10 only start epoch indices

    """


    task_epochs = {}

    # take extracted timings from behavioral task, for rest
    # take every three-second window as an epoch

    if self.task == 'rest':
        task_dict = {'rest': {'start': []}}

        for sec in np.arange(np.round(times_to_use[0]) + .5 + INCL_EDGE,  # add 1 for edge, half for rounding up
                             np.round(times_to_use[-1] - INCL_EDGE),
                             REST_EPOC_LEN):  # create distance between imaginary rest epochs
            task_dict['rest']['start'].append(sec)

    else:
        task_dict = self.tasktimings


    for task, timings in task_dict.items():

        task_epochs[task] = []

        for t0 in timings['start']:

            i0 = np.argmin(abs(times_to_use - (t0 - INCL_EDGE)))

            i1 = np.argmin(abs(times_to_use - (t0 + 3)))
            if i0 == i1: continue  # happened when task was incorrectly started before real start
            if (i1 - i0) < (3 * sfreq):
                print(f'skipped epoch with sample length {i1-i0}')
                continue  # too short
            
            task_epochs[task].append(i0)

    return task_epochs



def get_epochs(acqClass, TMIN=-1, TMAX=3):



    emg_epochs = Epochs(
        raw=acqClass.EMG, events=acqClass.aux_event_arr,
        event_id=acqClass.aux_event_codes,
        tmin=TMIN, tmax=TMAX, baseline=None, preload=True,
        reject=None,
    )
    acc_epochs = Epochs(
        raw=acqClass.ACC, events=acqClass.aux_event_arr,
        event_id=acqClass.aux_event_codes,
        tmin=TMIN, tmax=TMAX, baseline=None, preload=True,
        reject=None,
    )
    opm_epochs = Epochs(
        raw=acqClass.OPM_Z, events=acqClass.opm_event_arr,
        event_id=acqClass.opm_event_codes,
        tmin=TMIN, tmax=TMAX, baseline=None, preload=True,
        reject=None,
    )

    return opm_epochs, emg_epochs, acc_epochs



