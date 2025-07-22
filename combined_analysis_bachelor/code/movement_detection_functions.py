import numpy as np
import scipy as sp
import pandas as pd
import mne

### Functions i still need to create ###
# 1. importing of EMG and ACC # von processed data , dann eine envelope funktion machen und eine tkeo - oder diese
# direkt auch so speichern? - jeroen gefragt...
# 2. getting Baseline sections (mean and std e.g)
# 3. movement detection itself (choosing method?, threshold param, activity length param -> output = on/offsets
 # --> for EMG and ACC seperated or together always?


## removing off and onsets that are too close together (splitting one arm raise)
def take_out_short_off_onset(onsets, offsets, min_time_period, sampling_freq):
    onsets_clean = onsets.copy()
    offsets_clean = offsets.copy()
    min_sample_period = min_time_period * sampling_freq

    if hasattr(onsets, 'tolist'):
        onsets_clean = onsets.tolist()
        offsets_clean = offsets.tolist()

    # Iterate in reverse to safely delete items
    for i in range(len(offsets) - 2, -1, -1):  # Start from the end
        time_between = onsets[i + 1] - offsets[i]
        if time_between <= min_sample_period:
            del onsets_clean[i + 1]
            del offsets_clean[i]

    return onsets_clean, offsets_clean


# get array of final on- and onsets pairs #
def new_on_offsets(new_onsets, new_offsets):
    """takes in the final on- and offsets and puts them into new list as
    on- and offset pairs"""
    onsets_and_offsets = []
    for idx, onset in enumerate(new_onsets):
        pair = [onset, new_offsets[idx]]
        onsets_and_offsets.append(pair)
    return onsets_and_offsets



## create binary array ##
def fill_activity_mask(on_and_offsets, sampling_freq, time_column):
    """takes in final on- and offsets and creates binary array where the periods between on- and offset are set to True. Time points outside of these periods are False
    args
    on_and_offsets: list that holds pairs of on- and onsets
    sampling_freq: sampling frequency
    time_column: time column (Sync_Time) of the recording

    returns binary (boolean) array"""

    zeros = np.zeros(len(time_column))
    mask = np.zeros_like(zeros, dtype=bool)  # all to False
    for onset, offset in on_and_offsets:
        start = int(round(onset))
        end = int(round(offset))
        mask[max(0, start):min(len(mask), end + 1)] = True # sicher, dass hier end+1 hin muss?
    return mask



def create_behavioral_array(emg_activities_binary, acc_activities_binary):
    """compares the EMG and ACC binary arrays and outputs
    a number for each detected activity state"""
    behavioral_list = []
    for i, active in enumerate(emg_activities_binary):
        if active == True and acc_activities_binary[i] == True:
            behavioral_list.append(1)
        if active == False and acc_activities_binary[i] == False:
            behavioral_list.append(2)
        if active == True and acc_activities_binary[i] == False:
            behavioral_list.append(3)
        if active == False and acc_activities_binary[i] == True:
            behavioral_list.append(4)
    behavioral_array = np.array(behavioral_list)
    return behavioral_array