import numpy as np
import scipy as sp
import pandas as pd
import mne
from hyperframe.frame import DataFrame
from mne.io import read_raw_ant
import os
from matplotlib.lines import lineStyles
from scipy.signal import butter, sosfiltfilt, savgol_filter
from scipy.stats import zscore
from scipy.ndimage import label
from combined_analysis_bachelor.code.functions_for_pipeline import get_ch_indices, plot_channel_overview, normalize_emg, \
    notched_and_filtered, create_df, envelope, rectify, tkeo
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

channel_custom_order = ["BIP3", "BIP4", "BIP5", "BIP9", "BIP10", "BIP12", "BIP6", "BIP1", "BIP2", "BIP11", "BIP8", "BIP7"]
EMG = ["BIP7", "BIP8", "BIP9", "BIP10", "BIP11", "BIP12"]
ACC = ["BIP1", "BIP2", "BIP3", "BIP4", "BIP5", "BIP6"]
locations = {"BIP7":"right M. tibialis anterior",
            "BIP8": "right M. deltoideus",
            "BIP11": "right M. brachioradialis",
            "BIP1" : "ACC right hand : y",
            "BIP2" : "ACC right hand : z",
            "BIP6" : "ACC right hand : x",
            "BIP3" : "ACC left hand : x",
            "BIP4" : "ACC left hand : y",
            "BIP5" : "ACC left hand : z",
            "BIP9" : "left M. brachioradialis",
            "BIP10" : "left M. deltoideus",
            "BIP12" : "left M. tibialis anterior"}
sf=1000

#================================= ACC Movement Detec tests - Setup A - Move1 - L ======================================
A_move1_filtered = pd.read_hdf("C:/Users/User/Documents/bachelorarbeit/data/EMG_ACC/sub_91/"
                              "PTB_01_data_processed/A_1.2_move_processed.h5", key="data")
A_move1_filtered_df = pd.DataFrame(A_move1_filtered)

ACC_left_x_move1 = A_move1_filtered_df["BIP3"]
ACC_left_y_move1 = A_move1_filtered_df["BIP4"] * -1
ACC_left_z_move1 = A_move1_filtered_df["BIP5"]

# smooth signal
ACC_left_move1_smooth = savgol_filter(ACC_left_z_move1, window_length=21, polyorder=3)

# RMS
window_size = 60
ACC_left_move1_rms = np.sqrt(np.convolve(ACC_left_move1_smooth**2, np.ones(window_size)/window_size, mode='same'))

# get baseline parts
A_move1_baseline = ACC_left_move1_rms[7*sf:17*sf] # 10 secs for baseline values
mu_move1_baseline, sigma_move1_baseline = A_move1_baseline.mean(), A_move1_baseline.std()
p99 = np.percentile(ACC_left_move1_rms, 99)

# define threshold and activity
ACC_thresh = mu_move1_baseline + 8 * sigma_move1_baseline
ACC_activity = ACC_left_move1_rms > ACC_thresh

# plt.figure()
# plt.plot(A_move1_filtered["Sync_Time (s)"], ACC_left_move1_rms)
# plt.show()

# get on and offsets
min_samples = int(1 * sf) # 1000-ms-Grenze

labels, n_lbl = label(ACC_activity)
valid = np.zeros_like(ACC_activity, dtype=bool)

for lbl in range(1, n_lbl + 1):
    idx = np.where(labels == lbl)[0]
    if idx.size >= min_samples:
        valid[idx] = True

# On- & Offsets
Acc_move1_Onsets  = np.where(np.diff(valid.astype(int)) ==  1)[0] + 1
Acc_move1_Offsets = np.where(np.diff(valid.astype(int)) == -1)[0] + 1

### plotting ###
#plt.figure()
#plt.plot(A_move1_filtered_df["Sync_Time (s)"], ACC_left_move1_rms, "b", label="Envelope")
##plt.plot(A_move1_filtered_df["Sync_Time (s)"][:-1], np.diff(enveloped_move), "green", label="diff envelope")
#plt.axhline(ACC_thresh, color="r", linestyle="--", label="Threshold")
##plt.axhline(diff_p99_thresh, color="r", linestyle="-.", label="thresh for diff")
#for x in Acc_move1_Onsets:
#    plt.axvline(A_move1_filtered_df["Sync_Time (s)"][x], ls="--", c="g")
#for x in Acc_move1_Offsets:
#    plt.axvline(A_move1_filtered_df["Sync_Time (s)"][x], ls="--", c="k")
#plt.legend()
#plt.xlabel("Time (s)")
#plt.ylabel("Acceleration")
#plt.title("ACC movement Detection for move 1 -> left Z - thresh: mean+8*std")
#plt.xlim(43,47)
#plt.show()


## removing off and onsets that are too close together (splitting one arm raise) ###
def take_out_short_off_onset(onsets, offsets, min_time_period, sampling_frequency):
    onsets_clean = onsets.copy()
    offsets_clean = offsets.copy()
    min_sample_period = min_time_period * sampling_frequency

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

new_Acc_move1_onsets, new_Acc_move1_offsets = take_out_short_off_onset(Acc_move1_Onsets, Acc_move1_Offsets,
                                                                       0.5, sf)

# plot again to check result
#plt.figure()
#plt.plot(A_move1_filtered_df["Sync_Time (s)"], ACC_left_move1_rms, "b", label="Envelope")
##plt.plot(A_move1_filtered_df["Sync_Time (s)"][:-1], np.diff(enveloped_move), "green", label="diff envelope")
#plt.axhline(ACC_thresh, color="r", linestyle="--", label="Threshold")
##plt.axhline(diff_p99_thresh, color="r", linestyle="-.", label="thresh for diff")
#for x in new_Acc_move1_onsets:
#    plt.axvline(A_move1_filtered_df["Sync_Time (s)"][x], ls="--", c="g")
#for x in new_Acc_move1_offsets:
#    plt.axvline(A_move1_filtered_df["Sync_Time (s)"][x], ls="--", c="k")
#plt.legend()
#plt.xlabel("Time (s)")
#plt.ylabel("Acceleration")
#plt.title("ACC Movement Detection for move 1 -> left Z - thresh: mean+8*std & new_on-/offsets")
#plt.xlim(43,47)
#plt.show()

## maybe try euclidian norm and then again? ##
euclidean_norm_move1_left = np.sqrt(ACC_left_x_move1**2 + ACC_left_y_move1**2 + ACC_left_z_move1**2)

ACC_euclidean_move1_left_smooth = savgol_filter(euclidean_norm_move1_left, window_length=21, polyorder=3)

# RMS
window_size = 50
ACC_euclidean_move1_left_rms = np.sqrt(np.convolve(ACC_euclidean_move1_left_smooth**2, np.ones(window_size)/window_size, mode='same'))

# get baseline parts
A_euclidean_move1_baseline = ACC_euclidean_move1_left_rms[7*sf:17*sf] # 10 secs for baseline values
mu_move1_baseline, sigma_move1_baseline = A_euclidean_move1_baseline.mean(), A_euclidean_move1_baseline.std()
p99 = np.percentile(ACC_left_move1_rms, 99)

# define threshold and activity
ACC_thresh = mu_move1_baseline + 23 * sigma_move1_baseline
ACC_activity = ACC_euclidean_move1_left_rms > ACC_thresh

#plt.figure()
#plt.plot(A_move1_filtered_df["Sync_Time (s)"], ACC_euclidean_move1_left_rms)
#plt.show()

# get on and offsets
min_samples = int(1.5 * sf) # 2500-ms-Grenze

labels, n_lbl = label(ACC_activity)
valid = np.zeros_like(ACC_activity, dtype=bool)

for lbl in range(1, n_lbl + 1):
    idx = np.where(labels == lbl)[0]
    if idx.size >= min_samples:
        valid[idx] = True

# On- & Offsets
Acc_move1_Onsets  = np.where(np.diff(valid.astype(int)) ==  1)[0] + 1
Acc_move1_Offsets = np.where(np.diff(valid.astype(int)) == -1)[0] + 1

### plotting ###
plt.figure()
plt.plot(A_move1_filtered_df["Sync_Time (s)"], ACC_euclidean_move1_left_rms, "b", label="Envelope")
plt.axhline(ACC_thresh, color="r", linestyle="--", label="Threshold")
#plt.axhline(diff_p99_thresh, color="r", linestyle="-.", label="thresh for diff")
for x in Acc_move1_Onsets:
    plt.axvline(A_move1_filtered_df["Sync_Time (s)"][x], ls="--", c="g")
for x in Acc_move1_Offsets:
    plt.axvline(A_move1_filtered_df["Sync_Time (s)"][x], ls="--", c="k")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Acceleration")
plt.title("ACC movement Detection for move 1 -> euclidean left - thresh: mean+23*std")
plt.show()


#=======================================================================================================================


#======================================== ACC Movement Detec tests - Setup A - move2 - L========================
A_move2_filtered = pd.read_hdf("C:/Users/User/Documents/bachelorarbeit/data/EMG_ACC/sub_91/"
                              "PTB_01_data_processed/A_1.5_move_processed.h5", key="data")
A_move2_filtered_df = pd.DataFrame(A_move2_filtered)

ACC_left_move2 = A_move2_filtered_df["BIP5"] # * -1

# smooth signal
ACC_left_move2_smooth = savgol_filter(ACC_left_move2, window_length=21, polyorder=3)

# RMS
window_size = 50
ACC_left_move2_rms = np.sqrt(np.convolve(ACC_left_move2_smooth**2, np.ones(window_size)/window_size, mode='same'))

# get baseline parts
A_move2_baseline = ACC_left_move2_rms[:15*sf] # 10 secs for baseline values
mu_move2_baseline, sigma_move2_baseline = A_move2_baseline.mean(), A_move2_baseline.std()
p99 = np.percentile(ACC_left_move2_rms, 99)

# define threshold and activity
ACC_thresh = mu_move2_baseline + 4 * sigma_move2_baseline
ACC_activity = ACC_left_move2_rms > ACC_thresh

# get on and offsets
min_samples = int(1.6 * sf) # 1600-ms-Grenze --> min länge der aktivitäten

labels, n_lbl = label(ACC_activity)
valid = np.zeros_like(ACC_activity, dtype=bool)

for lbl in range(1, n_lbl + 1):
    idx = np.where(labels == lbl)[0]
    if idx.size >= min_samples:
        valid[idx] = True

# On- & Offsets
Acc_move2_Onsets  = np.where(np.diff(valid.astype(int)) ==  1)[0] + 1
Acc_move2_Offsets = np.where(np.diff(valid.astype(int)) == -1)[0] + 1

# plotting
plt.figure()
plt.plot(A_move2_filtered_df["Sync_Time (s)"], ACC_left_move2_rms, "b", label="Envelope")
#plt.plot(A_move1_filtered_df["Sync_Time (s)"][:-1], np.diff(enveloped_move), "green", label="diff envelope")
plt.axhline(ACC_thresh, color="r", linestyle="--", label="Threshold")
#plt.axhline(diff_p99_thresh, color="r", linestyle="-.", label="thresh for diff")
for x in Acc_move2_Onsets:
    plt.axvline(A_move1_filtered_df["Sync_Time (s)"][x], ls="--", c="g")
for x in Acc_move2_Offsets:
    plt.axvline(A_move1_filtered_df["Sync_Time (s)"][x], ls="--", c="k")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Acceleration")
plt.title("ACC Movement Detection for move 2 -> left forearm - thresh: mean+4*std")
#plt.xlim(43,47)
plt.show()

#sieht gut aus, jetzt noch einmal die funktion anwenden!!
new_Acc_move2_onsets, new_Acc_move2_offsets = take_out_short_off_onset(Acc_move2_Onsets, Acc_move2_Offsets,
                                                                       0.17, sf) # -> max länge der zwischendinger?

plt.figure()
plt.plot(A_move2_filtered_df["Sync_Time (s)"], ACC_left_move2_rms, "b", label="Envelope")
plt.axhline(ACC_thresh, color="r", linestyle="--", label="Threshold")
for x in new_Acc_move2_onsets:
    plt.axvline(A_move1_filtered_df["Sync_Time (s)"][x], ls="--", c="g")
for x in new_Acc_move2_offsets:
    plt.axvline(A_move1_filtered_df["Sync_Time (s)"][x], ls="--", c="k")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Acceleration")
plt.title("ACC Movement Detection for move 2 -> left forearm - thresh: mean+4*std + fkt active")
#plt.xlim(43,47)
plt.show()

