import numpy as np
import scipy as sp
import pandas as pd
import mne
from hyperframe.frame import DataFrame
from mne.io import read_raw_ant
import os
from matplotlib.lines import lineStyles
from scipy.signal import butter, sosfiltfilt
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


### prepare rest rec values ###
rest_rec_filtered = pd.read_hdf("C:/Users/User/Documents/bachelorarbeit/data/EMG_ACC/sub-91/"
                              "PTB_01_data_processed/A_1.1_rest_processed.h5", key="data")
rest_rec_filtered_df =pd.DataFrame(rest_rec_filtered)
left_delt_rest = rest_rec_filtered_df["BIP10"]

# enveloping rest #
rectified_rest = left_delt_rest.abs()
low_pass = 4/(1000/2)
sos = butter(4, low_pass, btype='lowpass', output="sos")
enveloped_rest = sosfiltfilt(sos, x=rectified_rest)


### left delt movement detection ###
A_move1_filtered = pd.read_hdf("C:/Users/User/Documents/bachelorarbeit/data/EMG_ACC/sub-91/"
                              "PTB_01_data_processed/A_1.2_move_processed.h5", key="data")
A_move1_filtered_df = pd.DataFrame(A_move1_filtered)


### location: left delt ###
left_delt_move = A_move1_filtered_df["BIP10"]
left_delt_move *= 1e6

# rectifying and building envelope #
rectified_move = left_delt_move.abs()
enveloped_move = sosfiltfilt(sos, x=rectified_move)


# get diff from envelope #
diff_move1 = np.diff(enveloped_move)

# taking baseline section #
A_move_1_baseline = enveloped_move[7*sf:17*sf] # 10 secs for baseline values
p99 = np.percentile(A_move_1_baseline, 99)
diff_baseline = diff_move1[7*sf:17*sf]
p99_diff = np.percentile(diff_baseline, 99)

# get std of baseline / rest_rec / diff baseline #
mu_move_baseline, sigma_move_baseline = A_move_1_baseline.mean(), A_move_1_baseline.std()
mu_rest, sigma_rest = enveloped_rest.mean(), enveloped_rest.std()
mu_diff_baseline, sigma_diff_baseline = diff_baseline.mean(), diff_baseline.std()

# set threshold #
std_thresh = mu_move_baseline + 14.125 * sigma_move_baseline
diff_std_thresh = mu_diff_baseline + 14.125 * sigma_diff_baseline
p99_thresh = p99 * 4
diff_p99_thresh = p99_diff * 4
envelope_activity = enveloped_move > p99_thresh
diff_activity = diff_move1 > diff_p99_thresh

starts = []
ends = []
for i in range(1, len(envelope_activity)):
    if not envelope_activity[i-1] and envelope_activity[i]:
        starts.append(i)
    elif envelope_activity[i-1] and not envelope_activity[i]:
        ends.append(i)
# if recording starts or ends with contraction
if envelope_activity[0]:
    starts = [0] + starts
if envelope_activity[-1]:
    ends = ends + [len(envelope_activity)-1]
# to seconds
start_times = np.array(starts) / sf
end_times = np.array(ends) / sf

# onset criteria : envelope has to be over threshold for min 1000ms for example #
min_samples = int(1 * sf)         # 1000-ms-Grenze

labels, n_lbl = label(envelope_activity)
valid = np.zeros_like(envelope_activity, dtype=bool)

for lbl in range(1, n_lbl + 1):
    idx = np.where(labels == lbl)[0]
    if idx.size >= min_samples:
        valid[idx] = True

# On- & Offsets
onsets  = np.where(np.diff(valid.astype(int)) ==  1)[0] + 1
offsets = np.where(np.diff(valid.astype(int)) == -1)[0] + 1

### onsets for diff ###
min_samples_diff = int(0.5 * sf)         # 1000-ms-Grenze

diff_labels, diff_n_lbl = label(diff_activity)
diff_valid = np.zeros_like(diff_activity, dtype=bool)

for diff_lbl in range(1, diff_n_lbl + 1):
    idx = np.where(diff_labels == diff_lbl)[0]
    if idx.size >= min_samples_diff:
        diff_valid[idx] = True

# On- & Offsets
diff_onsets  = np.where(np.diff(diff_valid.astype(int)) ==  1)[0] + 1
diff_offsets = np.where(np.diff(diff_valid.astype(int)) == -1)[0] + 1


plt.plot(A_move1_filtered_df["Sync_Time (s)"], enveloped_move, "b", label="Envelope")
plt.plot(A_move1_filtered_df["Sync_Time (s)"][:-1], np.diff(enveloped_move), "green", label="diff envelope")
plt.axhline(p99_thresh, color="r", linestyle="--", label="Threshold")
plt.axhline(diff_p99_thresh, color="r", linestyle="-.", label="thresh for diff")
# for t in start_times:
#     plt.axvline(t, color="g", linestyle="--", label="Start" if t == start_times[0] else "")
# for t in end_times:
#     plt.axvline(t, color="k", linestyle="--", label="End" if t == end_times[0] else "")
for x in onsets:
    plt.axvline(A_move1_filtered_df["Sync_Time (s)"][x], ls="--", c="g")
for x in offsets:
    plt.axvline(A_move1_filtered_df["Sync_Time (s)"][x], ls="--", c="k")
for y in diff_onsets:
    plt.axvline(A_move1_filtered_df["Sync_Time (s)"][y], ls="-.", c="g")
for y in diff_offsets:
    plt.axvline(A_move1_filtered_df["Sync_Time (s)"][y], ls="-.", c="k")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Envelope / Movement")
plt.title("Muscle Activity Detection")
#plt.xlim(43,47)
plt.show()
#plt.savefig("../images/zoom_movement_detec_p99x4_and_diff.png")
