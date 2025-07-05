import numpy as np
import pandas as pd
import seaborn as sns
import mne
import os
import scipy as sp
from matplotlib.lines import lineStyles
from scipy.signal import butter, sosfiltfilt
from scipy.stats import zscore
from EMG_analysis_bachelor.functions_for_pipeline import get_ch_indices, plot_channel_overview, normalize_emg, \
    notched_and_filtered, create_df, envelope, rectify, tkeo, notched
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
sf = 1000

directory = r"D:\PTB_01_data";

for file in os.listdir(directory):
    if file.endswith(".cnt"):
        filedir = mne.io.read_raw_ant(f"{directory}/{file}", preload=True)
        data, times = filedir[channel_custom_order, :]
        data[1] *= -1 # inverting y-Axis from left hand sensor
        data[6] *= -1 # invert x-Axis from right hand sensor

        #raw_df = create_df(data, channel_custom_order, times)
        ##plot_channel_overview(channel_custom_order, raw_df, f"raw_signals of {file}", locations, EMG)
        #notched_and_filtered_df = notched_and_filtered(raw_df, channel_custom_order, ACC, EMG, times)
        #plot_channel_overview(channel_custom_order, notched_and_filtered_df, f"filtered_signals of {file}", locations, EMG)
        ##rectified_df = rectify(notched_and_filtered_df, channel_custom_order, EMG)
        ##plot_channel_overview(channel_custom_order, rectified_df, f"rectified_signals of {file}", locations, EMG)
        #tkeo_df = tkeo(notched_and_filtered_df, channel_custom_order, EMG, times)
        #plot_channel_overview(channel_custom_order, tkeo_df, f"tkeo for {file}", locations, EMG)
        #emg_normalized_df = normalize_emg(rectified_df, channel_custom_order, EMG, times)
        #emg_envelopes = envelope(rectified_df, channel_custom_order, EMG, times, 3)
        #plot_channel_overview(EMG, emg_envelopes, f"enveloped EMG signals of {file}", locations, EMG)

## getting all file directories for individual plotting ##
filedirs = []
for file in os.listdir(directory):
    if file.endswith(".cnt"):
        filedir = f"{directory}/{file}"
        filedirs.append(filedir)
filedirs = filedirs[1:]

### ===================== Recording 1 : Setup A & Rest Recording ====================== ###
A_rest = mne.io.read_raw_ant(filedirs[0], preload=True)
A_rest_data, A_rest_times = A_rest[channel_custom_order, :]
onsets = A_rest.annotations.onset
onsets = onsets * sf
sync_trig = onsets[0] / sf
A_rest_data[1] *= -1 # inverting y-Axis from left hand sensor
A_rest_data[6] *= -1 # invert x-Axis from right hand sensor

# check where to trim
plt.figure()
plt.plot(A_rest_times, A_rest_data[6])
plt.axvline(x=onsets, color="black")
plt.show()

# trimming #
A_rest_start = int(onsets[0] + 10 * sf) # = 10 secs after Trigger Signal (=5 secs after Tap) (10 * 1000Hz)
A_rest_end = int(A_rest_start + 300 * sf) # = end after 5 mins from the start
A_rest_data_trimmed = A_rest_data[:, A_rest_start:A_rest_end]
A_rest_times_trimmed = A_rest_times[A_rest_start:A_rest_end]

# check that we trimmed correctly
#plt.figure()
#plt.plot(A_rest_times_trimmed, A_rest_data_trimmed[0])
#plt.show()

# get data into dataframe structure
# A_rest_raw_df = create_df(A_rest_data_trimmed, channel_custom_order, A_rest_times_trimmed)

# filtering of raw signal
#A_rest_filtered_df = notched_and_filtered(A_rest_raw_df, channel_custom_order, ACC, EMG, A_rest_times_trimmed)

# z-score of filtered signal (needed? look into literature!!)

# tkeo of filtered signal (should be very small!)
# rectify and smooth tkeo ("")



# ============== Recording 2: Setup A & Movement Task ====================== #
A_move_1 = mne.io.read_raw_ant(filedirs[1], preload=True)
A_move_1_data, A_move_1_times = A_move_1[channel_custom_order, :]
onsets = A_move_1.annotations.onset
onsets = onsets * sf
sync_trig = onsets[0] / sf
A_move_1_data[1] *= -1
A_move_1_data[6] *= -1

# trimming #
A_move_1_start = int(onsets[1] - 3 * sf) # = start 3 secs before first "Go" Signal
A_move_1_end = int(onsets[-1] + 7 * sf) # = 3 sec after last arm raise is finished = 7 secs after last trigger
A_move_1_data_trimmed = A_move_1_data[:, A_move_1_start:A_move_1_end]
A_move_1_times_trimmed = A_move_1_times[A_move_1_start:A_move_1_end]

# check that we trimmed correctly
#plt.figure()
#plt.plot(A_move_1_times_trimmed, A_move_1_data_trimmed[2], label="left arm")
#plt.plot(A_move_1_times_trimmed, A_move_1_data_trimmed[-4], label="right arm")
#plt.legend()
#plt.show()

# turn into raw_dataframe
#A_move_1_raw_df = create_df(A_move_1_data_trimmed, )

# ============== Recording 3: Setup A & Rest - mock dysk.  ====================== #
A_rest_mock = mne.io.read_raw_ant(filedirs[2], preload=True)
A_rest_mock_data, A_rest_mock_times = A_rest_mock[channel_custom_order, :]
onsets = A_rest_mock.annotations.onset
onsets = onsets * sf
sync_trig = onsets[2] / sf # i think there were two trigger sig. in my software for rec. 3 thats why nr. 3 of signals represents first rest period start
A_rest_mock_data[1] *= -1
A_rest_mock_data[6] *= -1

# trimming #
A_rest_mock_start = int(onsets[3] - 60 * sf) # = 60 secs * 1000Hz sf (taking the first move trig and going 60 secs back
A_rest_mock_end = int(onsets[-1] + 60 * sf) # = 60 secs after last trig
A_rest_mock_data_trimmed = A_rest_mock_data[:, A_rest_mock_start:]
A_rest_mock_times_trimmed = A_rest_mock_times[A_rest_mock_start:]

plt.figure()
plt.plot(A_rest_mock_times_trimmed, A_rest_mock_data_trimmed[2], label="left arm")
plt.plot(A_rest_mock_times_trimmed, A_rest_mock_data_trimmed[-4], label="right arm")
plt.legend()
plt.show()

# ============== Recording 4: Setup A & Move - mock dysk.  ====================== #
A_move_mock = mne.io.read_raw_ant(filedirs[3], preload=True)
A_move_mock_data, A_move_mock_times = A_move_mock[channel_custom_order, :]
onsets = A_move_mock.annotations.onset
onsets = onsets * sf
sync_trig = onsets[0] / sf
A_move_mock_data[1] *= -1
A_move_mock_data[6] *= -1

# trimming #
A_move_mock_start = int(onsets[1] - 3 * sf) # same reason as in Rec 2
A_move_mock_end = int(onsets[-1] + 7 * sf) # same reason as in Rec 2
A_move_mock_data_trimmed = A_move_mock_data[:, A_move_mock_start:A_move_mock_end]
A_move_mock_times_trimmed = A_move_mock_times[A_move_mock_start:A_move_mock_end]

# ============== Recording 5: Setup A & Move (2nd time)  ====================== #
A_move_2 = mne.io.read_raw_ant(filedirs[4], preload=True)
A_move_2_data, A_move_2_times = A_move_2[channel_custom_order, :]
onsets = A_move_2.annotations.onset
onsets = onsets * sf
sync_trig = onsets[0] / sf
A_move_2_data[1] *= -1
A_move_2_data[6] *= -1

# trimming #
A_move_2_start = int(onsets[1] - 3 * sf) # same reason as in Rec 2
A_move_2_end = int(onsets[-1] + 7 * sf) # same reason as in Rec 2
A_move_2_data_trimmed = A_move_2_data[:, A_move_2_start:A_move_2_end]
A_move_2_times_trimmed = A_move_2_times[A_move_2_start:A_move_2_end]

# ============== Recording 6: Setup B & Rest ====================== #
B_rest = mne.io.read_raw_ant(filedirs[5], preload=True)
B_rest_data, B_rest_times = B_rest[channel_custom_order, :]
onsets = B_rest.annotations.onset
onsets = onsets * sf
sync_trig = onsets[0] / sf
B_rest_data[1] *= -1
B_rest_data[6] *= -1

# trimming #
B_rest_start = int(onsets[0] + 11 * sf)
B_rest_end = int(B_rest_start + 300 * sf) # = ends 5 mins after the start
B_rest_data_trimmed = B_rest_data[:, B_rest_start:B_rest_end]
B_rest_times_trimmed = B_rest_times[B_rest_start:B_rest_end]

# ============== Recording 7: Setup B & Movement Task ====================== #
B_move = mne.io.read_raw_ant(filedirs[6], preload=True)
B_move_data, B_move_times = B_move[channel_custom_order, :]
onsets = B_move.annotations.onset
onsets = onsets * sf
sync_trig = onsets[0] / sf
B_move_data[1] *= -1
B_move_data[6] *= -1

# trimming #
B_move_start = int(onsets[1] - 3 * sf) # same reason as in Rec 2
B_move_end = int(onsets[-1] + 6 * sf) # = 3 sec after last arm raise = 6 secs after last trigger
B_move_data_trimmed = B_move_data[:, B_move_start:B_move_end]
B_move_times_trimmed = B_move_times[B_move_start:B_move_end]

# ============== Recording 8: Setup B & Rest - mock dysk. ====================== #
B_rest_mock = mne.io.read_raw_ant(filedirs[7], preload=True)
B_rest_mock_data, B_rest_mock_times = B_rest_mock[channel_custom_order, :]
onsets = B_rest_mock.annotations.onset
onsets = onsets * sf
sync_trig = onsets[0] / sf
B_rest_mock_data[1] *= -1
B_rest_mock_data[6] *= -1

# trimming #
B_rest_mock_start = int(onsets[1]) # = start of first rest min
B_rest_mock_end = int(onsets[-1] + 60 * sf) # last trigger plus 60 secs
B_rest_mock_data_trimmed = B_rest_mock_data[:, B_rest_mock_start:B_rest_mock_end]
B_rest_mock_times_trimmed = B_rest_mock_times[B_rest_mock_start:B_rest_mock_end]

# ============== Recording 9: Setup B & Movement Task - mock dysk. ====================== #
B_move_mock = mne.io.read_raw_ant(filedirs[8], preload=True)
B_move_mock_data, B_move_mock_times = B_move_mock[channel_custom_order, :]
onsets = B_move_mock.annotations.onset
onsets = onsets * sf
sync_trig = onsets[0] / sf
B_move_mock_data[1] *= -1
B_move_mock_data[6] *= -1

# trimming #
B_move_mock_start = int(onsets[1] - 3 * sf) # = 3 secs before first "go" trig
B_move_mock_end = int(onsets[-1] + 6 * sf) # last trigger plus 6 secs
B_move_mock_data_trimmed = B_move_mock_data[:, B_move_mock_start:B_move_mock_end]
B_move_mock_times_trimmed = B_move_mock_times[B_move_mock_start:B_move_mock_end]

# plt.figure()
# plt.plot(B_move_mock_times_trimmed, B_move_mock_data_trimmed[2])
# plt.plot(B_move_mock_times_trimmed, B_move_mock_data_trimmed[-2])
# plt.axvline(x=int(onsets[-1]/1000), color="black")
# plt.show()





# instead of enveloping? good for onset detection!
#def tkeo(signal):
#    """Teager-Kaiser Energy Operator."""
#    return signal[1:-1] ** 2 - signal[0:-2] * signal[2:]
