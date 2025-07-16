import numpy as np
import pandas as pd
import seaborn as sns
import mne
import scipy as sp
from matplotlib.lines import lineStyles
from scipy.signal import butter, sosfiltfilt
from scipy.stats import zscore
from combined_analysis_bachelor.code.functions_for_pipeline import get_ch_indices, plot_channel_overview, normalize_emg, \
    notched_and_filtered, create_df, envelope, rectify
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

# import example data
leg_checks = mne.io.read_raw_cnt("C:/Users/User/Documents/bachelorarbeit/data/EMG/Test_Leg_positions_synchron.cnt", preload=True)

onsets = leg_checks.annotations.onset
onsets = onsets * 1000
custom_order_names = ["BIP11", "BIP12"] # list of the disired order
location = {"BIP7":"left forearm",
            "BIP8": "left delt",
            "BIP1" : "Charité ACC : y",
            "BIP2" : "Charité ACC : z",
            "BIP6" : "Charité ACC : x",
            "BIP3" : "2nd Charité ACC: x",
            "BIP4" : "2nd Charité ACC: y",
            "BIP5" : "2nd Charité ACC: z",
            "BIP11" : "right calf upper",
            "BIP12" : "right calf lower"}
EMG = ["BIP11", "BIP12"]
ACC = ["BIP6", "BIP1", "BIP2", "BIP3", "BIP4", "BIP5"]

data, times = leg_checks[custom_order_names, : ]
#data[0] *= -1  # convert inverted x-axis of charite sensor 1
# data[4] *= -1

# creating raw df
raw_df = create_df(data, custom_order_names, times)

#plotting the raw signals
plot_channel_overview(custom_order_names, raw_df,"raw_signals", location, EMG)

# notch filter and bandpass filtering
notched_and_filtered_df = notched_and_filtered(raw_df, custom_order_names, ACC, EMG, times)

# plot filtered data
plot_channel_overview(custom_order_names, notched_and_filtered_df,"filtered_signals", location, EMG)

# retrification and plotting
#rectified_df = abs(notched_and_filtered_df) #technically also took the absolute values of the time column, but that doesnt matter i think
rectified_df = rectify(notched_and_filtered_df, custom_order_names, EMG)

plot_channel_overview(custom_order_names, rectified_df, "rectified_signals", location, EMG)

# normalizing
emg_normalized_df = normalize_emg(rectified_df, custom_order_names, EMG, times) # -> normalize to baseline recording??

# plot normalized data
plot_channel_overview(EMG, emg_normalized_df, "normalized_df", location, EMG)

# build envelope
emg_envelopes = envelope(emg_normalized_df, custom_order_names, EMG, times, 3)
plot_channel_overview(EMG, emg_envelopes, "envelope_df", location, EMG)

# movement detection
task_start_end = [1000, 15000] # pretending that that was trigger signal of start and end of task
forearm_envelope = emg_envelopes["BIP7"]
task_signal = forearm_envelope[task_start_end[0]:task_start_end[1]]

sfreq = 1000
times = np.arange(len(task_signal)) / sfreq

single_thresh = 0.03
adaptive_thresh = np.mean(task_signal[:1000]) + np.max(zscore(task_signal[:1000])) * np.std(task_signal[:1000])
#thresh = np.mean(task_signal[:1000]) + np.max(zscore(task_signal[:1000]))
#thresh = 0.035
movement = task_signal > adaptive_thresh
movement = movement.to_numpy()

plt.figure()
plt.plot(times, task_signal, "b")
plt.plot(times, movement, "g")
plt.show()

starts = []
ends = []

for i in range(1, len(movement)):
    if not movement[i-1] and movement[i]:
        starts.append(i)
    elif movement[i-1] and not movement[i]:
        ends.append(i)

# if recording starts or ends with contraction
if movement[0]:
    starts = [0] + starts
if movement[-1]:
    ends = ends + [len(movement)-1]

# to seconds
start_times = np.array(starts) / sfreq
end_times = np.array(ends) / sfreq

plt.figure(figsize=(10, 4))
plt.plot(times, task_signal, "b", label="Envelope")
plt.axhline(adaptive_thresh, color="r", linestyle="--", label="Threshold")
for t in start_times:
    plt.axvline(t, color="g", linestyle="--", label="Start" if t == start_times[0] else "")
for t in end_times:
    plt.axvline(t, color="k", linestyle="--", label="End" if t == end_times[0] else "")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Envelope / Movement")
plt.title("Contraction Detection")
plt.show()

# build pairs
contractions = []
for i in range(0, len(start_times)):
    if len(start_times) == len(end_times):
        contraction = [start_times[i], end_times[i]]
        contractions.append(contraction)
    else:
        print("no equal amount of starts and ends")

print(contractions)