from os import remove

import numpy as np
import pandas as pd
import seaborn as sns
import sys
import os
import mne
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

source_folder = "C:/Users/User/Documents/bachelorarbeit/data/EMG_rest_last_year"
filepaths = []
names = os.listdir(source_folder)
for name in names:
    filepath = f"{source_folder}/{name}"
    filepaths.append(filepath)

raw = mne.io.read_raw_kit(filepaths[0], preload=True)
# plt.plot(raw.times[:10000], raw.get_data(picks=['E09'])[0][:10000])
# plt.title("E05: Check for flat peaks (saturation)")
# plt.ylabel("Amplitude (V)")
# plt.show()

def apply_filter(data, sfreq, l_freq, h_freq):
    """Apply bandpass filter to data using MNE"""
    # Convert data to MNE RawArray format
    info = mne.create_info(ch_names=['signal'], sfreq=sfreq, ch_types=['misc'])
    raw = mne.io.RawArray(data.reshape(1, -1), info, verbose=False)

    # Apply filter with explicit picks and suppress output
    raw.filter(l_freq=l_freq, h_freq=h_freq, picks=[0], verbose=False)

    return raw.get_data()[0]

def extract_con_data(file):

    """Extract EEG, EMG and ACC data from .con file"""

    # Read the .con file
    raw = mne.io.read_raw_kit(file, preload=True)
    channels = raw.ch_names
    print(channels)

    print(f"\nFile: {file}")
    for ch in ["E05", "E09", "E08", "E11", "E13"]:
        data = raw.get_data(picks=[ch])[0]
        print(f"{ch}: min={data.min():.2e} V, max={data.max():.2e} V")

    # Define channel mappings
    emg_channels = {'right_arm':'E05',
                    'neck': ['E07','E06'], # 135-134
                    'left_arm': ['E09','E08'], # 137-136
                    'right_leg': ['E11','E10'], # 139-138
                    'left_leg': ['E13','E12']} # 141-140


    acc_channels = {'x': 'E17', # 145
                    'y': 'E18', # 146
                    'z': 'E19' # 147
                                }

    trigger_channel = 'E29' # 157

    # Create filtered copy for EEG channels only
    raw_filtered = raw.copy()

    # Apply filters only to EEG channels
    eeg_picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, misc=False)
    raw_filtered.filter(l_freq=20, h_freq=450, picks=eeg_picks)

    # Extract EMG data (bipolar) with filtering
    emg_data = {}
    for location, channels in emg_channels.items():
        if isinstance(channels,list):
            # Filter individual channels before subtraction

            ch1 = raw.get_data(picks=[channels[0]])[0]
            ch2 = raw.get_data(picks=[channels[1]])[0]
            ch1_filtered = apply_filter(ch1, raw.info['sfreq'], 20, 450)
            ch2_filtered = apply_filter(ch2, raw.info['sfreq'], 20, 450)
            emg_data[location] = ch1_filtered - ch2_filtered


        else:

            data = raw.get_data(picks=[channels])[0]
            emg_data[location] = apply_filter(data, raw.info['sfreq'], 20, 450)


    # Extract and filter ACC data
    acc_data = {}
    for axis, channel in acc_channels.items():

        data = raw.get_data(picks=[channel])[0]

        acc_data[axis] = apply_filter(data, raw.info['sfreq'], 2, 20)

    for location in emg_data:
        emg_data[location] *= 1000

    # Get filtered EEG
    eeg_data = raw_filtered.get_data(picks=eeg_picks)

    return emg_data, acc_data, eeg_data

#for filepath in filepaths:
#    extract_con_data(filepath)

def rest_recordings_df(location):
    absolutes = []
    recs = ["rest_rec01", "rest_rec03", "rest_rec05", "rest_rec07", "rest_rec09", "rest_rec11"]
    for i, filepath in enumerate(filepaths):
        emg_data, _, _  = extract_con_data(filepath)
        #absolute = np.abs(emg_data[location])
        df= pd.DataFrame({
            "emg_value": emg_data[location],
            "recording": recs[i],
            "location": location
        })
        absolutes.append(df)
    return pd.concat(absolutes)

rest_recs_left_arm = rest_recordings_df("left_arm")
rest_recs_left_leg = rest_recordings_df("left_leg")
rest_recs_right_arm = rest_recordings_df("right_arm")
rest_recs_right_leg = rest_recordings_df("right_leg")

all_locs = pd.concat([rest_recs_left_arm, rest_recs_left_leg, rest_recs_right_arm, rest_recs_right_leg])

#def plot_histograms(data, remove_outliers=False, use_log_y=False, y_max=None):
#    # Filter nach Körperstelle
#    #data = all_locs[all_locs["location"] == location].copy()
#
#    # Optional: Outlier entfernen (z.B. 1. und 99. Perzentil)
#    if remove_outliers:
#        lower = data["emg_value"].quantile(0.01)
#        upper = data["emg_value"].quantile(0.99)
#        data = data[(data["emg_value"] >= lower) & (data["emg_value"] <= upper)]
#
#    # Plot
#    plt.figure(figsize=(10, 5))
#    sns.histplot(data["emg_value"], bins=100, kde=True, hue="location")
#    plt.title(f"Histogramm der EMG-Werte) {'(ohne Ausreißer)' if remove_outliers else ''}")
#    plt.xlabel("Absolute EMG-Werte (µV)")
#    plt.ylabel("Häufigkeit")
#
#    if use_log_y:
#        plt.yscale("log")
#    if y_max:
#        plt.ylim(top=y_max)
#
#    plt.tight_layout()
#    plt.show()


def plot_recording_histograms(data, recordings):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    location_palette = {
        "right_arm": "blue",
        "right_leg": "orange",
        "left_arm": "red",
        "left_leg": "green"}

    for i, rec in enumerate(recordings):
        subset = data[data["recording"] == rec].copy()

        #lower = subset["emg_value"].quantile(0.01)
        #upper = subset["emg_value"].quantile(0.99)
        #subset = subset[(subset["emg_value"] >= lower) & (subset["emg_value"] <= upper)]


        sns.histplot(
            data=subset,
            x="emg_value",
            hue="location",
            bins=100,
            kde=False,
            ax=axes[i],
            element="step",
            palette=location_palette
        )

        axes[i].set_xlim(0,1100)
        axes[i].set_ylim(0,150)
        axes[i].set_title(f"{rec}")
        axes[i].set_xlabel("Absolute EMG-values (µV)")
        axes[i].set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig("../images/zoomed_histplot_last_year_EMG_PTB.png")
    #plt.show()


plot_recording_histograms(all_locs, recordings=["rest_rec01", "rest_rec07"])
#plot_histograms(all_locs_long, remove_outliers=True, use_log_y=False, y_max=None)


def plot_errorbar(location):
    absolutes = []
    for filepath in filepaths:
        emg_data, _, _ = extract_con_data(filepath)
        absolutes.append(np.abs(emg_data[location]))

    means = [np.mean(rec) for rec in absolutes]
    stds = [np.std(rec) for rec in absolutes]

    plt.figure()
    plt.errorbar(x=range(len(means)), y=means, yerr=stds, fmt='o', label=location,
               markersize=8, capsize=5, capthick=2)
    plt.title("Errorbars of EMG values (left arm) of progressing (rest) recordings")
    plt.xticks(ticks=range(len(means)), labels=["rec01", "rec03", "rec05", "rec07", "rec09", "rec11"])
    plt.ylabel("mean + std of measurement")
    plt.tight_layout()
    plt.show()

#plot_errorbar("left_arm")


#x_labels = [f"Recording{i}" for i in [1, 3, 5, 7, 9, 11]]
#x_pos = np.arange(len(absolutes))

#lt.figure()
#lt.errorbar(x_pos, means, yerr=stds, fmt='o',
#            markersize=8, capsize=5, capthick=2)
#lt.title("Errorbars of EMG values (left arm) of progressing (rest) recordings")
#lt.xticks(6, x_labels, rotation=45)
#lt.ylabel("mean + std of measurement")
#lt.tight_layout()
#lt.show()

plt.figure(figsize=(12, 6))
sns.histplot(
    data=all_locs,
    x='recording',
    y='emg_value',
    hue='location'
)

plt.title("EMG Signal Distribution by Recording and Location")
plt.xlabel("Recording Session")
plt.ylabel("Absolute EMG Value (µV)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


