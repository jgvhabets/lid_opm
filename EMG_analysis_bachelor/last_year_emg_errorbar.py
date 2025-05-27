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

print(filepaths)

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
    raw_filtered.filter(l_freq=25, h_freq=248, picks=eeg_picks)

    # Extract EMG data (bipolar) with filtering
    emg_data = {}
    for location, channels in emg_channels.items():
        if isinstance(channels,list):
            # Filter individual channels before subtraction

            ch1 = raw.get_data(picks=[channels[0]])[0]
            ch2 = raw.get_data(picks=[channels[1]])[0]
            ch1_filtered = apply_filter(ch1, raw.info['sfreq'], 2, 248)
            ch2_filtered = apply_filter(ch2, raw.info['sfreq'], 2, 248)
            emg_data[location] = ch1_filtered - ch2_filtered


        else:

            data = raw.get_data(picks=[channels])[0]
            emg_data[location] = apply_filter(data, raw.info['sfreq'], 2, 248)


    # Extract and filter ACC data
    acc_data = {}
    for axis, channel in acc_channels.items():

        data = raw.get_data(picks=[channel])[0]

        acc_data[axis] = apply_filter(data, raw.info['sfreq'], 2, 48)


    # Get filtered EEG
    eeg_data = raw_filtered.get_data(picks=eeg_picks)

    return emg_data, acc_data, eeg_data


all_recs_left_arm = []
for filepath in filepaths:
    emg_data, acc_data, eeg_data  = extract_con_data(filepath)
    all_recs_left_arm.append(emg_data["left_arm"])

print(all_recs_left_arm)

absolutes = [abs(rec) for rec in all_recs_left_arm]
print(absolutes)
for i, arr in enumerate(absolutes):
    print(f"Recording {i+1} has {sum(arr < 0)} negative values")

means = [np.mean(absolute) for absolute in absolutes]
stds = [np.std(absolute) for absolute in absolutes]

x_labels = [f"Recording{i}" for i in [1, 3, 5, 7, 9, 11]]
x_pos = np.arange(len(absolutes))

plt.figure()
plt.errorbar(x_pos, means, yerr=stds, fmt='o',
             markersize=8, capsize=5, capthick=2)
plt.title("Errorbars of EMG values (left arm) of progressing (rest) recordings")
plt.xticks(x_pos, x_labels, rotation=45)
plt.ylabel("mean + std of measurement")
plt.tight_layout()
plt.show()

