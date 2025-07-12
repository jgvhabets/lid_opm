import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

############################################################################
####### MAIN ##############################################################
############################################################################

# Paths and filenames
con_file_path = "/Users/federicobonato/Developer/WORK/lid_opm/MEG-EMG-Analysis/Data/dyskensie_opm_data_230625/"
con_file_name = 'pilot_dyst_230625_arm_move.con'
processed_h5_path = "/Users/federicobonato/Developer/WORK/lid_opm/MEG-EMG-Analysis/Data/dyskensie_opm_data_230625/EMG_ACC_data _cutted/opm_healthy_control_data_230625/"
processed_h5_name = 'PTB-01_EmgAcc_setupA_move1_processed.h5'
source_path = '/Users/federicobonato/Developer/WORK/lid_opm/MEG-EMG-Analysis/Data/dyskensie_opm_data_230625/EMG_ACC_data'
source_file_name = 'PTB_01_A_1.2_move.cnt'

# Read the .con file using MNE
con_raw = mne.io.read_raw_kit(con_file_path + con_file_name, preload=True)
con_data = con_raw.get_data()
channel_names = con_raw.ch_names

# Read the processed .h5 file (contains EMG and ACC data)
processed_h5_full_path = processed_h5_path + '/' + processed_h5_name
processed_h5_df = pd.read_hdf(processed_h5_full_path)
processed_h5_channel_names = list(processed_h5_df.columns)
processed_h5_data = processed_h5_df.values.T  # shape: (n_channels, n_times)
processed_h5_sfreq = 1000  # Hz
processed_h5_times = np.arange(processed_h5_data.shape[1]) / processed_h5_sfreq

# Extract trigger channel from MEG (.con) file (channel 157, index 156)
trigger_channel = con_raw.get_data(picks=[156])[0]
sfreq = con_raw.info["sfreq"]  # MEG sampling frequency

# Define EMG and ACC channel mappings for the h5 file
emg_channels = {
    "right_forearm": "BIP11",
    "right_delt": "BIP8",
    "left_forearm": "BIP9",
    "left_delt": "BIP10",
    "right_tibialis_anterior": "BIP7",
    "left_tibialis_anterior": "BIP12",
    "reference": "Olecranon"
}
acc_channels = {
    "right_hand": {
        "y": "BIP1",
        "z": "BIP2",
        "x": "BIP6"
    },
    "left_hand": {
        "x": "BIP3",
        "y": "BIP4",
        "z": "BIP5"
    }
}

# --- Defining MEG channels ---
meg_channel_map = {
    "C3": "MEG001",
    "C4": "MEG002",
    "CZ": "MEG003",
    "F3": "MEG004",
    "F4": "MEG005",
    "P3": "MEG006",
    "P4": "MEG007"
}
meg_channel_indices = list(range(7))
meg_channel_names = [con_raw.ch_names[i] for i in meg_channel_indices]
meg_selected_data = con_raw.get_data(picks=meg_channel_indices)

# --- Filtering MEG channels ---
'''
Filtering process for MEG channels:
1. Band-pass filter from 1 Hz to 100 Hz (zero-phase FIR).
2. Notch filter at 50 Hz to remove line noise.
3. Convert data from Tesla to picoTesla (pT) for easier interpretation.
'''
print("Starting MEG filtering process...")

lpass = 100  # Hz
Hpass = 1    # Hz
con_raw_filtered = con_raw.copy()
data_channels = mne.pick_types(con_raw.info, meg=True, eeg=False, stim=False, eog=False, misc=False)

print(f"Applying band-pass filter: {Hpass} - {lpass} Hz")
con_raw_filtered.filter(l_freq=Hpass, h_freq=lpass, picks=data_channels)

print("Applying 50 Hz notch filter to remove line noise...")
con_raw_filtered.notch_filter(freqs=50, picks=data_channels)

# Convert to pT
print("Extracting filtered MEG data and converting to picoTesla (pT)...")
meg_data = con_raw_filtered.get_data(picks=meg_channel_indices) * 1e12  # Convert to pT

print("MEG filtering process completed.\n")
###################################################################################
## SYNCHRONISING THE FILES (align MEG to first minimum of its trigger channel):
###################################################################################

# Invert the signal to find minima as peaks
inverted_trigger = -trigger_channel
all_minima, properties = find_peaks(inverted_trigger, distance=int(sfreq*0.1))  # at least 100ms apart

# Get the global minimum value
global_min = np.min(trigger_channel)

# Only keep minima that are less than half the global minimum (i.e., deep enough)
threshold = 0.5 * global_min
deep_minima = [idx for idx in all_minima if trigger_channel[idx] < threshold]

if not deep_minima:
    raise RuntimeError("No deep minima found in trigger channel. Adjust threshold or check data.")

# Take the first deep minimum as t=0
t_zero_idx = deep_minima[0]
t_zero = t_zero_idx / sfreq
print(f"First robust trigger minimum at index {t_zero_idx}, time {t_zero:.3f} s, value {trigger_channel[t_zero_idx]:.2f}")

# Align MEG time axis so that t=0 is at the first robust minimum
meg_time_aligned = np.arange(len(trigger_channel)) / sfreq - t_zero

# ...existing code up to t_zero_idx and t_zero...

# --- Trim MEG data and trigger channel to start from t_zero_idx ---
trigger_channel_trimmed = trigger_channel[t_zero_idx:]
meg_data_trimmed = meg_data[:, t_zero_idx:]
meg_time_trimmed = np.arange(len(trigger_channel_trimmed)) / sfreq  # Now t=0 at first robust minimum

# Prepare time axes (use trimmed MEG time for all)
common_time = meg_time_trimmed  # t=0 at first robust minimum

# --- Align EMG and ACC using time, not index ---
# Find the EMG/ACC index closest to t_zero (in their own sampling rate)
emg_start_idx = np.searchsorted(processed_h5_times, t_zero)
emg_end_idx = emg_start_idx + len(common_time)
emg_time_trimmed = processed_h5_times[emg_start_idx:emg_end_idx]

# 1. EMG right forearm (trim to match MEG start)
emg_ch = emg_channels["right_forearm"]
if emg_ch in processed_h5_channel_names:
    emg_idx = processed_h5_channel_names.index(emg_ch)
    emg_right_forearm = processed_h5_data[emg_idx]
    emg_right_forearm_trimmed = emg_right_forearm[emg_start_idx:emg_end_idx]
else:
    raise ValueError(f"EMG channel {emg_ch} not found in processed_h5_channel_names.")

# 2. ACC right hand (all axes, trimmed)
acc_axes = ["y", "z", "x"]
acc_signals_trimmed = []
for axis in acc_axes:
    acc_ch = acc_channels["right_hand"][axis]
    if acc_ch in processed_h5_channel_names:
        acc_idx = processed_h5_channel_names.index(acc_ch)
        acc_signal = processed_h5_data[acc_idx]
        acc_signals_trimmed.append(acc_signal[emg_start_idx:emg_end_idx])
    else:
        acc_signals_trimmed.append(np.zeros(len(common_time)))  # Fallback if missing

################################################
# --- Plotting the EMG, ACC and MEG signals ---
################################################
# --- Zoom to t >= 110s for all signals ---
t_min = 105  # seconds

# Mask for EMG/ACC (emg_time_trimmed) and MEG (common_time)
emg_mask = emg_time_trimmed >= t_min
meg_mask = common_time >= t_min

# Plot all three, sharing x-axis (use common_time for MEG, emg_time_trimmed for EMG/ACC)
fig, axs = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

# EMG right forearm
axs[0].plot(emg_time_trimmed[emg_mask], emg_right_forearm_trimmed[emg_mask], color='tab:blue')
axs[0].set_title("EMG: Right Forearm (BIP11)")
axs[0].set_ylabel("Amplitude")
axs[0].grid(True)

# ACC right hand (all axes)
for i, axis in enumerate(acc_axes):
    axs[1].plot(emg_time_trimmed[emg_mask], acc_signals_trimmed[i][emg_mask], label=axis)
axs[1].set_title("ACC: Right Hand (axes y, z, x)")
axs[1].set_ylabel("Amplitude")
axs[1].legend()
axs[1].grid(True)

# Trimmed MEG C3 channel (index 0)
axs[2].plot(common_time[meg_mask], meg_data_trimmed[0][meg_mask], color='purple')
axs[2].set_title("C3 - Left Motor Cortex")
axs[2].set_xlabel("Time from t_zero (s)")
axs[2].set_ylabel("Magnetiic Field (pT)")
axs[2].grid(True)

plt.tight_layout()
plt.show()
exit()


###################################################################################
###################################################################################
## PLOT THE CHANNELS:
###################################################################################
###################################################################################

# Plot the trigger channel to visualize events
time = np.arange(len(trigger_channel)) / sfreq
plt.figure(figsize=(12, 6))
plt.plot(time, trigger_channel, label="Trigger Channel (157)", color="red")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Trigger Channel Data")
plt.legend()
plt.grid(True)
plt.show()

# Detect trigger events (assuming non-zero values indicate events)
trigger_times = trigger_events / sfreq
print(f"Detected trigger events at times (s): {trigger_times}")

# Plot all channels of the h5 dataset to visually identify the trigger channel
plt.figure(figsize=(15, 10))
offset = 0
for i, ch_name in enumerate(raw_h5_channel_names):
    plt.plot(raw_h5_times, raw_h5_data[i] + offset, label=ch_name)
    offset += np.ptp(raw_h5_data[i]) * 1.2  # Add vertical offset for visibility

plt.xlabel("Time (s)")
plt.ylabel("Amplitude + offset")
plt.title("All channels in h5 file (offset for visibility)")
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0), fontsize='small', ncol=2)
plt.tight_layout()
plt.show()

'''
1- SYNCHRONIZE THE DATAS
2- EXTRACT THE MEG CHANNELS
3- PLOT THE CHANNELS AND COMPARES
'''

# After inspecting the output above, set the correct MEG channel names.
meg_channel_indices = list(range(7))  # indices 0 to 6
meg_channel_names = [con_raw.ch_names[i] for i in meg_channel_indices]

print("Selected MEG channel names:", meg_channel_names)
print("Selected MEG channel indices:", meg_channel_indices)
print('MEG mapped channels:')
for meg, sensor in meg_channel_map.items():
    print(meg, sensor)

# Extract MEG data for these channels
meg_selected_data = con_raw.get_data(picks=meg_channel_indices)

# 1. Extract right_forearm EMG data
emg_ch = emg_channels["right_forearm"]
if emg_ch in raw_h5_channel_names:
    emg_idx = raw_h5_channel_names.index(emg_ch)
    emg_right_forearm = raw_h5_data[emg_idx]
else:
    raise ValueError(f"EMG channel {emg_ch} not found in raw_h5_channel_names.")

# Prepare time axes
emg_time = raw_h5_times
acc_time = raw_h5_times
meg_time = con_raw.times

# Plot
fig, axs = plt.subplots(3, 1, figsize=(15, 10), sharex=False)

# 1. EMG right forearm
axs[0].plot(emg_time, emg_right_forearm, color='tab:blue')
axs[0].set_title("EMG: Right Forearm (BIP11)")
axs[0].set_ylabel("Amplitude")
axs[0].grid(True)

# 2. ACC right_hand (plot all axes: y, z, x)
acc_axes = ["y", "z", "x"]
for i, axis in enumerate(acc_axes):
    acc_ch = acc_channels["right_hand"][axis]
    if acc_ch in raw_h5_channel_names:
        acc_idx = raw_h5_channel_names.index(acc_ch)
        acc_signal = raw_h5_data[acc_idx]
        axs[1].plot(acc_time, acc_signal, label=axis)
    else:
        print(f"ACC channel {acc_ch} not found in raw_h5_channel_names.")
axs[1].set_title("ACC: Right Hand (axes y, z, x)")
axs[1].set_ylabel("Amplitude")
axs[1].legend()
axs[1].grid(True)

# 3. MEG C4
meg_c4 = meg_selected_data[1]  # C4 is the second channel
axs[2].plot(meg_time, meg_c4, color='tab:orange')
axs[2].set_title("MEG: C4 (MEG002)")
axs[2].set_xlabel("Time (s)")
axs[2].set_ylabel("Amplitude")

plt.show()