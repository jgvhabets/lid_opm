'''
I need to fix and adapt the power spectrum plotting function in the MEG analysis code,
in order to make it compatible with any component data.
'''


############################################################################
####### LIBRARIES ##########################################################
############################################################################

import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy.signal import find_peaks
from source.plot_functions import (plot_channels_comparison,
                                   plot_meg_2x3_grid,
                                   plot_ica_components,
                                   plot_ica_max_amplitudes,
                                   plot_single_ica_power_spectrum,
                                   plot_ica_power_spectra_grid,
                                   )
from source.MEG_analysis_functions import (
    apply_fastica_to_channels,             
)
from source.find_paths import (get_onedrive_path,
                               get_available_subs,)

############################################################################
####### MAIN ##############################################################
############################################################################

# READING THE FILE AND DATAFRAME CREATION:

# Use the find_paths functions instead of hardcoded paths
try:
    # Try to get the data path using the OneDrive function
    data_dir = get_onedrive_path('data')  
    print(f"Data directory found: {data_dir}")
except:
    # Fallback to the original method if OneDrive path doesn't work
    print("OneDrive path not found, using fallback method...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, '..', 'Data')

source_data_path = get_onedrive_path('source_data')
processed_data_path = get_onedrive_path('processed_data')

available_subs_source = get_available_subs('data', source_data_path)
available_subs_processed = get_available_subs('data', processed_data_path)

print(f'availabe subjects in {source_data_path}', available_subs_source)
print(f'availabe subjects in {processed_data_path}', available_subs_processed)
# ------ Define the subject for this analysis ------
SUB = 'sub-91'  # dataset from June 2025

# create new source_data subject directory
sub_source_data_dir = os.path.join(source_data_path, SUB)

if not os.path.exists(sub_source_data_dir):
    os.makedirs(sub_source_data_dir)

# Paths and filenames - now using dynamic paths
con_file_path = os.path.join(sub_source_data_dir, "OPM_data/")
con_file_name = 'pilot_dyst_230625_arm_move.con'
processed_h5_path = os.path.join(processed_data_path, "EMG_ACC_data/opm_healthy_control_data_230625/")
processed_h5_name = 'PTB-01_EmgAcc_setupA_move1_processed.h5'


# Read the .con file using MNE
con_raw = mne.io.read_raw_kit(con_file_path + con_file_name, preload=True)
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
MEG_sfreq = con_raw.info["sfreq"]  # MEG sampling frequency

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

print("Applying 100 Hz notch filter to remove line noise...")
con_raw_filtered.notch_filter(freqs=lpass, picks=data_channels)

# Convert to pT
print("Extracting filtered MEG data and converting to picoTesla (pT)...")
meg_filtered_data = con_raw_filtered.get_data(picks=meg_channel_indices) * 1e12  # Convert to pT

print("MEG filtering process completed.\n")

###################################################################################
# extract and keep RAW MEG data for further processing

# Raw MEG data (convert to pT for fair comparison)
meg_raw_data = con_raw.get_data(picks=meg_channel_indices) * 1e12  # shape: (7, n_times)
meg_time = con_raw.times


###################################################################################
## SYNCHRONISING THE FILES (align MEG to first minimum of its trigger channel):
###################################################################################

# find minima as peaks
all_minima, properties = find_peaks(trigger_channel, distance=int(MEG_sfreq*0.1))

# Get the global minimum value
global_min = np.min(trigger_channel)

# Only keep minima that are less than half the global minimum (i.e., deep enough)
threshold = 0.5 * global_min
deep_minima = [idx for idx in all_minima if trigger_channel[idx] < threshold]

if not deep_minima:
    raise RuntimeError("No deep minima found in trigger channel. Adjust threshold or check data.")

# Take the first deep minimum as t=0
t_zero_idx = deep_minima[0]
t_zero = t_zero_idx / MEG_sfreq
print(f"First robust trigger minimum at index {t_zero_idx}, time {t_zero:.3f} s, value {trigger_channel[t_zero_idx]:.2f}")


# --- Trim MEG data and trigger channel to start from t_zero_idx ---
trigger_channel_trimmed = trigger_channel[t_zero_idx:]
meg_data_trimmed = meg_filtered_data[:, t_zero_idx:]
meg_time_trimmed = np.arange(len(trigger_channel_trimmed)) / MEG_sfreq  # Creates a new time axis starting from 0 seconds

# Prepare time axes (use trimmed MEG time for all)
common_time = meg_time_trimmed  # t=0 at first robust minimum

# --- Align EMG and ACC using time, not index ---
# Find the EMG/ACC index closest to t_zero (in their own sampling rate)
emg_start_idx = np.searchsorted(processed_h5_times, t_zero)
emg_end_idx = emg_start_idx + len(common_time)
emg_time_trimmed = processed_h5_times[emg_start_idx:emg_end_idx]

# Multiply EMG channels by 1e6 (convert to µV) at extraction
emg_trimmed_dict = {}
for emg_label, emg_ch in emg_channels.items():
    if emg_ch in processed_h5_channel_names:
        emg_idx = processed_h5_channel_names.index(emg_ch)
        emg_signal = processed_h5_data[emg_idx]
        emg_trimmed_dict[emg_label] = emg_signal[emg_start_idx:emg_end_idx] * 1e6  # Convert to µV
    else:
        emg_trimmed_dict[emg_label] = np.zeros(len(common_time))  # Fallback if missing


# 1. EMG right forearm (already trimmed and converted to µV)
emg_right_forearm_trimmed = emg_trimmed_dict["right_forearm"]

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


###################################################################################
## ICA ANALYSIS ON MEG DATA:
###################################################################################

# Apply FastICA to the filtered MEG data (shape: 7, n_times)
n_components = meg_filtered_data.shape[0]  # 7
ica_signals, ica_model = apply_fastica_to_channels(meg_filtered_data, n_components=n_components)

###################################################################################
###################################################################################
## PLOT THE CHANNELS:
###################################################################################
###################################################################################

# --- Plotting the EMG, ACC and MEG signals ---
# --- Zoom to t >= 105s for all signals ---
t_min = 105  # seconds

# Mask for EMG/ACC (emg_time_trimmed) and MEG (common_time)
emg_mask = emg_time_trimmed >= t_min
meg_mask = common_time >= t_min

# Plot all three, sharing x-axis (use common_time for MEG, emg_time_trimmed for EMG/ACC)
fig, axs = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

# EMG right forearm
axs[0].plot(emg_time_trimmed[emg_mask], emg_right_forearm_trimmed[emg_mask], color='tab:blue')
axs[0].set_title("EMG: Right Forearm (BIP11)")
axs[0].set_ylabel("Amplitude (µV)")
axs[0].grid(True)

# ACC right hand (all axes)
for i, axis in enumerate(acc_axes):
    axs[1].plot(emg_time_trimmed[emg_mask], acc_signals_trimmed[i][emg_mask], label=axis)
axs[1].set_title("ACC: Right Hand (axes y, z, x)")
axs[1].set_ylabel("Acceleration (g)")
axs[1].legend()
axs[1].grid(True)

# Trimmed MEG C3 channel (index 0)
axs[2].plot(common_time[meg_mask], meg_data_trimmed[0][meg_mask], color='purple')
axs[2].set_title("C3 - Left Motor Cortex")
axs[2].set_xlabel(f"Time from t = {t_min} s")
axs[2].set_ylabel("Magnetiic Field (pT)")
axs[2].grid(True)

plt.tight_layout()
plt.show()
##############################################################################
# --- Plotting comparison of MEG raw vs filtered channels ---

# Use the same channel names for both
channel_labels = list(meg_channel_map.keys())
colors = plt.cm.rainbow(np.linspace(0, 1, len(channel_labels)))

# Plot comparison
plot_channels_comparison(
    meg_time,
    meg_raw_data,
    meg_filtered_data,
    channel_labels,
    channel_labels,
    colors,
    rec_label="Raw vs Filtered MEG",
    y_label="Amplitude (pT)",
    axis_label="All"
)
###########################################################################
# --- Plotting 2x3 GRID comparison of MEG raw vs filtered channels ---

# Define time frames as boolean masks for the grid plot
MEG_time_start = meg_time  # Use the full MEG time axis for raw/filtered comparison

t_1_start = (MEG_time_start >= 50) & (MEG_time_start < 65)
t_2_start = (MEG_time_start >= 100) & (MEG_time_start < 115)
t_3_start = (MEG_time_start >= 185) & (MEG_time_start < 200)
time_windows_start = [t_1_start, t_2_start, t_3_start]
time_labels = ["50-65 s", "100-115 s", "185-200 s"]

plot_meg_2x3_grid(
    meg_raw_data,
    meg_filtered_data,
    MEG_time_start,
    time_windows_start,
    channel_labels,
    channel_labels,
    time_labels,
    colors,
    colors,
    "Raw MEG",
    "Filtered MEG",
    "Raw vs Filtered MEG - All Channels"
)

###########################################################################
# --- Plotting ICA components ---

# Plot ICA components (all 7)
plot_ica_components(ica_signals, meg_time, 'All', 'Filtered MEG')

# Plot max amplitude of ICA components
plot_ica_max_amplitudes(ica_signals, title='Max Amplitude of ICA Components - Filtered MEG')

# Plot power spectra of ICA components
component_names = [f"C{i+1}" for i in range(ica_signals.shape[0])]
plot_ica_power_spectra_grid(
    ica_signals,
    lambda component, ax: plot_single_ica_power_spectrum(component, ax, sfreq=MEG_sfreq),
    component_names=component_names,
    title="ICA Components Power Spectra - Filtered MEG",
)


#############################################################################
'''
ICA Artifact Removal:
Based on the max amplitude analysis of the ICA components, we are excluding components C1 and C4 (indices 0 and 3).
These components showed the highest amplitudes and are likely to represent artifacts.
They are set to zero before reconstructing the cleaned MEG data.
'''
'''
# --- Remove ICA components C1 and C4 (indices 0 and 3) ---
ica_signals_clean = ica_signals.copy()
ica_signals_clean[0, :] = 0  # Remove C1
ica_signals_clean[3, :] = 0  # Remove C4

# --- Reconstruct cleaned MEG data ---
meg_data_ica_cleaned = ica_model.inverse_transform(ica_signals_clean.T).T  # shape: (7, n_times)
'''
