#################
### LIBRARIES ###
#################
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from mne.filter import filter_data
from mne.time_frequency import Spectrum
from source.ssp_function import apply_ssp_from_baseline
from source.plot_functions import (
    calculate_individual_power_spectra,
    plot_all_channel_power_spectra,
    plot_meg_spectrogram,
    plot_component_comparison,
    plot_meg_3x3_grid
)

#######################################################
#################
### FUNCTIONS ###
#################
#######################################################

# Function that calculates the norm of each channel;

def calculate_channel_norms(X_channels, Y_channels, Z_channels):
    """
    Calculate the Euclidean norm for each sensor using its X, Y, Z components.
    
    Args:
        X_channels: List of X component arrays for each sensor
        Y_channels: List of Y component arrays for each sensor
        Z_channels: List of Z component arrays for each sensor
    
    Returns:
        List of norm arrays for each sensor
    """
    norms = []
    n_channels = len(X_channels)
    
    for i in range(n_channels):
        # Calculate norm for each time point: sqrt(x² + y² + z²)
        norm = np.sqrt(
            X_channels[i]**2 + 
            Y_channels[i]**2 + 
            Z_channels[i]**2
        )
        norms.append(norm)
    
    return norms

# Function to calculate the power spectrum of a signal:

def calculate_power_spectrum(signal):
    """Calculate FFT and power spectrum of a signal"""
    # Apply FFT
    n = len(signal)
    fft = np.fft.fft(signal) / n  #Normalization
    freqs = np.fft.fftfreq(n, d=0.002667)  # 375 Hz sampling rate
    power = np.abs(fft) ** 2  # Power calculation
    
    # Get positive frequencies only and double the values to compensate for removing negatives
    pos_mask = freqs >= 0
    return freqs[pos_mask], 2 * power[pos_mask]

# Function for applying the filters to the data:

def apply_meg_filters(data, sfreq=375):
    """
    Apply bandpass and notch filters to MEG data using MNE.
    
    Args:
        data: Input MEG signal
        sfreq: Sampling frequency (Hz)
    Returns:
        filtered_data: Filtered MEG signal
    """
    # Bandpass filter (1-100 Hz)
    data_bandpass = filter_data(data, sfreq=sfreq, l_freq=3, h_freq=100, 
                              method='fir', verbose=False)
        
    # Apply notch filters (50 Hz and harmonics)
    filtered_data = data_bandpass
    for freq in [50, 100, 150]:
        filtered_data = mne.filter.notch_filter(
            filtered_data, 
            Fs=sfreq,  
            freqs=freq,
            verbose=False
        )
    

    return filtered_data


def create_meg_raw(channels, ch_names, sfreq=375):
    """
    Create an MNE RawArray from MEG channels.
    channels: list of arrays (n_channels, n_times)
    ch_names: list of channel names
    sfreq: sampling frequency
    Returns: MNE RawArray
    """
    n_channels = len(channels)
    ch_types = ['mag'] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    # Explicitly set the unit to Tesla for each channel
    data = np.array(channels)
    return mne.io.RawArray(data, info, verbose=False)


#################
### MAIN CODE ###
#################

#######################################################
#######################################################


# READING THE FILE AND DATAFRAME CREATION:

# Get the directory where this script is located
base_dir = os.path.dirname(os.path.abspath(__file__))

# Go up one directory and then into Data
data_dir = os.path.join(base_dir, '..', 'Data')

file_path_1 = os.path.join(data_dir, "plfp65_rec11_13.11.2024_14-18-30_array1.lvm")
file_path_11 = os.path.join(data_dir, "plfp65_rec7_13.11.2024_13-42-47_array1.lvm")
ssp_baseline_file = os.path.join(data_dir, "adxl_mov_sensor__12.12.2024_12-07-05_array1.lvm")


df_start = pd.read_csv(file_path_1, header= 22, sep='\t')
df_last = pd.read_csv(file_path_11, header=22, sep='\t')
df_baseline = pd.read_csv(ssp_baseline_file, header=22, sep=',')

# Extract recording names from file paths
rec1_name = file_path_1.split('/')[-1].split('_')[0:2]  # ['plfp65', 'rec3']
rec11_name = file_path_11.split('/')[-1].split('_')[0:2]  # ['plfp65', 'rec11']
rec1_label = '_'.join(rec1_name)  # 'plfp65_rec3'
rec11_label = '_'.join(rec11_name) 

# It seems that the column "Comment" is composed by Nan values, so I decide to remove it from the frame
df_start = df_start.drop(columns=["Comment"])
df_last = df_last.drop(columns=["Comment"])

# ORGANIZING DATAS
X_channels_names = [col for col in df_start.columns if "X" in col]
Y_channels_names = [col for col in df_start.columns if "Y" in col]
Z_channels_names = [col for col in df_start.columns if "Z" in col]

X_extras = ['X_Value', 'MUX_Counter1', 'MUX_Counter2']
X_channels_names = [col for col in X_channels_names if col not in X_extras]



# Extracting the channels from start and last files:

X_channels_start = [df_start[col].values for col in X_channels_names]
Y_channels_start = [df_start[col].values for col in Y_channels_names]
Z_channels_start = [df_start[col].values for col in Z_channels_names]

X_channels_last = [df_last[col].values for col in X_channels_names]
Y_channels_last = [df_last[col].values for col in Y_channels_names]
Z_channels_last = [df_last[col].values for col in Z_channels_names]

# Excluding channels filled with zeros in both files:
X_channels_start = X_channels_start[:20]
Y_channels_start = Y_channels_start[:20]
Z_channels_start = Z_channels_start[:20]

X_channels_last = X_channels_last[:20]
Y_channels_last = Y_channels_last[:20]
Z_channels_last = Z_channels_last[:20]

X_channels_names = X_channels_names[:20]  # Also trim the names to match

######################################################
# Create MNE RawArray objects for both recordings

# Convert start recording channels to pT
X_channels_start_raw = [channel * 1e-12 for channel in X_channels_start]
Y_channels_start_raw = [channel * 1e-12 for channel in Y_channels_start]
Z_channels_start_raw = [channel * 1e-12 for channel in Z_channels_start]

# Convert last recording channels to pT
X_channels_last_raw = [channel * 1e-12 for channel in X_channels_last]
Y_channels_last_raw = [channel * 1e-12 for channel in Y_channels_last]
Z_channels_last_raw = [channel * 1e-12 for channel in Z_channels_last]

norms_start_raw = calculate_channel_norms(X_channels_start_raw, Y_channels_start_raw, Z_channels_start_raw)
norms_last_raw = calculate_channel_norms(X_channels_last_raw, Y_channels_last_raw, Z_channels_last_raw)
X_channels_names_raw = X_channels_names[:20] 
Y_channels_names_raw = Y_channels_names[:20] 
Z_channels_names_raw = Z_channels_names[:20] 

######################################################

# Exclude channels 5, 6, 13, and 20 from all components
channels_to_exclude = [4, 5, 12, 19]
print("\nExcluding channels 5, 6, 13, and 20 from all components...")

def remove_channels(channel_list, indices):
    return [channel for i, channel in enumerate(channel_list) if i not in indices] 

# Remove channels from start recording
X_channels_start = remove_channels(X_channels_start, channels_to_exclude)
Y_channels_start = remove_channels(Y_channels_start, channels_to_exclude)
Z_channels_start = remove_channels(Z_channels_start, channels_to_exclude)

# Remove channels from last recording
X_channels_last = remove_channels(X_channels_last, channels_to_exclude)
Y_channels_last = remove_channels(Y_channels_last, channels_to_exclude)
Z_channels_last = remove_channels(Z_channels_last, channels_to_exclude)

# Remove channel names
X_channels_names = remove_channels(X_channels_names, channels_to_exclude)

print(f'After excluding channels {channels_to_exclude}')
print('we are considering: ', len(X_channels_names), ' channels for each component')

# Create list of channel numbers (excluding 5, 6, 13, and 20)
channel_numbers = [i+1 for i in range(20) if i not in channels_to_exclude]

# Convert MEG data to picoTesla after extracting channels
print("\nConverting MEG data to picoTesla...")

# Convert start recording channels to pT
X_channels_start = [channel * 1e-12 for channel in X_channels_start]
Y_channels_start = [channel * 1e-12 for channel in Y_channels_start]
Z_channels_start = [channel * 1e-12 for channel in Z_channels_start]

# Convert last recording channels to pT
X_channels_last = [channel * 1e-12 for channel in X_channels_last]
Y_channels_last = [channel * 1e-12 for channel in Y_channels_last]
Z_channels_last = [channel * 1e-12 for channel in Z_channels_last]

print('Data converted to picoTesla')

################################ Filter the data using MNE:
print("\nApplying filters to MEG data...")
print("\nBandpass filter (1-100 Hz)")
print("\nApply notch filters (50 Hz and harmonics)")


# Filter start recording channels
X_channels_start = [apply_meg_filters(channel) for channel in X_channels_start]
Y_channels_start = [apply_meg_filters(channel) for channel in Y_channels_start]
Z_channels_start = [apply_meg_filters(channel) for channel in Z_channels_start]

# Filter last recording channels
X_channels_last = [apply_meg_filters(channel) for channel in X_channels_last]
Y_channels_last = [apply_meg_filters(channel) for channel in Y_channels_last]
Z_channels_last = [apply_meg_filters(channel) for channel in Z_channels_last]

# Normalize the filtered data:
norms_start = calculate_channel_norms(X_channels_start, Y_channels_start, Z_channels_start)
norms_last = calculate_channel_norms(X_channels_last, Y_channels_last, Z_channels_last)

print("Filtering complete")


# Pick a sensor
time_start = df_start["X_Value"]
time_last = df_last["X_Value"]

#########################################################################
#################
###  PLOTS ###
#################
#########################################################################
print("\n=== Plotting MEG Raw and Normalized Data ===")

# First figure for rec_1
fig_raw, axes_raw = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Create colormap for MEG channels
n_raw = len(X_channels_names_raw)
n_channels = n_raw +1
colors = plt.cm.rainbow(np.linspace(0, 1, n_channels))

# 1. First subplot: MEG X Component (rec1_label)
for i, channel in enumerate(X_channels_start_raw):
    axes_raw[0].plot(time_start, channel, 
                     color=colors[i], linewidth=0.6, 
                     label=f'Channel {X_channels_names_raw[i]}')
axes_raw[0].set_title(f'MEG X Component - {rec1_label} - Raw')
axes_raw[0].set_ylabel('Amplitude (pT)')
axes_raw[0].grid(True, alpha=0.3)
axes_raw[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

# 2. Second subplot: MEG normalized channels
for i, norm in enumerate(norms_start_raw):
    axes_raw[1].plot(time_start, norm, 
                     color=colors[i], linewidth=0.6,
                     label=f'Channel {X_channels_names_raw[i]}')
axes_raw[1].set_title(f'MEG Vector Norm - {rec1_label} - Raw')
axes_raw[1].set_ylabel('Magnitude')
axes_raw[1].grid(True, alpha=0.3)
axes_raw[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

plt.tight_layout()
plt.subplots_adjust(top=0.95)

# Second figure for rec_11
fig_raw, axes_raw = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# 1. First subplot: MEG X Component (rec11_label)
for i, channel in enumerate(X_channels_last_raw):
    axes_raw[0].plot(time_last, channel, 
                     color=colors[i], linewidth=0.6, 
                     label=f'Channel {X_channels_names_raw[i]}')
axes_raw[0].set_title(f'MEG X Component - {rec11_label} - Raw')
axes_raw[0].set_ylabel('Amplitude (pT)')
axes_raw[0].grid(True, alpha=0.3)
axes_raw[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

# 2. Second subplot: MEG normalized channels
for i, norm in enumerate(norms_last_raw):  # Remove zip() here
    axes_raw[1].plot(time_last, norm, 
                     color=colors[i], linewidth=0.6,
                     label=f'Ch{X_channels_names_raw[i]}')
axes_raw[1].set_title(f'MEG Vector Norm - {rec11_label} - Raw')
axes_raw[1].set_ylabel('Magnitude')
axes_raw[1].grid(True, alpha=0.3)
axes_raw[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.show()
##########################################################
##########################################################

print("\n=== Filtering ===")
print("\n=== Plotting FILTERED MEG and Normalized Data ===")

# Create list of channel numbers (excluding 5 and 13)
channel_numbers = [i+1 for i in range(20) if i not in channels_to_exclude]

# First figure for rec_1
fig_raw, axes_raw = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Create colormap for MEG channels
n_channels = len(X_channels_names)
colors = plt.cm.rainbow(np.linspace(0, 1, n_channels))

# 1. First subplot: MEG X Component (rec1_label)
for i, channel in enumerate(X_channels_start):
    axes_raw[0].plot(time_start, channel, 
                     color=colors[i], linewidth=0.6, 
                     label=f'Channel {channel_numbers[i]}')
axes_raw[0].set_title(f'MEG X Component - {rec1_label} - Filtered')
axes_raw[0].set_ylabel('Amplitude (pT)')
axes_raw[0].grid(True, alpha=0.3)
axes_raw[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

# 2. Second subplot: MEG normalized channels
for i, norm in enumerate(norms_start):
    axes_raw[1].plot(time_start, norm, 
                     color=colors[i], linewidth=0.6,
                     label=f'Channel {channel_numbers[i]}')
axes_raw[1].set_title(f'MEG Vector Norm - {rec1_label} - Filtered')
axes_raw[1].set_ylabel('Magnitude')
axes_raw[1].grid(True, alpha=0.3)
axes_raw[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

plt.tight_layout()
plt.subplots_adjust(top=0.95)

# Second figure for rec_11
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# 1. First subplot: MEG X Component (rec11_label)
for i, channel in enumerate(X_channels_last):
    axes[0].plot(time_last, channel, 
                     color=colors[i], linewidth=0.6, 
                     label=f'Channel {X_channels_names[i]}')
axes[0].set_title(f'MEG X Component - {rec11_label} - Filtered')
axes[0].set_ylabel('Amplitude (pT)')
axes[0].grid(True, alpha=0.3)
axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

# 2. Second subplot: MEG normalized channels
for i, norm in enumerate(norms_last):
    axes[1].plot(time_last, norm, 
                     color=colors[i], linewidth=0.6,
                     label=f'Ch{X_channels_names[i]}')
axes[1].set_title(f'MEG Vector Norm - {rec11_label} - Filtered')
axes[1].set_ylabel('Magnitude')
axes[1].grid(True, alpha=0.3)
axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.show()

##########################################################
##########################################################
'''
# APPLICATION OF SSP METHOD TO DENOISE THE DATA
print("\n=== Applying SSP Method to Denoise the Data ===")

#   We need a baseline recording to compute the projectors.
#   We will use the empty room recording for this purpose.
#   I've selected this time range because it contains no significant activity.

# Select baseline values between 230 and 350 seconds
baseline_time = df_baseline["X_Value"]
baseline_mask = (baseline_time >= 230) & (baseline_time <= 350)
df_baseline_selected = df_baseline[baseline_mask].reset_index(drop=True)

# Prepare baseline Raw object (using the same channel names as your MEG data)
X_baseline = [df_baseline_selected[col].values for col in X_channels_names]
Y_baseline = [df_baseline_selected[col].values for col in Y_channels_names]
Z_baseline = [df_baseline_selected[col].values for col in Z_channels_names]

# Concatenate all baseline channels (X, Y, Z) and names
baseline_channels = X_baseline + Y_baseline + Z_baseline
baseline_ch_names = X_channels_names + Y_channels_names + Z_channels_names

raw_baseline = create_meg_raw(baseline_channels, baseline_ch_names, sfreq=375)
# Prepare MEG Raw object for the start recording (already in pT)
X_meg = X_channels_start_raw
Y_meg = Y_channels_start_raw
Z_meg = Z_channels_start_raw
meg_channels = X_meg + Y_meg + Z_meg
meg_ch_names = X_channels_names_raw[:20] + Y_channels_names_raw[:20] + Z_channels_names_raw[:20]

raw_meg = create_meg_raw(meg_channels, meg_ch_names, sfreq=375)

raw_denoised = apply_ssp_from_baseline(raw_baseline, raw_meg)
denoised_data = raw_denoised.get_data()  # shape: (n_channels, n_times)
# Split back into X, Y, Z lists if needed
n = len(X_channels_names)
X_denoised = [denoised_data[i] for i in range(n)]
Y_denoised = [denoised_data[i+n] for i in range(n)]
Z_denoised = [denoised_data[i+2*n] for i in range(n)]

print("\n=== Plotting DENOISED MEG X and Normalized Channels ===")

# Calculate norms for denoised data
norms_denoised = calculate_channel_norms(X_denoised, Y_denoised, Z_denoised)

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# 1. First subplot: Denoised MEG X Component
for i, channel in enumerate(X_denoised):
    axes[0].plot(time_start, channel, 
                 color=colors[i], linewidth=0.6, 
                 label=f'Channel {X_channels_names_raw[i]}')
axes[0].set_title(f'MEG X Component - {rec1_label} - Denoised (SSP)')
axes[0].set_ylabel('Amplitude (pT)')
axes[0].grid(True, alpha=0.3)
axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

# 2. Second subplot: Denoised MEG normalized channels
for i, norm in enumerate(norms_denoised):
    axes[1].plot(time_start, norm, 
                 color=colors[i], linewidth=0.6,
                 label=f'Channel {X_channels_names_raw[i]}')
axes[1].set_title(f'MEG Vector Norm - {rec1_label} - Denoised (SSP)')
axes[1].set_ylabel('Magnitude')
axes[1].grid(True, alpha=0.3)
axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.show()
'''
##########################################################
##########################################################

print("\n=== Plotting Power Spectra for Normalized MEG Channels ===")

# Plot power spectra for first recording (Vector Norm)
plot_all_channel_power_spectra(
    norms_start, 
    channel_numbers, 
    f'Vector Norm - {rec1_label}'
)

# Plot power spectra for second recording (Vector Norm)
plot_all_channel_power_spectra(
    norms_last, 
    channel_numbers, 
    f'Vector Norm - {rec11_label}'
)

##########################################################


# --- Select channels 3, 9, 19 for RAW data (indices 2, 8, 18) ---
selected_indices_raw = [2, 8, 18]
X_channels_start_raw_selected = [X_channels_start_raw[i] for i in selected_indices_raw]
X_channels_last_raw_selected = [X_channels_last_raw[i] for i in selected_indices_raw]
channel_labels_raw = [f"Ch {i+1}" for i in selected_indices_raw]

# --- Select channels 3, 9, 19 for FILTERED data (handle exclusion) ---
wanted_channels = [3, 9, 19]
selected_indices_filtered = [channel_numbers.index(ch) for ch in wanted_channels]
X_channels_start_selected = [X_channels_start[i] for i in selected_indices_filtered]
X_channels_last_selected = [X_channels_last[i] for i in selected_indices_filtered]
channel_labels_filtered = [f"Ch {ch}" for ch in wanted_channels]

# --- Define time frames as boolean masks ---
# For rec1 (start)
t_1_start = (time_start >= 0) & (time_start < 10)
t_2_start = (time_start >= 100) & (time_start < 110)
t_3_start = (time_start >= 280) & (time_start < 290)
time_windows_start = [t_1_start, t_2_start, t_3_start]

# For rec11 (last)
t_1_last = (time_last >= 0) & (time_last < 10)
t_2_last = (time_last >= 100) & (time_last < 110)
t_3_last = (time_last >= 280) & (time_last < 290)
time_windows_last = [t_1_last, t_2_last, t_3_last]
time_labels = ["0-10 s", "100-110 s", "280-290 s"]

# Use the same colormap as before
colors = plt.cm.rainbow(np.linspace(0, 1, 3))  # 3 channels

# For rec1 raw
plot_meg_3x3_grid(
    X_channels_start_raw_selected,
    np.array(time_start),
    time_windows_start,
    channel_labels_raw,
    time_labels,
    colors,
    f"Raw MEG X Channels - {rec1_label} - Selected Channels and Time Windows"
)

# For rec1 filtered
plot_meg_3x3_grid(
    X_channels_start_selected,
    np.array(time_start),
    time_windows_start,
    channel_labels_filtered,
    time_labels,
    colors,
    f"Filtered MEG X Channels - {rec1_label} - Selected Channels and Time Windows"
)

# For rec11 raw
plot_meg_3x3_grid(
    X_channels_last_raw_selected,
    np.array(time_last),
    time_windows_last,
    channel_labels_raw,
    time_labels,
    colors,
    f"Raw MEG X Channels - {rec11_label} - Selected Channels and Time Windows"
)

# For rec11 filtered
plot_meg_3x3_grid(
    X_channels_last_selected,
    np.array(time_last),
    time_windows_last,
    channel_labels_filtered,
    time_labels,
    colors,
    f"Filtered MEG X Channels - {rec11_label} - Selected Channels and Time Windows"
)

# For normalized vector norm (example for rec1)
# Compute vector norm for each selected filtered channel
norms_start_selected = [
    np.sqrt(
        X_channels_start_selected[i]**2 +
        Y_channels_start[selected_indices_filtered[i]]**2 +
        Z_channels_start[selected_indices_filtered[i]]**2
    )
    for i in range(3)
]
plot_meg_3x3_grid(
    norms_start_selected,
    np.array(time_start),
    time_windows_start,
    channel_labels_filtered,
    time_labels,
    colors,
    f" MEG Vector Norm - {rec1_label} - Selected Channels and Time Windows"
)

# Compute vector norm for each selected filtered channel for rec11
norms_last_selected = [
    np.sqrt(
        X_channels_last_selected[i]**2 +
        Y_channels_last[selected_indices_filtered[i]]**2 +
        Z_channels_last[selected_indices_filtered[i]]**2
    )
    for i in range(3)
]

plot_meg_3x3_grid(
    norms_last_selected,
    np.array(time_last),
    time_windows_last,
    channel_labels_filtered,
    time_labels,
    colors,
    f"MEG Vector Norm - {rec11_label} - Selected Channels and Time Windows"
)

##########################################################################
# Print sample counts per window:

print("rec1 (start) sample counts per window:")
for i, mask in enumerate(time_windows_start):
    print(f"  {time_labels[i]}: {np.sum(mask)} samples")
for i, mask in enumerate(time_windows_start):
    print(f"rec1 {time_labels[i]}: time range = {time_start[mask].min()} to {time_start[mask].max()}, samples = {np.sum(mask)}")

print("rec11 (last) sample counts per window:")
for i, mask in enumerate(time_windows_last):
    print(f"  {time_labels[i]}: {np.sum(mask)} samples")


for i, mask in enumerate(time_windows_last):
    print(f"rec11 {time_labels[i]}: time range = {time_last[mask].min()} to {time_last[mask].max()}, samples = {np.sum(mask)}")