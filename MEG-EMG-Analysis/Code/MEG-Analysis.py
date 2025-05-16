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
# Add to the top of MEG-Analysis.py
from source.plot_functions import (
    calculate_individual_power_spectra,
    plot_all_channel_power_spectra,
    plot_meg_spectrogram,
    plot_component_comparison,
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
    data_bandpass = filter_data(data, sfreq=sfreq, l_freq=1, h_freq=100, 
                              method='fir', verbose=False)
        
    # Apply notch filters (50 Hz and harmonics)
    filtered_data = data_bandpass
    for freq in [50, 100, 150]:
        filtered_data = mne.filter.notch_filter(
            filtered_data, 
            Fs=sfreq,  # Changed from sfreq to Fs
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

file_path_1 = os.path.join(data_dir, "plfp65_rec1_13.11.2024_12-51-13_array1.lvm")
file_path_11 = os.path.join(data_dir, "plfp65_rec11_13.11.2024_14-18-30_array1.lvm")


df_start = pd.read_csv(file_path_1, header= 22, sep='\t')
df_last = pd.read_csv(file_path_11, header=22, sep='\t')

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



# Extracting the channels from startting and last files:

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