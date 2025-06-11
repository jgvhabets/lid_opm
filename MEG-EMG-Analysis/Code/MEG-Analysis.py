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
from sklearn.decomposition import FastICA
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
    data_bandpass = filter_data(data, sfreq=sfreq, l_freq=1, h_freq=100, 
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

def apply_fastica_to_channels(channels, n_components=None, random_state=0, max_iter=1000):
    """
    Apply FastICA to a list of MEG channel arrays (shape: n_channels x n_times).
    
    Args:
        channels: list or np.ndarray, shape (n_channels, n_times)
        n_components: int or None, number of ICA components (default: n_channels)
        random_state: int, random seed
        max_iter: int, max iterations for FastICA
        
    Returns:
        ica_signals: np.ndarray, shape (n_channels, n_times), ICA components
        ica_model: fitted FastICA object
    """
    data = np.array(channels)
    if n_components is None:
        n_components = data.shape[0]
    ica = FastICA(n_components=n_components, random_state=random_state, max_iter=max_iter)
    ica_signals = ica.fit_transform(data.T).T  # Transpose to (n_times, n_channels), then back
    return ica_signals, ica

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
######################################################

# Exclude channels 5, 6, 13, and 20 from all components
channels_to_exclude = [4, 5, 12, 19]
print("\nExcluding channels 5, 6, 13, and 20 from all components...")

def remove_channels(channel_list, indices):
    return [channel for i, channel in enumerate(channel_list) if i not in indices] 

# Remove channels from start recording
X_channels_start = remove_channels(X_channels_start_raw, channels_to_exclude)
Y_channels_start = remove_channels(Y_channels_start_raw, channels_to_exclude)
Z_channels_start = remove_channels(Z_channels_start_raw, channels_to_exclude)

# Remove channels from last recording
X_channels_last = remove_channels(X_channels_last_raw, channels_to_exclude)
Y_channels_last = remove_channels(Y_channels_last_raw, channels_to_exclude)
Z_channels_last = remove_channels(Z_channels_last_raw, channels_to_exclude)

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

######################################################
#As a part of pre-processing, we will apply ICA (indipendent component analysis) to the baseline data
# --- ICA on filtered data ---

# Apply FastICA to all filtered components (example for 'last' dataset)
X_ica_last, ica_X = apply_fastica_to_channels(X_channels_last)
Y_ica_last, ica_Y = apply_fastica_to_channels(Y_channels_last)
Z_ica_last, ica_Z = apply_fastica_to_channels(Z_channels_last)

# For the 'start' dataset:
X_ica_start, ica_X_start = apply_fastica_to_channels(X_channels_start)
Y_ica_start, ica_Y_start = apply_fastica_to_channels(Y_channels_start)
Z_ica_start, ica_Z_start = apply_fastica_to_channels(Z_channels_start)

######################################################
## Let's consider a smaller sample of channels for plotting
## In this way it's easier to visualize the data

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

# Norm for selected channels
norms_start_raw_selected = [
    np.sqrt(
        X_channels_start_raw_selected[i]**2 +
        Y_channels_start_raw[selected_indices_raw[i]]**2 +
        Z_channels_start_raw[selected_indices_raw[i]]**2
    )
    for i in range(3)
]

norms_last_raw_selected = [
    np.sqrt(
        X_channels_last_raw_selected[i]**2 +
        Y_channels_last_raw[selected_indices_raw[i]]**2 +
        Z_channels_last_raw[selected_indices_raw[i]]**2
    )
    for i in range(3)
]

#########################################################################
#################
###  PLOTS ###
#################
#########################################################################



print("\n=== Plotting MEG Raw and Normalized Data ===")

# Create colormap for MEG channels
n_raw = len(X_channels_names_raw)
n_channels = n_raw +1
colors = plt.cm.rainbow(np.linspace(0, 1, 3))  # 3 channels

# RAW rec1
fig_raw, axes_raw = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
for i, channel in enumerate(X_channels_start_raw_selected):
    axes_raw[0].plot(time_start, channel, color=colors[i], linewidth=0.6, label=channel_labels_raw[i])
axes_raw[0].set_title(f'MEG X Component - {rec1_label} - Raw (Selected)')
axes_raw[0].set_ylabel('Amplitude (pT)')
axes_raw[0].grid(True, alpha=0.3)
axes_raw[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

for i, norm in enumerate(norms_start_raw_selected):
    axes_raw[1].plot(time_start, norm, color=colors[i], linewidth=0.6, label=channel_labels_raw[i])
axes_raw[1].set_title(f'MEG Vector Norm - {rec1_label} - Raw (Selected)')
axes_raw[1].set_ylabel('Magnitude')
axes_raw[1].grid(True, alpha=0.3)
axes_raw[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.subplots_adjust(top=0.95)

# RAW rec11
fig_raw, axes_raw = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
for i, channel in enumerate(X_channels_last_raw_selected):
    axes_raw[0].plot(time_last, channel, color=colors[i], linewidth=0.6, label=channel_labels_raw[i])
axes_raw[0].set_title(f'MEG X Component - {rec11_label} - Raw (Selected)')
axes_raw[0].set_ylabel('Amplitude (pT)')
axes_raw[0].grid(True, alpha=0.3)
axes_raw[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
norms_last_raw_selected = [
    np.sqrt(
        X_channels_last_raw_selected[i]**2 +
        Y_channels_last_raw[selected_indices_raw[i]]**2 +
        Z_channels_last_raw[selected_indices_raw[i]]**2
    )
    for i in range(3)
]
for i, norm in enumerate(norms_last_raw_selected):
    axes_raw[1].plot(time_last, norm, color=colors[i], linewidth=0.6, label=channel_labels_raw[i])
axes_raw[1].set_title(f'MEG Vector Norm - {rec11_label} - Raw (Selected)')
axes_raw[1].set_ylabel('Magnitude')
axes_raw[1].grid(True, alpha=0.3)
axes_raw[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.show()
##########################################################
# ICA COMPONENT PLOTS - START
# Example for X_ica_last (repeat for Y_ica_last, Z_ica_last)
def plot_ica_components(ica_signals, time, axis_label, rec_label):
    n_components = ica_signals.shape[0]
    n_cols = 4
    n_rows = int(np.ceil(n_components / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 2.5 * n_rows), sharex=True)
    axes = axes.flatten()
    for i in range(n_components):
        axes[i].plot(time, ica_signals[i])
        axes[i].set_title(f'ICA Component {i+1}')
        axes[i].set_ylabel('Amplitude')
    for i in range(n_components, len(axes)):
        axes[i].axis('off')
    fig.suptitle(f'ICA Components ({axis_label} axis) - {rec_label}', fontsize=16)
    plt.xlabel('Time (s)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# Plot for X
plot_ica_components(np.array(X_ica_last), time_last, 'X', rec11_label)
# Plot for Y
plot_ica_components(np.array(Y_ica_last), time_last, 'Y', rec11_label)
# Plot for Z
plot_ica_components(np.array(Z_ica_last), time_last, 'Z', rec11_label)

print('Analyzing ICA components...')
print('These are the components that we will consider as artifacts:')
print('X: C12, Y: C3, Z: C6')
print('We will zero out these components in the next step.')

# Zero out the identified artifact components
X_ica_last_clean = X_ica_last.copy()
Y_ica_last_clean = Y_ica_last.copy()
Z_ica_last_clean = Z_ica_last.copy()

X_ica_last_clean[11, :] = 0  # C12 (Python is 0-based index, so 12 -> 11)
Y_ica_last_clean[2, :]  = 0  
Z_ica_last_clean[5, :]  = 0  
# Reconstruct the cleaned signals using the ICA model's inverse_transform
X_channels_last_clean = ica_X.inverse_transform(X_ica_last_clean.T).T
Y_channels_last_clean = ica_Y.inverse_transform(Y_ica_last_clean.T).T
Z_channels_last_clean = ica_Z.inverse_transform(Z_ica_last_clean.T).T
print('ICA components cleaned')


# Plot comparison: Filtered vs ICA-cleaned (rec7/last)
# Compute vector norm for each channel after ICA cleaning
norms_last_clean = calculate_channel_norms(
    X_channels_last_clean, Y_channels_last_clean, Z_channels_last_clean
)

# Plot power spectra for filtered (before ICA)
plot_all_channel_power_spectra(
    norms_last, 
    channel_numbers, 
    f'Vector Norm - {rec11_label} (Filtered, Before ICA)'
)

# Plot power spectra for ICA-cleaned data
plot_all_channel_power_spectra(
    norms_last_clean, 
    channel_numbers, 
    f'Vector Norm - {rec11_label} (Filtered + ICA Cleaned)'
)
plt.show()
exit()
##########################################################

print("\n=== Filtering ===")
print("\n=== Plotting FILTERED MEG and Normalized Data ===")



# Create list of channel numbers (excluding 5 and 13)
channel_numbers = [i+1 for i in range(20) if i not in channels_to_exclude]

# Create colormap for MEG channels
n_channels = len(X_channels_names)
colors = plt.cm.rainbow(np.linspace(0, 1, 3))  # 3 channels

# FILTERED rec1
fig_filt, axes_filt = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
for i, channel in enumerate(X_channels_start_selected):
    axes_filt[0].plot(time_start, channel, color=colors[i], linewidth=0.6, label=channel_labels_filtered[i])
axes_filt[0].set_title(f'MEG X Component - {rec1_label} - Filtered (Selected)')
axes_filt[0].set_ylabel('Amplitude (pT)')
axes_filt[0].grid(True, alpha=0.3)
axes_filt[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
norms_start_selected = [
    np.sqrt(
        X_channels_start_selected[i]**2 +
        Y_channels_start[selected_indices_filtered[i]]**2 +
        Z_channels_start[selected_indices_filtered[i]]**2
    )
    for i in range(3)
]
for i, norm in enumerate(norms_start_selected):
    axes_filt[1].plot(time_start, norm, color=colors[i], linewidth=0.6, label=channel_labels_filtered[i])
axes_filt[1].set_title(f'MEG Vector Norm - {rec1_label} - Filtered (Selected)')
axes_filt[1].set_ylabel('Magnitude')
axes_filt[1].grid(True, alpha=0.3)
axes_filt[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.subplots_adjust(top=0.95)

# FILTERED rec11
fig_filt, axes_filt = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
for i, channel in enumerate(X_channels_last_selected):
    axes_filt[0].plot(time_last, channel, color=colors[i], linewidth=0.6, label=channel_labels_filtered[i])
axes_filt[0].set_title(f'MEG X Component - {rec11_label} - Filtered (Selected)')
axes_filt[0].set_ylabel('Amplitude (pT)')
axes_filt[0].grid(True, alpha=0.3)
axes_filt[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
norms_last_selected = [
    np.sqrt(
        X_channels_last_selected[i]**2 +
        Y_channels_last[selected_indices_filtered[i]]**2 +
        Z_channels_last[selected_indices_filtered[i]]**2
    )
    for i in range(3)
]
for i, norm in enumerate(norms_last_selected):
    axes_filt[1].plot(time_last, norm, color=colors[i], linewidth=0.6, label=channel_labels_filtered[i])
axes_filt[1].set_title(f'MEG Vector Norm - {rec11_label} - Filtered (Selected)')
axes_filt[1].set_ylabel('Magnitude')
axes_filt[1].grid(True, alpha=0.3)
axes_filt[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
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
plt.show()
# Plot power spectra for second recording (Vector Norm)
plot_all_channel_power_spectra(
    norms_last, 
    channel_numbers, 
    f'Vector Norm - {rec11_label}'
)
plt.show()

##########################################################


# --- Define time frames as boolean masks ---
# For rec1 (start)
t_1_start = (time_start >= 5) & (time_start < 10)
t_2_start = (time_start >= 105) & (time_start < 110)
t_3_start = (time_start >= 285) & (time_start < 290)
time_windows_start = [t_1_start, t_2_start, t_3_start]

# For rec11 (last)
t_1_last = (time_last >= 0) & (time_last < 10)
t_2_last = (time_last >= 100) & (time_last < 110)
t_3_last = (time_last >= 280) & (time_last < 290)
time_windows_last = [t_1_last, t_2_last, t_3_last]
time_labels = ["0-10 s", "100-110 s", "280-290 s"]

# Use the same colormap as before
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
