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
    plot_meg_2x3_grid,
    plot_channels_comparison,
    plot_ica_max_amplitudes,
    plot_single_ica_power_spectrum,
    plot_ica_power_spectra_grid
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

def plot_ica_components(ica_signals, time, axis_label, rec_label):
    n_components = ica_signals.shape[0]
    n_cols = 1
    n_rows = n_components
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.2 * n_rows), sharex=True)
    if n_components == 1:
        axes = [axes]
    for i in range(n_components):
        axes[i].plot(time, ica_signals[i])
        axes[i].set_ylabel(f'C{i+1}')
    fig.suptitle(f'ICA Components ({axis_label} axis) - {rec_label}', fontsize=10)
    plt.xlabel('Time (s)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

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

file_path_1 = os.path.join(data_dir, "plfp65_rec7_13.11.2024_13-42-47_array1.lvm")
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

# Exclude channels 5, 6, 13, 17 and 20 from all components
channels_to_exclude = [4, 5, 12, 16, 19]
print("\nExcluding channels 5, 6, 13, 17 and 20 from all components...")

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
# Create colormaps for MEG channels
colors_all = plt.cm.rainbow(np.linspace(0, 1, len(X_channels_start_raw)))
colors_filtered_start = plt.cm.rainbow(np.linspace(0, 1, len(X_channels_start)))
colors_filtered_last = plt.cm.rainbow(np.linspace(0, 1, len(X_channels_last)))

# Plot all channels for rec1 (start)
plot_channels_comparison(
    time_start,
    X_channels_start_raw,
    X_channels_start,
    X_channels_names_raw,
    X_channels_names,
    colors_all,
    rec1_label,
    y_label="Amplitude (pT)",
    axis_label="X"
)
# Plot all channels for rec11 (last)
plot_channels_comparison(
    time_last,
    X_channels_last_raw,
    X_channels_last,
    X_channels_names_raw,
    X_channels_names,
    colors_all,
    rec11_label,
    y_label="Amplitude (pT)",
    axis_label="X"
)
########################################
# Plot selected channels for rec1 (start)
plot_channels_comparison(
    time_start,
    X_channels_start_raw_selected,
    X_channels_start_selected,
    channel_labels_raw,
    channel_labels_filtered,
    colors,
    rec1_label,
    y_label="Amplitude (pT)",
    axis_label="X"
)
# Plot selected channels for rec11 (last)
plot_channels_comparison(
    time_last,
    X_channels_last_raw_selected,
    X_channels_last_selected,
    channel_labels_raw,
    channel_labels_filtered,
    colors,
    rec11_label,
    y_label="Amplitude (pT)",
    axis_label="X"
)

##########################################################
##########################################################


# --- Define time frames as boolean masks ---
# For rec1 (start)
t_1_start = (time_start >= 5) & (time_start < 15)
t_2_start = (time_start >= 100) & (time_start < 110)
t_3_start = (time_start >= 280) & (time_start < 290)
time_windows_start = [t_1_start, t_2_start, t_3_start]

# For rec11 (last)
t_1_last = (time_last >= 0) & (time_last < 10)
t_2_last = (time_last >= 100) & (time_last < 110)
t_3_last = (time_last >= 280) & (time_last < 290)
time_windows_last = [t_1_last, t_2_last, t_3_last]
time_labels = ["0-10 s", "100-110 s", "280-290 s"]

#############################################################
# The following grid plots allow visual comparison of MEG signals before and after filtering.
# Each 2x3 grid figure shows three different time windows(t_1, t_2, t_3) (columns):
# the first two figures, consider all channels, while the other two show only the selected channels (3, 9, 19).
# The first row in each grid shows the raw (unfiltered) data, while the second row shows the filtered data.
# This is done for both the initial (rec1) and final (rec11) recordings.

# PLOTTING THE GRID WITH ALL CHANNELS:
# Create colormap for MEG channels
colors_all = plt.cm.rainbow(np.linspace(0, 1, len(X_channels_start_raw)))
colors_selected = plt.cm.rainbow(np.linspace(0, 1, len(X_channels_start_raw_selected)))

# For rec1 (start) - ALL CHANNELS
plot_meg_2x3_grid(
    X_channels_start_raw,
    X_channels_start,
    time_start,
    time_windows_start,
    [],  # No labels for raw channels
    [],  # No labels for filtered channels
    time_labels,
    colors_all,
    colors_filtered_start,
    f'Raw {rec1_label}', f'Filtered {rec1_label}',
    f'Raw vs Filtered {rec1_label} - All Channels'
)

# For rec11 (last) - ALL CHANNELS
plot_meg_2x3_grid(
    X_channels_last_raw,
    X_channels_last,
    time_last,
    time_windows_last,
    [],  # No labels for raw channels
    [],  # No labels for filtered channels
    time_labels,
    colors_all,
    colors_filtered_last,
    f'Raw {rec11_label}', f'Filtered {rec11_label}',
    f'Raw vs Filtered {rec11_label} - All Channels'
)

# No channel labels here to keep the figure uncluttered

#############################################################
# PLOTTING THE GRID JUST FOR CHANNELS 3, 9, 19 (indices 2, 8, 18)
# For rec1 (start) - SELECTED CHANNELS
plot_meg_2x3_grid(
    X_channels_start_raw_selected,
    X_channels_start_selected,
    time_start,
    time_windows_start,
    channel_labels_filtered,
    channel_labels_filtered,
    time_labels,
    colors_selected,
    colors_selected,
    f'Raw {rec1_label}', f'Filtered {rec1_label}',
    f'Raw vs Filtered {rec1_label} - All Channels'
)

# For rec11 (last) - SELECTED CHANNELS
plot_meg_2x3_grid(
    X_channels_last_raw_selected,
    X_channels_last_selected,
    time_last,
    time_windows_last,
    channel_labels_filtered,
    channel_labels_filtered,
    time_labels,
    colors_selected,
    colors_selected,
    f'Raw {rec11_label}', f'Filtered {rec11_label}',
    f'Raw vs Filtered {rec11_label} - All Channels'
)

##########################################################
##########################################################
# ICA COMPONENT PLOTS:

# Plot ICA components for rec1 (start)
plot_ica_components(np.array(X_ica_start), time_start, 'X', rec1_label)
plot_ica_components(np.array(Y_ica_start), time_start, 'Y', rec1_label)
plot_ica_components(np.array(Z_ica_start), time_start, 'Z', rec1_label)

# Plot for X
plot_ica_components(np.array(X_ica_last), time_last, 'X', rec11_label)
# Plot for Y
plot_ica_components(np.array(Y_ica_last), time_last, 'Y', rec11_label)
# Plot for Z
plot_ica_components(np.array(Z_ica_last), time_last, 'Z', rec11_label)

##########################################################
#let's analyze the ICA components to identify artifacts
# Analyze ICA max amplitude to check if there are any random values or artifacts

# For the first recording (rec1)
max_X_ica_start = np.max(np.abs(X_ica_start), axis=1)
max_Y_ica_start = np.max(np.abs(Y_ica_start), axis=1)
max_Z_ica_start = np.max(np.abs(Z_ica_start), axis=1)

# Bar plots for rec1 (start):
plot_ica_max_amplitudes(X_ica_start, title=f'Max Amplitude of ICA Components - {rec1_label} (X)')
plot_ica_max_amplitudes(Y_ica_start, title=f'Max Amplitude of ICA Components - {rec1_label} (Y)')
plot_ica_max_amplitudes(Z_ica_start, title=f'Max Amplitude of ICA Components - {rec1_label} (Z)')

# For the last recording (rec11)
max_X_ica_last = np.max(np.abs(X_ica_last), axis=1)
max_Y_ica_last = np.max(np.abs(Y_ica_last), axis=1)
max_Z_ica_last = np.max(np.abs(Z_ica_last), axis=1)

# Bar plots for rec1 (start):
plot_ica_max_amplitudes(X_ica_last, title=f'Max Amplitude of ICA Components - {rec11_label} (X)')
plot_ica_max_amplitudes(Y_ica_last, title=f'Max Amplitude of ICA Components - {rec11_label} (Y)')
plot_ica_max_amplitudes(Z_ica_last, title=f'Max Amplitude of ICA Components - {rec11_label} (Z)')

##########################################################
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

##########################################################
##########################################################
# ICA compare using power spectra:

# PS of the first recording (rec1)
component_names = [f"C{i+1}" for i in range(X_ica_start.shape[0])]

plot_ica_power_spectra_grid(
    X_ica_start,
    plot_single_ica_power_spectrum,
    component_names=component_names,
    title=f"X Components Power Spectra - {rec1_label}"
)
plot_ica_power_spectra_grid(
    Y_ica_start,
    plot_single_ica_power_spectrum,
    component_names=component_names,
    title=f"Y Components Power Spectra - {rec1_label}"
)
plot_ica_power_spectra_grid(
    Z_ica_start,
    plot_single_ica_power_spectrum,
    component_names=component_names,
    title=f"Z Components Power Spectra - {rec1_label}"
)

# PS of the last recording (rec11)
component_names = [f"C{i+1}" for i in range(X_ica_last.shape[0])]

plot_ica_power_spectra_grid(
    X_ica_last,
    plot_single_ica_power_spectrum,
    component_names=component_names,
    title=f"X Components Power Spectra - {rec11_label}"
)
plot_ica_power_spectra_grid(
    Y_ica_last,
    plot_single_ica_power_spectrum,
    component_names=component_names,
    title=f"Y Components Power Spectra - {rec11_label}"
)
plot_ica_power_spectra_grid(
    Z_ica_last,
    plot_single_ica_power_spectrum,
    component_names=component_names,
    title=f"Z Components Power Spectra - {rec11_label}"
)


##########################################################
##########################################################
print("\n=== Plotting Power Spectra before and after ICA method MEG Channels ===")

# Plot comparison: Filtered vs ICA-cleaned (last)
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
