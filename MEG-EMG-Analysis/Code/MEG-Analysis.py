#################
### LIBRARIES ###
#################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from mne.filter import filter_data
from mne.time_frequency import Spectrum


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

# Function to create component subplot
def plot_component_comparison(channels_start, channels_last, component_name, channel_names, rec1_label, rec11_label):
    # Calculate grid layout
    n_channels = len(channel_names)
    n_cols = 5
    n_rows = (n_channels + n_cols - 1) // n_cols
    
    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 15), layout="constrained")
    axes = axes.flatten()
    
    for i in range(n_channels):
        # Plot both recordings on the same subplot
        axes[i].plot(time_start, channels_start[i], color='#1f77b4', 
                    label=rec1_label, linewidth=1.5, alpha=0.8)
        axes[i].plot(time_last, channels_last[i], color='#ff7f0e', 
                    label=rec11_label, linewidth=1.5, alpha=0.7)
        
        # Add title and labels
        axes[i].set_title(f'Channel {channel_names[i]}', fontsize=7)
        axes[i].grid(True, alpha=0.3)
        
        # Add legend only for first subplot
        if i == 0:
            axes[i].legend()
        
        # Add y-label for leftmost plots
        if i % n_cols == 0:
            axes[i].set_ylabel('Magnitude (pT)')
        
        # Add x-label for bottom plots
        if i >= n_channels - n_cols:
            axes[i].set_xlabel('Time (sec)')
    
    # Hide unused subplots
    for i in range(n_channels, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'{component_name} Component Comparison: {rec1_label} vs {rec11_label}', fontsize=8)
    plt.tight_layout()
    plt.show()

#################
### MAIN CODE ###
#################

#######################################################
#######################################################


# READING THE FILE AND DATAFRAME CREATION:

#file rec1:
file_path_1 = "/Users/federicobonato/Developer/WORK/lid_opm/MEG-EMG-Analysis/Data/plfp65_rec3_13.11.2024_13-10-36_array1.lvm"
df_start = pd.read_csv(file_path_1, header= 22, sep='\t')

# file rec11:
file_path_11 = "/Users/federicobonato/Developer/WORK/lid_opm/MEG-EMG-Analysis/Data/plfp65_rec11_13.11.2024_14-18-30_array1.lvm"
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

# Exclude channels 5 and 13 from all components
channels_to_exclude = [4, 12]
print("\nExcluding channels 5 and 13 from all components...")

# Helper function to remove specific indices from lists
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

print("Filtering complete")


# Assign a specific color to each component
component_colors = {
    'X': 'blue',     # Blue for X component
    'Y': 'green',    # Green for Y component
    'Z': 'red',      # Red for Z component
    'Norm': 'purple' # Purple for the norm component
}

# Pick a sensor
time_start = df_start["X_Value"]
time_last = df_last["X_Value"]

#########################################################################
#################
###  PLOTS ###
#################
#########################################################################

# Create separate plots for each component
plot_component_comparison(X_channels_start, X_channels_last, 'X', X_channels_names, rec1_label, rec11_label)
plot_component_comparison(Y_channels_start, Y_channels_last, 'Y', X_channels_names, rec1_label, rec11_label)
plot_component_comparison(Z_channels_start, Z_channels_last, 'Z', X_channels_names, rec1_label, rec11_label)


# Now let's normalize the data and plot the channels:
print("\nCalculating channel norms...")
norms_start = calculate_channel_norms(X_channels_start, Y_channels_start, Z_channels_start)
norms_last = calculate_channel_norms(X_channels_last, Y_channels_last, Z_channels_last)

# Plotting the normalized channels:

# Create figure with subplots for norm comparisons
n_channels = len(X_channels_names)
n_cols = 5
n_rows = (n_channels + n_cols - 1) // n_cols

# Create figure
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 15), layout="constrained")
axes = axes.flatten()

# Plot norms for each channel
for i in range(n_channels):
    
    axes[i].plot(time_start, norms_start[i], color='#1f77b4', 
            label=rec1_label, linewidth=1.5, alpha=0.7)
    axes[i].plot(time_last, norms_last[i], color='#ff7f0e', 
            label=rec11_label, linewidth=1.5, alpha=0.7)
    
    # Add title and labels
    axes[i].set_title(f'Channel {X_channels_names[i]}', fontsize=7)
    axes[i].grid(True, alpha=0.3)
    
    # Add legend only for the first subplot
    if i == 0:
        axes[i].legend()
    
    # Add y-label for leftmost subplots
    if i % n_cols == 0:
        axes[i].set_ylabel('Magnitude')
    
    # Add x-label for bottom subplots
    if i >= n_channels - n_cols:
        axes[i].set_xlabel('Time (sec)')

# Hide unused subplots
for i in range(n_channels, len(axes)):
    axes[i].set_visible(False)

plt.suptitle(f'Channel Norms Comparison: {rec1_label} vs {rec11_label}', fontsize=8)
plt.tight_layout()
plt.show()

# Create figure for power spectrum comparison
fig_ps, axes_ps = plt.subplots(n_rows, n_cols, figsize=(20, 15), layout="constrained")
axes_ps = axes_ps.flatten()

# Plot power spectrum for each channel
for i in range(n_channels):
    # Calculate power spectrum
    freqs_start, power_start = calculate_power_spectrum(norms_start[i])
    freqs_last, power_last = calculate_power_spectrum(norms_last[i])
    
    # Plot on log scale
    axes_ps[i].semilogy(freqs_start, power_start, color='#1f77b4', 
                    label=rec1_label, linewidth=1.5, alpha=0.7)
    axes_ps[i].semilogy(freqs_last, power_last, color='#ff7f0e', 
                    label=rec11_label, linewidth=1.5, alpha=0.7)
    
    # Add title and labels
    axes_ps[i].set_title(f'{X_channels_names[i]}', fontsize=5)
    axes_ps[i].grid(True, alpha=0.3)
    
      # Add legend to first subplot only
    if i == 0:
        axes_ps[i].legend()
    
    # Add y-label for leftmost subplots
    if i % n_cols == 0:
        axes_ps[i].set_ylabel('Power (magnitude²)')
    
    # Add x-label for bottom subplots
    if i >= n_channels - n_cols:
        axes_ps[i].set_xlabel('Frequency (Hz)')

# Hide unused subplots
for i in range(n_channels, len(axes_ps)):
    axes_ps[i].set_visible(False)

plt.suptitle(f'Power Spectrum Comparison: {rec1_label} vs {rec11_label}', fontsize=8)
plt.tight_layout()
plt.show()

########################################################################
# Now let's perform frequency band analysis using MNE
# and plot the results for each channel.

print("\nPerforming frequency band analysis using MNE...")

# Define frequency bands
freq_bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 100)
}

# Create info object for the data, necessary to use mne functions on the data
info = mne.create_info(
    ch_names=X_channels_names,
    sfreq=375,  # Your sampling frequency
    ch_types=['mag'] * len(X_channels_names)
)

# Convert numpy arrays to MNE Raw objects
raw_start = mne.io.RawArray(np.array(norms_start), info)
raw_last = mne.io.RawArray(np.array(norms_last), info)


# Create Spectrum objects with 2-second windows
window_length = 2  # seconds
samples_per_window = int(375 * window_length)  # 750 samples
start_time = 100
spectrum_start = Spectrum(
    raw_start,
    tmin=start_time,                    # Start time
    tmax= start_time + window_length,                 # Full duration
    fmin=0.5,
    fmax=100,
    method='welch',
    n_fft=samples_per_window,  # 2-second windows
    n_per_seg=samples_per_window,
    n_overlap=samples_per_window // 2,  # 50% overlap
    picks=None,
    exclude=[],
    proj=False,
    remove_dc=True,
    reject_by_annotation=True,
    n_jobs=1,
    verbose=False
)

spectrum_last = Spectrum(
    raw_last,
    tmin=start_time,
    tmax=start_time + window_length,
    fmin=0.5,
    fmax=100,
    method='welch',
    n_fft=samples_per_window,
    n_per_seg=samples_per_window,
    n_overlap=samples_per_window // 2,
    picks=None,
    exclude=[],
    proj=False,
    remove_dc=True,
    reject_by_annotation=True,
    n_jobs=1,
    verbose=False
)
# Plot for each channel
for ch_idx in range(len(X_channels_names)):
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    
    # Get power spectra data
    psd_start = spectrum_start.get_data(picks=[ch_idx])
    psd_last = spectrum_last.get_data(picks=[ch_idx])
    freqs = spectrum_start.freqs
    
    # Plot power spectra
    ax.semilogy(freqs, psd_start.squeeze(), 
            color='#1f77b4', label=rec1_label)
    ax.semilogy(freqs, psd_last.squeeze(), 
            color='#ff7f0e', label=rec11_label)
    # Highlight frequency bands
    for band_name, (low, high) in freq_bands.items():
        ax.axvspan(low, high, color='gray', alpha=0.3)
        ax.axvline(x=low, color='black', linewidth=2, alpha=0.4)
        ax.axvline(x=high, color='black', linewidth=2, alpha=0.4)
        ax.text((low + high)/2, ax.get_ylim()[1], band_name, 
                horizontalalignment='center', verticalalignment='bottom')
    
    ax.set_title(f'Channel {X_channels_names[ch_idx]} - Power Spectrum')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
