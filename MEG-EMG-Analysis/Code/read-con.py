########################################################################
####### LIBRARIES ########################################################
########################################################################

import mne
import matplotlib.pyplot as plt
import numpy as np

############################################################################
####### FUNCTIONS ########################################################
############################################################################

def apply_filter(data, sfreq, l_freq, h_freq):
    """Apply bandpass filter to data using MNE"""
    # Convert data to MNE RawArray format
    info = mne.create_info(ch_names=['signal'], sfreq=sfreq, ch_types=['misc'])
    raw = mne.io.RawArray(data.reshape(1, -1), info, verbose=False)
    
    # Apply filter with explicit picks and suppress output
    raw.filter(l_freq=l_freq, h_freq=h_freq, picks=[0], verbose=False)
    
    return raw.get_data()[0]

############################################################################
####### MAIN ##############################################################
############################################################################


# Path to your .con file 
file_path = "/Users/federicobonato/Developer/WORK/lid_opm/MEG-EMG-Analysis/Data/dyskensie_opm_data_230625/pilot_dyst_230625_arm_move.con"
# Read the .con file
raw = mne.io.read_raw_kit(file_path, preload=True)
dat, times = raw[:]
data = raw.get_data()
channel_names = raw.ch_names


###################################################################################
###################################################################################



# Print info about the data
print(raw.info)
print(f'Channel names within .con data, (n={len(raw.ch_names)}): {raw.ch_names}')
print(f'Timestamps within .con data (n={len(raw.times)}): {raw.times}')
print(f'Sampling freq within .con data: {raw.info["sfreq"]}')  
print(f'\ndata shape: {dat.shape}, times shape: {times.shape}')

###################################################################################
###################################################################################
## EXTRAPOLATION DATA:
# First we check if there are any bad channels:

# Check for bad channels in all data
data = raw.get_data()
bad_channels = []
for i, ch_name in enumerate(raw.ch_names):
    ch_data = data[i]
    if np.isnan(ch_data).any() or np.isinf(ch_data).any() or np.all(ch_data == 0):
        bad_channels.append(ch_name)

if bad_channels:
    print(f"Bad channels detected: {bad_channels}")
    raw.info["bads"].extend(bad_channels)
else:
    print("No bad channels found.")


###################################################################################
# Now we extract all the channels:
print("\n=== Channel extraction and bad channel detection ===")
# Extract all channel types first
meg_channels = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=False, misc=False)
eeg_channels = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, misc=False)

print(f"Found {len(meg_channels)} MEG channels")
print(f"Found {len(eeg_channels)} EEG channels")

# Let's apply the filters to the MEG and EEG channels only:

## Frequencies:
sfreq = raw.info["sfreq"] #2000Hz
lpass = 100 #Hz
Hpass = 1 #Hz

# Create filtered copy for MEG and EEG channels
raw_filtered = raw.copy()

# Apply filter only to MEG and EEG channels
print("\n=== Applying filters to MEG and EEG channels ===")
data_channels = mne.pick_types(raw.info, meg=True, eeg=True, stim=False, eog=False, misc=False)
raw_filtered.filter(l_freq=Hpass, h_freq=lpass, picks=data_channels)


# Get filtered MEG and EEG data
meg_data = raw_filtered.get_data(picks=meg_channels)
eeg_data = raw_filtered.get_data(picks=eeg_channels)

# Get unfiltered STIM data
stim_data = raw.copy().pick_channels(['STI 014']).get_data()

###################################################################################
###################################################################################
## PLOT THE CHANNELS:



# Extract trigger channel data (channel 157)
trigger_channel = raw.get_data(picks=[156])[0]

# Plot the trigger channel to visualize events
sfreq = raw.info['sfreq']
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
trigger_events = np.where(trigger_channel > 0)[0]
trigger_times = trigger_events / sfreq
# Print detected trigger times
print(f"Detected trigger events at times (s): {trigger_times}")



def plot_eeg_channels_with_trigger(raw, trigger_channel):
    """
    Plots each EEG channel individually over time with the trigger channel on top.
    
    Parameters:
    raw : mne.io.Raw
        The raw data object containing EEG channels.
    trigger_channel : np.ndarray
        The data of the trigger channel.
    """
    # Get EEG channel indices
    eeg_channels = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, misc=False)
    
    # Sampling frequency
    sfreq = raw.info['sfreq']
    
    # Extract EEG data
    eeg_data = raw.get_data(picks=eeg_channels)
    filtered_eeg_data = raw.copy().filter(l_freq=1, h_freq=100, picks=eeg_channels).get_data()
    time = np.arange(eeg_data.shape[1]) / sfreq  # Time vector
    
    # Loop through each EEG channel and plot
    for i, ch_idx in enumerate(eeg_channels):
        plt.figure(figsize=(12, 8))
        
        # Top subplot: Trigger channel
        plt.subplot(2, 1, 1)
        plt.plot(time, trigger_channel, label="Trigger Channel", color="red")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title("Trigger Channel Over Time")
        plt.legend()
        plt.grid(True)
        
        # Bottom subplot: Current EEG channel
        plt.subplot(2, 1, 2)
        plt.plot(time, filtered_eeg_data[i], label=f"EEG Channel {raw.ch_names[ch_idx]}", color="blue")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title(f"EEG Channel {raw.ch_names[ch_idx]} Over Time")
        plt.legend()
        plt.grid(True)
        
        # Show the figure
        plt.tight_layout()
        plt.show()

# Call the function to plot EEG channels with the trigger channel
plot_eeg_channels_with_trigger(raw, trigger_channel)