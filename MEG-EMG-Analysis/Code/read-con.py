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
con_file_path = "/Users/federicobonato/Developer/WORK/lid_opm/MEG-EMG-Analysis/Data/dyskensie_opm_data_230625/"
con_file_name = 'pilot_dyst_230625_arm_move.con'
cnt_file_path = "/Users/federicobonato/Developer/WORK/lid_opm/MEG-EMG-Analysis/Data/dyskensie_opm_data_230625/EMG_ACC_data"
cnt_file_name = 'PTB_01_A_1.2_move.cnt'

# Read the .con file
con_raw = mne.io.read_raw_kit(con_file_path+con_file_name, preload=True)
dat, times = con_raw[:]
data = con_raw.get_data()
channel_names = con_raw.ch_names

# Read the .cnt file 
cnt_raw = mne.io.read_raw_ant(cnt_file_path + '/' + cnt_file_name, preload=True)
cnt_dat, cnt_times = cnt_raw[:]
cnt_data = cnt_raw.get_data()
cnt_channel_names = cnt_raw.ch_names

###################################################################################
###################################################################################

# Print info about the data
print(con_raw.info)
print(f'Channel names within .con data, (n={len(con_raw.ch_names)}): {con_raw.ch_names}')
print(f'Timestamps within .con data (n={len(con_raw.times)}): {con_raw.times}')
print(f'Sampling freq within .con data: {con_raw.info["sfreq"]}')  
print(f'\ndata shape: {dat.shape}, times shape: {times.shape}')

# Print info about the cnt data
print(cnt_raw.info)
print(f'Channel names within .cnt data, (n={len(cnt_raw.ch_names)}): {cnt_raw.ch_names}')
print(f'Timestamps within .cnt data (n={len(cnt_raw.times)}): {cnt_raw.times}')
print(f'Sampling freq within .cnt data: {cnt_raw.info["sfreq"]}')  
print(f'\ndata shape: {cnt_dat.shape}, times shape: {cnt_times.shape}')

###################################################################################
###################################################################################
## EXTRAPOLATION DATA:
# First we check if there are any bad channels:

# Check for bad channels in all data
data = con_raw.get_data()
bad_channels = []
for i, ch_name in enumerate(con_raw.ch_names):
    ch_data = data[i]
    if np.isnan(ch_data).any() or np.isinf(ch_data).any() or np.all(ch_data == 0):
        bad_channels.append(ch_name)

if bad_channels:
    print(f"Bad channels detected: {bad_channels}")
    con_raw.info["bads"].extend(bad_channels)
else:
    print("No bad channels found.")

# Same for cnt data
# Check for bad channels in all cnt data
cnt_bad_channels = []
for i, ch_name in enumerate(cnt_raw.ch_names):
    ch_data = cnt_data[i]
    if np.isnan(ch_data).any() or np.isinf(ch_data).any() or np.all(ch_data == 0):
        cnt_bad_channels.append(ch_name)

if cnt_bad_channels:
    print(f"Bad channels detected in cnt: {cnt_bad_channels}")
    cnt_raw.info["bads"].extend(cnt_bad_channels)
else:
    print("No bad channels found in cnt.")

###################################################################################
# Now we extract all the channels:
print("\n=== Channel extraction and bad channel detection ===")
# Extract all channel types first
meg_channels = mne.pick_types(con_raw.info, meg=True, eeg=False, stim=False, eog=False, misc=False)

print(f"Found {len(meg_channels)} MEG channels")

# Let's apply the filters to the MEG and EEG channels only:

## Frequencies:
sfreq = con_raw.info["sfreq"] #2000Hz
lpass = 100 #Hz
Hpass = 1 #Hz

# Create filtered copy for MEG and EEG channels
con_raw_filtered = con_raw.copy()

# Apply filter only to MEG and EEG channels
print("\n=== Applying filters to MEG channels ===")
data_channels = mne.pick_types(con_raw.info, meg=True, eeg=True, stim=False, eog=False, misc=False)
con_raw_filtered.filter(l_freq=Hpass, h_freq=lpass, picks=data_channels)


# Get filtered MEG data
meg_data = con_raw_filtered.get_data(picks=meg_channels)


# Get unfiltered STIM data
stim_data = con_raw.copy().pick_channels(['STI 014']).get_data()

################################################################################
print("\n=== Extracting trigger channel ===")
# Extract trigger channel data (channel 157)
trigger_channel = con_raw.get_data(picks=[156])[0]

###################################################################################
# Define EMG and ACC channel mappings for the cnt file
emg_channels = {
    "right_forearm": "BIP11",  # protocol BIP7
    "right_delt": "BIP8",
    "left_forearm": "BIP9",
    "left_delt": "BIP10",
    "right_tibialis_anterior": "BIP7",  # protocol BIP11
    "left_tibialis_anterior": "BIP12",
    "reference": "Olecranon"
}

acc_channels = {
    "A_right_hand_backside": {"channels": [1, 2, 6], "axes": ["y", "z", "x"]},
    "A_left_hand_backside": {"channels": [3, 4, 5], "axes": ["x", "y", "z"]},
    "B_left_hand_backside": {"channels": [3, 4, 5], "axes": ["x", "y", "z"]},
    "B_left_knee": {"channels": [1, 2, 6], "axes": ["y", "z", "x"]}
}

print("EMG channel mapping (cnt file):", emg_channels)
print("ACC channel mapping (cnt file):", acc_channels)

###################################################################################
###################################################################################
## PLOT THE CHANNELS:
###################################################################################
###################################################################################

# Plot the trigger channel to visualize events
sfreq = con_raw.info['sfreq']
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

