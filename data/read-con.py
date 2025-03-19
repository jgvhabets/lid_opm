import mne
import matplotlib.pyplot as plt
import numpy as np

# Path to your .con file
file_path = "plfp65/plfp65_rec4.con" #reading the plfp65_rec4.con file

# Read the .con file
raw = mne.io.read_raw_kit(file_path, preload=True)
dat, times = raw[:]
chnames = raw.ch_names
data = raw.get_data()

###################################################################################
###################################################################################
# I don't know why, but if i read this file it results that there are 128 MEG channels and none of those are empty.
# Reading the .lvm file both with matlab and python it results there are 192 MEG channels,
#  and just 60 of those are non-empty.
# In this way i can't distinguish which are the X, Y, Z components and at the same time 
# i don't understand the mismatch.
###################################################################################
###################################################################################
'''
Here i try to visualize the MEG channels, trying to understand what they represent 
sfreq = 500 #Hz
lpass = 200 #Hz
Hpass = 1 #Hz

x_channels = raw.copy().pick_channels(raw.ch_names[0:64])   # MEG 001 - MEG 064 (X)
y_channels = raw.copy().pick_channels(raw.ch_names[64:128]) # MEG 065 - MEG 128 (Y)

x_data = x_channels.get_data()  # Shape: (64, n_samples)
y_data = y_channels.get_data()  # Shape: (64, n_samples)

x_channels.plot(duration=5, n_channels=30, show=True, highpass=Hpass, lowpass=lpass, block=True, scalings="auto",title="X-Component MEG Sensors")
y_channels.plot(duration=5, n_channels=30, show=True, highpass=Hpass, lowpass=lpass, block=True, scalings="auto",title="Y-Component MEG Sensors")

print("X Data Shape:", x_data.shape)
print("Y Data Shape:", y_data.shape)
exit()
'''
###################################################################################


# Print info about the data
print(raw.info)
print(f'Channel names within .con data, (n={len(raw.ch_names)}): {raw.ch_names}')
print(f'Timestamps within .con data (n={len(raw.times)}): {raw.times}')
print(f'Sampling freq within .con data: {raw.info['sfreq']}')

print(f'\ndata shape: {dat.shape}, times shape: {times.shape}')

## Frequencies:

sfreq = 500 #Hz
lpass = 200 #Hz
Hpass = 1 #Hz

#Setting up band-pass filter from 1 - 40 Hz
raw.filter(l_freq=Hpass, h_freq=lpass)

#Selecting the channels:
meg_channels = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=False, misc=False)
eeg_channels = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, misc=False)
channel_names = raw.ch_names


data = raw.get_data()
bad_channels = []

##Check if there's any empty channel:

for i, ch_name in enumerate(channel_names):
    ch_data = data[i]
    if np.isnan(ch_data).any() or np.isinf(ch_data).any() or np.all(ch_data == 0):
        bad_channels.append(ch_name)

# Mark bad channels
if bad_channels:
    print(f"Bad channels detected and excluded: {bad_channels}")
    raw.info["bads"].extend(bad_channels)
    raw = raw.drop_channels(raw.info["bads"])

else:
    print("No bad channels found.")

# Plot raw MEG data
meg_data = raw.copy().pick(picks=meg_channels)
meg_data.plot(duration=5, n_channels=30, scalings="auto", title="MEG Channels", block=True, show=True)

# Plot raw EEG data
eeg_data = raw.copy().pick(picks=eeg_channels)
eeg_data.plot(duration=5, n_channels=len(eeg_channels), scalings="auto", title="EEG Channels", block=True, show=True)


##############################################
# !!!!
#From eeg plot i figured out that EEG(1, 2, 3, 4, 5, 14, 15, 16) have a common pattern.
# The channels EEG(6, 7, 8, 9, 10, 11, 12 ,13) have a different common pattern.
# I separate now the two different groups:
group_1 = ["E01", "E02", "E03", "E04", "E05", "E14", "E15", "E16"]
group_2 = ["E06", "E07", "E08", "E09", "E10", "E11", "E12", "E13"]

# Ensure channel names exist
group_1 = [ch for ch in group_1 if ch in raw.ch_names]
group_2 = [ch for ch in group_2 if ch in raw.ch_names]

# Extract EEG data for each group
data_1 = raw.copy().pick_channels(group_1).get_data()
data_2 = raw.copy().pick_channels(group_2).get_data()

# Ensure shapes match before subtraction
if data_1.shape == data_2.shape:
    data_subtracted = data_2 - data_1
else:
    raise ValueError(f"Shape mismatch: {data_1.shape} vs {data_2.shape}")

# Create a new MNE Raw object with the subtracted data
info = mne.create_info(ch_names=group_1, sfreq=raw.info['sfreq'], ch_types='eeg')
raw_subtracted = mne.io.RawArray(data_subtracted, info)

raw_subtracted.plot(duration=5, n_channels=len(group_1), scalings="auto", title="Subtracted EEG (Group 1 - Group 2)",block=True, show=True)

'''
# Plot raw data
#raw.plot(duration=3, n_channels=21, show=True, highpass=Hpass, lowpass=lpass, block=True, scalings="auto")

# Plot sensor locations
#raw.plot_sensors( show=True, block=True )'
'''