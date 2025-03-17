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

# Set the filters:

sfreq = 500 #Hz
lpass = 200 #Hz
Hpass = 1 #Hz

# Plot sensor locations
raw.plot_sensors( show=True, block=True )

sfreq = 500 #Hz
lpass = 200 #Hz
Hpass = 1 #Hz

#Setting up band-pass filter from 1 - 2e+02 Hz
raw.filter(l_freq=1.0, h_freq=40.0)

empty_channels = [raw.ch_names[i] for i in range(data.shape[0]) if np.all(data[i] == 0)]
if empty_channels:
    print("Empty channels (all zeros):", empty_channels)
else:
    print("No empty channels found.")

# Plot raw data
raw.plot(duration=3, n_channels=21, show=True, highpass=Hpass, lowpass=lpass, block=True, scalings="auto")
