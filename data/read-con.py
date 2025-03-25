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
channel_names = raw.ch_names


###################################################################################
###################################################################################
# It seems that the channel from E20 to E28 have the same periodical pattern. 
#i'll check if they are exactly the same, so we could maybe consider them as triggers.
# From E29 to E32 there are periodical pitches, i want to check more in details,
#maybe some of those are related to the ACC.
###################################################################################



# Print info about the data
print(raw.info)
print(f'Channel names within .con data, (n={len(raw.ch_names)}): {raw.ch_names}')
print(f'Timestamps within .con data (n={len(raw.times)}): {raw.times}')
print(f'Sampling freq within .con data: {raw.info['sfreq']}')

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
sfreq = 500 #Hz
lpass = 200 #Hz
Hpass = 1 #Hz

# Create filtered copy for MEG and EEG channels
raw_filtered = raw.copy()

# Apply filter only to MEG and EEG channels
print("\n=== Applying filters to MEG and EEG channels ===")
data_channels = mne.pick_types(raw.info, meg=True, eeg=True, stim=False, eog=False, misc=False)
raw_filtered.filter(l_freq=Hpass, h_freq=lpass, picks=data_channels)

'''
TRYING TO CHECK IF THERE'S A SIMILARITY BTWN E20-E28 CHANNELS:

print("\n=== Checking if E20-E28 channels are identical ===")
# Extract E20-E28 channels
trigger_group = [f'E{i:02d}' for i in range(20, 29)]
trigger_data = raw_filtered.copy().pick_channels(trigger_group).get_data()

print("\n=== Checking if E20-E28 channels are identical ===")
# Extract E20-E28 channels
trigger_group = [f'E{i:02d}' for i in range(20, 29)]
trigger_data = raw.copy().pick_channels(trigger_group).get_data()

# Compare each channel with every other channel
print("\nComparing all channels with each other:")
for i in range(len(trigger_group)):
    for j in range(i + 1, len(trigger_group)):  # Compare with channels we haven't compared yet
        ch1_name = trigger_group[i]
        ch2_name = trigger_group[j]
        is_identical = np.array_equal(trigger_data[i], trigger_data[j])
        print(f"{ch1_name} vs {ch2_name}: {'Identical' if is_identical else 'Different'}")

exit()
'''

# Get filtered MEG and EEG data
meg_data = raw_filtered.get_data(picks=meg_channels)
eeg_data = raw_filtered.get_data(picks=eeg_channels)

# Get unfiltered STIM data
stim_data = raw.copy().pick_channels(['STI 014']).get_data()

###################################################################################
###################################################################################
## PLOT THE CHANNELS:


# Plot trigger channel
trigger_plot = raw.copy().pick_channels(['STI 014'])
trigger_plot.plot(duration=5, scalings='auto', 
                 title="Trigger Channel (STI 014)", 
                 block=True, show=True)

# Plot MEG channels (filtered)
meg_plot = raw_filtered.copy().pick_types(meg=True)
meg_plot.plot(duration=5, n_channels=30, 
              scalings="auto", 
              title="MEG Channels (filtered)", 
              block=True, show=True)

# Plot EEG channels (filtered)
eeg_plot = raw_filtered.copy().pick_types(eeg=True)
eeg_plot.plot(duration=5, 
              n_channels=len(eeg_channels), 
              scalings="auto", 
              title="EEG Channels (filtered)", 
              block=True, show=True)

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

# Extract filtered EEG data for each group
data_1 = raw_filtered.copy().pick_channels(group_1).get_data()
data_2 = raw_filtered.copy().pick_channels(group_2).get_data()

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