############ COMMENTS ############
# The two files have different lengths and different sample frequencies.
# The .lvm file has a sample frequency of 375 Hz, 
# while the .con file has a sample frequency of 500 Hz.



#################
### LIBRARIES ###
#################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import mne

#################
### FUNCTIONS ###
#################

def apply_filter(data, sfreq, l_freq, h_freq):
    """Apply bandpass filter to data using MNE"""
    # Convert data to MNE RawArray format
    info = mne.create_info(ch_names=['signal'], sfreq=sfreq, ch_types=['misc'])
    raw = mne.io.RawArray(data.reshape(1, -1), info, verbose=False)
    
    # Apply filter with explicit picks and suppress output
    raw.filter(l_freq=l_freq, h_freq=h_freq, picks=[0], verbose=False)
    
    return raw.get_data()[0]

###### LVM DATA ######
def extract_lvm_data(filepath):
    """Extract MEG data from .lvm file"""
    # Read the LVM file
    df = pd.read_csv(filepath, header=22, sep='\t')
    
    # Remove Comment column if exists
    if 'Comment' in df.columns:
        df = df.drop(columns=["Comment"])
    
    # Organize channels by component
    X_channels_names = [col for col in df.columns if "X" in col]
    Y_channels_names = [col for col in df.columns if "Y" in col]
    Z_channels_names = [col for col in df.columns if "Z" in col]
    
    # Remove extra X columns
    X_extras = ['X_Value', 'MUX_Counter1', 'MUX_Counter2']
    X_channels_names = [col for col in X_channels_names if col not in X_extras]
    
    # Extract channel data (first 20 channels only)
    X_channels = [df[col].values for col in X_channels_names][:20]
    Y_channels = [df[col].values for col in Y_channels_names][:20]
    Z_channels = [df[col].values for col in Z_channels_names][:20]
    
    # Get LVM sampling frequency (375 Hz)
    sfreq_lvm = 375
    
    # Filter X, Y, Z channels between 1-100 Hz and convert to pT
    conversion_factor = 1e-12  # Convert to pT 
    X_channels_filtered = [apply_filter(ch, sfreq_lvm, 1, 100) * conversion_factor for ch in X_channels]
    Y_channels_filtered = [apply_filter(ch, sfreq_lvm, 1, 100) * conversion_factor for ch in Y_channels]
    Z_channels_filtered = [apply_filter(ch, sfreq_lvm, 1, 100) * conversion_factor for ch in Z_channels]
    
    # Get timing
    time = df["X_Value"].values
    
    meg_data = {
        'X': X_channels_filtered,
        'Y': Y_channels_filtered,
        'Z': Z_channels_filtered,
        'time': time,
        'channel_names': X_channels_names[:20]
    }
    
    return meg_data, df.shape[0]

###### CON DATA ######
def extract_con_data(filepath):
    """Extract EEG, EMG and ACC data from .con file"""
    # Read the .con file
    raw = mne.io.read_raw_kit(filepath, preload=True)

    # Define channel mappings
    emg_channels = {
        'right_arm': 'E05',
        'neck': ['E07', 'E06'],  # 135-134
        'left_arm': ['E09', 'E08'],  # 137-136
        'right_leg': ['E11', 'E10'],  # 139-138
        'left_leg': ['E13', 'E12']   # 141-140
    }
    
    acc_channels = {
        'x': 'E17',  # 145
        'y': 'E18',  # 146
        'z': 'E19'   # 147
    }
    
    trigger_channel = 'E29'  # 157
    
    # Create filtered copy for EEG channels only
    raw_filtered = raw.copy()
    
    # Apply filters only to EEG channels
    eeg_picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, misc=False)
    raw_filtered.filter(l_freq=25, h_freq=248, picks=eeg_picks)
    
    # Extract EMG data (bipolar) with filtering
    emg_data = {}
    for location, channels in emg_channels.items():
        if isinstance(channels, list):
            # Filter individual channels before subtraction
            ch1 = raw.get_data(picks=[channels[0]])[0]
            ch2 = raw.get_data(picks=[channels[1]])[0]
            ch1_filtered = apply_filter(ch1, raw.info['sfreq'], 2, 248)
            ch2_filtered = apply_filter(ch2, raw.info['sfreq'], 2, 248)
            emg_data[location] = ch1_filtered - ch2_filtered
        else:
            data = raw.get_data(picks=[channels])[0]
            emg_data[location] = apply_filter(data, raw.info['sfreq'], 2, 248)
    
    # Extract and filter ACC data
    acc_data = {}
    for axis, channel in acc_channels.items():
        data = raw.get_data(picks=[channel])[0]
        acc_data[axis] = apply_filter(data, raw.info['sfreq'], 2, 48)
    
    # Get filtered EEG
    
    eeg_data = raw_filtered.get_data(picks=eeg_picks)
    
    return emg_data, acc_data, eeg_data, raw.info['sfreq']


##### MEG norm #####

def calculate_meg_norm(meg_data):
    """Calculate and plot the norm of MEG X, Y, Z components"""
    n_channels = len(meg_data['X'])
    norm_channels = []
    
    # Calculate norms
    for i in range(n_channels):
        norm = np.sqrt(
            np.array(meg_data['X'][i])**2 + 
            np.array(meg_data['Y'][i])**2 + 
            np.array(meg_data['Z'][i])**2
        )
        norm_channels.append(norm)
    
    return norm_channels

###### ACC norm ######

def calculate_acc_norm(acc_data):
    """Calculate and plot the norm of accelerometer X, Y, Z components"""
    # Calculate norm
    norm = np.sqrt(
        acc_data['x']**2 + 
        acc_data['y']**2 + 
        acc_data['z']**2
    )
    
    return norm
 
def create_time_mask(time_array, t_start, t_end):
    """Create a boolean mask for time window selection"""
    return (time_array >= t_start) & (time_array <= t_end)

def synchronize_data(meg_data, emg_data, acc_data, eeg_data, time, time_con, Tzero, Tfinal):
    """Synchronize and process data within the specified time window"""
    
    # Create time masks for both files
    lvm_mask = create_time_mask(time, Tzero, Tfinal)
    con_mask = create_time_mask(time_con, Tzero, Tfinal)
    
    # Create synchronized data dictionary
    sync_data = {
        'time_window': {
            'start': Tzero,
            'end': Tfinal
        },
        'meg': {
            'time': time,
            'time_mask': lvm_mask,
            'X': meg_data['X'],
            'Y': meg_data['Y'],
            'Z': meg_data['Z']
        },
        'con': {
            'time': time_con,
            'time_mask': con_mask,
            'emg': emg_data,
            'acc': acc_data,
            'eeg': eeg_data
        }
    }
    
    return sync_data

#######################################################################
#######################################################################


#################
### MAIN CODE ###
#################

# File paths
os.chdir(os.path.dirname(os.path.abspath(__file__)))
lvm_path = "../Data/plfp65_rec7_13.11.2024_13-42-47_array1.lvm"
con_path = "../Data/plfp65_rec7.con"

print("=== Extracting Data from Files ===")

# Get LVM data (MEG)
print("\nExtracting LVM data...")
meg_data, n_samples_lvm = extract_lvm_data(lvm_path)
print(f"Found {len(meg_data['X'])} MEG channels")

# Get CON data (EEG, EMG, ACC)
print("\nExtracting CON data...")
emg_data, acc_data, eeg_data, sfreq = extract_con_data(con_path)
print(f"Found {len(emg_data)} EMG locations")
print(f"Found {len(acc_data)} ACC axes")
print(f"Found {eeg_data.shape[0]} EEG channels")

# Create an array 'time' for .con data:
time_con = []
for i in range(len(eeg_data[0])):
    time_con.append(i*0.002)  # 0.002 sec = 500 Hz
time_con = np.array(time_con)

# Get command signal from LVM:
df = pd.read_csv(lvm_path, header=22, sep='\t')
all_columns = df.columns.tolist()

#########################################################
#### SYNCHRONIZATION BETWEEN LVM AND CON FILES ####
#########################################################
## Let's select the trigger channel from LVM and CON files:

# LVM trigger channel
time = df["X_Value"].values
trigger_lvm = df["A16"].values

# CON trigger channel
raw = mne.io.read_raw_kit(con_path, preload=True)
trigger_con = raw.get_data(picks=['E29'])[0] * (-1)

# For LVM trigger
max_idx_lvm = np.argmax(trigger_lvm)
t_zero_lvm = time[max_idx_lvm]
print(f"\nLVM trigger max at time: {t_zero_lvm:.3f} seconds")

# For CON trigger
max_idx_con = np.argmax(trigger_con)
t_zero_con = time_con[max_idx_con]
print(f"CON trigger max at time: {t_zero_con:.3f} seconds")

# Apply time shift to CON data (shift back by 0.15 seconds)
time_shift = 0.174
t_zero_con_shifted = t_zero_con + time_shift

# Create relative time arrays (align triggers to t=0)
time_lvm_rel = time - t_zero_lvm
time_con_rel = time_con - t_zero_con_shifted  # Use shifted time

# Define duration for analysis (5 minutes = 300 seconds)
duration = 300

# Create masks for 300s after trigger
lvm_mask = (time_lvm_rel >= 0) & (time_lvm_rel <= duration)
con_mask = (time_con_rel >= 0) & (time_con_rel <= duration)

print(f"""
SYNCHRONIZATION:
---------------
- LVM trigger peak: {t_zero_lvm:.3f} s
- CON trigger peak: {t_zero_con:.3f} s
- Applied time_shift: {time_shift:.3f} s (empirically determined)
- Analysis window: 0 to {duration} seconds after trigger
""")

'''
SYNCHRONIZATION SUMMARY:
-----------------------
1. Trigger Alignment:
   - Found peak times in both LVM and CON triggers
   - LVM trigger peak at {t_zero_lvm:.3f} seconds
   - CON trigger peak at {t_zero_con:.3f} seconds
   
2. Time Shift Adjustment:
   - Observed a consistent delay between triggers
   - Empirically determined time_shift = 0.174 seconds by visual inspection
   - Applied shift to CON data to align with LVM trigger
   
3. Time Window Selection:
   - Set t=0 at trigger peaks
   - Defined analysis window: 0 to 300 seconds after trigger
   - Created boolean masks for both datasets
   
4. Data Access:
   To work with synchronized data, use the mask arrays:
   - For MEG: data[lvm_mask]
   - For EMG/ACC/EEG: data[con_mask]
'''
########################################################################
# Reload raw (unfiltered) MEG data from the LVM file
df_raw = pd.read_csv(lvm_path, header=22, sep='\t')

# Get all X, Y, Z channel names (first 20)
X_raw_names = [col for col in df_raw.columns if "X" in col and col not in ['X_Value', 'MUX_Counter1', 'MUX_Counter2']]
Y_raw_names = [col for col in df_raw.columns if "Y" in col and col != 'Y_Value']
Z_raw_names = [col for col in df_raw.columns if "Z" in col and col != 'Z_Value']

X_raw = [df_raw[col].values for col in X_raw_names][:20]
Y_raw = [df_raw[col].values for col in Y_raw_names][:20]
Z_raw = [df_raw[col].values for col in Z_raw_names][:20]
channel_numbers_raw = list(range(1, 21))  # Channels 1-20

# Calculate norms for raw data
meg_norms_raw = [
    np.sqrt(X_raw[i]**2 + Y_raw[i]**2 + Z_raw[i]**2)
    for i in range(20)
]

# Extract raw (unfiltered) ACC data
acc_raw = {}
acc_raw['x'] = raw.get_data(picks=['E17'])[0]
acc_raw['y'] = raw.get_data(picks=['E18'])[0]
acc_raw['z'] = raw.get_data(picks=['E19'])[0]
acc_norm_raw = np.sqrt(acc_raw['x']**2 + acc_raw['y']**2 + acc_raw['z']**2)


### Removing bad channels 5 and 13###
channels_to_exclude = [4, 12]  
print("\nExcluding channels 5 and 13 from all MEG components...")

def remove_channels(channel_list, indices):
    return [channel for i, channel in enumerate(channel_list) if i not in indices]

# Remove channels from MEG data
meg_data['X'] = remove_channels(meg_data['X'], channels_to_exclude)
meg_data['Y'] = remove_channels(meg_data['Y'], channels_to_exclude)
meg_data['Z'] = remove_channels(meg_data['Z'], channels_to_exclude)
meg_data['channel_names'] = remove_channels(meg_data['channel_names'], channels_to_exclude)

#######################################################################
## RAW DATA PLOT: MEG (X) + ACC (x, y, z) + ACC norm
#######################################################################

print("\n=== Plotting Raw MEG and ACC Data ===")

# 1. MEG X channels (already extracted as X_raw)
# 2. MEG raw vector norm
# 3. ACC norm (calculated from raw axes)

# Create figure with 3 subplots
fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

# Colors for MEG channels
colors_meg = plt.cm.rainbow(np.linspace(0, 1, len(X_raw)))

# 1. First subplot: MEG X channels (raw)
for i, channel in enumerate(X_raw):
    axes[0].plot(time_lvm_rel[lvm_mask], channel[lvm_mask], 
                 color=colors_meg[i], linewidth=0.6, label=f'Channel {i+1}')
axes[0].set_title('MEG X Component - All Channels (Raw)')
axes[0].set_ylabel('Amplitude (a.u.)')
axes[0].grid(True, alpha=0.3)
axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

# 2. Second subplot: MEG raw vector norm
for i, norm in enumerate(meg_norms_raw):
    axes[1].plot(time_lvm_rel[lvm_mask], norm[lvm_mask], 
                 color=colors_meg[i], linewidth=0.6, label=f'Channel {i+1}')
axes[1].set_title('MEG Vector Norm - All Channels (Raw)')
axes[1].set_ylabel('Magnitude (a.u.)')
axes[1].grid(True, alpha=0.3)

# 3. Third subplot: ACC norm (raw)
axes[2].plot(time_con_rel[con_mask], acc_norm_raw[con_mask], color='purple', linewidth=1, label='ACC magnitude')
axes[2].set_title('Accelerometer Vector Norm (Raw)')
axes[2].set_xlabel('Time from Trigger (s)')
axes[2].set_ylabel('Magnitude (a.u.)')
axes[2].grid(True, alpha=0.3)
axes[2].legend()

plt.suptitle('Synchronized MEG (Raw) and ACC (Raw) Data', fontsize=14)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()

#######################################################################
##FILTERED DATA PLOT:
#######################################################################

print("\n=== Plotting Filtered Data ===")

# Create figure with 3 subplots
fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

# Create colormap for MEG channels
n_channels = len(meg_data['X'])
colors = plt.cm.rainbow(np.linspace(0, 1, n_channels))

# Create list of channel numbers (excluding 5 and 13)
channel_numbers = [i+1 for i in range(20) if i not in channels_to_exclude]

# 1. First subplot: MEG X channels
for i, channel in enumerate(meg_data['X']):
    axes[0].plot(time_lvm_rel[lvm_mask], channel[lvm_mask], 
                 color=colors[i], linewidth=0.6, label=f'Channel {channel_numbers[i]}')
axes[0].set_title('MEG X Component - All Channels')
axes[0].set_ylabel('Amplitude (pT)')
axes[0].grid(True, alpha=0.3)
axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

# 2. Second subplot: MEG normalized channels
meg_norms = calculate_meg_norm(meg_data)
for i, norm in enumerate(meg_norms):
    axes[1].plot(time_lvm_rel[lvm_mask], norm[lvm_mask], 
                 color=colors[i], linewidth=0.6, label=f'Channel {channel_numbers[i]}')
axes[1].set_title('MEG Vector Norm - All Channels')
axes[1].set_ylabel('Magnitude')
axes[1].grid(True, alpha=0.3)

# 3. Third subplot: ACC normalized magnitude
acc_norm = calculate_acc_norm(acc_data)
axes[2].plot(time_con_rel[con_mask], acc_norm[con_mask], 
             color='purple', linewidth=1, label='ACC magnitude')
axes[2].set_title('Accelerometer Vector Norm')
axes[2].set_xlabel('Time from Trigger (s)')
axes[2].set_ylabel('Magnitude')
axes[2].grid(True, alpha=0.3)
axes[2].legend()

plt.suptitle('Synchronized MEG and ACC Data', fontsize=14)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()

###################################################################################
# Plot synchronized data for EMG - MEG - ACC channels
print("\n=== Plotting Synchronized Filtered MEG Norm, ACC Norm, and Right Arm EMG ===")

fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

# 1. MEG norm (filtered)
meg_norms = calculate_meg_norm(meg_data)
for i, norm in enumerate(meg_norms):
    axes[0].plot(time_lvm_rel[lvm_mask], norm[lvm_mask], 
                 color=colors[i], linewidth=0.6, label=f'Channel {channel_numbers[i]}')
axes[0].set_title('MEG Vector Norm - All Channels')
axes[0].set_ylabel('Magnitude')
axes[0].grid(True, alpha=0.3)
axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)


# 2. ACC norm (filtered)
acc_norm = calculate_acc_norm(acc_data)
axes[1].plot(time_con_rel[con_mask], acc_norm[con_mask], 
             color='purple', linewidth=1.2, label='ACC magnitude')
axes[1].set_title('Accelerometer Vector Norm - Filtered')
axes[1].set_ylabel('Magnitude')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

# 3. Right arm EMG (filtered)
axes[2].plot(time_con_rel[con_mask], emg_data['right_arm'][con_mask], 
             color='black', linewidth=1, label='Right Arm EMG')
axes[2].set_title('Right Arm EMG - Filtered')
axes[2].set_xlabel('Time from Trigger (s)')
axes[2].set_ylabel('Amplitude')
axes[2].grid(True, alpha=0.3)
axes[2].legend()

plt.suptitle('Synchronized Filtered MEG Norm, ACC Norm, and Right Arm EMG', fontsize=14)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()

###################################################################################
# Plot synchronized data for EMG channels

print("\n=== Plotting Synchronized Filtered EMG Channels ===")

emg_locations = ['right_arm', 'neck', 'left_arm', 'right_leg', 'left_leg']
emg_labels = [
    'Right Arm EMG',
    'Neck EMG',
    'Left Arm EMG',
    'Right Leg EMG',
    'Left Leg EMG'
]

fig, axes = plt.subplots(5, 1, figsize=(15, 18), sharex=True)

for idx, (loc, label) in enumerate(zip(emg_locations, emg_labels)):
    emg_signal = emg_data[loc]
    # If the signal is 2D (from bipolar subtraction), flatten it
    if hasattr(emg_signal, "ndim") and emg_signal.ndim > 1:
        emg_signal = emg_signal[0]
    axes[idx].plot(time_con_rel[con_mask], emg_signal[con_mask], color='black', linewidth=1, label=label)
    axes[idx].set_title(label, fontsize=8)
    axes[idx].set_ylabel('Amplitude')
    axes[idx].grid(True, alpha=0.3)

axes[-1].set_xlabel('Time from Trigger (s)')
plt.suptitle('Synchronized Filtered EMG Channels', fontsize=12)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()