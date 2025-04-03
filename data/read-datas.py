
############ COMMENTS ############
# The two files have different lengths and different sample frequencies.
# The .lvm file has a sample frequency of 375 Hz, 
# while the .con file has a sample frequency of 500 Hz.
# My idea consists in interpolating the .lvm data to match the .con data (switch from 375 Hz to 500 Hz).
# Then consider just the number of samples that are common to both files.
# I still don't find any trigger channel in the .con file.


#################
### LIBRARIES ###
#################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from scipy.stats import pearsonr
import mne

#################
### FUNCTIONS ###
#################

####### CHECK ZERO #######

def check_zero(channel_list, channel_names):
    """Helper function to check for zero-filled channels"""
    c = 0
    for i, channel in enumerate(channel_list):
        if np.all(channel == 0):
            c = c + 1
            print(f"The channel {channel_names[i]} is filled with 0's")
    print(f"There are {c} empty channels.")

###### LVM DATA ######

def extract_lvm_data(filepath):
    """Extract MEG and trigger data from .lvm file"""
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
    
    # Get timing and trigger
    time = df["X_Value"].values
    trigger_lvm = df["MUX_Counter1"].values
    
    meg_data = {
        'X': X_channels,
        'Y': Y_channels,
        'Z': Z_channels,
        'time': time,
        'channel_names': X_channels_names[:20]
    }
    
    return meg_data, trigger_lvm, df.shape[0]

###### CON DATA ######

def extract_con_data(filepath):
    """Extract EEG, EMG, ACC and trigger data from .con file"""
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
    raw_filtered.filter(l_freq=1, h_freq=200, picks=eeg_picks)
    
    # Extract EMG data (bipolar)
    emg_data = {}
    for location, channels in emg_channels.items():
        if isinstance(channels, list):
            ch1 = raw.get_data(picks=[channels[0]])[0]
            ch2 = raw.get_data(picks=[channels[1]])[0]
            emg_data[location] = ch1 - ch2
        else:
            emg_data[location] = raw.get_data(picks=[channels])[0]
    
    # Extract ACC data
    acc_data = {
        axis: raw.get_data(picks=[channel])[0]
        for axis, channel in acc_channels.items()
    }
    
    # Get trigger and filtered EEG
    trigger_con = raw.get_data(picks=[trigger_channel])[0]
    eeg_data = raw_filtered.get_data(picks=eeg_picks)
    
    return emg_data, acc_data, eeg_data, trigger_con, raw.info['sfreq']

##### MEG norm #####

def calculate_meg_norm(meg_data, save_plot=False):
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
    
    # Plot setup
    n_cols = 5
    n_rows = (n_channels + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 12), sharex=True)
    axes = axes.flatten()
    
    # Plot each norm
    for i in range(n_channels):
        axes[i].plot(meg_data['time'], norm_channels[i], 
                    color='purple', linewidth=1)
        channel_num = ''.join(filter(str.isdigit, meg_data['channel_names'][i]))
        norm_name = f"Norm{channel_num}"
        
        axes[i].set_title(norm_name, fontsize=10)
        axes[i].grid(True, alpha=0.3)
        
        if i % n_cols == 0:
            axes[i].set_ylabel('Magnitude')
    
    # Hide unused subplots
    for i in range(n_channels, len(axes)):
        axes[i].set_visible(False)
    
    # Add x-labels
    for i in range(n_cols * (n_rows-1), n_cols * n_rows):
        if i < len(axes) and axes[i].get_visible():
            axes[i].set_xlabel('Time (sec)')
    
    plt.suptitle("MEG Vector Norm - Individual Channels", fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    if save_plot:
        if not os.path.exists('plot'):
            os.makedirs('plot')
        plt.savefig('plot/MEG_norm_channels.png')
        print('*** MEG norm plot saved in plot folder ***')
    
    
    return norm_channels

###### ACC norm ######

def calculate_acc_norm(acc_data, save_plot=False):
    """Calculate and plot the norm of accelerometer X, Y, Z components"""
    # Calculate norm
    norm = np.sqrt(
        acc_data['x']**2 + 
        acc_data['y']**2 + 
        acc_data['z']**2
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 5))
    
    # Plot norm
    ax.plot(norm, color='purple', linewidth=1)
    ax.set_title('Accelerometer Vector Norm', fontsize=12)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Magnitude')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plot:
        if not os.path.exists('plot'):
            os.makedirs('plot')
        plt.savefig('plot/ACC_norm.png')
        print('*** ACC norm plot saved in plot folder ***')
    
    
    return norm


#######################################################################
#######################################################################


#################
### MAIN CODE ###
#################

# File paths
lvm_path = "plfp65/plfp65_rec4_13.11.2024_13-17-33_array1.lvm"
con_path = "plfp65/plfp65_rec4.con"

print("=== Extracting Data from Files ===")

# Get LVM data (MEG)
print("\nExtracting LVM data...")
meg_data, trigger_lvm, n_samples_lvm = extract_lvm_data(lvm_path)
print(f"Found {len(meg_data['X'])} MEG channels")

# Get CON data (EEG, EMG, ACC)
print("\nExtracting CON data...")
emg_data, acc_data, eeg_data, trigger_con, sfreq = extract_con_data(con_path)
print(f"Found {len(emg_data)} EMG locations")
print(f"Found {len(acc_data)} ACC axes")
print(f"Found {eeg_data.shape[0]} EEG channels")

# Get command signal from LVM:
df = pd.read_csv(lvm_path, header=22, sep='\t')
all_columns = df.columns.tolist()
command_signal = df[all_columns[218]].values



#######################################################################
##RAW DATA PLOT:


print("\n=== Plotting Data ===")

# Define colors for components
component_colors = {
    'X': 'blue',
    'Y': 'green',
    'Z': 'red'
}

# 1. Plot MEG components (X, Y, Z) in subplots
for component in ['X', 'Y', 'Z']:
    n_channels = len(meg_data[component])
    n_cols = 5
    n_rows = (n_channels + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 12), sharex=True)
    axes = axes.flatten()
    
    for i, channel in enumerate(meg_data[component]):
        # Use full time series instead of just first 1000 samples
        axes[i].plot(meg_data['time'], channel, 
                    color=component_colors[component], linewidth=1)
        axes[i].set_title(f'{meg_data["channel_names"][i]}', fontsize=10)
        axes[i].grid(True, alpha=0.3)
        
        if i % n_cols == 0:
            axes[i].set_ylabel('Amplitude')
    
    # Hide unused subplots
    for i in range(n_channels, len(axes)):
        axes[i].set_visible(False)
    
    # Add x-label only for bottom plots
    for i in range(n_cols * (n_rows-1), n_cols * n_rows):
        if i < len(axes) and axes[i].get_visible():
            axes[i].set_xlabel('Time (sec)')
    
    plt.suptitle(f"MEG {component} Component - Individual Channels", fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()

# 2. Plot EMG channels in subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
axes = axes.flatten()

for i, (location, data) in enumerate(emg_data.items()):
    axes[i].plot(data[:1000], 'black', linewidth=1)
    axes[i].set_title(f'EMG {location}', fontsize=10)
    axes[i].grid(True, alpha=0.3)
    axes[i].set_ylabel('Amplitude')
    axes[i].set_xlabel('Sample')

# Hide the last unused subplot
axes[-1].set_visible(False)

plt.suptitle("EMG Channels", fontsize=12)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()

# 3. Plot ACC components in subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)

for i, (axis, data) in enumerate(acc_data.items()):
    axes[i].plot(data[:1000], linewidth=1)
    axes[i].set_title(f'ACC {axis}-axis', fontsize=10)
    axes[i].grid(True, alpha=0.3)
    axes[i].set_xlabel('Sample')
    if i == 0:
        axes[i].set_ylabel('Amplitude')

plt.suptitle("Accelerometer Components", fontsize=12)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()

# 4. Plot LVM trigger channel
plt.figure(figsize=(15, 4))
plt.plot(meg_data['time'], trigger_lvm, color='red', linewidth=1)
plt.title('LVM Trigger Channel', fontsize=12)
plt.xlabel('Time (sec)')
plt.ylabel('Amplitude')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


###################################################################################
###################################################################################
## PROCESSING DATA:
# Now let's proceed with 2 steps:
# 1. Combining the x, y, z components of MEG data:

print("\n=== Processing MEG Components ===")
meg_norms = calculate_meg_norm(meg_data, save_plot=True)
print("\n=== MEG normalized plots are saved in 'plot' folder ===")

# 2. Combing the x, y, z components of ACC data:

print("\n=== Processing ACC Components ===")
acc_norm = calculate_acc_norm(acc_data, save_plot=True)
print("\n=== ACC normalized plot is saved in 'plot' folder ===")

