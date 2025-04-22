############ COMMENTS ############
# The two files have different lengths and different sample frequencies.
# The .lvm file has a sample frequency of 375 Hz, 
# while the .con file has a sample frequency of 500 Hz.
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

def apply_filter(data, sfreq, l_freq, h_freq):
    """Apply bandpass filter to data using MNE"""
    # Convert data to MNE RawArray format
    info = mne.create_info(ch_names=['signal'], sfreq=sfreq, ch_types=['misc'])
    raw = mne.io.RawArray(data.reshape(1, -1), info, verbose=False)
    
    # Apply filter with explicit picks and suppress output
    raw.filter(l_freq=l_freq, h_freq=h_freq, picks=[0], verbose=False)
    
    return raw.get_data()[0]

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
    
    # Get LVM sampling frequency (375 Hz)
    sfreq_lvm = 375
    
    # Filter X, Y, Z channels between 1-100 Hz and convert to pT
    conversion_factor = 1e-12  # Convert to pT 
    X_channels_filtered = [apply_filter(ch, sfreq_lvm, 1, 100) * conversion_factor for ch in X_channels]
    Y_channels_filtered = [apply_filter(ch, sfreq_lvm, 1, 100) * conversion_factor for ch in Y_channels]
    Z_channels_filtered = [apply_filter(ch, sfreq_lvm, 1, 100) * conversion_factor for ch in Z_channels]
    
    # Get timing and trigger
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
    
    # Get trigger and filtered EEG
    trigger_con = raw.get_data(picks=[trigger_channel])[0]
    eeg_data = raw_filtered.get_data(picks=eeg_picks)
    
    return emg_data, acc_data, eeg_data, trigger_con, raw.info['sfreq']

##### MEG norm #####

def calculate_meg_norm(meg_data, save_plot=False, lvm_filename=None):
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
    
    if save_plot and lvm_filename:
        # Create plot/MEG folder structure if it doesn't exist
        plot_folder = '../plot'
        meg_folder = os.path.join(plot_folder, 'MEG')
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
        if not os.path.exists(meg_folder):
            os.makedirs(meg_folder)
            
        # Extract filename without path and extension
        base_filename = os.path.splitext(os.path.basename(lvm_filename))[0]
        
        # Save plot with filename in plot/MEG folder
        plot_path = os.path.join(meg_folder, f'{base_filename}_norm.png')
        plt.savefig(plot_path)
        print(f'*** MEG norm plot saved as {plot_path} ***')
    
    return norm_channels

###### ACC norm ######

def calculate_acc_norm(acc_data, save_plot=False, con_filename=None):
    """Calculate and plot the norm of accelerometer X, Y, Z components"""
    # Calculate norm
    norm = np.sqrt(
        acc_data['x']**2 + 
        acc_data['y']**2 + 
        acc_data['z']**2
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 5))
    
    # Plot norm using time_con
    ax.plot(time_con, norm, color='purple', linewidth=1)
    ax.set_title('Accelerometer Vector Norm', fontsize=12)
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Magnitude')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plot and con_filename:
        # Create plot/ACC folder structure if it doesn't exist
        plot_folder = '../plot'
        acc_folder = os.path.join(plot_folder, 'ACC')
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
        if not os.path.exists(acc_folder):
            os.makedirs(acc_folder)
            
        # Extract filename without path and extension
        base_filename = os.path.splitext(os.path.basename(con_filename))[0]
        
        # Save plot with filename in plot/ACC folder
        plot_path = os.path.join(acc_folder, f'{base_filename}_acc_norm.png')
        plt.savefig(plot_path)
        print(f'*** ACC norm plot saved as {plot_path} ***')
    
    return norm

def plot_emg_channels(emg_data, time_con, save_plot=False, con_filename=None):
    """Plot and optionally save EMG channel plots"""
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    axes = axes.flatten()

    for i, (location, data) in enumerate(emg_data.items()):
        axes[i].plot(time_con, data, 'black', linewidth=1)
        axes[i].set_title(f'EMG {location}', fontsize=10)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylabel('Amplitude')
        axes[i].set_xlabel('Time (sec)')

    # Hide the last unused subplot
    axes[-1].set_visible(False)

    plt.suptitle("EMG Channels", fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    if save_plot and con_filename:
        # Create plot/EMG folder structure if it doesn't exist
        plot_folder = '../plot'
        emg_folder = os.path.join(plot_folder, 'EMG')
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
        if not os.path.exists(emg_folder):
            os.makedirs(emg_folder)
            
        # Extract filename without path and extension
        base_filename = os.path.splitext(os.path.basename(con_filename))[0]
        
        # Save plot with filename in plot/EMG folder
        plot_path = os.path.join(emg_folder, f'{base_filename}_emg.png')
        plt.savefig(plot_path)
        print(f'*** EMG plots saved as {plot_path} ***')
 

#######################################################################
#######################################################################


#################
### MAIN CODE ###
#################

# File paths
os.chdir(os.path.dirname(os.path.abspath(__file__)))
lvm_path = "../Data/plfp65_rec4_13.11.2024_13-17-33_array1.lvm"
con_path = "../Data/plfp65_rec4.con"

print("=== Extracting Data from Files ===")

# Get LVM data (MEG)
print("\nExtracting LVM data...")
meg_data, n_samples_lvm = extract_lvm_data(lvm_path)
print(f"Found {len(meg_data['X'])} MEG channels")

# Get CON data (EEG, EMG, ACC)
print("\nExtracting CON data...")
emg_data, acc_data, eeg_data, trigger_con, sfreq = extract_con_data(con_path)
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
    axes[i].plot(time_con, data, 'black', linewidth=1)  # Removed [:1000] to show all data
    axes[i].set_title(f'EMG {location}', fontsize=10)
    axes[i].grid(True, alpha=0.3)
    axes[i].set_ylabel('Amplitude')
    axes[i].set_xlabel('Time (sec)')


# Hide the last unused subplot
axes[-1].set_visible(False)

plt.suptitle("EMG Channels", fontsize=12)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()

plot_emg_channels(emg_data, time_con, save_plot=True, con_filename=con_path)
print("\n=== EMG plots are saved in 'plot/EMG' folder ===")


# 3. Plot ACC components in subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)

for i, (axis, data) in enumerate(acc_data.items()):
    axes[i].plot(time_con, data, linewidth=1)  # Removed [:1000] to show all data
    axes[i].set_title(f'ACC {axis}-axis', fontsize=10)
    axes[i].grid(True, alpha=0.3)
    axes[i].set_xlabel('Time (sec)')
    if i == 0:
        axes[i].set_ylabel('Amplitude')

plt.suptitle("Accelerometer Components", fontsize=12)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()

###################################################################################
###################################################################################
## PROCESSING DATA:
# Now let's proceed with 2 steps:
# 1. Combining the x, y, z components of MEG data:

print("\n=== Processing MEG Components ===")
meg_norms = calculate_meg_norm(meg_data, save_plot=True, lvm_filename=lvm_path)
print("\n=== MEG normalized plots are saved in 'MEG' folder ===")

# 2. Combing the x, y, z components of ACC data:


print("\n=== Processing ACC Components ===")
acc_norm = calculate_acc_norm(acc_data, save_plot=True, con_filename=con_path)
print("\n=== ACC normalized plot is saved in 'ACC' folder ===")

plot_emg_channels(emg_data, time_con, save_plot=True, con_filename=con_path)
