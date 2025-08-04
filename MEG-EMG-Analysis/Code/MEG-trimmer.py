'''
MEG Trimmer - Uses the same file import logic as MEG-h5-synchro.py
This script is used to trimm the MEG files for the raw data folder

'''

############################################################################
####### LIBRARIES ##########################################################
############################################################################

import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy.signal import find_peaks
from source.find_paths import (get_onedrive_path,
                               get_available_subs,)
from source.trimmer_functions import (trim_meg_from_trigger_minimum,
                                      trim_meg_to_match_emg_duration)
from source.MEG_analysis_functions import (apply_meg_filters)

############################################################################
####### FILE IMPORT #######################################################
############################################################################

# Use the find_paths functions instead of hardcoded paths
try:
    # Try to get the data path using the OneDrive function
    data_dir = get_onedrive_path('data')  
    print(f"Data directory found: {data_dir}")
except:
    # Fallback to the original method if OneDrive path doesn't work
    print("OneDrive path not found, using fallback method...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, '..', 'Data')

source_data_path = get_onedrive_path('source_data')
processed_data_path = get_onedrive_path('processed_data')

available_subs_source = get_available_subs('data', source_data_path)
available_subs_processed = get_available_subs('data', processed_data_path)

print(f'availabe subjects in {source_data_path}', available_subs_source)
print(f'availabe subjects in {processed_data_path}', available_subs_processed)

# ------ Define the subject for this analysis ------
SUB = 'sub-91'  # dataset from June 2025

# create new source_data subject directory
sub_source_data_dir = os.path.join(source_data_path, SUB)
sub_processed_data_dir = os.path.join(processed_data_path, SUB)

# Paths and filenames:
con_file_path = os.path.join(sub_source_data_dir, "OPM_MEG/")
con_file_name = 'sub-91_OPM-MEG_setupB_RestMockDys.con'
processed_h5_path = os.path.join(sub_processed_data_dir, "EMG_ACC")
processed_h5_name = 'sub-91_EmgAcc_setupB_RestMockDys_processed.h5'

# Read the .con file using MNE
con_raw = mne.io.read_raw_kit(con_file_path + con_file_name, preload=True)
channel_names = con_raw.ch_names

# Read the processed .h5 file (contains EMG and ACC data)
processed_h5_full_path = processed_h5_path + '/' + processed_h5_name
processed_h5_df = pd.read_hdf(processed_h5_full_path)
processed_h5_channel_names = list(processed_h5_df.columns)
processed_h5_data = processed_h5_df.values.T  # shape: (n_channels, n_times)
processed_h5_sfreq = 1000  # Hz
processed_h5_times = np.arange(processed_h5_data.shape[1]) / processed_h5_sfreq

# Extract trigger channel from MEG (.con) file (channel 157, index 156)
MEG_trigger_channel = con_raw.get_data(picks=[156])[0]
MEG_sfreq = con_raw.info["sfreq"]  # MEG sampling frequency

# Define EMG and ACC channel mappings for the h5 file
print("Defined EMG channels:", list(processed_h5_channel_names))

"""
Available channels in processed H5 file:
- EMG channels: 'brachioradialis_L', 'deltoideus_L', 'tibialisAnterior_L', 
                'brachioradialis_R', 'deltoideus_R', 'tibialisAnterior_R'
- ACC channels: 'acc_x_hand_L', 'acc_y_hand_L', 'acc_z_hand_L', 
                'acc_x_hand_R', 'acc_y_hand_R', 'acc_z_hand_R'
- Other channels: 'Sync_Time (s)', 'Source_Time (s)', 'SVM_L', 'SVM_R'

EMG channels correspond to:
- brachioradialis: forearm muscles (flexion/extension)
- deltoideus: shoulder muscles  
- tibialisAnterior: leg muscles (dorsiflexion)
- L/R suffix indicates Left/Right side
"""

emg_channels = {
    "left_forearm": "brachioradialis_L",
    "right_forearm": "brachioradialis_R",
    "left_shoulder": "deltoideus_L", 
    "right_shoulder": "deltoideus_R",
    "left_leg": "tibialisAnterior_L",
    "right_leg": "tibialisAnterior_R"
}



# --- Defining MEG channels ---
meg_channel_map = {
    "C3": "MEG001",
    "C4": "MEG002",
    "CZ": "MEG003",
    "F3": "MEG004",
    "F4": "MEG005",
    "P3": "MEG006",
    "P4": "MEG007"
}
meg_channel_indices = list(range(7))
meg_channel_names = [con_raw.ch_names[i] for i in meg_channel_indices]


############################################################################
####### TRIMMING ######################################
############################################################################
trigger_channel_index = 156

# Apply the trimming function
print("Trimming MEG data from first deep trigger minimum...")
con_raw_trimmed, t_zero_idx, t_zero_time = trim_meg_from_trigger_minimum(con_raw, trigger_channel_index)

print(f"\nOriginal MEG data duration: {con_raw.times[-1]:.2f} seconds")
print(f"Trimmed MEG data duration: {con_raw_trimmed.times[-1]:.2f} seconds")
print(f"Time removed: {t_zero_time:.2f} seconds")

# Create timelines for comparison
# EMG timeline
emg_n_samples = len(processed_h5_df)
EMG_timeline = emg_n_samples / processed_h5_sfreq

# MEG timeline (using trimmed data)
meg_n_samples = len(con_raw_trimmed.times)
MEG_timeline = meg_n_samples / con_raw_trimmed.info['sfreq']

# Print comparison
print(f"\n=== TIMELINE COMPARISON ===")
print(f"EMG samples: {emg_n_samples}")
print(f"EMG sample frequency: {processed_h5_sfreq} Hz")
print(f"EMG timeline: {EMG_timeline:.3f} seconds")
print(f"")
print(f"MEG samples (trimmed): {meg_n_samples}")
print(f"MEG sample frequency: {con_raw_trimmed.info['sfreq']} Hz")
print(f"MEG timeline: {MEG_timeline:.3f} seconds")
print(f"")
print(f"Timeline difference: {abs(EMG_timeline - MEG_timeline):.3f} seconds")


#############################################################
### CREATE NEW DATAFRAME
#############################################################

# Calculate time difference and trim MEG to match EMG duration
time_difference = MEG_timeline - EMG_timeline
print(f"\nTime difference (MEG - EMG): {time_difference:.3f} seconds")

if time_difference > 0:
    print("MEG is longer than EMG. Trimming MEG to match EMG duration...")
    con_raw_final, time_removed = trim_meg_to_match_emg_duration(con_raw_trimmed, time_difference)
    
    # Recalculate final timeline
    final_meg_samples = len(con_raw_final.times)
    final_MEG_timeline = final_meg_samples / con_raw_final.info['sfreq']
    
    print(f"\n=== FINAL TIMELINE COMPARISON ===")
    print(f"EMG timeline: {EMG_timeline:.3f} seconds")
    print(f"MEG timeline (final): {final_MEG_timeline:.3f} seconds")
    print(f"Final difference: {abs(EMG_timeline - final_MEG_timeline):.3f} seconds")
else:
    print("EMG is longer than or equal to MEG. No additional trimming needed.")
    con_raw_final = con_raw_trimmed

#############################################################
### FILTERING ###

lpass = 100  # Hz
Hpass = 1    # Hz

print(f"Applying band-pass filter: {Hpass} - {lpass} Hz")

# Get MEG data from the Raw object
meg_data = con_raw_final.get_data(picks=mne.pick_types(con_raw_final.info, meg=True))

# Apply filters to MEG data
meg_data_filtered = apply_meg_filters(meg_data, MEG_sfreq, l_freq=Hpass, h_freq=lpass)

# Create a copy of the raw object and replace the MEG data with filtered data
con_raw_filtered = con_raw_final.copy()
meg_picks = mne.pick_types(con_raw_filtered.info, meg=True)
con_raw_filtered._data[meg_picks] = meg_data_filtered

print("MEG data filtering completed!")


############################################################################
############################################################################
####### PLOTTING ###########################################################
############################################################################

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Extract C3 MEG channel data (filtered)
# C3 corresponds to MEG001 (index 0 based on your meg_channel_map)
c3_channel_idx = 0  # MEG001 is the first MEG channel
c3_filtered_data = con_raw_filtered.get_data(picks=[c3_channel_idx])[0]
filtered_times = con_raw_filtered.times

# Plot 1: Filtered C3 MEG channel
ax1.plot(filtered_times, c3_filtered_data, 'g-', linewidth=1)
ax1.set_title('Filtered C3 MEG Channel (1-100 Hz bandpass + notch filters)', fontsize=12)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('MEG Signal (T)')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, filtered_times[-1])

# Plot 2: EMG right forearm channel
emg_right_forearm_data = processed_h5_df[emg_channels["right_forearm"]].values
ax2.plot(processed_h5_times, emg_right_forearm_data, 'r-', linewidth=1)
ax2.set_title('EMG Right Forearm Channel', fontsize=12)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('EMG Signal (µV)')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, processed_h5_times[-1])

# Adjust layout and show
plt.tight_layout()
plt.show()

############################################################################
####### SAVE TRIMMED MEG DATA ##############################################
############################################################################

# Define output path and filename
processed_data_path = get_onedrive_path('processed_data')
available_subs_processed = get_available_subs('data', processed_data_path)
sub_processed_data_dir = os.path.join(processed_data_path, SUB)

output_dir = os.path.join(sub_processed_data_dir, "OPM_MEG/")
os.makedirs(output_dir, exist_ok=True)

# Create output filename with trimming info following MNE conventions
original_name = con_file_name.replace('.con', '')
output_filename = f"{original_name}_processed.fif"  # Changed to follow MNE naming convention
output_full_path = os.path.join(output_dir, output_filename)

print(f"\n=== SAVING TRIMMED MEG DATA ===")
print(f"Output directory: {output_dir}")
print(f"Output filename: {output_filename}")

# Save the trimmed MEG data as .fif file
# This preserves all channel information, sampling rate, and data integrity
con_raw_filtered.save(output_full_path, overwrite=True, verbose=False)

print(f"Trimmed MEG data saved successfully!")
print(f"Full path: {output_full_path}")

