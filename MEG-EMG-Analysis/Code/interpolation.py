#################
### LIBRARIES ###
#################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re  # For regular expressions to extract numbers
from scipy.stats import pearsonr
import mne

#########################################################
#########################################################

print("\n=== Comparing Channel Lengths ===")

# Read LVM file
lvm_path = "plfp65/plfp65_rec1_13.11.2024_12-51-13_array1.lvm"
df = pd.read_csv(lvm_path, header=22, sep='\t')
x_value = df['X_Value']
last_x_Value = x_value[len(x_value)-1]


# Get STI 014 from CON file
# Read CON file
con_path = "plfp65/plfp65_rec1.con"
raw = mne.io.read_raw_kit(con_path, preload=True)
sti_data = raw.get_data(picks=['STI 014'])[0]

# Compare time and sample frequency:
lvm_time = (last_x_Value)/60 #minutes
con_time = ((len(sti_data))*0.002)/60 #minutes
print('In lvm file there are: ', len(x_value), 'samples')
print('In con file there are: ', len(sti_data), 'samples')
print('THe lvm covers: ', lvm_time, 'minutes')
print('The con covers: ', con_time, 'minutes')

exit()
#########################################################
#########################################################
'''
# Let's interpolate the data to match the length of the CON file:


print("\n=== Interpolating LVM data from 375Hz to 500Hz ===")

# Create new time base at 500Hz
time_orig = x_value.values
time_500hz = np.linspace(time_orig[0], time_orig[-1], int(len(time_orig) * 500/375))

# Interpolate X_Value to 500Hz
x_value_500hz = np.interp(time_500hz, time_orig, x_value)

# Read and interpolate MEG channels
meg_channels = []
for col in df.columns:
    if any(x in col for x in ['X', 'Y', 'Z']) and col != 'X_Value':
        meg_channels.append(col)
        
meg_data_500hz = {}
for ch in meg_channels:
    meg_data_500hz[ch] = np.interp(time_500hz, time_orig, df[ch].values)

# Interpolate trigger channel
trigger_500hz = np.interp(time_500hz, time_orig, df['MUX_Counter1'].values)

# Print results
print(f"Original length: {len(x_value)} samples")
print(f"Interpolated length: {len(x_value_500hz)} samples")
print(f"STI 014 length: {len(sti_data)} samples")
print(f"Time covered (original): {x_value_time:.2f} minutes")
print(f"Time covered (interpolated): {time_500hz[-1]/60:.2f} minutes")'
'''