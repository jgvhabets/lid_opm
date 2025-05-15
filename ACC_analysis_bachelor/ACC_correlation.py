import numpy as np
import mne
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

test = mne.io.read_raw_ant("C:/Users/User/Documents/bachelorarbeit/data/Accelerometer/"
                           "Test_Accelerometer_2025-04-22_14-22-47.cnt", preload=True)
full_array = test.get_data()

# taking only first 6 channels = ACC channels
ACC_array = full_array[0:6]

# notch filter and high-pass filtering
ACC_array = mne.filter.notch_filter(x=ACC_array[0:6], freqs=[50,100,150], Fs=1000)
low_filtered = mne.filter.filter_data(ACC_array, sfreq=1000, l_freq=2, h_freq=None)
low_filtered[5] *= -1

# correlation coefficient
corr = np.corrcoef(low_filtered[5], low_filtered[2]) [0,1]
print(f"Similarity for X: {corr:.2f}")
corr = np.corrcoef(low_filtered[0], low_filtered[3]) [0,1]
print(f"Similarity for Y: {corr:.2f}")
corr = np.corrcoef(low_filtered[1], low_filtered[4]) [0,1]
print(f"Similarity for Z: {corr:.2f}")