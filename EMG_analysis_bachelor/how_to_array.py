import numpy as np
import pandas as pd
import seaborn as sns
import mne
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

#test = mne.io.read_raw_ant("C:/Users/User/Documents/bachelorarbeit/data/Accelerometer/"
 #                          "Test_Accelerometer_2025-04-22_16-01-45_withNotch.cnt", preload=True)
#test = mne.io.read_raw_ant("C:/Users/User/Documents/bachelorarbeit/data/Accelerometer/"
#                           "Test_Accelerometer_2025-04-22_14-22-47.cnt", preload=True)
#
#full_array = test.get_data()
## taking only first 6 channels = ACC channels
#ACC_array = full_array[0:6]
#
## hier jetzt notch filter entweder selber machen oder file mit notch nehmen + low pass filter implementieren
#ACC_array = mne.filter.notch_filter(x=ACC_array[0:6], freqs=[50,100,150], Fs=1000)
#low_filtered = mne.filter.filter_data(ACC_array, sfreq=1000, l_freq=2, h_freq=None)
#
## zwei mal integrieren:
#def integration_to_velocity_position(data, sfreq):
#    """Integrate acceleration (g) to velocity (m/s) and position (m).
#    Returns: (velocity_channels, position_channels), both shaped (6, N)"""
#
#    dt = 1.0 / sfreq
#    g_to_ms2 = 9.81
#
#    acc_in_ms2 = data * g_to_ms2
#
#    velocity_ms = np.cumsum(acc_in_ms2, axis=1) * dt
#    position_m = np.cumsum(velocity_ms, axis=1) * dt
#
#    return velocity_ms, position_m
#
#velocity_channels, position_channels = integration_to_velocity_position(low_filtered, 1000)
#
## raw und nach high pass filter vergleichen
#plt.figure()
#plt.plot(test.times, ACC_array[1], "b", label="raw")
#plt.plot(test.times, low_filtered[1], "r", label="filtered 2Hz")
#plt.xlabel("Time (s)")
#plt.ylabel(f"Acceleration (g)")
#plt.title(test.ch_names[1])
#plt.show()
#
## jeweils ein plot für jeden channel mit 3 signalen : filtered, nach 1. integration und nach 2. integr.:
#fig, axs = plt.subplots(2, 3, figsize=(15,8))
#custom_order = [5, 0, 1, 2, 3, 4]
## Loop through channels in custom order
#for idx, i in enumerate(custom_order):
#    row = idx // 3  # 0 or 1 for 2x3 grid
#    col = idx % 3  # 0, 1, or 2
#
#    axs[row, col].plot(test.times, low_filtered[i], 'r', label='Filtered (g)', linewidth=1)
#    axs[row, col].plot(test.times, velocity_channels[i], 'g', label='Velocity (m/s)', linewidth=1)
#    axs[row, col].plot(test.times, position_channels[i], 'purple', label='Position (m)', linewidth=1)
#
#    axs[row, col].set_title(test.ch_names[i])
#    axs[row, col].set_xlabel('Time (s)')
#    axs[row, col].grid(True)
#
#    if idx == 0:
#        axs[row, col].legend()
#
#plt.tight_layout()
#plt.show()



# testing signals with two boxes:
file = mne.io.read_raw_ant("C:/Users/User/Documents/bachelorarbeit/data/Accelerometer/Test_Accelerometer_2025-04-29_12-43-27_two_boxes_test.cnt", preload=True)
custom_order_names = ["BIP6", "BIP1", "BIP2", "BIP9", "BIP10", "BIP11"]

data, times = file[custom_order_names, : ]

data_notched = mne.filter.notch_filter(data, freqs=[50, 100, 150], Fs=1000)
two_box = mne.filter.filter_data(data_notched, sfreq = 1000, l_freq=2, h_freq=None)
two_box[0, :] *= -1

fig, axs = plt.subplots(2, 3, figsize=(15,8))
axs = axs.ravel()
for idx, channel_name in enumerate(custom_order_names):
    axs[idx].plot(times, two_box[idx], "r", linewidth=1)
    #axs[idx].set_title(f"{channel_name} - {file.ch_names[file.ch_names.index(channel_name)]}")

    axs[idx].set_title(f"{channel_name}")
    axs[idx].set_xlabel('Time (s)')
    axs[idx].grid(True)

plt.tight_layout()
plt.show()

