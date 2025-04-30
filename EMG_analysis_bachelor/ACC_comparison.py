import numpy as np
import pandas as pd
import seaborn as sns
import mne
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

#test = mne.io.read_raw_ant("C:/Users/User/Documents/bachelorarbeit/data/Accelerometer/Test_Accelerometer_2025-04-22_11-53-42.cnt", preload=True) # before any changes in software
#test = mne.io.read_raw_ant("C:/Users/User/Documents/bachelorarbeit/data/Accelerometer/Test_Accelerometer_2025-04-22_16-01-45_withNotch.cnt", preload=True) # after the changed settings
#test = mne.io.read_raw_ant("C:/Users/User/Documents/bachelorarbeit/data/Accelerometer/Test_Accelerometer_2025-04-24_11-57-41_after_changed2.cnt", preload=True) # again after changes
test = mne.io.read_raw_ant("C:/Users/User/Documents/bachelorarbeit/data/Accelerometer/Test_Accelerometer_2025-04-24_14-04-43_onHand.cnt", preload=True) # sensors on hand

channel_names = ["BIP3", "BIP4", "BIP5", "BIP1", "BIP2", "BIP6"]
#test = mne.filter.notch_filter(x=test[channel_names], freqs=[50,100,150])
test = mne.filter.filter_data(data=test, picks=channel_names, sfreq = 1000, l_freq=2, h_freq=500)

location = {"BIP3" : "AntNeuro : x",
            "BIP4" : "AntNeuro : y",
            "BIP5" : "AntNeuro : z",
            "BIP1" : "Charité ACC : y",
            "BIP2" : "Charité ACC : z",
            "BIP6 inverted" : "Charité ACC * -1 : x"}
title = {"BIP3" : "X axis in the two sensors (inverted back for charité sensor)",
            "BIP4" : "Y axis in the two sensors",
            "BIP5" : "Z axis in the two sensors"}

data, times = test[channel_names, :]
ACC_df = pd.DataFrame(data.T, columns=channel_names)
ACC_df["Time (s)"] = times
ACC_df["BIP6 inverted"] = ACC_df["BIP6"] * (-1)

pairs = [
    ("BIP3", "BIP6 inverted"),  # X-axis comparison
    ("BIP4", "BIP1"),  # Y-axis comparison
    ("BIP5", "BIP2")   # Z-axis comparison
]

fig, axs = plt.subplots(3, 1, figsize=(15,8))
axs = axs.ravel()
for i, (ch1, ch2) in enumerate(pairs):
    axs[i].plot(ACC_df["Time (s)"], ACC_df[ch1], label=location[ch1])
    axs[i].plot(ACC_df["Time (s)"], ACC_df[ch2], label=location[ch2])
    axs[i].set_title(title[ch1])
    axs[i].set_ylabel("Acceleration (g)")
    axs[i].set_xlabel("Time (s)")
    axs[i].legend()
    axs[i].grid(True)
plt.tight_layout()
plt.show()
