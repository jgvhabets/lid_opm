import numpy as np
import pandas as pd
import seaborn as sns
import mne
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

#test = mne.io.read_raw_ant("C:/Users/User/Documents/bachelorarbeit/data/Accelerometer/Test_Accelerometer_2025-04-22_11-53-42.cnt", preload=True) # before any changes in software
#test = mne.io.read_raw_ant("C:/Users/User/Documents/bachelorarbeit/data/Accelerometer/Test_Accelerometer_2025-04-22_16-01-45_after_changed.cnt", preload=True) # after the changed settings
test = mne.io.read_raw_ant("C:/Users/User/Documents/bachelorarbeit/data/Accelerometer/Test_Accelerometer_2025-04-24_11-57-41_after_changed2.cnt", preload=True) # again after changes


channel_names = ["BIP3", "BIP4", "BIP5", "BIP1", "BIP2", "BIP6"]
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

#fig, axs = plt.subplots(3, 1, figsize=(12,8))
#axs = axs.ravel()
for (ch1,ch2) in pairs:
    plt.figure(figsize=(15,10))
    plt.plot(ACC_df["Time (s)"], ACC_df[ch1], label=location[ch1])
    plt.plot(ACC_df["Time (s)"], ACC_df[ch2], label=location[ch2])
    plt.title(title[ch1])
    plt.ylabel("Acceleration (g)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.show()