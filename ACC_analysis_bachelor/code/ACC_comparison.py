import numpy as np
import pandas as pd
import seaborn as sns
import mne
import matplotlib
from matplotlib.lines import lineStyles

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from EMG_analysis_bachelor.functions_for_pipeline import create_df, filtered

#test = mne.io.read_raw_ant("C:/Users/User/Documents/bachelorarbeit/data/Accelerometer/Test_Accelerometer_2025-04-22_11-53-42.cnt", preload=True) # before any changes in software
#test = mne.io.read_raw_ant("C:/Users/User/Documents/bachelorarbeit/data/Accelerometer/Test_Accelerometer_2025-04-22_16-01-45_withNotch.cnt", preload=True) # after the changed settings
test = mne.io.read_raw_ant("C:/Users/User/Documents/bachelorarbeit/data/Accelerometer/Test_Accelerometer_2025-04-24_11-57-41_after_changed2.cnt", preload=True) # again after changes
#test = mne.io.read_raw_ant("C:/Users/User/Documents/bachelorarbeit/data/Accelerometer/Test_Accelerometer_2025-04-24_14-04-43_onHand.cnt", preload=True) # sensors on hand

channel_names = ["BIP3", "BIP4", "BIP5", "BIP1", "BIP2", "BIP6"]
ACC = ["BIP3", "BIP4", "BIP5", "BIP1", "BIP2", "BIP6"]
EMG = []
#test = mne.filter.notch_filter(x=test[channel_names], freqs=[50,100,150])
#test = mne.filter.filter_data(data=test, picks=channel_names, sfreq = 1000, l_freq=2, h_freq=48)


location = {"BIP3" : "AntNeuro",
            "BIP4" : "AntNeuro",
            "BIP5" : "AntNeuro",
            "BIP1" : "Charité",
            "BIP2" : "Charité",
            "BIP6" : "Charité"}
title = {"BIP3" : "X axis",
            "BIP4" : "Y axis",
            "BIP5" : "Z axis"}

data, times = test[channel_names, :]
ACC_df = pd.DataFrame(data.T, columns=channel_names)
ACC_df["Time (s)"] = times
ACC_df["BIP6"] = ACC_df["BIP6"] * (-1)
filtered_ACC = filtered(ACC_df,channel_names, ACC, EMG, times)

pairs = [
    ("BIP3", "BIP6"),  # X-axis comparison
    ("BIP4", "BIP1"),  # Y-axis comparison
    ("BIP5", "BIP2")   # Z-axis comparison
]

lims = [15, 35]

fig, axs = plt.subplots(3, 1, figsize=(15,8))
axs = axs.ravel()
for i, (ch1, ch2) in enumerate(pairs):
    axs[i].plot(filtered_ACC["Time (s)"], filtered_ACC[ch1], label=location[ch1], color="b")
    axs[i].plot(filtered_ACC["Time (s)"], filtered_ACC[ch2], label=location[ch2], linestyle="dashed", color="red")
    axs[i].set_xlim(lims[0], lims[1])
    axs[i].set_title(title[ch1])
    axs[i].set_ylabel("Acceleration (g)")
    axs[i].set_xlabel("Time (s)")
    axs[i].legend()
    axs[i].grid(True)
plt.tight_layout()
plt.savefig("ACC_ch_ant_comparison.png")

