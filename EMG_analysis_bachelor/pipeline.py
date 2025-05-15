import numpy as np
import pandas as pd
import seaborn as sns
import mne
from functions_for_pipeline import create_raw_df, plot_raw_signals, notch_filter, bandpass_filter, create_filtered_df
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

# import data
EMG_ACC_data = mne.io.read_raw_ant("C:/Users/User/Documents/bachelorarbeit/data/EMG_ACC/"
                                   "PTB_measurement_14.04/Bonato_Federico_2025-04-14_13-02-56.cnt", preload=True)
custom_order_names = ["BIP6", "BIP1", "BIP2", "BIP7", "BIP8"]
location = {"BIP7":"left forearm",
            "BIP8": "left delt",
            "BIP1" : "Charité ACC : y",
            "BIP2" : "Charité ACC : z",
            "BIP6" : "Charité ACC : x"}
EMG = ["BIP7", "BIP8"]

data, times = EMG_ACC_data[custom_order_names, : ]
data[0] *= -1

# creating raw df
raw_df = create_raw_df(data, custom_order_names, times)

# plotting the raw signals
plot_raw_signals(custom_order_names, raw_df, location, EMG)

# apply notch filter
notched_data = notch_filter(data, [50, 100, 150], 1000)

# apply band pass filters (specify which Frequencies) -> differences for EMG and ACC! - noch machen!
notched_and_filtered_data = bandpass_filter(notched_data, 1000, 2, None)

# create notched and filtered df
filtered_df = create_filtered_df(notched_and_filtered_data, custom_order_names, times)

# plot filtered data (overview?)
fig, axs = plt.subplots(2,3, figsize=(12,8))
axs = axs.ravel()
for i, channel in enumerate(custom_order_names):
    axs[i].plot(filtered_df["Time (s)"], filtered_df[channel])
    axs[i].set_title(f"{channel} : signal of {location[channel]}")
    axs[i].set_xlabel("Time (s)")
    axs[i].set_ylabel("Amplitude (V)" if channel in EMG else "g")

    if len(axs) > len(custom_order_names):
        axs[-1].axis("off")

plt.tight_layout()
#plt.show()

# retrification and plotting
df_rectified = abs(filtered_df) #technically also took the absolute values of the time column, but that doesnt matter i think

fig, axs = plt.subplots(2,3, figsize=(12,8))
axs = axs.ravel()
for i, channel in enumerate(custom_order_names):
    axs[i].plot(df_rectified["Time (s)"], df_rectified[channel])
    axs[i].set_title(f"{channel} : signal of {location[channel]}")
    axs[i].set_xlabel("Time (s)")
    axs[i].set_ylabel("Amplitude (V)" if channel in EMG else "g")

    if len(axs) > len(custom_order_names):
        axs[-1].axis("off")
plt.tight_layout()
#plt.show()

# getting mean of EMG
mean_forearm = df_rectified.loc[:, "BIP7"].mean()
mean_delt = df_rectified.loc[:, "BIP8"].mean()

plt.figure()
plt.errorbar(["Forearm"], mean_forearm)
plt.errorbar(["Delt"], mean_delt)
plt.show()

# movement detection...