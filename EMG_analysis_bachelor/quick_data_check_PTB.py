import numpy as np
import pandas as pd
import seaborn as sns
import mne
from mne.viz.utils import plt_show
from seaborn import color_palette

forearm_upperside = mne.io.read_raw_ant("C:/Users/User/Documents/bachelorarbeit/data/EMG/test_first_2025-04-07_14-10-04_forearm_upperSide.cnt", preload=True)
forearm_downside = mne.io.read_raw_ant("C:/Users/User/Documents/bachelorarbeit/data/EMG/test_first_2025-04-07_14-23-02_forearm_UnderSide.cnt", preload=True)
forearm_downside_with_post_proc = mne.io.read_raw_ant("C:/Users/User/Documents/bachelorarbeit/data/EMG/test_first_2025-04-07_14-23-02_forearm_UnderSide_exported_with_post_processing.cnt", preload=True)

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

#channel_names = ["BIP7", "BIP8", "BIP9", "BIP10", "BIP11", "BIP12", "BIP3", "BIP4", "BIP5"]
channel_names = ["BIP7", "BIP8", "BIP9", "BIP10", "BIP11", "BIP12"]
location = {"BIP7":"left forearm",
            "BIP8": "left delt",
            "BIP9": "right forearm",
            "BIP10" : "right delt",
            "BIP11" : "left calf",
            "BIP12" : "right calf",
            "BIP3" : "Accelerometer : x",
            "BIP4" : "Accelerometer : y",
            "BIP5" : "Accelerometer : z"}

data, times = forearm_downside_with_post_proc[channel_names, :]
df_upside_with_post = pd.DataFrame(data.T, columns=channel_names)

fig, axs = plt.subplots(3, 3, figsize=(12,8))
axs = axs.ravel()
for ax, channel in zip(axs, channel_names):
    ax.plot(times, df_upside_with_post[channel])
    ax.set_title(f"{channel} : signal of {location[channel]}")
plt.tight_layout()
plt.show()

# loop through the channels , individual plots
for channel in channel_names:
    plt.figure()
    plt.plot(times, df_upside_with_post[channel])
    plt.title(f" {channel} : signal of {location[channel]}")
    plt.show()







