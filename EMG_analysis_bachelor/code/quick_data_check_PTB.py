import numpy as np
import pandas as pd
import seaborn as sns
import mne
from mne.viz.utils import plt_show
from seaborn import color_palette

#forearm_upperside = mne.io.read_raw_ant("C:/Users/User/Documents/bachelorarbeit/data/EMG/test_first_2025-04-07_14-10-04_forearm_upperSide.cnt", preload=True)
#forearm_downside = mne.io.read_raw_ant("C:/Users/User/Documents/bachelorarbeit/data/EMG/test_first_2025-04-07_14-23-02_forearm_UnderSide.cnt", preload=True)
#test = mne.io.read_raw_ant("C:/Users/User/Documents/bachelorarbeit/data/EMG_ACC/PTB_measurement_14.04/Bonato_Federico_2025-04-14_13-02-56.cnt", preload=True)
two_cha_acc = mne.io.read_raw_ant("C:/Users/User/Documents/bachelorarbeit/data/Accelerometer/Test_Two_Charite_ACC_2025-06-10_11-53-04.cnt", preload=True)

test = two_cha_acc.notch_filter(picks=["BIP1", "BIP2", "BIP3", "BIP4", "BIP5", "BIP6"], freqs=50)

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

channel_names = ["BIP7", "BIP8", "BIP9", "BIP10", "BIP11", "BIP12", "BIP3", "BIP4", "BIP5", "BIP1", "BIP2", "BIP6"]
#channel_names = ["BIP7", "BIP8", "BIP9", "BIP10", "BIP11", "BIP12"]
location = {"BIP7":"right forearm",
            "BIP8": "right delt",
            "BIP9": "right calf",
            "BIP10" : "left forearm",
            "BIP11" : "left delt",
            "BIP12" : "left calf",
            "BIP3" : "Accelerometer : x",
            "BIP4" : "Accelerometer : y",
            "BIP5" : "Accelerometer : z",
            "BIP1" : "Charité ACC ...",
            "BIP2" : "Charité ACC ...",
            "BIP6" : "Charité ACC ..."}

data, times = test[channel_names, :]
test_df = pd.DataFrame(data.T, columns=channel_names)

fig, axs = plt.subplots(4, 3, figsize=(12,8))
axs = axs.ravel()
for ax, channel in zip(axs, channel_names):
    ax.plot(times, test_df[channel])
    ax.set_title(f"{channel} : signal of {location[channel]}")
plt.tight_layout()
plt.show()

# loop through the channels , individual plots
for channel in channel_names:
    plt.figure()
    plt.plot(times, test_df[channel])
    plt.title(f" {channel} : signal of {location[channel]}")
    plt.show()

#channel_name = ["BIP7"]
#data, times = test[channel_name, :]
#data = data * 10**6
#test_df = pd.DataFrame(data.T, columns=channel_name)
#
#for channel in channel_name:
#    plt.figure()
#    plt.plot(times, test_df[channel])
#    plt.title(f" {channel} : signal of {location[channel]}")
#    plt.show()





