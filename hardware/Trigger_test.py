import numpy as np
import pandas as pd
import seaborn as sns
import mne
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

trigger_test = mne.io.read_raw_ant("C:/Users/User/Documents/bachelorarbeit/data/test_Trigger_2025-05-13_09-54-51_zweiter_Test.cnt", preload=True)

channel_names = ["BIP6", "BIP1", "BIP2", "BIP7", "BIP8", "BIP9", "BIP10", "BIP11", "BIP12"]
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

data, times = trigger_test[channel_names, :]
test_df = pd.DataFrame(data.T, columns=channel_names)

fig, axs = plt.subplots(4, 3, figsize=(12,8))
axs = axs.ravel()
for ax, channel in zip(axs, channel_names):
    ax.plot(times, test_df[channel])
    ax.set_title(f"{channel} : signal of {location[channel]}")
plt.tight_layout()
plt.show()