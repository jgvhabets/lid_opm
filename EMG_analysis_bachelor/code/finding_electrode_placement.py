import numpy as np
import pandas as pd
import seaborn as sns
import mne
from mne.viz.utils import plt_show
from seaborn import color_palette

forearm_downside_with_post_proc = mne.io.read_raw_ant("C:/Users/User/Documents/bachelorarbeit/data/EMG/test_first_2025-04-07_14-23-02_forearm_UnderSide_exported_with_post_processing.cnt", preload=True)
forearm_upperside_with_post_proc = mne.io.read_raw_ant("C:/Users/User/Documents/bachelorarbeit/data/EMG/test_first_2025-04-07_14-10-04_forearm_upperSide_with_post_proc.cnt", preload=True)


import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

channel_names = ["BIP7", "BIP8", "BIP9", "BIP10", "BIP11", "BIP12"]
data, times = forearm_downside_with_post_proc[channel_names, :]
df_upside_with_post = pd.DataFrame(data.T, columns=channel_names)

channel_name = ["BIP7"]
data2, times2 = forearm_downside_with_post_proc[channel_name, :]
df_downside_with_post = pd.DataFrame(data2.T, columns=channel_name)

fig, ax = plt.subplots(1, 2, figsize=(12, 8))

ax[0].plot(times, df_upside_with_post["BIP7"])
ax[0].set_title("electrodes on the upper side of the forearm")
ax[1].plot(times2, df_downside_with_post["BIP7"])
ax[1].set_title("electrodes on the downside of the forearm")
plt.show()

# für mich sieht das aus wie das selbe signal - hab ich ausversehen das gleiche recording doppelt exportiert?
# nochmal auschecken!!