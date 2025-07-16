import numpy as np
import pandas as pd
import seaborn as sns
import mne
from mne.viz.utils import plt_show
from seaborn import color_palette

test = mne.io.read_raw_ant("C:/Users/User/Documents/bachelorarbeit/data/Accelerometer/test_raising_arm.cnt", preload=True)

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

# mne plot overview
test.plot(n_channels=3, duration=35, scalings="auto")

# melting and matplotlib plot
channels_to_plot = [test.ch_names[30], test.ch_names[31], test.ch_names[32]]
data, times = test[channels_to_plot, :]

df = pd.DataFrame(data.T, columns=channels_to_plot)
df['Time (s)'] = times

df_melted = df.melt(id_vars='Time (s)', var_name='Channel', value_name='Amplitude')

fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

ax[0].plot(df_melted[df_melted["Channel"] == "BIP3"]["Time (s)"],
         df_melted[df_melted["Channel"] == "BIP3"]["Amplitude"],
         color="green",
         alpha=1.0,
         label="BIP3 - x?")

ax[1].plot(df_melted[df_melted["Channel"] == "BIP4"]["Time (s)"],
         df_melted[df_melted["Channel"] == "BIP4"]["Amplitude"],
         color="pink",
         alpha= 1,
         label="BIP4 - y?")

ax[2].plot(df_melted[df_melted["Channel"] == "BIP5"]["Time (s)"],
         df_melted[df_melted["Channel"] == "BIP5"]["Amplitude"],
         color="blue",
         alpha=1.0,
         label="BIP5 - z?")

plt.suptitle("first ACC recording, montage: 10-20 common average")
ax[0].legend()
ax[1].legend()
ax[2].legend()
plt.show()

# second acc recording - with different montage
acc_diff_montage = mne.io.read_raw_ant("C:/Users/User/Documents/bachelorarbeit/data/Accelerometer/test_first_accelerometer1_diff_montage.cnt", preload=True)


acc_diff_montage.plot(n_channels=10, duration=35, scalings="auto")

# melting and matplotlib plot
channels_to_plot = [acc_diff_montage.ch_names[34], acc_diff_montage.ch_names[35], acc_diff_montage.ch_names[36]]
data, times = acc_diff_montage[channels_to_plot, :]

df = pd.DataFrame(data.T, columns=channels_to_plot)
df['Time (s)'] = times

df_melted = df.melt(id_vars='Time (s)', var_name='Channel', value_name='Amplitude')

fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

ax[0].plot(df_melted[df_melted["Channel"] == "BIP3"]["Time (s)"],
         df_melted[df_melted["Channel"] == "BIP3"]["Amplitude"],
         color="green",
         alpha=1.0,
         label="BIP3 - x?")

ax[1].plot(df_melted[df_melted["Channel"] == "BIP4"]["Time (s)"],
         df_melted[df_melted["Channel"] == "BIP4"]["Amplitude"],
         color="pink",
         alpha= 1,
         label="BIP4 - y?")

ax[2].plot(df_melted[df_melted["Channel"] == "BIP5"]["Time (s)"],
         df_melted[df_melted["Channel"] == "BIP5"]["Amplitude"],
         color="blue",
         alpha=1.0,
         label="BIP5 - z?")

plt.suptitle("2nd recording, montage: BIP-XS...")
ax[0].legend()
ax[1].legend()
ax[2].legend()
plt.show()