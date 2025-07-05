import numpy as np
import pandas as pd
import seaborn as sns
import mne
from mne.viz.utils import plt_show
from seaborn import color_palette

raw = mne.io.read_raw_ant("C:/Users/User/Documents/bachelorarbeit/data/Subject_Test_2025-03-24_17-43-47.cnt", preload=True)

print(raw.info)

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

raw.set_channel_types({"BIP1": "emg"})

#raw.plot(n_channels=1, duration=10, scalings={"emg": 1e-3})
#plt.show()

parallel_recording_bip_unip = mne.io.read_raw_ant("C:/Users/User/Documents/bachelorarbeit/data/Test_Liliane_2025-03-26_15-03-49_parallel_recording_2_boxes.cnt", preload=True)
#parallel_recording_bip_unip.plot(n_channels=7, duration=10, scalings="auto")
#plt.show()

channels_to_plot = [parallel_recording_bip_unip.ch_names[34]]
data, times = parallel_recording_bip_unip[channels_to_plot, :]

df = pd.DataFrame(data.T, columns=channels_to_plot)
df['Time (s)'] = times
df_melted = df.melt(id_vars='Time (s)', var_name='Channel', value_name='Amplitude')

parallel_rec_under_clothes = mne.io.read_raw_ant("C:/Users/User/Documents/bachelorarbeit/data/Test_Liliane_2025-03-26_15-16-49_under_clothes_parallel_rec.cnt", preload=True)
#parallel_rec_under_clothes.plot(n_channels=7, duration=10, scalings="auto")
#plt.show()
print(parallel_rec_under_clothes.info['sfreq'])
print("Available channels:", parallel_rec_under_clothes.ch_names)

channels_to_plot1 = [parallel_rec_under_clothes.ch_names[28], parallel_rec_under_clothes.ch_names[34]]
data1, times1 = parallel_rec_under_clothes[channels_to_plot1, :]

df1 = pd.DataFrame(data1.T, columns=channels_to_plot1)
df1['Time (s)'] = times1

df_melted1 = df1.melt(id_vars='Time (s)', var_name='Channel', value_name='Amplitude')

#how to with seaborn
# alpha_palette = {"BIP7" : 0.4}
# color_palette = {"BIP1" : "skyblue", "BIP7": "pink"}
# g = sns.lineplot(data=df_melted, x="Time (s)", y="Amplitude", hue="Channel", alpha=alpha_palette["BIP7"], palette=color_palette)

fig, ax = plt.subplots(1, 2, figsize=(15, 10), sharex=True, sharey=True)

ax[0].plot(df_melted1[df_melted1["Channel"] == "BIP1"]["Time (s)"],
         df_melted1[df_melted1["Channel"] == "BIP1"]["Amplitude"],
         color="skyblue",
         alpha=1.0,
         label="BIP1 - bipolar cable")

ax[0].plot(df_melted1[df_melted1["Channel"] == "BIP7"]["Time (s)"],
         df_melted1[df_melted1["Channel"] == "BIP7"]["Amplitude"],
         color="pink",
         alpha=0.4,
         label="BIP7 - unipolar cables")

ax[1].plot(df_melted[df_melted["Channel"] == "BIP7"]["Time (s)"],
         df_melted[df_melted["Channel"] == "BIP7"]["Amplitude"],
         color="yellow",
         alpha=1.0,
         label="BIP7 - wrapped + above clothes")

ax[1].plot(df_melted1[df_melted1["Channel"] == "BIP7"]["Time (s)"],
         df_melted1[df_melted1["Channel"] == "BIP7"]["Amplitude"],
         color="pink",
         alpha=0.4,
         label="BIP7 - wrapped + under clothes")


plt.xlabel("Time (s)")
plt.ylabel("Amplitude (V)")
ax[0].set_title("different electrode setups")
ax[0].legend()
ax[1].set_title("cables above vs under clothes")
ax[1].legend()
plt.tight_layout()
plt.savefig("vergleich_elektroden.png")



