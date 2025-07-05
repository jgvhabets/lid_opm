import numpy as np
import pandas as pd
import seaborn as sns
import mne
import tkinter as tk
from tkinter import *
from tkinter import ttk
import scipy as sp
from matplotlib.lines import lineStyles
from scipy.signal import butter, sosfiltfilt
from scipy.stats import zscore
from EMG_analysis_bachelor.functions_for_pipeline import get_ch_indices, plot_channel_overview, normalize_emg, notched, \
    create_df, envelope, rectify, notched_and_filtered
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

leg_checks = mne.io.read_raw_ant("C:/Users/User/Documents/bachelorarbeit/data/EMG_ACC/Test_Leg_positions_2025-06-17_17-51-24.cnt", preload=True)

onsets = leg_checks.annotations.onset
onsets = np.insert(onsets,0,0)

custom_order_names = ["BIP6", "BIP1", "BIP2", "BIP11"]
EMG = ["BIP11"]
ACC = ["BIP6", "BIP1", "BIP2", "BIP3", "BIP4", "BIP5"]

location = {"BIP7":"left forearm",
            "BIP8": "left delt",
            "BIP1" : "Charité ACC : y",
            "BIP2" : "Charité ACC : z",
            "BIP6" : "Charité ACC : x",
            "BIP3" : "2nd Charité ACC: x",
            "BIP4" : "2nd Charité ACC: y",
            "BIP5" : "2nd Charité ACC: z",
            "BIP11" : "right calf"}

data, times = leg_checks[custom_order_names, : ]
data[0] *= -1

raw_df = create_df(data, custom_order_names, times)

recordings = []
for i in range(len(onsets)):
    start_time = onsets[i]
    end_time = onsets[i + 1] if i < len(onsets) - 1 else (raw_df['Time (s)'].iloc[-1])

    mask = (raw_df['Time (s)'] >= start_time) & (raw_df['Time (s)'] < end_time)
    recordings.append(raw_df.loc[mask].copy())


A_kni_E_ob = recordings[0]
A_knoe_E_ob = recordings[1]
A_knoe_E_un = recordings[2]
A_kni_E_un = recordings[3]

recording_dfs = [A_kni_E_ob, A_knoe_E_ob, A_knoe_E_un, A_kni_E_un]
locations = ["ACC am knie & EMG obere wade", "ACC am knöchel & EMG obere wade", "ACC am knöchel & EMG untere wade", "ACC am knie & EMG untere wade"]

#for idx, df in enumerate(recording_dfs):
#    notched_df = notched(df, custom_order_names, df["Time (s)"])
#    filtered_df = filtered(notched_df, custom_order_names, ACC, EMG, df["Time (s)"])
#    rectified_df = rectify(filtered_df, custom_order_names, EMG)
#    envelope_df = envelope(rectified_df, custom_order_names, EMG, df["Time (s)"], 3)
#    plot_channel_overview(custom_order_names, envelope_df, f"{locations[idx]}", location, EMG)

#fig, axs = plt.subplots(2, 2, figsize=(12,8))
#axs = axs.ravel()
#row_titles = ["Elektroden Placement: Oberer Teil der Wade", "Elektroden Placement: Unterer Teil der Wade"]
#
#for i, (ax, df) in enumerate(zip(axs, recording_dfs)):
#    notched_df = notched(df, custom_order_names, df["Time (s)"])
#    filtered_df = filtered(notched_df, custom_order_names, ACC, EMG, df["Time (s)"])
#    rectified_df = rectify(filtered_df, custom_order_names, EMG)
#    envelope_df = envelope(rectified_df, custom_order_names, EMG, df["Time (s)"], 3)
#    ax.plot(df["Time (s)"], envelope_df["BIP11"])
#    ax.set_ylim(0.00002, 0.00025)
#    ax.set_title(f"Leg movement test recording {i}")
#
#fig.text(
#    x=0.5, y=0.965,
#    s=row_titles[0],
#    ha='center', va='center', fontsize=12, weight='bold'
#)
#fig.text(
#    x=0.5, y=0.47,
#    s=row_titles[1],
#    ha='center', va='center', fontsize=12, weight='bold')
#
#plt.tight_layout()
#plt.subplots_adjust(top=0.9, hspace=0.35)
#plt.show()

### synchron aufgenommen ###
leg_check_syn = mne.io.read_raw_ant("C:/Users/User/Documents/bachelorarbeit/data/EMG/Test_Leg_positions_synchron.cnt")

onsets = leg_check_syn.annotations.onset
onsets = onsets * 1000
custom_order_names = ["BIP11", "BIP12"]
location = {"BIP11" : "right calf upper",
            "BIP12" : "right calf lower"}
EMG = ["BIP11", "BIP12"]
ACC = ["BIP6", "BIP1", "BIP2", "BIP3", "BIP4", "BIP5"]

data, times = leg_check_syn[custom_order_names, : ]

raw_df = create_df(data, custom_order_names, times)

notched_and_filtered_df = notched_and_filtered(raw_df, custom_order_names, ACC, EMG, times)
rectified_df = rectify(notched_and_filtered_df, custom_order_names, EMG)
envelope_df = envelope(rectified_df, custom_order_names, EMG, times, 3)
plot_channel_overview(custom_order_names, envelope_df, f"leg movements recorded synchronously", location, EMG)


### brachioradialis check ###

brachoradialis = mne.io.read_raw_ant("C:/Users/User/Documents/bachelorarbeit/data/EMG/test_brachioradialis_compare_2025-06-17_18-11-05.cnt", preload=True)
custom_order_names = ["BIP7"]
EMG = ["BIP7"]

data, times = brachoradialis[custom_order_names, : ]
raw_df = create_df(data, custom_order_names, times)

notched_and_filtered_df = notched_and_filtered(raw_df, custom_order_names,ACC, EMG, times)
rectified_df = rectify(notched_and_filtered_df, custom_order_names, EMG)
envelope_df = envelope(rectified_df, custom_order_names, EMG, times, 3)
plot_channel_overview(custom_order_names, envelope_df, "brachoradialis check", location, EMG)



