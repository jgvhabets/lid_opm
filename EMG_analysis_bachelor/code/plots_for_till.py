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
from combined_analysis_bachelor.code.functions_for_pipeline import get_ch_indices, plot_channel_overview, normalize_emg, notched, \
    notched_and_filtered, create_df, envelope, rectify
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

base_mini_big = mne.io.read_raw_ant("C:/Users/User/Documents/bachelorarbeit/data/EMG/test_baseline_small_big_2025-06-20_10-54-02.cnt", preload=True)

onsets = base_mini_big.annotations.onset
onsets = np.insert(onsets,0,0)

custom_order_names = ["BIP7", "BIP12"]
EMG = ["BIP7", "BIP12"]
ACC = ["BIP6", "BIP1", "BIP2", "BIP3", "BIP4", "BIP5"]

location_dict = {"BIP7":"right forearm",
            "BIP8": "left delt",
            "BIP1" : "Charité ACC : y",
            "BIP2" : "Charité ACC : z",
            "BIP6" : "Charité ACC : x",
            "BIP3" : "2nd Charité ACC: x",
            "BIP4" : "2nd Charité ACC: y",
            "BIP5" : "2nd Charité ACC: z",
            "BIP11" : "right calf",
            "BIP12" : "right calf"}

data, times = base_mini_big[custom_order_names, : ]
data = data * 1e6

raw_df = create_df(data, custom_order_names, times)
filtered_df = notched_and_filtered(raw_df, ACC, EMG, [1,20], [20,450])

recordings = []
for i in range(len(onsets)):
    start_time = onsets[i]
    end_time = onsets[i + 1] if i < len(onsets) - 1 else (filtered_df['Time (s)'].iloc[-1])

    mask = (filtered_df['Time (s)'] >= start_time) & (filtered_df['Time (s)'] < end_time)
    recordings.append(filtered_df.loc[mask].copy())


baseline = recordings[0]
small_moves = recordings[1]
big_moves = recordings[2]

rec_dfs = [baseline, small_moves, big_moves]
rec_dfs_names = ["rest", "small moves", "bigger moves"]

plt.figure()
plt.plot(baseline["Time (s)"], baseline["BIP7"])
plt.show()

melted_dfs = []
for df in rec_dfs:
    df_melted = df.melt(id_vars='Time (s)', var_name='Channel', value_name='Amplitude')
    melted_dfs.append(df_melted)


def plot_recording_histograms(rec_melted_dfs, location):
    dfs_with_location = []
    for df in rec_melted_dfs:
        df_copy = df.copy()
        df_copy['Location'] = df_copy['Channel'].map(location_dict)
        dfs_with_location.append(df_copy)

    fig, axs = plt.subplots(1, 3, figsize=(14, 6), sharey=True)
    for i, (ax, df) in enumerate(zip(axs, dfs_with_location)):

        g = sns.histplot(
            data=df,
            x="Amplitude",
            hue="Location",
            bins=100,
            kde=False,
            ax=axs[i],
            element="step"
        )

        axs[i].set_title(f"{rec_dfs_names[i]}")
        axs[i].set_xlabel("EMG-values (V)")
        axs[i].set_ylabel("Frequency")
        axs[i].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

plot_recording_histograms(melted_dfs, rec_dfs_names)
