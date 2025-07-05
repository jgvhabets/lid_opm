import numpy as np
import pandas as pd
import seaborn as sns
import mne
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

federico_test = mne.io.read_raw_ant("C:/Users/User/Documents/bachelorarbeit/data/EMG_ACC/"
                                    "PTB_measurement_14.04/Bonato_Federico_2025-04-14_13-02-56.cnt", preload=True)
channel_names  = ["BIP6", "BIP1", "BIP2", "BIP7", "BIP8"]
location = {"BIP7":"left forearm",
            "BIP8": "left delt",
            "BIP1" : "Charité ACC : y",
            "BIP2" : "Charité ACC : z",
            "BIP6" : "Charité ACC : x"}
data, times = federico_test[channel_names, :]
fed_test_df = pd.DataFrame(data.T, columns=channel_names)
fed_test_df["Time (s)"] = times

time_filter_baseline = (fed_test_df["Time (s)"] >= 369) & (fed_test_df["Time (s)"] <= 409)
baseline_df = fed_test_df[time_filter_baseline].copy()

time_filter_fast_move = (fed_test_df["Time (s)"] >= 420) & (fed_test_df["Time (s)"] <= 450)
fast_move_df = fed_test_df[time_filter_fast_move].copy()

time_filter_squeeze = (fed_test_df["Time (s)"] >= 492) & (fed_test_df["Time (s)"] <= 580)
squeeze_df = fed_test_df[time_filter_squeeze].copy()

dfs = [baseline_df, fast_move_df, squeeze_df]
EMG = ["BIP7", "BIP8"]
df_names = ["Baseline", "Arm Fast Move Up-Down", "Squeeze"]

for i, df in enumerate(dfs):
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 8))
    axs = axs.ravel()

    for j, channel in enumerate(channel_names):
        axs[j].plot(df["Time (s)"], df[channel])
        axs[j].set_title(f"{channel} : signal of {location[channel]}")
        axs[j].set_xlabel("Time (s)")
        axs[j].set_ylabel("Amplitude (V)" if channel in EMG else "g")

    if len(axs) > len(channel_names):
        axs[-1].axis("off")

    fig.suptitle(f"ACC and EMG for {df_names[i]}", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    print(f"Saving plot for {df_names[i]}")
    plt.savefig(f"plot_for_{df_names[i]}")
    plt.show()

