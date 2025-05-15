import numpy as np
import pandas as pd
import seaborn as sns
import mne
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

# here, I will write the functions that I can then use in the pipeline (to spare some space)

def create_raw_df(data, column_names, times):
    raw_df = pd.DataFrame(data.T, columns=column_names)
    raw_df["Time (s)"] = times
    return raw_df

def plot_raw_signals(columns, raw_df, location, emg):
    fig, axs = plt.subplots(3, 3, figsize=(12, 8))
    axs = axs.ravel()
    for i, channel in enumerate(columns):
        axs[i].plot(raw_df["Time (s)"], raw_df[channel])
        axs[i].set_title(f"{channel} : signal of {location[channel]}")
        axs[i].set_xlabel("Time (s)")
        axs[i].set_ylabel("Amplitude (V)" if channel in emg else "g")

        #if len(axs) > len(columns):
        #    axs[-1].axis("off")

    plt.tight_layout()
    plt.show()

def notch_filter(data, frequencies, sampling_rate):
    notched_data = mne.filter.notch_filter(data, freqs=frequencies, Fs=sampling_rate)
    return notched_data

def bandpass_filter(data, sampling_rate, low_freq, high_freq):
    bandpassed_data = mne.filter.filter_data(data, sfreq=sampling_rate, l_freq=low_freq, h_freq=high_freq)
    return bandpassed_data

def create_filtered_df(data, columns, times):
    filtered_df = pd.DataFrame(data.T, columns=columns)
    filtered_df["Time (s)"] = times
    return filtered_df