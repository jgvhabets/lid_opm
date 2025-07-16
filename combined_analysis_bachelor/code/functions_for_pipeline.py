import numpy as np
import pandas as pd
import mne
from scipy.signal import butter, sosfiltfilt
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

def create_df(data, column_names, times):
    df = pd.DataFrame(data.T, columns=column_names)
    df["Time (s)"] = times
    return df

def notched(df, all_columns, times):
    notched_df = df.copy()
    for col in all_columns:
        col_data = df[col].to_numpy()
        notched_col = mne.filter.notch_filter(col_data, freqs=[50, 100, 150], Fs=1000)
        notched_df[col] = notched_col
    notched_df["Time (s)"] = times
    return notched_df

def notched_and_filtered(raw_df, all_columns, acc_columns, emg_columns, acc_filter, emg_filter):
    filtered_df = raw_df.copy()
    for col in all_columns:
        col_data = raw_df[col].to_numpy()
        notched_col = mne.filter.notch_filter(col_data, freqs=[50, 100, 150], Fs=1000)
        if col in acc_columns:
            col_filtered = mne.filter.filter_data(notched_col, sfreq=1000, l_freq=acc_filter[0], h_freq=acc_filter[1])
            filtered_df[col] = col_filtered

        if col in emg_columns:
            col_filtered = mne.filter.filter_data(notched_col, sfreq=1000, l_freq=emg_filter[0], h_freq=emg_filter[1])
            filtered_df[col] = col_filtered
        else:
            continue

    return filtered_df

def plot_channel_overview(channels, df, title_name, location, emg):
    fig, axs = plt.subplots(4, 3, figsize=(12, 8))
    axs = axs.ravel()
    for i, channel in enumerate(channels):
        axs[i].plot(df["Time (s)"], df[channel])
        axs[i].set_title(f"{channel} : signal of {location[channel]}")
        axs[i].set_xlabel("Time (s)")
        axs[i].set_ylabel("Amplitude (V)" if channel in emg else "g")
        plt.suptitle(f"plots for {title_name}")
        #plt.ylim()
        if i > len(channels): #abändern, dass man nur so viele ax hat wie auch channel!
            axs.axis("off")

    plt.tight_layout()
    plt.show()

def rectify(df, all_columns, emg_columns):
    if not all(col in all_columns for col in emg_columns):
        print("Warning: Some EMG channels are not in your custom_order")

    df[emg_columns] = df[emg_columns].abs()
    print("Min value after rectification:", df[emg_columns].min().min())
    return df


def envelope(df, all_columns, emg_columns, freq):
    emg_envelope_df = df.copy()
    low_pass = freq/(1000/2)
    sos = butter(4, low_pass, btype='lowpass', output="sos")
    for col in all_columns:
        if col in emg_columns:
            col_data = df[col].to_numpy()
            enveloped = sosfiltfilt(sos, x=col_data)
            emg_envelope_df[col] = enveloped
        else:
            continue

    return emg_envelope_df


def normalize_emg(df, all_columns, emg_columns, times):
    normalized_emg_df = df.copy()
    for col in all_columns:
        if col in emg_columns:
            col_data = df[col].to_numpy()
            normalized_emg = col_data / np.max(np.abs(col_data))
            normalized_emg_df[col] = normalized_emg
            print(f"Normalized to range: [{np.min(normalized_emg): .2f}, {np.max(normalized_emg): .2f}]")
        else:
            continue
    normalized_emg_df["Time (s)"] = times
    return normalized_emg_df

def tkeo(df, all_columns, emg_columns, times):
    tkeo_df = df.copy()
    for col in all_columns:
        if col in emg_columns:
            col_data = df[col].to_numpy()
            tkeo_col = col_data[1:-1] ** 2 - col_data[0:-2] * col_data[2:]
            tkeo_padded = np.pad(tkeo_col, (1, 1), 'constant', constant_values=0)
            tkeo_df[col] = tkeo_padded
    tkeo_df["Time (s)"] = times
    return tkeo_df

def movement_detection(data, threshold):
    over_thresh = data >= threshold
    return over_thresh

def get_ch_indices(all_columns, emg, acc):
    emg_cols = [all_columns.index(ch) for ch in emg if ch in all_columns]
    acc_cols = [all_columns.index(ch) for ch in acc if ch in all_columns]
    return emg_cols, acc_cols
