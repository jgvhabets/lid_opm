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

def filtered(notched_dataframe, all_columns, acc_columns, emg_columns, times):
    filtered_df = notched_dataframe.copy()
    for col in all_columns:
        if col in acc_columns:
            col_data = notched_dataframe[col].to_numpy()
            col_filtered = mne.filter.filter_data(col_data, sfreq=1000, l_freq=1, h_freq=48)
            filtered_df[col] = col_filtered

        if col in emg_columns:
            col_data = notched_dataframe[col].to_numpy()
            col_filtered = mne.filter.filter_data(col_data, sfreq=1000, l_freq=30, h_freq=300)
            filtered_df[col] = col_filtered
        else:
            continue
    filtered_df["Time (s)"] = times
    return filtered_df #könnte das auch noch flexibler machen mit h_freq und l_freq indem ich ne liste als input tu und dann [0]und[1]!

def plot_channel_overview(channels, df, title_name, location, emg): # only this function is needed right? does the same thing!! delete others?
    fig, axs = plt.subplots(3, 3, figsize=(12, 8))
    axs = axs.ravel()
    for i, channel in enumerate(channels):
        axs[i].plot(df["Time (s)"], df[channel])
        axs[i].set_title(f"{channel} : signal of {location[channel]}")
        axs[i].set_xlabel("Time (s)")
        axs[i].set_ylabel("Amplitude (V)" if channel in emg else "g")
        plt.suptitle(f"plots for {title_name}")

        # if len(axs) > len(channels):
        #    axs[-1].axis("off")

    plt.tight_layout()
    plt.show()

def envelope(df, all_columns, emg_columns, times, freq):
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
    emg_envelope_df["Time (s)"] = times
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

def movement_detection(data, threshold):
    over_thresh = data >= threshold
    return over_thresh

def get_ch_indices(all_columns, emg, acc):
    emg_cols = [all_columns.index(ch) for ch in emg if ch in all_columns]
    acc_cols = [all_columns.index(ch) for ch in acc if ch in all_columns]
    return emg_cols, acc_cols



# df_filte[acc_cols] = df_raw[acc_cols].apply(filter_f, min_f = 10, max_f = 1000)
# df_filter[meg_cols]