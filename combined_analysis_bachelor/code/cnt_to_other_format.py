import numpy as np
import scipy as sp
import pandas as pd
import mne
from mne.io import read_raw_ant
import os
from matplotlib.lines import lineStyles
from scipy.signal import butter, sosfiltfilt
from scipy.stats import zscore
from combined_analysis_bachelor.code.functions_for_pipeline import get_ch_indices, plot_channel_overview, normalize_emg, \
    notched_and_filtered, create_df, envelope, rectify, tkeo
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt



channel_custom_order = ["BIP3", "BIP4", "BIP5", "BIP9", "BIP10", "BIP12", "BIP6", "BIP1", "BIP2", "BIP11", "BIP8", "BIP7"]
EMG = ["BIP7", "BIP8", "BIP9", "BIP10", "BIP11", "BIP12"]
ACC = ["BIP1", "BIP2", "BIP3", "BIP4", "BIP5", "BIP6"]
locations = {"BIP7":"right M. tibialis anterior",
            "BIP8": "right M. deltoideus",
            "BIP11": "right M. brachioradialis",
            "BIP1" : "ACC right hand : y",
            "BIP2" : "ACC right hand : z",
            "BIP6" : "ACC right hand : x",
            "BIP3" : "ACC left hand : x",
            "BIP4" : "ACC left hand : y",
            "BIP5" : "ACC left hand : z",
            "BIP9" : "left M. brachioradialis",
            "BIP10" : "left M. deltoideus",
            "BIP12" : "left M. tibialis anterior"}
sf = 1000
source_dir = r"C:/Users/User/Documents/bachelorarbeit/data/EMG_ACC/PTB_01_data_source";
target_dir = "C:/Users/User/Documents/bachelorarbeit/data/EMG_ACC/PTB_01_data_raw"


# ================ Recording 1: Setup A & Rest ====================== #
#A_rest = mne.io.read_raw_ant(filedirs[0], preload=True)
#A_rest_data, A_rest_times = A_rest[channel_custom_order, :]
#A_rest_data[1] *= -1
#A_rest_data[6] *= -1
#
#onsets_time = A_rest.annotations.onset
#onsets_samples = onsets_time * sf
#
#
#### trimming from sync start ###
#A_rest_start = int(onsets_samples[0])
#A_rest_data_trimmed = A_rest_data[:, A_rest_start:]
#A_rest_source_times = A_rest_times[A_rest_start:]
#
#### adding sync_time ###
#A_rest_sync_time = A_rest_source_times - sync_trig
#
#### put together as raw dataframe ###
#A_rest_data_trimmed_T = A_rest_data_trimmed.T
#A_rest_raw_df = pd.DataFrame(A_rest_data_trimmed_T, columns=channel_custom_order)
#A_rest_raw_df["Sync_Time (s)"] = A_rest_sync_time
#A_rest_raw_df["Source_Time (s)"] = A_rest_source_times

### into Hdf5 ###
#A_rest_raw_df.to_hdf("C:/Users/User/Documents/bachelorarbeit/data/EMG_ACC/PTB_01_data_raw/PTB_01_A_1.1_rest_raw.h5", key="data", mode="w")

## put together in new file format ##


def cnt_to_raw_hdf(source_directory, target_directory, channels, sampling_freq):
    filepaths = []
    for file in os.listdir(source_directory):
        if file.endswith(".cnt"):
            filepath = f"{source_directory}/{file}"
            filepaths.append(filepath)


    print(f"retrieved data: {filepaths}")

    # ========= looping trough filedirectory ======== #
    for filepath in filepaths:

        # --- read in file --- #
        source = mne.io.read_raw_ant(filepath, preload=True)
        data, times = source[channels, :]

        # --- correcting inverted channels --- #
        data[1] *= -1
        data[6] *= -1

        # --- finding trigger --- #
        onsets = source.annotations.onset
        onsets_samples = onsets * sf

        print(onsets)

        if filepath == filepaths[2]: # in Rec 3 there were random trig. signals too early
            sync_trig = onsets[3]
            sync_trig_samples = int(onsets_samples[3])

        else:
            sync_trig = onsets[0] # in seconds
            sync_trig_samples = int(onsets_samples[0]) # in samples

        # ----------- trimming ----------- #
        data_trimmed = data[:, sync_trig_samples:]
        source_times = times[sync_trig_samples:]

        # ---------- getting sync_times --------- #
        sync_times = source_times - sync_trig

        
        # ---------- creating dataframe ---------- #
        data_trimmed_T = data_trimmed.T
        raw_df = pd.DataFrame(data_trimmed_T, columns=channels)
        raw_df["Sync_Time (s)"] = sync_times
        raw_df["Source_Time (s)"] = source_times

        # --------- creating new file -------- #
        filename = os.path.basename(filepath)
        if filename.startswith("PTB_01_"):
            filename = filename.replace("PTB_01_", "")
        if filename.endswith(".cnt"):
            filename = filename[:-4]
        filename += "_raw.h5"

        target_path = os.path.join(target_directory, filename)

        print(f"Saving to: {target_path}")

        raw_df.to_hdf(target_path, key="data", mode="w")

    print("\n✅ Every File processed and saved!")


cnt_to_raw_hdf(source_dir, target_dir, channel_custom_order, sf)

