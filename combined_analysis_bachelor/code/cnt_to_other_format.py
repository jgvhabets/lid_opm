import numpy as np
import scipy as sp
import pandas as pd
import mne
from mne.io import read_raw_ant
import os
import antio
from matplotlib.lines import lineStyles
from scipy.signal import butter, sosfiltfilt
from scipy.stats import zscore
from combined_analysis_bachelor.code.functions_for_pipeline import get_ch_indices, plot_channel_overview, normalize_emg, \
    notched_and_filtered, create_df, envelope, rectify, tkeo
from utils.get_sub_dir import get_sub_folder_dir
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt



channel_custom_order = ["BIP3", "BIP4", "BIP5", "BIP9", "BIP10", "BIP12", "BIP6", "BIP1", "BIP2", "BIP11", "BIP8", "BIP7"]
EMG = ["BIP7", "BIP8", "BIP9", "BIP10", "BIP11", "BIP12"]
ACC = ["BIP1", "BIP2", "BIP3", "BIP4", "BIP5", "BIP6"]
LOCS = {"BIP7":"tibialisAnterior_R",
            "BIP8": "deltoideus_R",
            "BIP11": "brachioradialis_R",
            "BIP1" : "acc_y_hand_R",
            "BIP2" : "acc_z_hand_R",
            "BIP6" : "acc_x_hand_R",
            "BIP3" : "acc_x_hand_L",
            "BIP4" : "acc_y_hand_L",
            "BIP5" : "acc_z_hand_L",
            "BIP9" : "brachioradialis_L",
            "BIP10" : "deltoideus_L",
            "BIP12" : "tibialisAnterior_L"}
sf = 1000
SUB = "91"
SUB_target = "test"

source_dir = "C:/Users/User/Documents/bachelorarbeit/data/EMG_ACC/sub_91/PTB_01_data_source"
target_dir = get_sub_folder_dir(SUB_target,"raw_data")


def cnt_to_raw_hdf(source_directory, target_directory, channels, sampling_freq):
    filepaths = []
    for file in os.listdir(source_directory):
        if file.endswith(".cnt"):
            filepath = f"{source_directory}\\{file}"
            filepaths.append(filepath)

    print(f"retrieved data: {filepaths}")

    # ========= looping trough filedirectory ======== #
    for filepath in filepaths:
        # --- read in file --- #
        if os.path.basename(filepath).startswith("PTB_01"):
            source = mne.io.read_raw_ant(filepath, preload=True)
            data, times = source[channels, :]


        # --- correcting inverted channels --- #
            data[1] *= -1
            data[6] *= -1


            # --- finding trigger --- #
            onsets = source.annotations.onset
            onsets_samples = onsets * sf

            print(onsets)

            if filepath == filepaths[2]: # in Rec 3 there were random trig. signals too early --> how can i make this modualar?
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
            raw_df = pd.DataFrame(data_trimmed_T, columns=[LOCS[ch] for ch in channel_custom_order])
            assert all(ch in LOCS for ch in channels), "Some channels missing in your location_dict!"

            raw_df["Sync_Time (s)"] = sync_times
            raw_df["Source_Time (s)"] = source_times

            # --------- creating new file -------- #
            filename = os.path.basename(filepath)
            if filename.startswith(f"PTB_01") and filename.endswith(".cnt"):
                filename = filename.replace(f"PTB_01", f"sub-{SUB}")
                print(filename)
                filename = filename[:-4]
            filename += "_raw.h5"

            target_filepath = os.path.join(target_directory, filename)

            print(f"Saving to: {target_filepath}")

            raw_df.to_hdf(target_filepath, key="data", mode="w")

    print("\n✅ Every File processed and saved!")


cnt_to_raw_hdf(source_dir, target_dir, channel_custom_order, sf)

