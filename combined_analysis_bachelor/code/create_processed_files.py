import numpy as np
import scipy as sp
import pandas as pd
import mne
from hyperframe.frame import DataFrame
from mne.io import read_raw_ant
import os
import matplotlib
from matplotlib.lines import lineStyles
from scipy.signal import butter, sosfiltfilt
from scipy.stats import zscore
from combined_analysis_bachelor.code.functions_for_pipeline import get_ch_indices, plot_channel_overview, normalize_emg, \
    notched_and_filtered, create_df, envelope, rectify, tkeo
from utils.find_paths import get_onedrive_path
from utils.get_sub_dir import get_sub_folder_dir
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

channel_custom_order = ["BIP3", "BIP4", "BIP5", "BIP9", "BIP10", "BIP12",
                        "BIP6", "BIP1", "BIP2", "BIP11", "BIP8", "BIP7", "SVM_L", "SVM_R"]
EMG = ["BIP7", "BIP8", "BIP9", "BIP10", "BIP11", "BIP12"]
ACC = ["BIP1", "BIP2", "BIP3", "BIP4", "BIP5", "BIP6", "SVM_L, SVM_R"]
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
SUB = "91"

source_dir = get_sub_folder_dir(SUB,"raw_data")
target_dir = get_sub_folder_dir(SUB, "processed_data")
print(source_dir)
print(target_dir)

def create_processed_files(source_directory, target_directory,
                           channels, acc_channels, emg_channels,
                           acc_filter_parameters, emg_filter_parameters):

    """takes in the raw (trimmed) files, generates SVM columns for ACC data, applies filtering, generates new filenames and saves files
    to target directory

    input:
    - source_directory
    - targes_directory
    - channels : list of channels (in preferred order)
    - acc_channels : list of channels that are ACC channels
    - emg_channels : list of channels that are EMG channels
    - acc_filter_parameters & emg_filter_parameters : lists that hold the desired lfreq and hfreq for filtering

    returns the files that hold the processed data in the target directory
    """
    ### ==== get all filepaths ==== ###
    filepaths = []
    for file in os.listdir(source_directory):
        if file.endswith(".h5"):
            filepath = f"{source_dir}/{file}"
            filepaths.append(filepath)

    print(f"retrieved data: {filepaths}")

    ### ===== looping trough files ===== ###
    for filepath in filepaths:

        # ----- read in raw file ----- #
        raw = pd.read_hdf(filepath, key="data")
        raw_df = pd.DataFrame(raw) # make sure its a df, not an object

        # --- create euclidean norm column per side --- #
        SVM_left = np.sqrt(raw_df["BIP3"]**2 + raw_df["BIP4"]**2 +  raw_df["BIP5"]**2)
        SVM_right = np.sqrt(raw_df["BIP6"]**2 + raw_df["BIP1"]**2 +  raw_df["BIP2"]**2)
        raw_df["SVM_L"] = SVM_left
        raw_df["SVM_R"] = SVM_right

        # ----- perform filtering ----- #
        filtered_df = notched_and_filtered(raw_df, channels, acc_channels, emg_channels, acc_filter_parameters, emg_filter_parameters)

        # ----- creating new file ----- #
        filename = os.path.basename(filepath)
        if filename.endswith(".h5"):
            filename = filename[:-6]
        filename += "processed.h5"

        target_path = os.path.join(target_directory, filename)

        print(f"Saving to: {target_path}")

        filtered_df.to_hdf(target_path, key="data", mode="w")

    print("\n✅ Every File processed and saved!")


create_processed_files(source_dir, target_dir, channel_custom_order, ACC, EMG, [1,20], [20,450])
