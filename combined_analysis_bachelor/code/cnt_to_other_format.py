import numpy as np
import scipy as sp
import pandas as pd
import mne
from mne.io import read_raw_ant
from mne.io.constants import FIFF
import os
import antio
import json
from matplotlib.lines import lineStyles
from scipy.signal import butter, sosfiltfilt
from scipy.stats import zscore
from combined_analysis_bachelor.code.functions_for_pipeline import get_ch_indices, plot_channel_overview, normalize_emg, \
create_df
from utils.get_sub_dir import get_sub_folder_dir
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


def cnt_to_raw_hdf(sub, source_directory, target_directory, channels, emg_channels, location_dict, sampling_freq):
    filepaths = []
    for file in os.listdir(source_directory):
        if file.endswith(".cnt"):
            filepath = f"{source_directory}\\{file}"
            filepaths.append(filepath)

    print(f"retrieved data: {filepaths}")

    # ---- initializing json file ---- #
    json_path = f"{target_directory}\\sub-{sub}_recording_durations.json"
    print(json_path)

    durations = {}
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            durations = json.load(f)

    # ========= looping trough filepaths ======== #
    for filepath in filepaths:
        print("Checking file:", filepath)
        print("Basename:", os.path.basename(filepath))
        # --- read in file --- #
        if os.path.basename(filepath).startswith(("PTB_01", f"sub_{sub}" ,f"sub-{sub}")):
            source = mne.io.read_raw_ant(filepath, preload=True)

            emg_idxs = mne.pick_channels(source.ch_names, include=emg_channels) # defining unit info, for documentation
            for idx in emg_idxs:
                source.info["chs"][idx]["unit"] = FIFF.FIFF_UNIT_V
                source.info["chs"][idx]["unit_mul"] = 0

            data, times = source[channels, :] # get out data from source


            # --- correcting inverted channels --- #
            data[channels.index("BIP4")] *= -1
            data[channels.index("BIP6")] *= -1

            # --- convert EMG data from V to qv --- #
            for idx in [channels.index(ch) for ch in emg_channels]:
                data[idx] *= 1e6

            # --- finding trigger --- #
            onsets = source.annotations.onset
            print(onsets)
            onsets_samples = onsets * sampling_freq
            start_trig = int(onsets_samples[0])
            stop_trig = int(onsets_samples[-1])


            #if filepath == filepaths[3]: # in Rec 3 (but in new order=Rec4) there were random trig. signals too early --> how can i make this modualar?
            #    sync_trig = onsets[3]
            #    sync_trig_samples = int(onsets_samples[3])

            #else:
            #    sync_trig = onsets[0] # in seconds
            #    sync_trig_samples = int(onsets_samples[0]) # in samples


            # ----------- trimming ----------- #
            data_trimmed = data[:, start_trig:stop_trig]
            source_times = times[start_trig:stop_trig]

            # ---------- getting sync_times --------- #
            sync_times = source_times - (start_trig/1000)


            # ---------- creating dataframe ---------- #
            data_trimmed_T = data_trimmed.T
            raw_df = pd.DataFrame(data_trimmed_T, columns=[location_dict[ch] for ch in channels])
            assert all(ch in location_dict for ch in channels), "Some channels missing in your location_dict!"


            raw_df["Sync_Time (s)"] = sync_times
            raw_df["Source_Time (s)"] = source_times


            # --------- creating new file -------- #
            filename = os.path.basename(filepath)
            if filename.startswith((f"sub_{sub}", "PTB_01")) and filename.endswith(".cnt"):
                filename = filename.replace(f"sub_{sub}", f"sub-{sub}").replace("PTB_01", f"sub-{sub}")
                print(filename)
                filename = filename[:-4]
            filename += "_raw.h5"

            target_filepath = os.path.join(target_directory, filename)

            print(f"Saving to: {target_filepath}")

            # --- saving processed file to raw_data --- #
            raw_df.to_hdf(target_filepath, key="data", mode="w")

            print("Datei existiert:", os.path.exists(target_filepath))

            # --- saving to json file --- #
            durations[filename[:-7]] = sync_times[-1]

            with open(json_path, 'w') as f:
                json.dump(durations, f, indent=4)

    print("\n✅ Every File and .JSON processed and saved!")

#cnt_to_raw_hdf(SUB, source_dir, target_dir, channels_c, EMG, LOCS, SF)

#def cnt_to_raw_hdf_safe(sub, source_directory, target_directory, channels, location_dict, sampling_freq):
#    filepaths = []
#    for file in os.listdir(source_directory):
#        if file.endswith(".cnt"):
#            filepath = f"{source_directory}\\{file}"
#            filepaths.append(filepath)
#
#    print(f"retrieved data: {filepaths}")
#
#    # ========= looping trough filedirectory ======== #
#    for filepath in filepaths:
#        # --- read in file --- #
#        if os.path.basename(filepath).startswith(f"sub_{sub}"):
#            source = mne.io.read_raw_ant(filepath, preload=True)
#            data, times = source[channels, :]
#
#
#        # --- correcting inverted channels --- #
#            data[channels.index("BIP4")] *= -1
#            #data[1] *= -1
#            #data[6] *= -1
#
#
#            # --- finding trigger --- #
#            onsets = source.annotations.onset
#            onsets_samples = onsets * sampling_freq
#            onsets_df = pd.DataFrame(onsets_samples)
#
#            print(onsets)
#
#            sync_trig = onsets[0]
#            sync_trig_samples = int(sync_trig * sampling_freq)
#            print(sync_trig_samples)
#
#
#
#            # ----------- trimming ----------- #
#            data_trimmed = data[:, sync_trig_samples:]
#            source_times = times[sync_trig_samples:]
#
#            # ---------- getting sync_times --------- #
#            sync_times = source_times - sync_trig
#
#            # ---------- creating dataframe ---------- #
#            data_trimmed_T = data_trimmed.T
#            raw_df = pd.DataFrame(data_trimmed_T, columns=[location_dict[ch] for ch in channels])
#            assert all(ch in location_dict for ch in channels), "Some channels missing in your location_dict!"
#
#            raw_df["Sync_Time (s)"] = sync_times
#            raw_df["Source_Time (s)"] = source_times
#
#            # --------- creating new file -------- #
#            filename = os.path.basename(filepath)
#            if filename.startswith(f"sub_{sub}") and filename.endswith(".cnt"):
#                filename = filename.replace(f"sub_{sub}", f"sub-{sub}")
#                print(filename)
#            filename += "_raw.h5"
#
#            target_filepath = os.path.join(target_directory, filename)
#
#            print(f"Saving to: {target_filepath}")
#
#            raw_df.to_hdf(target_filepath, key="data", mode="w")
#
#    print("\n✅ Every File processed and saved!")
#
#
#