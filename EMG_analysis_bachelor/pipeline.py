import numpy as np
import pandas as pd
import seaborn as sns
import mne
import scipy as sp
from scipy.signal import butter, sosfiltfilt
from EMG_analysis_bachelor.functions_for_pipeline import get_ch_indices, plot_channel_overview, normalize_emg, notched, \
    filtered, create_df, envelope
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

# import data
EMG_ACC_data = mne.io.read_raw_ant("C:/Users/User/Documents/bachelorarbeit/data/EMG_ACC/"
                                   "PTB_measurement_14.04/Bonato_Federico_2025-04-14_13-02-56.cnt", preload=True)
EMG_only_test = mne.io.read_raw_ant("C:/Users/User/Documents/bachelorarbeit/data/EMG/"
                                   "test_first_2025-04-07_14-10-04_forearm_upperSide.cnt", preload=True)
custom_order_names = ["BIP6", "BIP1", "BIP2", "BIP7", "BIP8"]
# custom_order_names = ["BIP7", "BIP8"]
location = {"BIP7":"left forearm",
            "BIP8": "left delt",
            "BIP1" : "Charité ACC : y",
            "BIP2" : "Charité ACC : z",
            "BIP6" : "Charité ACC : x"}
EMG = ["BIP7", "BIP8"]
ACC = ["BIP6", "BIP1", "BIP2"]
emg_idx, acc_idx = get_ch_indices(custom_order_names, EMG, ACC)

data, times = EMG_ACC_data[custom_order_names, : ]
data[0] *= -1

# creating raw df
raw_df = create_df(data, custom_order_names, times)

# plotting the raw signals
plot_channel_overview(custom_order_names, raw_df,"raw_signals", location, EMG)

# apply notch filter
notched_df = notched(raw_df, custom_order_names, times)

# filtering
notched_and_filtered_df = filtered(notched_df, custom_order_names, ACC, EMG, times)

# plot filtered data (overview?)
plot_channel_overview(custom_order_names, notched_and_filtered_df,"filtered_signals", location, EMG)

# retrification and plotting
rectified_df = abs(notched_and_filtered_df) #technically also took the absolute values of the time column, but that doesnt matter i think

plot_channel_overview(custom_order_names, rectified_df, "rectified_signals", location, EMG)

# normalizing
emg_normalized_df = normalize_emg(rectified_df, custom_order_names, EMG, times)

# plot normalized data
plot_channel_overview(EMG, emg_normalized_df, "normalized_df", location, EMG)

# build envelope
emg_envelopes = envelope(emg_normalized_df, custom_order_names, EMG, times, 3)
plot_channel_overview(EMG, emg_envelopes, "normalized_df", location, EMG)







emg_envelopes = pd.DataFrame()

low_pass = 10/(1000/2)
sos = butter(4, low_pass, btype='lowpass', output="sos")

emg_columns = [col for col in df_rectified.columns if col != 'Time (s)']
for col in emg_columns:
    signal = df_rectified.loc[:, col].values
    filtered = sosfiltfilt(sos, x=signal)
    emg_envelopes[col] = filtered

emg_envelopes["Time (s)"] = times

plot_channel_overview(custom_order_names, emg_envelopes, "emg_envelopes", location, EMG)

