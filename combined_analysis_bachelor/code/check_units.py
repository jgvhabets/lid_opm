import numpy as np
import pandas as pd
import seaborn as sns
import mne
import os
import scipy as sp
from matplotlib.lines import lineStyles
from scipy.signal import butter, sosfiltfilt
from scipy.stats import zscore
from combined_analysis_bachelor.code.functions_for_pipeline import get_ch_indices, plot_channel_overview, normalize_emg, \
    notched_and_filtered, create_df, envelope, rectify, tkeo, filtered_and_notched
from my_utils.get_sub_dir import get_sub_folder_dir
from read_in_emg_acc import read_in_h5
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

### TESTING PTB FILES , SAME FILE , first is source, second is processed.h5!! ###

ACC = ["BIP6", "BIP1", "BIP2", "BIP3", "BIP4", "BIP5"]
EMG = ["BIP7", "BIP8", "BIP9", "BIP10", "BIP11", "BIP12"]
EMG_names = ["brachioradialis_R", "brachioradialis_L",
             "deltoideus_R", "deltoideus_L", "tibialisAnterior_R", "tibialisAnterior_L"]
EMG_names1 = ["brachioradialis_R", "brachioradialis_L",
             "deltoideus_R", "deltoideus_L"]
channel_custom_order = ["BIP3", "BIP4", "BIP5", "BIP9", "BIP10", "BIP12", "BIP6", "BIP1", "BIP2", "BIP11", "BIP8", "BIP7"]

#directory = "E:/PTB_01_data/PTB_01_A_1.5_move.cnt"
#raw = mne.io.read_raw_ant(directory, preload=True)
#onsets = raw.annotations.onset
#start = int(onsets[0] * 1000)
#data, times = raw[channel_custom_order, :]
#trimmed_data = data[:, start:] #
#trimmed_times = times[start:]
#df = create_df(trimmed_data, channel_custom_order, trimmed_times)
#filtered_df = notched_and_filtered(df, ACC, EMG, [1,20], [20,450])
# hab hier versucht, zu trimmedn und zu notchen und filtern um schauen zu können, dass es der selbe plot dann
# ist wie move2_processed.h5.. schaff ich grad aber nicht.. da kommt nen error wo markiert


#file = "E:/PTB_unit_test_proc/sub-91_EmgAcc_setupA_Move2_raw_processed.h5" # sollte selbe aufnahme wie oben sein!#
#move2 = pd.read_hdf(file, key="data")
#move2_df = pd.DataFrame(move2)

direc = get_sub_folder_dir("91", "processed_data")
path = f"{direc}/sub-91_EmgAcc_setupA_Move2_processed.h5"
correct_proc_df, _ = read_in_h5(path)
#rectified_df = rectify(correct_proc_df, EMG_names)
#enveloped_df = envelope(rectified_df, EMG_names, 3)


#fig, axs = plt.subplots(3, 1, figsize=(15,5))
#axs = axs.ravel()

#axs[0].plot(filtered_df["Time (s)"], filtered_df["BIP10"], label="manually")
#axs[0].set_title("manually read in, not scaled, filtered")
#
#axs[1].plot(move2_df["Source_Time (s)"], move2_df["deltoideus_L"], label="through function")
#axs[1].set_title("put through functions, processed, should be scaled")
#
#axs[2].plot(correct_proc_df["Source_Time (s)"], correct_proc_df["deltoideus_L"], label="correct")
#axs[2].set_title("correct processed file, should be scaled, enveloped")
#
#plt.tight_layout()
#plt.show()
#

# okay hier sind die units schonmal mormal wieder! nachdem sie
# cnt to other format safe (github kopieren) durchgelaufen sind und to proc. --> aber komisch
# dass es oben andere values sind als dann beim raw...

## ==================================================================================================== ##

# Jetzt dann mal: Fed.cnt durchlaufen lassen, vllt so abändern dass nichts geschnitten wird
# und onsets gespeichert werden. Dann trimmen mit trimm_dict und dann durch proc laufen lassen. Schauen was rauskommt.


### put full Federico rec into cnt_to_hdf and then also into processed --> still weird vals ###
#filepath = "E:/classification_raw_h5/sub-91_F.cnt_raw.h5"
#F_raw_h5 = pd.read_hdf(filepath, key="data")
#F_raw_h5_df = pd.DataFrame(F_raw_h5)
#F_filtered_df =

#plt.figure()
#plt.plot(F_raw_h5_df["Source_Time (s)"], F_raw_h5_df["deltoideus_R"])
#plt.show()

# vllt einmal processen? -- hab dann von folder classification_raw_h5 die files proc und in class_processed_test
# gespeichert --> auschecken! (gucken ob eh erst nach filtern die richtigen values da sind)
#file = "E:/classification_processed_test/sub-91_F.cnt__processed.h5"
#F_proc_h5 = pd.read_hdf(file, preload=True)
#F_proc_h5_df = pd.DataFrame(F_proc_h5)
#rectified_df = rectify(F_proc_h5_df, EMG_names1)
#enveloped_df = envelope(rectified_df, EMG_names1, 3)

#plt.figure()
#plt.plot(enveloped_df["Source_Time (s)"], enveloped_df["brachioradialis_R"])
#plt.show() ## okay ne immernoch komisch.






### WANTED to check how the data that I recorded on myself looks like (seperated rec already) ###

#filepa = "E:/sub_class_test2.cnt"
#me = mne.io.read_raw_ant(filepa, preload=True)
#data_me, times_me = me[channel_custom_order, :]


#plt.figure()
#plt.plot(times_me, data_me[-1])
#plt.show() #sieht auch weird aus...


### I think resolved : I just thought Values were off bc something happened in the functions
# Turns out --> These were probably just normal values, since they can go up to 10.000 microV!
















# checking out notch_filter function!

file = get_sub_folder_dir("91", "processed_data")
filepath = f"{file}/sub-91_EmgAcc_setupA_Move2_processed.h5"
processed_df = pd.DataFrame(pd.read_hdf(filepath, key="data"))

file = get_sub_folder_dir("91", "raw_data")
filepath = f"{file}/sub-91_EmgAcc_setupA_Move2_raw.h5"
raw_df = pd.DataFrame(pd.read_hdf(filepath, key="data"))

emg_cols = raw_df.columns[raw_df.columns.str.contains('brachioradialis|deltoideus|tibialis',
                                                                               case=False, regex=True)].tolist()
acc_cols = raw_df.columns[raw_df.columns.str.contains("hand", case=False, regex=True)].tolist()

# filtering raw manually
manually_processed_df = filtered_and_notched(raw_df, acc_cols, emg_cols, [1,20], [20,450], 1000)

# new function in fkt
target_dir = get_sub_folder_dir("91", "processed_data")
file = f"{target_dir}/test/sub-91_EmgAcc_setupA_Move2_processed.h5"
new_df = pd.DataFrame(pd.read_hdf(file, key="data"))

fig, axs = plt.subplots(4, 1)

axs[0].plot(raw_df["Sync_Time (s)"], raw_df["deltoideus_L"], label="raw_df")

axs[1].plot(processed_df["Sync_Time (s)"], processed_df["deltoideus_L"], label="first notched then filtered")

axs[2].plot(manually_processed_df["Sync_Time (s)"], manually_processed_df["deltoideus_L"], label="first filtered then notched")

axs[3].plot(new_df["Sync_Time (s)"], new_df["deltoideus_L"], label="new filtered then notched")

plt.legend()
plt.tight_layout()
plt.show()
