import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from mne.io import read_raw_ant
import json
import antio
import os
from combined_analysis_bachelor.code.functions_for_pipeline import filtered_and_notched, rectify, envelope, add_tkeo, \
    add_tkeo_add_envelope, crop_edges, sliding_windows, extract_features_from_window, apply_savgol_rms_acc
from my_utils.get_sub_dir import get_sub_folder_dir
from combined_analysis_bachelor.code.create_processed_files import create_processed_files

#
#source_dir = get_sub_folder_dir("92", "processed_data")
#file = f"{source_dir}/sub-92_electrode_compare_processed.h5"
#df = pd.DataFrame(pd.read_hdf(file, key="data"))
#df = crop_edges(df)
#
#
#fig, axs = plt.subplots(2, 2, sharex=True)
#axs = axs.ravel()
#
#axs[0].plot(df["time_sec"], df["deltoideus_L"], label="klebe elektrode, delt")
#axs[0].set_xlabel("time (s)")
#axs[0].set_ylabel("Amplitude (microV)")
#axs[0].set_title("Klebe Elektrode Beispiel")
#axs[0].legend()
#
#axs[1].plot(df["time_sec"], df["deltoideus_R"], label="teller elektrode, delt")
#axs[1].set_xlabel("time (s)")
#axs[1].set_ylabel("Amplitude (microV)")
#axs[1].set_title("Teller Elektrode Beispiel")
#axs[1].legend()
#
#axs[2].plot(df["time_sec"], df["brachioradialis_L"], label="klebe elektrode, unterarm")
#axs[2].set_xlabel("time (s)")
#axs[2].set_ylabel("Amplitude (microV)")
#axs[2].legend()
#
#axs[3].plot(df["time_sec"], df["brachioradialis_R"], label="teller elektrode, unterarm")
#axs[3].set_xlabel("time (s)")
#axs[3].set_ylabel("Amplitude (microV)")
#axs[3].legend()
#
#plt.tight_layout()
#plt.show()



# ====== measuring SNR ====== #
rest = [4,14]
delt_move = [34,44]
brachioradialis_suppr = [18,24]

SF = 1000
window_Size = 100

def rms(signal, window_size):
    return np.sqrt(np.convolve(signal**2, np.ones(window_size)/window_size, mode='valid'))

# Ruhe- und Aktivbereiche extrahieren
# emg_rest_delt_klebe = df["deltoideus_L"][int(rest[0]*SF):int(rest[1]*SF)]
# emg_active_delt_klebe = df["deltoideus_L"][int(delt_move[0]*SF):int(delt_move[1]*SF)]
#
# emg_rest_delt_teller = df["deltoideus_R"][int(rest[0]*SF):int(rest[1]*SF)]
# emg_active_delt_teller = df["deltoideus_R"][int(delt_move[0]*SF):int(delt_move[1]*SF)]
#
# emg_rest_brachio_klebe = df["brachioradialis_L"][int(rest[0]*SF):int(rest[1]*SF)]
# emg_active_brachio_klebe = df["brachioradialis_L"][int(brachioradialis_suppr[0]*SF):int(brachioradialis_suppr[1]*SF)]
#
# emg_rest_brachio_teller = df["brachioradialis_R"][int(rest[0]*SF):int(rest[1]*SF)]
# emg_active_brachio_teller = df["brachioradialis_R"][int(brachioradialis_suppr[0]*SF):int(brachioradialis_suppr[1]*SF)]

# params
fs = 1000  # Hz
window_size_ms = 100
window_samples = int(window_size_ms * fs / 1000)

# rms
#rms_rest_delt_klebe = rms(emg_rest_delt_klebe, window_samples)
#rms_active_delt_klebe = rms(emg_active_delt_klebe, window_samples)
#
#rms_rest_delt_teller = rms(emg_rest_delt_teller, window_samples)
#rms_active_delt_teller = rms(emg_active_delt_teller, window_samples)
#
#rms_rest_brachio_klebe = rms(emg_rest_brachio_klebe, window_samples)
#rms_active_brachio_klebe = rms(emg_active_brachio_klebe, window_samples)
#
#rms_rest_brachio_teller = rms(emg_rest_brachio_teller, window_samples)
#rms_active_brachio_teller = rms(emg_active_brachio_teller, window_samples)
#
## means
#mean_rms_rest_delt_klebe = np.mean(rms_rest_delt_klebe)
#mean_rms_active_delt_klebe = np.mean(rms_active_delt_klebe)
#
#mean_rms_rest_delt_teller = np.mean(rms_rest_delt_teller)
#mean_rms_active_delt_teller = np.mean(rms_active_delt_teller)
#
#mean_rms_rest_brachio_klebe = np.mean(rms_rest_brachio_klebe)
#mean_rms_active_brachio_klebe = np.mean(rms_active_brachio_klebe)
#
#mean_rms_rest_brachio_teller = np.mean(rms_rest_brachio_teller)
#mean_rms_active_brachio_teller = np.mean(rms_active_brachio_teller)

# SNR delt klebe
# snr_db = 20 * np.log10(mean_rms_active_delt_klebe / mean_rms_rest_delt_klebe)
# print(f"SNR für deltoideus mit klebe-Elektroden: {snr_db:.0f}")
#
# # SNR delt teller
# snr_db = 20 * np.log10(mean_rms_active_delt_teller / mean_rms_rest_delt_teller)
# print(f"SNR für deltoideus mit teller-Elektroden: {snr_db:.0f}")
#
# # SNR brachio klebe
# snr_db = 20 * np.log10(mean_rms_active_brachio_klebe / mean_rms_rest_brachio_klebe)
# print(f"SNR für brachio mit klebe-Elektroden: {snr_db:.0f}")
#
# # SNR brachio teller
# snr_db = 20 * np.log10(mean_rms_active_brachio_teller / mean_rms_rest_brachio_teller)
# print(f"SNR für brachio mit teller-Elektroden: {snr_db:.0f}")






###### ======================== jetzt für Annica ======================== ######
source_dir = get_sub_folder_dir("91", "processed_data")
print(source_dir)
file = f"{source_dir}/sub-91_EmgAcc_setupA_move2_processed.h5"
file_tib = f"{source_dir}/sub-91_EmgAcc_setupA_RestMockDys_processed.h5"
df = pd.DataFrame(pd.read_hdf(file, key="data"))
df_tib = pd.DataFrame(pd.read_hdf(file_tib, key="data"))

fig, axs = plt.subplots(1, 3, sharex=True)
axs = axs.ravel()

axs[0].plot(df["Sync_Time (s)"], df["deltoideus_L"], label="delt")
axs[0].set_xlabel("Sync_Time (s)")
axs[0].set_ylabel("Amplitude (microV)")
axs[0].set_title("delt links")
axs[0].legend()

axs[1].plot(df["Sync_Time (s)"], df["brachioradialis_L"], label="brachio")
axs[1].set_xlabel("Sync_Time (s)")
axs[1].set_ylabel("Amplitude (microV)")
axs[1].set_title("brachio links")
axs[1].legend()

axs[2].plot(df_tib["Sync_Time (s)"], df_tib["tibialisAnterior_L"], label="tibialis anterior")
axs[2].set_xlabel("Sync_Time (s)")
axs[2].set_ylabel("Amplitude (microV)")
axs[2].set_title("tibialis anterior links")
axs[2].legend()

plt.tight_layout()
plt.show()


rest_delt_list = [7,14]
active_delt_list = [31.5, 33.5]
rest_brach_list = [9,15]
active_brach_list = [66.5, 67.5]
rest_tib_list = [25,50]
active_tib_list = [80,110]

rest_delt = df["deltoideus_L"][int(rest_delt_list[0]*SF):int(rest_delt_list[1]*SF)]
active_delt = df["deltoideus_L"][int(active_delt_list[0]*SF):int(active_delt_list[1]*SF)]

rest_brachio = df["brachioradialis_L"][int(rest_brach_list[0]*SF):int(rest_brach_list[1]*SF)]
active_brachio = df["brachioradialis_L"][int(active_brach_list[0]*SF):int(active_brach_list[1]*SF)]

rest_tibialis = df_tib["brachioradialis_L"][int(rest_tib_list[0]*SF):int(rest_tib_list[1]*SF)]
active_tibialis = df_tib["brachioradialis_L"][int(active_tib_list[0]*SF):int(active_tib_list[1]*SF)]

rms_rest_delt = rms(rest_delt, window_samples)
rms_active_delt = rms(active_delt, window_samples)

rms_rest_brachio = rms(rest_brachio, window_samples)
rms_active_brachio = rms(active_brachio, window_samples)

rms_rest_tibialis = rms(rest_tibialis, window_samples)
rms_active_tibialis = rms(active_tibialis, window_samples)

# -----------------------means-----------------------------
mean_rms_rest_delt = np.mean(rms_rest_delt)
mean_rms_active_delt = np.mean(rms_active_delt)

mean_rms_rest_brachio = np.mean(rms_rest_brachio)
mean_rms_active_brachio = np.mean(rms_active_brachio)

mean_rms_rest_tibialis = np.mean(rms_rest_tibialis)
mean_rms_active_tibialis = np.mean(rms_active_tibialis)

# --------------------------SNR------------------------------
snr_db_delt = 20 * np.log10(mean_rms_active_delt / mean_rms_rest_delt)
print(f"SNR für delt PTB-measurement: {snr_db_delt:.0f}")

snr_db_brachio = 20 * np.log10(mean_rms_active_brachio / mean_rms_rest_brachio)
print(f"SNR für brachioradialis PTB-measurement: {snr_db_brachio:.0f}")

snr_db_tibialis = 20 * np.log10(mean_rms_active_brachio / mean_rms_rest_brachio)
print(f"SNR für tibialis PTB-measurement: {snr_db_tibialis:.0f}")