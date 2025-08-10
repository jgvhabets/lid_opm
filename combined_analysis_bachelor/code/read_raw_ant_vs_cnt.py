import numpy as np
import pandas as pd
import seaborn as sns
import mne
from mne.io import read_raw_cnt
from mne.io.constants import FIFF
import os
import scipy as sp
from matplotlib.lines import lineStyles
from scipy.signal import butter, sosfiltfilt
from scipy.stats import zscore
from combined_analysis_bachelor.code.functions_for_pipeline import get_ch_indices, plot_channel_overview, normalize_emg, \
    notched_and_filtered, create_df, envelope, rectify, tkeo
from utils.get_sub_dir import get_sub_folder_dir
from read_in_emg_acc import read_in_h5
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


## === Neuroscan File === #
filepath = "E:/classification/classification_Elisa_2025-07-21_15-14-53_Neuroscan.cnt"
emg_ch = ["BIP7", "BIP8", "BIP9", "BIP10", "BIP11", "BIP12"]

raw = read_raw_cnt(filepath, emg=emg_ch, preload=True)
print(raw.info["chs"][7])


## === EEGprobe File (same file) === #
filepath2 = "E:/classification_cnt/sub_class_E.cnt"

raw2 = mne.io.read_raw_ant(filepath2, preload=True)
emg_idxs = mne.pick_channels(raw2.ch_names, include=emg_ch)
for idx in emg_idxs:
    raw2.info["chs"][idx]["unit"] = FIFF.FIFF_UNIT_V
    raw2.info["chs"][idx]["unit_mul"] = 0


filepath2 = "E:/classification_cnt/sub_class_F.cnt"

raw3 = mne.io.read_raw_ant(filepath2, preload=True)
emg_idxs = mne.pick_channels(raw3.ch_names, include=emg_ch)
for idx in emg_idxs:
    raw3.info["chs"][idx]["unit"] = FIFF.FIFF_UNIT_V
    raw3.info["chs"][idx]["unit_mul"] = 0


# Better to use read_raw_ant for our use, since we use AntNeuro Amp and software.





