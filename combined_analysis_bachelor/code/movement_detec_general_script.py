import numpy as np
import scipy as sp
import pandas as pd
import mne
import os
import importlib
import matplotlib.pyplot as plt
from my_utils.get_sub_dir import get_sub_folder_dir
import my_utils.find_paths as find_paths
from matplotlib.widgets import Slider
from combined_analysis_bachelor.code.read_in_emg_acc import show_processed_data, read_in_h5
from combined_analysis_bachelor.code.functions_for_pipeline import tkeo, rectify, envelope, savgol, root_mean_square, \
    plot_overview, plot_overview_threshs, interactive_multichannel_plot, plot_sliding_window_detection, sliding_window_threshold_detection

SUB = 91
SF= 1000
fig_dir = get_sub_folder_dir(SUB, "figures")
baselines = {
    "arm_L_setupA_Move1": [10,20],
    "arm_R_setupA_Move1": [10,20],
    "arm_L_setupA_Move2": [10,15],
    "arm_R_setupA_Move2": [10,15],
    "arm_L_setupA_MoveMockDys": [3,13],
    "arm_R_setupA_MoveMockDys": [8,14],
    "arm_L_setupA_Rest": [10,30],
    "arm_R_setupA_Rest": [10,30],
    "arm_L_setupA_RestMockDys": [10,30],
    "arm_R_setupA_RestMockDys": [10,30],
    "arm_L_setupB_Move": [12,16],
    "leg_L_setupB_Move": [12,15],
    "arm_L_setupB_MoveMockDys": [10,15],
    "leg_L_setupB_MoveMockDys": [10,15],
    "arm_L_setupB_Rest": [10,20],
    "leg_L_setupB_Rest": [10,20],
    "arm_L_setupB_RestMockDys": [20,30],
    "leg_L_setupB_RestMockDys": [20,30]
}

processed_data_path = find_paths.get_onedrive_path("processed_data")
sub91_filepaths, processed_files = show_processed_data(sub=SUB, processed_paths=processed_data_path)
figure_dir = get_sub_folder_dir(SUB, "figures")


# hier fehlt code!!!! ersetzen!!!! in loop einbauen, wo task_name genommen wird (damit task name gleich bleibt!)


#### === plotting per task per limb === ###
#if "setupA".lower() in task_name.lower():
#    print("going trough setupA")
#    #plot_overview("arm_L", ready_for_detec_df, task_name, emg_cols, fig_dir)
#    #plot_overview("arm_R", ready_for_detec_df, task_name, emg_cols, fig_dir)
#    #plot_overview_threshs("arm_L", ready_for_detec_df, task_name, emg_cols, fig_dir, baselines, SF)
#    #plot_overview_threshs("arm_R", ready_for_detec_df, task_name, emg_cols, fig_dir, baselines, SF)
#    interactive_multichannel_plot("arm_L", ready_for_detec_df, task_name, emg_cols, baselines, SF)
#    interactive_multichannel_plot("arm_R", ready_for_detec_df, task_name, emg_cols, baselines, SF)
#elif "setupB".lower() in task_name.lower(): #### noch einbauen dass es bis 25 gibt, tight layout - geht nicht!! + noch letzte funktion??
#    print("going trough setupB")
#    #plot_overview("arm_L", ready_for_detec_df, task_name, emg_cols, fig_dir)
#    #plot_overview("leg_L", ready_for_detec_df, task_name, emg_cols, fig_dir)
#    #plot_overview_threshs("arm_L", ready_for_detec_df, task_name, emg_cols, fig_dir, baselines, SF)
#    #plot_overview_threshs("leg_L", ready_for_detec_df, task_name, emg_cols, fig_dir, baselines, SF)
#    interactive_multichannel_plot("arm_L", ready_for_detec_df, task_name, emg_cols, baselines, SF)
#    interactive_multichannel_plot("leg_L", ready_for_detec_df, task_name, emg_cols, baselines, SF)