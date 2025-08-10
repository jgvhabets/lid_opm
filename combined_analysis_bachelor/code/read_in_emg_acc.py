import numpy as np
import pandas as pd
import os


### make into one function! ###

def get_filepaths_filenames(sub, paths, dType='emg_acc'): # similar one in file "get_sub_dir"
    """
    loops through paths to get out filepaths and filenames
    :param sub: sub-ID
    :param dType: set to EMG_ACC data
    :return: filepaths and filenames for each file
    """

    if dType == 'emg_acc': folder = 'EMG_ACC'

    sub_rawdata_dir = os.path.join(paths,
                                   f'sub-{sub}',
                                   folder,"class_processed")

    raw_files = os.listdir(sub_rawdata_dir)

    sub_rawdata_filepaths = []
    for file in raw_files:
        sub_rawdata_filepaths.append(os.path.join(sub_rawdata_dir, file))

    return sub_rawdata_filepaths, raw_files


def show_processed_data(sub, processed_paths, dType="emg_acc"):
    """
    loops through paths to get out filepaths and filenames
    :param sub: sub-ID
    :param dType: set to EMG_ACC data
    :return: filepaths and filenames for each file
    """
    if dType == "emg_acc": folder = "EMG_ACC"

    sub_procdata_dir = os.path.join(processed_paths,
                                    f"sub-{sub}",
                                    folder)
    processed_files = os.listdir(sub_procdata_dir)

    sub_procdata_filepaths = []
    for file in processed_files:
        sub_procdata_filepaths.append(os.path.join(sub_procdata_dir, file))

    return sub_procdata_filepaths, processed_files


def read_in_h5(filepath):
    """
    - reads in an h5 file (output is a df)
    - splits the filepath to the task_name

    :param filepath: list of filepaths for one SUB
    :return: df for one file (=one task) and the corresponding name of the task for
    better overview
    """

    processed = pd.read_hdf(filepath, key="data")
    processed_df = pd.DataFrame(processed)

    filename = os.path.basename(filepath)
    filename_split = filename.split("_")
    print(filename_split)
    task_name = filename_split[2] + "_" + filename_split[3] + "_" + filename_split[4][:-3]

    return processed_df, task_name





