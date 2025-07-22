import numpy as np
import pandas as pd
import os


### make into one function! ###

def show_raw_data(sub, raw_paths, dType='emg_acc'): # similar one in file "get_sub_dir"
    """
    loops through paths to get out filepaths and filenames
    :param sub: sub-ID
    :param dType: set to EMG_ACC data
    :return: filepaths and filenames for each file
    """

    if dType == 'emg_acc': folder = 'EMG_ACC_data'

    sub_rawdata_dir = os.path.join(raw_paths,
                                   f'sub-{sub}',
                                   folder)

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
    if dType == "emg_acc": folder = "EMG_ACC_data"

    sub_procdata_dir = os.path.join(processed_paths,
                                    f"sub-{sub}",
                                    folder)
    processed_files = os.listdir(sub_procdata_dir)

    sub_procdata_filepaths = []
    for file in processed_files:
        sub_procdata_filepaths.append(os.path.join(sub_procdata_dir, file))

    return sub_procdata_filepaths, processed_files


def read_in_h5_to_df(h5_filepaths):
    """
    takes in a list of filepaths (of h5 files)
    - cuts filename accordingly to only get out the name of the task
    - reads them in and turns them into dataframes for each file

    output:
    list of the dataframes
    task_names (for usage in plotting)
    """
    task_names = []
    dfs = []
    for filepath in h5_filepaths:

        filename = os.path.basename(filepath)
        filename_split = filename.split("_")
        task_name = filename_split[2] +  "_" + filename_split[3] + "_" + filename_split[4][:-3]
        task_names.append(task_name)

        data = pd.read_hdf(filepath, key="data")
        df = pd.DataFrame(data) # make sure its a df, not an object
        dfs.append(df)
    return task_names, dfs

