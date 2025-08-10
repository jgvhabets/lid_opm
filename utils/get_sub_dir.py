import numpy as np
import os
from utils.find_paths import get_onedrive_path


def get_sub_folder_dir(sub, data_subfolder: str, dType="emg_acc" ):
    """
    get the directory of the sub folder
    for getting or saving data to sub directory
    """

    subfolder_dir = get_onedrive_path("onedrive_charite", data_subfolder)
    print(subfolder_dir)

    subfolder_options = ['project', 'figures', 'data',
                                'raw_data', 'source_data',
                                    'processed_data', 'results',
                                "classification_data", "onset_data"]

    if data_subfolder.lower() not in subfolder_options:
        raise ValueError(
            f'given folder: {data_subfolder} is incorrect, '
            f'should be {subfolder_options}')

    if dType == 'emg_acc': folder = 'EMG_ACC'

    dir = os.path.join(subfolder_dir,
                            f'sub-{sub}',
                                folder)

    return dir

