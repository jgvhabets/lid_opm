import numpy as np
import os
from utils.find_paths import get_onedrive_path


def get_sub_folder_dir(sub, data_subfolder: str, dType="emg_acc" ):
    """
    used to get the directory of the sub folder
    for getting or saving data to sub directory
    """

    subfolder_dir = get_onedrive_path(data_subfolder)

    subfolder_options = ['project', 'figures', 'data',
                                'raw_data', 'source_data',
                                    'processed_data', 'results', ]

    if data_subfolder.lower() not in subfolder_options:
        raise ValueError(
            f'given folder: {data_subfolder} is incorrect, '
            f'should be {subfolder_options}')

    if dType == 'emg_acc': folder = 'EMG_ACC_data'

    dir = os.path.join(subfolder_dir,
                            f'sub-{sub}',
                                folder)

    return dir