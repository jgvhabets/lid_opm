"""

"""

# import
import json
import os
from typing import Dict, Any
from pandas import read_csv, read_excel



def get_onedrive_path(folder: str = 'project',):
    """
    Device and OS independent function to find
    the synced-OneDrive folder where data is stored
    """
    folder_options = ['project', 'figures','data',
                      'raw_data', 'source_data',
                      'processed_data', 'results',]
    
    if folder.lower() not in folder_options:
        raise ValueError(
            f'given folder: {folder} is incorrect, '
            f'should be {folder_options}')

    path = os.getcwd()

    while_count = 0

    while os.path.dirname(path)[-5:].lower() != 'users':
        path = os.path.dirname(path)
        while_count += 1
        if while_count > 20: return False

    # path is now Users/username
    onedrive_dirs = [f for f in os.listdir(path)
                     if 'charit' in f.lower()]

    for dir in onedrive_dirs:

        if 'onedrive' in dir.lower():
            dir_files = os.listdir(os.path.join(path, dir))
            project_folder = [f for f in dir_files if f.endswith('LID_MEG')][0]
            project_directory = dir

        
    project_path = os.path.join(path, project_directory, project_folder)

    
    if folder == 'project': return project_path

    elif folder == 'data': return os.path.join(project_path, 'data')

    elif folder == 'raw_data': return os.path.join(project_path, 'data', 'raw_data')

    elif folder == 'processed_data': return os.path.join(project_path, 'data', 'processed_data')

    elif folder == 'source_data': return os.path.join(project_path, 'data', 'source_data')

    elif folder == 'figures': return os.path.join(project_path, 'figures')

    elif folder == 'results': return os.path.join(project_path, 'results')


def load_subject_config(subject_id: str, config_dir: str = '../configs',) -> Dict[str, Any]:
    """
    Load configuration file for a specific subject and version.
    """
    config_file = os.path.join(config_dir, f'config_sub{subject_id}.json')
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in config file {config_file}: {e}")
    
    return config


def get_sub_rec_metainfo(config_sub):

    metainfo = read_excel(
        os.path.join(get_onedrive_path('source_data'),
                     config_sub["subject_id"],
                     f'rec_admin_{config_sub["subject_id"]}.xlsx'),
        header=0, index_col=0,
    )
    
    return metainfo



def load_preproc_config(version: str, config_dir: str = '../configs',) -> Dict[str, Any]:
    """
    Load preproc setting configurations for a version.
    """
    config_file = os.path.join(config_dir, f'preproc_settings_{version}.json')
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in config file {config_file}: {e}")
    
    return config