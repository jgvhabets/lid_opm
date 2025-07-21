"""
general functions to navigate through file structures
"""
import os
from os import getcwd, listdir
from os.path import dirname, join
from numpy import logical_and



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

    path = getcwd()

    while_count = 0

    while dirname(path)[-5:].lower() != 'users':
        path = dirname(path)
        while_count += 1
        if while_count > 20: return False

    # path is now Users/username
    onedrive_dirs = [f for f in listdir(path)
                     if 'charit' in f.lower()]
    # print(onedrive_dirs)

    for dir in onedrive_dirs:

        if not 'onedrive' in dir.lower():
            dir_files = listdir(join(path, dir))
            project_folder = [f for f in dir_files if f.endswith('LID_MEG')][0]
            project_directory = dir

        
        else:
            # Handle OneDrive folders - search recursively for LID_MEG project
            onedrive_path = join(path, dir)
            project_folder = None
            project_directory = None
            
            try:
                # Walk through OneDrive directory to find LID_MEG folder
                for root, dirs, files in os.walk(onedrive_path):
                    lid_meg_dirs = [d for d in dirs if d.endswith('LID_MEG')]
                    if lid_meg_dirs:
                        project_folder = lid_meg_dirs[0]
                        # Get relative path from the base OneDrive directory
                        project_directory = os.path.relpath(root, path)
                        break
                
                if not project_folder:
                    print(f'No project_folder found in OneDrive directory: {dir}')
                    continue
                    
            except Exception as e:
                print(f'Error searching OneDrive folder {dir}: {e}')
                continue
            
    project_path = join(path, project_directory, project_folder)

    print(f'project folder found: {project_path}')

    
    if folder == 'project': return project_path

    elif folder == 'data': return join(project_path, 'data')

    elif folder == 'raw_data': return join(project_path, 'data', 'raw_data')

    elif folder == 'processed_data': return join(project_path, 'data', 'processed_data')

    elif folder == 'source_data': return join(project_path, 'data', 'source_data')

    elif folder == 'figures': return join(project_path, 'figures')

    elif folder == 'results': return join(project_path, 'results')

'''
raw_data_path = get_onedrive_path('raw_data')
processed_data_path = get_onedrive_path('processed_data')
source_data_path = get_onedrive_path('source_data')
'''

def get_available_subs(data, subs_folder):
    "Function to get available subjects in the given data folder."
    "Arguments: data: str, subs_folder: str"
    "Returns: list of available subjects in the given folder."
    
    data_path = get_onedrive_path(folder=data)

    subs = listdir(subs_folder)

    return subs