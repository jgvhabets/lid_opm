"""
general functions to navigate through file structures
"""

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
            # print(f'Folder {dir} is skipped, check onedrive finding!!')
            # TODO: possibly alternative project_folder finding
            print('TODO: create altearntive onedrive finding')
            
    project_path = join(path, project_directory, project_folder)

    # print(f'project folder found: {project_path}')

    
    if folder == 'project': return project_path

    elif folder == 'data': return join(project_path, 'data')

    elif folder == 'raw_data': return join(project_path, 'data', 'raw_data')

    elif folder == 'processed_data': return join(project_path, 'data', 'processed_data')

    elif folder == 'source_data': return join(project_path, 'data', 'source_data')

    elif folder == 'figures': return join(project_path, 'figures')

    elif folder == 'results': return join(project_path, 'results')

    