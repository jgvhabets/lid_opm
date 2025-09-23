"""
Contains functions to load source data stored with LSL,
typically containing AntNeuro acc + emg, AntNeuro triggers,
and behavioral markers from the PyGame Stream
"""

import os
import pyxdf

from utils.load_utils import get_onedrive_path


def get_source_streams(SUB, ACQ, TASK):
    """
    Input:
    - SUB
    - ACQ
    - TASK

    Returns:
    - streams
    - fileheader
    """

    lsl_source_path = os.path.join(get_onedrive_path('source_data'),
                                   f'sub-{SUB}', 'lsl')
    # gets folder with defined task and acquisition
    try:
        sel_folder = [f for f in os.listdir(lsl_source_path)
                      if ACQ in f.lower() and TASK in f.lower()][0]
    except IndexError:
        raise ValueError(f'IndexError catched, no folder for sub-{SUB, ACQ, TASK}')
    
    sel_path = os.path.join(lsl_source_path, sel_folder, 'eeg')  # convert to path
    sel_fname = sel_f = [f for f in os.listdir(sel_path)][0]

    # load defined LSL file
    streams, fileheader = pyxdf.load_xdf(os.path.join(sel_path, sel_fname, ))

    return streams, fileheader


def define_streams(streams):
    """
    Input:
    - streams

    Returns:
    - AN_DATA stream
    - AN_MRK stream
    - Pygame stream (if found), otherwise None
    """

    AN_DATA, AN_MRK, PYGAME = None, None, None

    print(f"### Found {len(streams)} streams")

    for i, stream in enumerate(streams):
        # print(f"\n--- Stream {i} ---")
        # print("Name:", stream['info']['name'][0])
        # print("Type:", stream['info']['type'][0])
        # print("Channel count:", stream['info']['channel_count'][0])
        
        # select streams
        if stream['info']['type'][0] == 'Markers':
            if 'TRG' in stream['info']['name'][0]:
                AN_MRK = stream
                print(f'\tinclude as AntNeuro Marker:', stream['info']['name'][0])
            
            elif 'gonogo' in stream['info']['name'][0].lower():
                PYGAME = stream
                print(f'\tinclude as PyGame:', stream['info']['name'][0])
    
            else:
                print('unknown MARKERS stream', stream['info']['name'][0])

        elif stream['info']['type'][0] == 'EEG':
            AN_DATA = stream
            print(f'\tinclude as AntNeuro DATA:', stream['info']['name'][0])
    
        else:
            print('unknown stream', stream['info']['type'][0], stream['info']['name'][0])
    
    if not PYGAME: print('\tno pygame stream found')
    
    return AN_DATA, AN_MRK, PYGAME