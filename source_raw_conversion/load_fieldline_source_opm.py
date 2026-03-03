import os
from mne.io import read_raw_fif
import numpy as np

from utils.load_utils import get_onedrive_path
from source_raw_conversion.time_syncing import find_arduino_triggers

def find_source_fl_file(SUB, SES, TASK, ACQ,):

    if len(SUB) == 2: SUB = 'sub-' + SUB

    # find filepath
    ses_path = os.path.join(
        get_onedrive_path('source_data'),
        SUB,
        f'ses-{SES}',
        'opm'
    )
    files = os.listdir(ses_path)
    sel_fname = [f for f in files if TASK in f and ACQ in f and f.endswith('.fif')][0]

    file_path = os.path.join(ses_path, sel_fname)
    assert os.path.exists(file_path), 'WARNING. FILEPATH NOTE EXISTING'

    return file_path


def get_fieldline_in_mne(
    SUB, SES, TASK, ACQ,
    TARGET_SFREQ=None,
    CROP_RETURN_TRIGGERS=False,
    CROP_MARGIN = 0,
):
    """
    Loads raw fif file of fieldline OPM data into mne, with option to resample and crop.
    
    resampling is done if target_sfreq is set
    CROP_RETURN_TRIGGERS defaults False, if True start and ending of recording
    iscropped based on present triggers.
    CROP_MARGIN is taken around triggers if > 0.

    if cropping is true, trigger_times/types are returned bcs of change in
    time-axis due to cropping
    """

    source_filepath = find_source_fl_file(SUB, SES, TASK, ACQ)
    raw = read_raw_fif(source_filepath, preload=True, verbose=True)

    if type(TARGET_SFREQ) != type(None):
        raw.resample(sfreq=TARGET_SFREQ)

    # crop between 2nd and last trigger
    if CROP_RETURN_TRIGGERS:
        (FL_trigger_times, FL_trigger_types) = find_arduino_triggers(raw_mne_opm=raw)
        # TODO include arduino specific start/end triggers
        print('in next arduino version: include specific start/end triggers')
        raw_cropped = raw.copy().crop(
            tmin=FL_trigger_times[0] - CROP_MARGIN,
            tmax=FL_trigger_times[-1] + CROP_MARGIN
        )
        # adjust triggers accodingly
        FL_trigger_times = np.array(FL_trigger_times[1:-1]) - (FL_trigger_times[0] + CROP_MARGIN)
        FL_trigger_types = FL_trigger_types[1:-1]
        assert len(FL_trigger_times) == len(FL_trigger_types), (
            f'after cropping, unequal FL trigger times ({len(FL_trigger_times)})'
             f' and types ({len(FL_trigger_types)})'
        )

        return raw_cropped, FL_trigger_times, FL_trigger_types

    else:
        # Display the data header (raw.info)
        print("\n" + "="*60)
        print("DATA HEADER:")
        print("="*60)
        print(raw.info)

        return raw
