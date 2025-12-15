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


def get_fieldline_in_mne(SUB, SES, TASK, ACQ, CROP_RETURN_TRIGGERS=False,):

    source_filepath = find_source_fl_file(SUB, SES, TASK, ACQ)
    raw = read_raw_fif(source_filepath, preload=True, verbose=True)

    # crop between 2nd and last trigger
    if CROP_RETURN_TRIGGERS:
        (FL_trigger_times, FL_trigger_types) = find_arduino_triggers(raw_mne_opm=raw)
         raw_cropped = raw.copy().crop(tmin=FL_trigger_times[1], tmax=FL_trigger_times[-1])
        # adjust triggers accodingly
        FL_trigger_times = np.array(FL_trigger_times) - FL_trigger_times[1]
        FL_trigger_types = FL_trigger_types[1:-1]

        return raw, FL_trigger_times, FL_trigger_times

    else:
        # Display the data header (raw.info)
        print("\n" + "="*60)
        print("DATA HEADER:")
        print("="*60)
        print(raw.info)

        return raw
