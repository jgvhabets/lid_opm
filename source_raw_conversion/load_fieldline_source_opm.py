import os
from mne.io import read_raw_fif

from utils.load_utils import get_onedrive_path


def find_source_fl_file(SUB, SES, TASK):

    # find filepath
    ses_path = os.path.join(
        get_onedrive_path('source_data'),
        SUB,
        f'ses-{SES}',
        'opm'
    )
    files = os.listdir(ses_path)
    sel_fname = [f for f in files if TASK in f and f.endswith('.fif')][0]

    file_path = os.path.join(ses_path, sel_fname)
    assert os.path.exists(file_path), 'WARNING. FILEPATH NOTE EXISTING'

    return file_path


def get_fieldline_in_mne(SUB, SES, TASK):

    source_filepath = find_source_fl_file(SUB, SES, TASK)
    raw = read_raw_fif(source_filepath, preload=True, verbose=True)

    # Display the data header (raw.info)
    print("\n" + "="*60)
    print("DATA HEADER:")
    print("="*60)
    print(raw.info)

    return raw
