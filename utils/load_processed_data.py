"""
Functions to load preprocessed MNE data (fif) saved by
signal_processing.preproc_functions.save_cleaned_data.
"""

import os
import mne

from utils.load_utils import get_onedrive_path


# --------------------------------------------------------------------------- #
# path helpers (mirrors save_cleaned_data naming exactly)
# --------------------------------------------------------------------------- #


def _find_file(modality: str, data_type: str, config_version: str,
               SUB: str, SES: str, ACQ: str, TASK: str) -> tuple[str, str]:
    """
    Tries 'raw' then 'epochs' to find the saved fif file.
    Returns (full_path, data_type) for the first match found.
    Raises FileNotFoundError if neither exists.
    """
    # get deriv_dir path
    deriv_dir = os.path.join(
        get_onedrive_path('cleaned_data'),
        f"cleaned_data_preproc_{config_version}",
        f"sub-{SUB}",
    )
    if not os.path.isdir(deriv_dir):
        raise FileNotFoundError(f"Derivatives directory not found: {deriv_dir}")

    # build expected filenames based on naming convention in save_cleaned_data
    fname = f"{modality}_cleaned_{data_type}_sub-{SUB}_ses-{SES}_acq-{ACQ}_task-{TASK}.fif"
    fpath = os.path.join(deriv_dir, fname)
    
    if os.path.exists(fpath):
        return fpath, data_type

    raise FileNotFoundError(
        f"No cleaned {modality} file found for "
        f"sub-{SUB} ses-{SES} acq-{ACQ} task-{TASK} "
        f"in {deriv_dir}"
    )


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #

def load_cleaned_data(
    SUB: str,
    SES: str,
    ACQ: str,
    TASK: str,
    config_version: str = 'v1',
    data_type: str = 'raw',  # 'raw' or 'epochs', only used for filename construction, not for loading
    load_meg: bool = True,
    load_aux: bool = False,
    preload: bool = True,
):
    """
    Load preprocessed MEG and/or AUX data saved by save_cleaned_data.
    Automatically detects whether the file is Raw or Epochs.

    Parameters
    ----------
    SUB, SES, ACQ, TASK : str
        Recording identifiers, must match those used when saving.
    config_version : str
        Preprocessing config version (e.g. 'v1'), must match saving call.
    load_meg : bool
        Load the MEG fif file (default True).
    load_aux : bool
        Load the AUX fif file (default False).
    preload : bool
        Whether to preload data into memory (passed to MNE readers).

    Returns
    -------
    meg_data : mne.io.Raw or mne.Epochs or None
    aux_data : mne.io.Raw or mne.Epochs or None
    """
    
    meg_data = None
    aux_data = None

    if load_meg:
        fpath, data_type = _find_file(
            modality='MEG', data_type=data_type,
            SUB=SUB, SES=SES, ACQ=ACQ, TASK=TASK,
            config_version=config_version,
        )
        print(f"[load] MEG ({data_type}): {fpath}")
        meg_data = _read_fif(fpath, data_type, preload)

    if load_aux:
        fpath, data_type = _find_file(
            modality='AUX', data_type=data_type,
            SUB=SUB, SES=SES, ACQ=ACQ, TASK=TASK,
            config_version=config_version,
        )
        print(f"[load] AUX ({data_type}): {fpath}")
        aux_data = _read_fif(fpath, data_type, preload)

    return meg_data, aux_data


def _read_fif(fpath: str, data_type: str, preload: bool):
    if data_type == 'raw':
        return mne.io.read_raw_fif(fpath, preload=preload, verbose=False)
    elif data_type == 'epochs':
        return mne.read_epochs(fpath, preload=preload, verbose=False)
    else:
        raise ValueError(f"Unknown data_type '{data_type}', expected 'raw' or 'epochs'")
