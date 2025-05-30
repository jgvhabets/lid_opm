import mne
import numpy as np

def apply_ssp_from_baseline(raw_baseline, raw_meg, n_proj=2):
    """
    Compute SSP projectors from a baseline MEG Raw object and apply them to a MEG data Raw object.

    Args:
        raw_baseline: MNE Raw object (baseline/empty room)
        raw_meg: MNE Raw object (data to denoise)
        n_proj: Number of projectors to use

    Returns:
        raw_denoised: MNE Raw object with SSP projectors applied
    """
    # Get data matrix (channels x times)
    data = raw_baseline.get_data()
    U, S, Vt = np.linalg.svd(data, full_matrices=False)
    # Stack the first n_proj vectors as columns
    proj_data = U[:, :n_proj]
    proj = mne.Projection(
        active=True,
        data=dict(
            col_names=raw_baseline.ch_names,
            row_names=None,
            data=proj_data,
            nrow=proj_data.shape[0],
            ncol=proj_data.shape[1]
        ),
        desc=f'SSP (first {n_proj} components)',
        kind=1,  # 1 = EEG/MEG
    )
    # Add the single Projection object (with n_proj vectors)
    raw_meg.add_proj([proj])
    raw_denoised = raw_meg.copy().apply_proj()
    return raw_denoised