"""
Movement artefact cleaning functions for MEG data.
Combines steps 4a (spike removal), 4b (highpass filter),
and 4c (regression against reference signals).
Works on both mne.Raw and mne.Epochs objects.
"""

import os
import numpy as np
import mne
import matplotlib.pyplot as plt
from mne import pick_types
from mne.time_frequency import psd_array_welch
from sklearn.linear_model import Ridge, LinearRegression

from utils.load_utils import get_onedrive_path


def clean_movement_artefacts(
    meg_data,
    ref_data=None,
    apply_spikeremoval: bool = True,
    apply_highpass: bool = True,
    apply_regression: bool = True,
    n_sd_threshold: float = 4,
    highpass_freq: float = 4,
    incl_timelag_regr: bool = True,
    lag_samples: list = None,
    regression_model: str = 'ridge',
    plot: bool = False,
    part_figname: str = None,
):
    """
    Applies movement artefact cleaning in-place on mne.Raw or mne.Epochs.
    Each step can be toggled independently. meg_data and ref_data may be
    different types (Raw vs Epochs); shapes are aligned automatically.

    Steps:
    - 4a: spike removal - samples exceeding n_sd_threshold * global std
          are replaced by the local median of surrounding 5 samples.
    - 4b: highpass filter for slow movement artefacts and drift.
    - 4c: regression against EMG/ACC reference signals, optionally
          including time-lagged versions of the reference channels.

    Parameters
    ----------
    meg_data : mne.io.Raw or mne.Epochs
        MEG data to clean. Modified in-place.
    ref_data : mne.io.Raw or mne.Epochs, optional
        Reference data (EMG/ACC channels) for regression (step 4c).
        Required when apply_regression=True.
    apply_spikeremoval : bool
        Whether to run spike removal (step 4a).
    apply_highpass : bool
        Whether to run highpass filtering (step 4b).
    apply_regression : bool
        Whether to run regression-based cleaning (step 4c).
    n_sd_threshold : float
        Spike threshold in number of standard deviations (step 4a).
    highpass_freq : float
        Highpass cutoff in Hz for the movement filter (step 4b).
    incl_timelag_regr : bool
        Whether to append time-lagged copies of reference signals as
        additional regressors (step 4c).
    lag_samples : list of int
        Sample lags to include when incl_timelag_regr=True.
        Defaults to [3, 6, 8, 10] (~6-20 ms at 512 Hz).
    regression_model : str
        Regression model for step 4c: 'ridge' (default) or 'linear'.
    plot : bool
        If True, plot temporal and spectral snapshots after each step
        in a single figure.
    part_figname : str, optional
        If provided (and plot=True), save the figure as
        <onedrive>/figures/preprocessing/move_artef_cleaning/autoMoveCleaning_<part_figname>.png
    """
    if lag_samples is None:
        lag_samples = [3, 6, 8, 10]

    is_epochs_meg = isinstance(meg_data, mne.Epochs)
    meg_picks = pick_types(meg_data.info, meg=True, exclude='bads')
    sfreq = meg_data.info['sfreq']

    print(
        f"[init] meg_data type: {type(meg_data).__name__}, "
        f"n_meg_picks: {len(meg_picks)}, sfreq: {sfreq}, "
        f"steps: spikeremoval={apply_spikeremoval}, "
        f"highpass={apply_highpass}, regression={apply_regression}"
    )

    snapshots = []   # list of (label, 2D ndarray) captured after each step

    if plot:
        print("[init] capturing pre-cleaning snapshot...")
        snapshots.append(("Before", _get_2d_snapshot(meg_data, meg_picks, is_epochs_meg)))
        print("[init] snapshot done")

    # ------------------------------------------------------------------ #
    # 4a: spike removal via median replacement
    # ------------------------------------------------------------------ #
    if apply_spikeremoval:
        print("[4a] loading MEG data...")
        tempdat = meg_data.get_data(picks=meg_picks)
        print(f"[4a] data loaded, shape: {tempdat.shape} — computing threshold...")
        thresh = n_sd_threshold * np.std(tempdat)
        spike_mask = np.abs(tempdat) > thresh
        print(
            f"[4a] spike samples: {spike_mask.sum()} "
            f"({round(spike_mask.sum() / spike_mask.size * 100, 3)} %)"
        )

        temp_clean = tempdat.copy()

        if not is_epochs_meg:                      # 2D: (n_ch, n_times)
            for ch in range(tempdat.shape[0]):
                for i in np.where(spike_mask[ch])[0]:
                    if 1 < i < tempdat.shape[1] - 2:
                        temp_clean[ch, i] = np.median(tempdat[ch, i - 2:i + 3])
            meg_data._data[meg_picks, :] = temp_clean

        else:                                      # 3D: (n_ep, n_ch, n_times)
            for ep in range(tempdat.shape[0]):
                for ch in range(tempdat.shape[1]):
                    for i in np.where(spike_mask[ep, ch])[0]:
                        if 1 < i < tempdat.shape[-1] - 2:
                            temp_clean[ep, ch, i] = np.median(tempdat[ep, ch, i - 2:i + 3])
            meg_data._data[:, meg_picks, :] = temp_clean

        if plot:
            snapshots.append(("After spike\nremoval (4a)", _get_2d_snapshot(meg_data, meg_picks, is_epochs_meg)))

    # ------------------------------------------------------------------ #
    # 4b: highpass filter for slow movement artefacts
    # ------------------------------------------------------------------ #
    if apply_highpass:
        print(f"[4b] highpass filter at {highpass_freq} Hz")
        meg_data.filter(l_freq=highpass_freq, h_freq=None, method='fir', verbose=False)

        if plot:
            snapshots.append((f"After highpass\n{highpass_freq} Hz (4b)", _get_2d_snapshot(meg_data, meg_picks, is_epochs_meg)))

    # ------------------------------------------------------------------ #
    # 4c: regression against reference channels
    # ------------------------------------------------------------------ #
    if apply_regression:
        if ref_data is None:
            raise ValueError("ref_data must be provided when apply_regression=True")

        if regression_model == 'ridge':
            model = Ridge()
        elif regression_model == 'linear':
            model = LinearRegression()
        else:
            raise ValueError(
                f"regression_model must be 'ridge' or 'linear', got '{regression_model}'"
            )

        is_epochs_ref = isinstance(ref_data, mne.Epochs)
        ref_picks = pick_types(ref_data.info, emg=True, misc=True)

        X_meg = meg_data.get_data(picks=meg_picks)   # (n_ch, n_t) or (n_ep, n_ch, n_t)
        X_ref = ref_data.get_data(picks=ref_picks)   # (n_ch, n_t) or (n_ep, n_ch, n_t)

        # flatten epochs independently to 2D: (n_ch, n_times)
        if is_epochs_meg:
            n_epochs_meg, n_meg_ch, n_times_meg = X_meg.shape
            print(f"[4c] meg epochs shape before flatten: {X_meg.shape}")
            X_meg = X_meg.transpose(1, 0, 2).reshape(n_meg_ch, -1)

        if is_epochs_ref:
            print(f"[4c] ref epochs shape before flatten: {X_ref.shape}")
            n_ref_ch = X_ref.shape[1]
            X_ref = X_ref.transpose(1, 0, 2).reshape(n_ref_ch, -1)

        # align time dimension
        if X_meg.shape[1] != X_ref.shape[1]:
            min_len = min(X_meg.shape[1], X_ref.shape[1])
            X_meg = X_meg[:, :min_len]
            X_ref = X_ref[:, :min_len]
            print(f"[4c] time lengths aligned to {min_len} samples")

        print(f"[4c] regression ({regression_model}): meg {X_meg.shape}, ref {X_ref.shape}")

        X_ref = X_ref.T    # (n_times, n_ref) for sklearn

        if incl_timelag_regr:
            # append time-lagged versions of reference signals as additional regressors
            # aims to remove slower emg-associated movement artefacts that may not be perfectly time-aligned with meg signals
            n_ref = X_ref.shape[1]
            X_ref_lag = np.hstack([np.roll(X_ref, lag, axis=0) for lag in lag_samples])
            for j, lag in enumerate(lag_samples):
                X_ref_lag[:lag, j * n_ref:(j + 1) * n_ref] = 0.0
            X_ref = np.hstack([X_ref, X_ref_lag])

        X_meg_clean = X_meg.copy()
        for i in range(len(meg_picks)):
            y = X_meg[i, :]
            model.fit(X_ref, y)
            X_meg_clean[i, :] = y - model.predict(X_ref)

        # write cleaned data back, reshaping to 3D if meg_data is Epochs
        if is_epochs_meg:
            X_meg_clean_3d = (
                X_meg_clean[:, :n_epochs_meg * n_times_meg]
                .reshape(n_meg_ch, n_epochs_meg, n_times_meg)
                .transpose(1, 0, 2)
            )
            meg_data._data[:, meg_picks, :] = X_meg_clean_3d
        else:
            meg_data._data[meg_picks, :X_meg_clean.shape[1]] = X_meg_clean

        print("[4c] regression complete")

        if plot:
            snapshots.append((f"After regression\n({regression_model}) (4c)", _get_2d_snapshot(meg_data, meg_picks, is_epochs_meg)))

    if plot and len(snapshots) > 1:
        _plot_cleaning_steps(snapshots, sfreq, part_figname=part_figname,)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _get_2d_snapshot(meg_data, meg_picks, is_epochs):
    """Returns MEG data as 2D array (n_ch, n_times), flattening epochs if needed."""
    dat = meg_data.get_data(picks=meg_picks)
    if is_epochs:
        n_ep, n_ch, n_t = dat.shape
        dat = dat.transpose(1, 0, 2).reshape(n_ch, -1)
    return dat


def _plot_cleaning_steps(snapshots, sfreq, part_figname=None, n_ch_plot=10):
    """
    Plots temporal and spectral state of MEG data after each cleaning step.
    One column per snapshot, two rows: time series (top) and mean PSD (bottom).

    Parameters
    ----------
    snapshots : list of (label: str, data: ndarray shape (n_ch, n_times))
    sfreq : float
    part_figname : str, optional
        If given, save to <onedrive>/figures/preprocessing/move_artef_cleaning/autoMoveCleaning_<part_figname>.png
    n_ch_plot : int
        Number of channels shown in the time-series row.
    """
    amp_scale = 1e12          # T -> pT
    win = int(sfreq) * 4
    step = max(1, int(sfreq) // 50)
    n_cols = len(snapshots)

    # evenly spaced channel subset (based on first snapshot)
    n_ch = snapshots[0][1].shape[0]
    ch_idx = np.linspace(0, n_ch - 1, min(n_ch_plot, n_ch), dtype=int)

    # reference offset from first snapshot so y-scale is consistent
    ref_traces = snapshots[0][1][ch_idx, ::step] * amp_scale
    offset = np.arange(len(ch_idx)) * np.std(ref_traces) * 3

    fig, axes = plt.subplots(2, n_cols, figsize=(4.5 * n_cols, 7), sharey='row')
    if n_cols == 1:
        axes = axes[:, np.newaxis]   # keep 2D indexing

    fig.suptitle("Movement cleaning steps", fontsize=13)

    for col, (label, data) in enumerate(snapshots):

        # --- time series ---
        ax_t = axes[0, col]
        traces = data[ch_idx, ::step] * amp_scale
        t_axis = np.arange(data.shape[1])[::step] / sfreq
        for k, tr in enumerate(traces):
            ax_t.plot(t_axis, tr - np.mean(tr) + offset[k], lw=0.5, color='steelblue', alpha=0.7)
        ax_t.set_title(label, fontsize=10)
        ax_t.set_xlabel("Time (s)")
        if col == 0:
            ax_t.set_ylabel("Amplitude (pT)")

        # --- PSD ---
        ax_p = axes[1, col]
        psd, freqs = psd_array_welch(
            data, sfreq=sfreq, fmin=0.5, fmax=min(sfreq / 2, 150),
            n_fft=win, n_overlap=win // 2, n_per_seg=win, verbose=False,
        )
        psd_db = 10 * np.log10(np.mean(psd, axis=0) * 1e30 + 1e-30)
        ax_p.plot(freqs, psd_db, color='steelblue')
        ax_p.set_xlabel("Frequency (Hz)")
        ax_p.set_xlim(0, min(sfreq / 2, 150))
        if col == 0:
            ax_p.set_ylabel("Power (dB) [fT²/Hz]")

    plt.tight_layout()

    if part_figname is not None:
        save_dir = os.path.join(
            get_onedrive_path('figures'), 'preprocessing', 'move_artef_cleaning'
        )
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"autoMoveCleaning_{part_figname}.png")
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[plot] saved to {save_path}")

    plt.show()
