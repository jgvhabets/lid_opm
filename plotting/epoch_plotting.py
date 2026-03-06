"""
Epoch-level plotting functions for MEG + auxiliary data.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import uniform_filter1d
from mne import pick_types

from utils.load_utils import get_onedrive_path


# Frequency bands for correlation row
# replace with greek symbols for theta alpha, beta, gamma
FREQ_BANDS = {
    'θ 4-8':    (4,  8),
    'α 8-12':   (8,  12),
    'low-β 8-16':  (8,  16),
    'high-β 16-25': (16, 25),
    'γ 60-90':  (60, 90),
}

# Five qualitatively distinct band colors
BAND_COLORS = ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8', '#984ea3']

# Font sizes
FS = {'suptitle': 13, 'title': 11, 'label': 10, 'tick': 9, 'legend': 9}


def plot_tfr_with_aux(
    meg_epochs,
    aux_epochs,
    epoch_type: str,
    freqs=None,
    n_cycles_factor: float = 2.0,
    tmin_plot: float = None,
    tmax_plot: float = None,
    vmin: float = -2,
    vmax: float = 2,
    motor_ch_nrs: list = None,
    emg_smooth_ms: float = 50,
    opm_smooth_ms: float = 50,
    z_score_aux: bool = True,
    corr_window_ms: float = 50,
    plot_corr_heatmap: bool = False,
    part_figname: str = None,
    donot_show: bool = False,
):
    """
    4-row × 2-column figure for one epoch type, all time-axes aligned:

    Row 0 — TFR         : mean z-scored time-frequency power per hemisphere
    Row 1 — Evoked MEG  : mean ± SEM across channels per hemisphere
    Row 2 — AUX traces  : contralateral EMG (purple shades) + ACC (orange shades)
    Row 3 — Correlations: per-band Pearson r(t) across trials vs. mean EMG / ACC

    Columns: left hemisphere | right hemisphere.
    AUX columns show the contralateral side.

    Parameters
    ----------
    meg_epochs : mne.Epochs
        Channel names must start with 'L' or 'R'.
    aux_epochs : mne.Epochs
        EMG (type='emg') and ACC (type='misc') channels whose names contain
        'left' or 'right' (e.g. 'emg_leftforearm', 'acc_righthand_z').
    epoch_type : str
        Event label to select (e.g. 'go_left', 'go_right').
    freqs : array-like, optional
        Frequencies for TFR. Defaults to np.arange(2, 95, 1).
    n_cycles_factor : float
        n_cycles = freqs / n_cycles_factor.
    tmin_plot, tmax_plot : float, optional
        Display time limits. Defaults to full epoch range.
    vmin, vmax : float
        TFR colormap limits.
    motor_ch_nrs : list of str, optional
        Restrict MEG channels to those whose name contains one of these
        substrings (e.g. ['206', '207', '305', '306', '404', '405']).
    emg_smooth_ms : float
        Moving-average smoothing for EMG traces in ms (0 = off).
    opm_smooth_ms : float
        Moving-average smoothing for OPM traces in ms (0 = off).
    z_score_aux : bool
        Whether to z-score the auxiliary channels.
    corr_window_ms : float
        Width of the sliding window (ms) used when computing Pearson r(t).
        At each time point, data from all trials within ±window/2 are pooled
        before computing r, yielding a smoother estimate. Default 50 ms.
    part_figname : str, optional
        If given, saves to <onedrive>/figures/spectral/epoch_tfr/epochTFRbehav_<part_figname>.png
    """
    if freqs is None:
        freqs = np.arange(2, 95, 1)
    n_cycles = freqs / n_cycles_factor
    sfreq_meg = meg_epochs.info['sfreq']

    # ------------------------------------------------------------------ #
    # MEG channel selection
    # ------------------------------------------------------------------ #
    meg_ch_names = meg_epochs.ch_names
    left_meg_idx  = [i for i, ch in enumerate(meg_ch_names) if ch.startswith('L')]
    right_meg_idx = [i for i, ch in enumerate(meg_ch_names) if ch.startswith('R')]

    if motor_ch_nrs is not None:
        left_meg_idx  = [i for i in left_meg_idx  if any(nr in meg_ch_names[i] for nr in motor_ch_nrs)]
        right_meg_idx = [i for i in right_meg_idx if any(nr in meg_ch_names[i] for nr in motor_ch_nrs)]

    if not left_meg_idx or not right_meg_idx:
        raise ValueError("No MEG channels found after hemisphere/motor split. "
                         "Check channel names and motor_ch_nrs.")

    motor_label = f"motor {motor_ch_nrs}" if motor_ch_nrs else "all channels"
    left_names  = [meg_ch_names[i] for i in left_meg_idx]
    right_names = [meg_ch_names[i] for i in right_meg_idx]

    # ------------------------------------------------------------------ #
    # AUX channel split
    # ------------------------------------------------------------------ #
    emg_picks = pick_types(aux_epochs.info, emg=True)
    acc_picks = pick_types(aux_epochs.info, misc=True)
    emg_names = [aux_epochs.ch_names[i] for i in emg_picks]
    acc_names = [aux_epochs.ch_names[i] for i in acc_picks]

    right_emg = [n for n in emg_names if 'right' in n.lower()]
    right_acc = [n for n in acc_names if 'right' in n.lower()]
    left_emg  = [n for n in emg_names if 'left'  in n.lower()]
    left_acc  = [n for n in acc_names if 'left'  in n.lower()]

    sfreq_aux = aux_epochs.info['sfreq']

    # ------------------------------------------------------------------ #
    # Averaged TFR (for rows 0 display)
    # ------------------------------------------------------------------ #
    print(f"[tfr] computing averaged TFR for '{epoch_type}'...")
    tfr_avg = meg_epochs[epoch_type].copy().pick(left_names + right_names).compute_tfr(
        method="multitaper", freqs=freqs, n_cycles=n_cycles,
        average=True, return_itc=False, verbose=False,
    )
    tfr_times = tfr_avg.times

    # channel indices within the picked TFR object
    tfr_ch_names  = tfr_avg.ch_names
    left_tfr_idx  = [i for i, ch in enumerate(tfr_ch_names) if ch.startswith('L')]
    right_tfr_idx = [i for i, ch in enumerate(tfr_ch_names) if ch.startswith('R')]

    tfr_left  = _zscore_tfr(np.mean(tfr_avg.data[left_tfr_idx],  axis=0))
    tfr_right = _zscore_tfr(np.mean(tfr_avg.data[right_tfr_idx], axis=0))

    # ------------------------------------------------------------------ #
    # Per-trial TFR (for row 3 correlations)
    # ------------------------------------------------------------------ #
    print("[tfr] computing per-trial TFR for correlations...")
    tfr_trials = meg_epochs[epoch_type].copy().pick(left_names + right_names).compute_tfr(
        method="multitaper", freqs=freqs, n_cycles=n_cycles,
        average=False, return_itc=False, verbose=False,
    )
    # shape: (n_trials, n_ch, n_freqs, n_times)
    n_trials = tfr_trials.data.shape[0]

    # mean across channels per hemisphere → (n_trials, n_freqs, n_times)
    left_trials_pow  = tfr_trials.data[:, left_tfr_idx,  :, :].mean(axis=1)
    right_trials_pow = tfr_trials.data[:, right_tfr_idx, :, :].mean(axis=1)

    # ------------------------------------------------------------------ #
    # Per-trial behavioral signals → mean across contralateral channels
    # ------------------------------------------------------------------ #
    def _trial_beh(ch_names_subset, all_picks, all_names):
        """Mean across selected channels, per trial, interpolated to tfr_times."""
        idx_in_picks = [list(all_names).index(n) for n in ch_names_subset if n in all_names]
        if not idx_in_picks:
            return np.zeros((n_trials, len(tfr_times)))
        dat = aux_epochs[epoch_type].get_data(picks=list(all_picks))  # (n_tr, n_ch, n_t_aux)
        dat = dat[:, idx_in_picks, :].mean(axis=1)                    # (n_tr, n_t_aux)
        aux_t = aux_epochs[epoch_type].times
        # interpolate each trial to tfr_times
        out = np.stack([np.interp(tfr_times, aux_t, dat[k]) for k in range(n_trials)])
        return out   # (n_trials, n_times_tfr)

    right_emg_trials = _trial_beh(right_emg, emg_picks, emg_names)
    right_acc_trials = _trial_beh(right_acc, acc_picks, acc_names)
    left_emg_trials  = _trial_beh(left_emg,  emg_picks, emg_names)
    left_acc_trials  = _trial_beh(left_acc,  acc_picks, acc_names)

    # ------------------------------------------------------------------ #
    # Correlations over time: r(t) per band × behavioral signal
    # ------------------------------------------------------------------ #
    def _band_r_over_time(trials_pow, beh_trials):
        """Windowed Pearson r across trials at each time point, per freq band.
        Returns dict {band_name: r_series (n_times,)}"""
        result = {}
        for band, (fmin, fmax) in FREQ_BANDS.items():
            fmask = (freqs >= fmin) & (freqs <= fmax)
            if not fmask.any():
                result[band] = np.zeros(len(tfr_times))
                continue
            band_pow = trials_pow[:, fmask, :].mean(axis=1)  # (n_trials, n_times)
            result[band] = _pearsonr_windowed(band_pow, beh_trials,
                                              sfreq_meg, corr_window_ms)
        return result

    # square EMG/ACC before correlation — captures signal power, not signed amplitude
    corr_left_emg  = _band_r_over_time(left_trials_pow,  right_emg_trials ** 2)
    corr_left_acc  = _band_r_over_time(left_trials_pow,  right_acc_trials ** 2)
    corr_right_emg = _band_r_over_time(right_trials_pow, left_emg_trials  ** 2)
    corr_right_acc = _band_r_over_time(right_trials_pow, left_acc_trials  ** 2)
    
    print(f'max corr left opm, right emg {[[k, np.max(v)] for k, v in corr_left_emg.items()]}')
    print(f'max corr right opm, left emg {[[k, np.max(v)] for k, v in corr_right_emg.items()]}')

    # significance threshold (two-tailed p=0.05, df = n_trials-2)
    from scipy.stats import t as t_dist
    if n_trials > 2:
        t_crit = t_dist.ppf(0.975, df=n_trials - 2)
        r_crit = t_crit / np.sqrt(t_crit ** 2 + n_trials - 2)
    else:
        r_crit = 1.0

    # ------------------------------------------------------------------ #
    # Evoked MEG (row 1)
    # ------------------------------------------------------------------ #
    meg_evoked = meg_epochs[epoch_type].copy().pick(left_names + right_names).average()
    meg_times  = meg_evoked.times
    amp        = 1e12   # T → pT

    evoked_ch = meg_evoked.ch_names
    left_ev_idx  = [i for i, ch in enumerate(evoked_ch) if ch.startswith('L')]
    right_ev_idx = [i for i, ch in enumerate(evoked_ch) if ch.startswith('R')]

    left_traces  = meg_evoked.data[left_ev_idx]  * amp
    right_traces = meg_evoked.data[right_ev_idx] * amp
    # smooth the evoked traces for better visualization (not for stats)
    if opm_smooth_ms and opm_smooth_ms > 0:
        win = max(1, int(opm_smooth_ms / 1000 * sfreq_meg))
        left_traces  = uniform_filter1d(left_traces,  size=win, axis=1)
        right_traces = uniform_filter1d(right_traces, size=win, axis=1)
    lm, ls = left_traces.mean(0),  left_traces.std(0)  / max(1, np.sqrt(len(left_ev_idx)))
    rm, rs = right_traces.mean(0), right_traces.std(0) / max(1, np.sqrt(len(right_ev_idx)))

    # ------------------------------------------------------------------ #
    # Averaged AUX traces (row 2)
    # ------------------------------------------------------------------ #
    aux_all_picks = emg_picks.tolist() + acc_picks.tolist()
    aux_evoked    = aux_epochs[epoch_type].average(picks=aux_all_picks)
    if z_score_aux:
        # zscore, using the mean-std per hemibody-side for emg/acc separately
        for ch_type, ch_names in zip(['EMG l', 'EMG r', 'ACC l', 'ACC r'],
                                     [right_emg, left_emg, right_acc, left_acc]):
            idxs = [aux_evoked.ch_names.index(ch) for ch in ch_names if ch in aux_evoked.ch_names]
            if idxs:
                temp_data = aux_evoked.data[idxs]
                mean = temp_data.mean(axis=1, keepdims=True)
                std  = temp_data.std(axis=1, keepdims=True) + 1e-30
                aux_evoked.data[idxs] = (temp_data - mean) / std


    aux_times     = aux_evoked.times

    def _get_aux_traces(ch_names):
        out = {}
        for ch in ch_names:
            if ch not in aux_evoked.ch_names:
                continue
            idx   = aux_evoked.ch_names.index(ch)
            trace = aux_evoked.data[idx].copy()
            if emg_smooth_ms and emg_smooth_ms > 0 and 'emg' in ch.lower():
                win = max(1, int(emg_smooth_ms / 1000 * sfreq_aux))
                trace = uniform_filter1d(trace, size=win)
            out[ch] = trace
            # # z-score each trace for better visualization (not for correlations)
            # if z_score_aux:
            #     out[ch] = (trace - trace.mean()) / (trace.std() + 1e-30)
        return out

    right_emg_tr = _get_aux_traces(right_emg)
    right_acc_tr = _get_aux_traces(right_acc)
    left_emg_tr  = _get_aux_traces(left_emg)
    left_acc_tr  = _get_aux_traces(left_acc)

    # ------------------------------------------------------------------ #
    # Time masks
    # ------------------------------------------------------------------ #
    t0 = tmin_plot if tmin_plot is not None else tfr_times[0]
    t1 = tmax_plot if tmax_plot is not None else tfr_times[-1]

    tfr_mask = (tfr_times >= t0) & (tfr_times <= t1)
    meg_mask = (meg_times  >= t0) & (meg_times  <= t1)
    aux_mask = (aux_times  >= t0) & (aux_times  <= t1)

    # ------------------------------------------------------------------ #
    # Figure layout
    # ------------------------------------------------------------------ #
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(
        f"Epoch: {epoch_type}  |  MEG: {motor_label}  |  n={n_trials} trials",
        fontsize=FS['suptitle'], y=1.01,
    )

    gs = gridspec.GridSpec(
        4, 2, figure=fig,
        height_ratios=[1.7, 0.8, 0.9, 0.9],
        hspace=0.28, wspace=0.30,
    )

    # row 0: TFR
    ax_tfr_l = fig.add_subplot(gs[0, 0])
    ax_tfr_r = fig.add_subplot(gs[0, 1], sharey=ax_tfr_l, sharex=ax_tfr_l)
    # row 1: evoked MEG
    ax_meg_l = fig.add_subplot(gs[1, 0], sharex=ax_tfr_l)
    ax_meg_r = fig.add_subplot(gs[1, 1], sharex=ax_tfr_l, sharey=ax_meg_l)
    # row 2: AUX
    ax_aux_l = fig.add_subplot(gs[2, 0], sharex=ax_tfr_l)
    ax_aux_r = fig.add_subplot(gs[2, 1], sharex=ax_tfr_l, sharey=ax_aux_l)
    # row 3: correlations
    ax_cor_l = fig.add_subplot(gs[3, 0], sharex=ax_tfr_l)
    ax_cor_r = fig.add_subplot(gs[3, 1], sharex=ax_tfr_l, sharey=ax_cor_l)

    # 1-second major ticks on all panels (shared x-axis propagates automatically)
    ax_tfr_l.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    # "Time (s)" label only on bottom row
    for ax in [ax_tfr_l, ax_tfr_r, ax_meg_l, ax_meg_r, ax_aux_l, ax_aux_r]:
        ax.set_xlabel("")

    # ------------------------------------------------------------------ #
    # Row 0: TFR
    # ------------------------------------------------------------------ #
    _plot_tfr_panel(ax_tfr_l, tfr_left[:, tfr_mask],  tfr_times[tfr_mask], freqs,
                    title=f"Left hemisphere OPM  (n={len(left_meg_idx)} ch)",
                    vmin=vmin, vmax=vmax)
    _plot_tfr_panel(ax_tfr_r, tfr_right[:, tfr_mask], tfr_times[tfr_mask], freqs,
                    title=f"Right hemisphere OPM  (n={len(right_meg_idx)} ch)",
                    vmin=vmin, vmax=vmax)
    ax_tfr_r.set_ylabel("")

    # ------------------------------------------------------------------ #
    # Row 1: Evoked MEG
    # ------------------------------------------------------------------ #
    _plot_evoked_panel(ax_meg_l, meg_times[meg_mask], lm[meg_mask], ls[meg_mask],
                       title="Evoked — Left hemisphere OPM")
    _plot_evoked_panel(ax_meg_r, meg_times[meg_mask], rm[meg_mask], rs[meg_mask],
                       title="Evoked — Right hemisphere OPM")
    ax_meg_r.set_ylabel("")

    # ------------------------------------------------------------------ #
    # Row 2: AUX traces
    # ------------------------------------------------------------------ #
    _plot_aux_panel(ax_aux_l, right_emg_tr, right_acc_tr, aux_times[aux_mask],
                    title="Contralateral AUX — right side", z_score_aux=z_score_aux)
    _plot_aux_panel(ax_aux_r, left_emg_tr,  left_acc_tr, aux_times[aux_mask],
                    title="Contralateral AUX — left side", z_score_aux=z_score_aux)
    ax_aux_r.set_ylabel("")

    # ------------------------------------------------------------------ #
    # Row 3: Correlations over time
    # ------------------------------------------------------------------ #
    _plot_corr_panel(ax_cor_l, corr_left_emg, corr_left_acc,
                     tfr_times[tfr_mask], r_crit,
                     title="Band-power × contra-AUX  correlation — Left hemi",
                     plot_corr_heatmap=plot_corr_heatmap)
    _plot_corr_panel(ax_cor_r, corr_right_emg, corr_right_acc,
                     tfr_times[tfr_mask], r_crit,
                     title="Band-power × contra-AUX  correlation — Right hemi",
                     plot_corr_heatmap=plot_corr_heatmap)
    ax_cor_r.set_ylabel("")

    ax_cor_l.set_xlabel("Time (s)", fontsize=FS['label'])
    ax_cor_r.set_xlabel("Time (s)", fontsize=FS['label'])

    # uniform x limits
    ax_tfr_l.set_xlim(t0, t1)

    fig.tight_layout()

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    if part_figname is not None:
        figures_path = get_onedrive_path('figures')
        assert figures_path, "get_onedrive_path('figures') returned False — OneDrive path not found"
        save_dir = os.path.join(figures_path, 'spectral', 'epoch_tfr')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"epochTFRbehav_{part_figname}.png")
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[plot] saved to {save_path}")

    if donot_show: plt.close(fig)
    else: plt.show() 


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _zscore_tfr(tfr_2d):
    mean = tfr_2d.mean(axis=1, keepdims=True)
    std  = tfr_2d.std(axis=1,  keepdims=True)
    return (tfr_2d - mean) / (std + 1e-30)


def _pearsonr_windowed(X, Y, sfreq, window_ms: float = 50):
    """Pearson r computed over a sliding window across trials.

    At each time point t, data from all trials within ±(window/2) samples are
    pooled into a single vector before computing r, giving a smoother estimate.

    Parameters
    ----------
    X, Y : ndarray, shape (n_trials, n_t)
    sfreq : float  — sampling frequency of TFR times
    window_ms : float — full window width in ms (default 50)

    Returns
    -------
    r : ndarray, shape (n_t,)
    """
    n_t = X.shape[1]
    half = max(0, int(window_ms / 1000 * sfreq) // 2)
    r = np.zeros(n_t)
    for t in range(n_t):
        t0 = max(0, t - half)
        t1 = min(n_t, t + half + 1)
        x_win = X[:, t0:t1].ravel()
        y_win = Y[:, t0:t1].ravel()
        x_c = x_win - x_win.mean()
        y_c = y_win - y_win.mean()
        denom = np.sqrt((x_c ** 2).sum() * (y_c ** 2).sum()) + 1e-30
        r[t] = (x_c * y_c).sum() / denom
    return r


def _plot_tfr_panel(ax, tfr_2d, times, freqs, title, vmin, vmax):
    im = ax.pcolormesh(times, freqs, tfr_2d,
                       shading='auto', cmap='coolwarm', vmin=vmin, vmax=vmax)
    # pcolormesh with shading='auto' extends each cell by half a bin beyond the
    # data range; clamp to the actual time range so sharex aligns correctly.
    ax.set_xlim(times[0], times[-1])
    ax.axvline(0, color='k', linestyle='--', linewidth=0.9)
    ax.set_ylabel("Frequency (Hz)", fontsize=FS['label'])
    ax.set_title(title, fontsize=FS['title'])
    ax.tick_params(labelsize=FS['tick'])
    plt.colorbar(im, ax=ax, label="Power (z)", pad=0.02).ax.tick_params(labelsize=FS['tick'])


def _plot_evoked_panel(ax, times, mean, sem, title):
    ax.plot(times, mean, color='#444', lw=1.5, alpha=.7, label='μ ± sem')
    ax.fill_between(times, mean - sem, mean + sem, color='#444', alpha=0.2)
    ax.axvline(0, color='k', linestyle='--', linewidth=0.9)
    ax.axhline(0, color='gray', linestyle=':', linewidth=0.6)
    ax.set_ylabel("Amplitude (pT)", fontsize=FS['label'])
    ax.set_title(title, fontsize=FS['title'])
    ax.tick_params(labelsize=FS['tick'])
    ax.legend(fontsize=FS['legend'], loc='upper right', ncol=2,
              framealpha=0.6, handlelength=1.5)


def _plot_aux_panel(ax, emg_traces: dict, acc_traces: dict, times, title,
                    z_score_aux: bool = True,):
    """EMG: purple shades (solid). ACC: orange shades (dashed). Per-channel labels."""
    acc_ax = ax
    emg_ax = ax.twinx() if acc_traces and emg_traces else ax
    
    n_emg = max(len(emg_traces), 1)
    n_acc = max(len(acc_traces), 1)
    emg_colors = plt.colormaps['Purples'](np.linspace(0.45, 0.88, n_emg))
    acc_colors = plt.colormaps['Oranges'](np.linspace(0.45, 0.88, n_acc))

    for (name, trace), color in zip(emg_traces.items(), emg_colors):
        name = name.replace('emg_left', 'EMG L ')
        name = name.replace('emg_right', 'EMG R ')
        emg_ax.plot(times, trace[-len(times):], color=color, lw=3,
                    alpha=0.5, linestyle='-', label=name)
    for (name, trace), color in zip(acc_traces.items(), acc_colors):
        name = name.replace('acc_left', 'L ').replace('_', ' ')
        name = name.replace('acc_right', 'R ')
        acc_ax.plot(times, trace[-len(times):], color=color, lw=3,
                alpha=0.5, linestyle='-', label=name)
    # add legend labels and handles emg_ax to ax if both emg and acc are present
    if emg_traces and acc_traces:
        handles_emg, labels_emg = emg_ax.get_legend_handles_labels()
        handles_acc, labels_acc = acc_ax.get_legend_handles_labels()
        ax.legend(handles_emg + handles_acc, labels_emg + labels_acc,
                    fontsize=FS['legend'], loc='upper right', ncol=3,
                    framealpha=0.6, handlelength=1.5)
    elif emg_traces or acc_traces:
        ax.legend(fontsize=FS['legend'], loc='upper right', ncol=3,
                  framealpha=0.6, handlelength=1.5)

    ax.axvline(0, color='k', linestyle='--', linewidth=0.9)
    ax.axhline(0, color='gray', linestyle=':', linewidth=0.6)
    # get correct ylabels, z-scored or not
    if emg_traces:
        emg_ax.set_ylabel("EMG amplitude (z)" if z_score_aux else "EMG amplitude (V)", fontsize=FS['label'])
    if acc_traces:
        acc_ax.set_ylabel("ACC amplitude (z)" if z_score_aux else "ACC amplitude (g)", fontsize=FS['label'])
    if z_score_aux:
        ax.set_ylabel("Amplitude (z)", fontsize=FS['label'])
    ax.set_title(title, fontsize=FS['title'])
    ax.tick_params(labelsize=FS['tick'])
    

def _plot_corr_panel(ax, corr_emg: dict, corr_acc: dict,
                     times, r_crit: float, title: str,
                     plot_corr_heatmap: bool = False):
    """
    Per-band r(t). Two styles controlled by plot_corr_heatmap:

    False (default) — line plots, one colour per band; EMG solid, ACC dashed.
    True            — heatmap (bands × time), EMG rows on top, ACC rows below,
                      separated by a black line. Colormap: RdBu_r centred at 0.
    """
    band_names = list(FREQ_BANDS.keys())
    n_bands    = len(band_names)

    if not plot_corr_heatmap:
        # ---- line-plot style ------------------------------------------ #
        for band, color in zip(band_names, BAND_COLORS):
            r_emg = corr_emg.get(band)
            r_acc = corr_acc.get(band)
            if r_emg is not None:
                r = r_emg[-len(times):]
                ax.plot(times, r, color=color, lw=1.4,
                        linestyle='-',  alpha=0.85, label=f"{band} EMG")
                # black outline on significant segments
                ax.plot(times, np.where(np.abs(r) > r_crit, r, np.nan),
                        color='k', lw=2.5, alpha=0.5, linestyle='-', zorder=3)
            if r_acc is not None:
                r = r_acc[-len(times):]
                ax.plot(times, r, color=color, lw=1.2,
                        linestyle='--', alpha=0.75, label=f"{band} ACC")
                # black outline on significant segments
                ax.plot(times, np.where(np.abs(r) > r_crit, r, np.nan),
                        color='k', lw=2.5, alpha=0.5, linestyle='--', zorder=3)

        ax.axhline( r_crit, color='gray', linestyle=':', linewidth=0.9,
                   label=f'p=0.05 (r={r_crit:.2f})')
        ax.axhline(-r_crit, color='gray', linestyle=':', linewidth=0.9)
        ax.axhline(0, color='k', linewidth=0.7)
        ax.axvline(0, color='k', linestyle='--', linewidth=0.9)
        ax.set_ylim(-1.05, 1.25)
        ax.set_ylabel("Pearson r over time", fontsize=FS['label'])
        ax.set_title(title, fontsize=FS['title'])
        ax.tick_params(labelsize=FS['tick'])
        ax.legend(fontsize=FS['legend'] - 1, loc='upper right', ncol=3,
                  framealpha=0.6, handlelength=1.5)

    else:
        # ---- heatmap style -------------------------------------------- #
        # matrix: (2*n_bands, n_times) — EMG rows [0..n_bands), ACC rows [n_bands..2*n_bands)
        mat = np.full((2 * n_bands, len(times)), np.nan)
        for i, band in enumerate(band_names):
            if corr_emg.get(band) is not None:
                mat[i, :] = corr_emg[band][-len(times):]
            if corr_acc.get(band) is not None:
                mat[n_bands + i, :] = corr_acc[band][-len(times):]

        # y_edges has n+1 values so pcolormesh treats them as cell edges:
        # row i spans [i, i+1], centres at i+0.5 — consistent with ticks/rects/separator
        y_edges = np.arange(2 * n_bands + 1, dtype=float)
        pcolor_shading = 'flat'
        if pcolor_shading == 'flat' and len(times) == mat.shape[1]:
            print('adjust array shapes pcolormesh for shading=flat', times.shape, y_edges.shape, mat.shape)
            times = np.concatenate([times, [times[-1] + (times[-1] - times[-2])]])  # add one more time point at the end
        elif pcolor_shading == 'auto' and len(times) == mat.shape[1] + 1:
            print('adjust array shapes pcolormesh for shading=auto', times.shape, y_edges.shape, mat.shape)
            times = times[:-1]  # remove the last time point
        im = ax.pcolormesh(times, y_edges, mat, vmin=-1, vmax=1,
                           shading=pcolor_shading, cmap='RdBu_r',)
        ax.set_xlim(times[0], times[-1])
        ax.set_ylim(0, 2 * n_bands)

        # black rectangles around significant time segments within each band row
        from matplotlib.patches import Rectangle
        dt = (times[1] - times[0]) if len(times) > 1 else 0.0
        for row_idx in range(2 * n_bands):
            sig = np.abs(mat[row_idx]) > r_crit
            sig_pad = np.concatenate([[False], sig, [False]])
            starts = np.where(np.diff(sig_pad.astype(int)) ==  1)[0]
            ends   = np.where(np.diff(sig_pad.astype(int)) == -1)[0]
            for s, e in zip(starts, ends):
                x0 = times[s] - dt / 2
                x1 = times[min(e, len(times) - 1)] + dt / 2
                ax.add_patch(Rectangle(
                    (x0, row_idx), x1 - x0, 1.0,
                    linewidth=1.4, edgecolor='black', facecolor='none', zorder=5,
                ))

        # separator between EMG and ACC
        ax.axhline(n_bands, color='k', linewidth=1.5)
        ax.axvline(0, color='k', linestyle='--', linewidth=0.9)

        # y-ticks: band names, one per row
        ax.set_yticks(np.arange(2 * n_bands) + 0.5)
        ax.set_yticklabels(band_names + band_names, fontsize=FS['tick'])

        # section labels (EMG / ACC) on a twin y-axis on the right
        ax2 = ax.twinx()
        ax2.set_ylim(0, 2 * n_bands)
        ax2.set_yticks([n_bands / 2, n_bands + n_bands / 2])
        ax2.set_yticklabels(['EMG', 'ACC'], fontsize=FS['label'])
        ax2.tick_params(length=0)

        ax.set_title(title, fontsize=FS['title'])
        ax.tick_params(labelsize=FS['tick'])
        plt.colorbar(im, ax=ax2,
                     label=f"Pearson r  (crit ±{r_crit:.2f})",
                     pad=0.12).ax.tick_params(labelsize=FS['tick'])
