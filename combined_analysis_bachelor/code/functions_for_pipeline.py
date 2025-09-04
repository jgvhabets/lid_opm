import numpy as np
import pandas as pd
import mne
from scipy.signal import butter, sosfiltfilt, savgol_filter, welch
from scipy.ndimage import label
import json
import matplotlib
from matplotlib.widgets import Slider
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import combined_analysis_bachelor.code.movement_detection_functions as move_funcs
from combined_analysis_bachelor.code.read_in_emg_acc import read_in_h5

def create_df(data, column_names, times):
    df = pd.DataFrame(data.T, columns=column_names)
    df["Time (s)"] = times
    return df

def notched(df, all_columns, times):
    notched_df = df.copy()
    for col in all_columns:
        col_data = df[col].to_numpy()
        notched_col = mne.filter.notch_filter(col_data, freqs=[50, 100, 150], Fs=1000)
        notched_df[col] = notched_col
    notched_df["Time (s)"] = times
    return notched_df


def filtered_and_notched(raw_df, acc_columns, emg_columns, acc_filter, emg_filter, sf):
    filtered_df = raw_df.copy()
    for col in raw_df.columns:
        col_data = raw_df[col].to_numpy()

        if col in acc_columns:
            col_filtered = mne.filter.filter_data(col_data, sfreq=sf, l_freq=acc_filter[0], h_freq=acc_filter[1])
            notched_col = mne.filter.notch_filter(col_filtered, freqs=[50, 100, 150], Fs=sf)
            filtered_df[col] = notched_col
        elif col in emg_columns:
            col_filtered = mne.filter.filter_data(col_data, sfreq=sf, l_freq=emg_filter[0], h_freq=emg_filter[1])
            notched_col = mne.filter.notch_filter(col_filtered, freqs=[50, 100, 150], Fs=sf)
            filtered_df[col] = notched_col

        else:
            pass

    return filtered_df

def plot_channel_overview(channels, df, title_name, location, emg):
    fig, axs = plt.subplots(4, 3, figsize=(12, 8))
    axs = axs.ravel()
    for i, channel in enumerate(channels):
        axs[i].plot(df["Time (s)"], df[channel])
        axs[i].set_title(f"{channel} : signal of {location[channel]}")
        axs[i].set_xlabel("Time (s)")
        axs[i].set_ylabel("Amplitude (V)" if channel in emg else "g")
        plt.suptitle(f"plots for {title_name}")
        #plt.ylim()
        if i > len(channels): #abändern, dass man nur so viele ax hat wie auch channel?
            axs.axis("off")

    plt.tight_layout()
    plt.show()

def rectify(df, emg_columns):
    if not all(col in df.columns for col in emg_columns):
        print("Warning: Some EMG channels are not in your custom_order")

    df[emg_columns] = df[emg_columns].abs()
    print("Min value after rectification:", df[emg_columns].min().min())
    return df


def envelope(df, emg_columns, freq):
    emg_envelope_df = df.copy()
    low_pass = freq/(1000/2)
    sos = butter(4, low_pass, btype='lowpass', output="sos")
    for col in emg_columns:
        col_data = df[col].to_numpy()
        enveloped = sosfiltfilt(sos, x=col_data)
        emg_envelope_df[col] = enveloped

    return emg_envelope_df


def normalize_emg(df, all_columns, emg_columns, times):
    normalized_emg_df = df.copy()
    for col in all_columns:
        if col in emg_columns:
            col_data = df[col].to_numpy()
            normalized_emg = col_data / np.max(np.abs(col_data))
            normalized_emg_df[col] = normalized_emg
            print(f"Normalized to range: [{np.min(normalized_emg): .2f}, {np.max(normalized_emg): .2f}]")
        else:
            continue
    normalized_emg_df["Time (s)"] = times
    return normalized_emg_df

def add_tkeo_add_envelope(df, emg_columns, window_size_tkeo, envelope_freq, sf, smoothing_method:str="rms"):
    added_to_df = df.copy()
    for col in emg_columns:
        col_data = df[col].to_numpy()
        tkeo_col = col_data[1:-1] ** 2 - col_data[0:-2] * col_data[2:]
        # tkeo_padded = np.pad(tkeo_col, (1, 1), 'constant', constant_values=np.nan)
        tkeo_padded = np.pad(tkeo_col, (1, 1), 'edge')
        tkeo_rectified = abs(tkeo_padded)
        if smoothing_method.lower() == "rms":
            tkeo_smoothed = np.sqrt(np.convolve(tkeo_rectified**2, np.ones(window_size_tkeo)/window_size_tkeo, mode='same'))
            added_to_df[f"{col}_tkeo_rms"] = tkeo_smoothed
        elif smoothing_method.lower() == "low-pass":
            low_pass = 50 / (1000 / 2)
            sos = butter(4, low_pass, btype='lowpass', output="sos")
            tkeo_smoothed = sosfiltfilt(sos, x=tkeo_rectified)
            added_to_df[f"{col}_tkeo_lowPass"] = tkeo_smoothed
        else:
            raise ValueError("smooting method not available, only: rms, low-pass")

        rectified_data = abs(col_data)
        low_pass = envelope_freq / (sf / 2)
        sos = butter(4, low_pass, btype='lowpass', output="sos")
        enveloped = sosfiltfilt(sos, x=rectified_data)
        added_to_df[f"{col}_envelope"] = enveloped

    return added_to_df

def add_tkeo(df, emg_columns, window_size):
    added_to_df = df.copy()
    for col in emg_columns:
        if col not in df.columns:
            continue
        col_data = df[col].to_numpy()
        if len(col_data) < 3:
            raise ValueError(f"col {col} does not have enough samples for TKEO")
        tkeo_col = col_data[1:-1] ** 2 - col_data[0:-2] * col_data[2:]
        tkeo_padded = np.pad(tkeo_col, (1, 1), 'edge')
        tkeo_rectified = np.abs(tkeo_padded)
        kernel = np.ones(window_size) / max(1, window_size)
        tkeo_smoothed = np.sqrt(np.convolve(tkeo_rectified ** 2, kernel, mode='same'))
        added_to_df[f"{col}_tkeo"] = tkeo_smoothed
    return added_to_df


def root_mean_square(df, chosen_columns, window_size=50):
    rms_df = df.copy()
    for col in chosen_columns:
        col_data = df[col].to_numpy()
        rms = np.sqrt(np.convolve(col_data**2, np.ones(window_size)/window_size, mode='same'))
        rms_df[f"{col}_rms"] = rms
    return rms_df


def movement_detection(data, threshold):
    over_thresh = data >= threshold
    return over_thresh

def get_ch_indices(all_columns, emg, acc):
    emg_cols = [all_columns.index(ch) for ch in emg if ch in all_columns]
    acc_cols = [all_columns.index(ch) for ch in acc if ch in all_columns]
    return emg_cols, acc_cols


def plot_overview(limb_and_side, df, task_name, emg_columns, directory_for_image, baseline_parts, sf):
    sync_times = df["Sync_Time (s)"]

    if "arm" in limb_and_side.lower():
        fig, axs = plt.subplots(3, 1, figsize=(12, 8))
        axs = axs.ravel()

        channels = []
        if "l" in limb_and_side.lower():
            channels = ["brachioradialis_L", "deltoideus_L", "SVM_L"]
        elif "r" in limb_and_side.lower():
            channels = ["brachioradialis_R", "deltoideus_R", "SVM_R"]

        for i, channel in enumerate(channels):
            axs[i].plot(sync_times, df[channel])
            axs[i].set_title(f"signal of {channel}")
            axs[i].set_xlabel("Sync_Time (s)")
            axs[i].set_ylabel("Amplitude (V)" if channel in emg_columns else "Acceleration (g)")

            if "l" in limb_and_side.lower():
                axs[i].set_xlim(0, 60)
            else:
                axs[i].set_xlim(100,160)

        plt.suptitle(f"{limb_and_side.upper()} in task: {task_name}")
        plt.tight_layout()
        plt.savefig(f"{directory_for_image}/plots_limb_overview/{limb_and_side.upper()}_{task_name}.png")

    elif "leg" in limb_and_side.lower():
        fig, axs = plt.subplots(2, 1, figsize=(12, 8))
        axs = axs.ravel()

        if "l" in limb_and_side.lower():
            channels = ["tibialisAnterior_L", "SVM_R"] # ACHTUNG hier noch wegen setupB: ACC auf linkem Bein war ACC
        elif "r" in limb_and_side.lower():                # vom rechten Arm vorher --> abändern wenn 4 ACCs da sind!!
            channels = ["tibialisAnterior_R", "SVM_R"]

        for i, channel in enumerate(channels):
            axs[i].plot(df["Sync_Time (s)"], df[channel])
            axs[i].set_title(f"signal of {channel}")
            axs[i].set_xlabel("Sync_Time (s)")
            axs[i].set_ylabel("Amplitude (V)" if channel in emg_columns else "Acceleration (g)")
            axs[i].set_xlim(0, 60)

        plt.suptitle(f"{limb_and_side.upper()} in task: {task_name}")
        plt.tight_layout()
        plt.savefig(f"{directory_for_image}/plots_limb_overview/{limb_and_side.upper()}_{task_name}.png")



def plot_overview_threshs(limb_and_side, df, task_name, emg_columns, directory_for_image, baseline_parts, sf):
    sync_times = df["Sync_Time (s)"]
    baseline_start = baseline_parts[f"{limb_and_side}_{task_name[:-10]}"][0]
    baseline_end = baseline_parts[f"{limb_and_side}_{task_name[:-10]}"][1]
    baseline_part = df.iloc[int(baseline_start * sf):int(baseline_end * sf)]  # wichtig: iloc statt []

    # Kanalwahl je nach Limb & Seite
    if "arm" in limb_and_side.lower():
        fig, axs = plt.subplots(3, 1, figsize=(12, 8))
        axs = axs.ravel()

        channels = []
        if "l" in limb_and_side.lower():
            channels = ["brachioradialis_L", "deltoideus_L", "SVM_L"]
        elif "r" in limb_and_side.lower():
            channels = ["brachioradialis_R", "deltoideus_R", "SVM_R"]

    elif "leg" in limb_and_side.lower():
        fig, axs = plt.subplots(2, 1, figsize=(12, 8))
        axs = axs.ravel()

        if "l" in limb_and_side.lower():
            channels = ["tibialisAnterior_L", "SVM_R"]
        elif "r" in limb_and_side.lower():
            channels = ["tibialisAnterior_R", "SVM_R"]

    # --- Plotting Loop ---
    for i, channel in enumerate(channels):
        signal = df[channel]
        axs[i].plot(sync_times, signal, label='Signal')
        axs[i].set_title(f"Signal of {channel}")
        axs[i].set_xlabel("Sync_Time (s)")

        # Baseline-Berechnung
        baseline_signal = baseline_part[channel]
        mu = baseline_signal.mean()
        sigma = baseline_signal.std()

        # Schwellenlinien
        for k in range(3, 16):
            threshold = mu + k * sigma
            axs[i].axhline(y=threshold, linestyle='--', linewidth=0.8, label=f"μ+{k}σ" if k in [3, 6, 9, 12, 15] else None)  # nicht alles labeln

        axs[i].legend(loc='upper right', fontsize='x-small', ncol=2)
        axs[i].set_ylabel("")  # keine Einheit, nur die Schwellen sichtbar

        # X-Lim spezifisch pro Seite
        #if "arm" in limb_and_side.lower():
         #   axs[i].set_xlim(0, 60) if "l" in limb_and_side.lower() else axs[i].set_xlim(100, 160)
        #else:
         #   axs[i].set_xlim(0, 60)

    plt.suptitle(f"{limb_and_side.upper()} in task: {task_name}")
    plt.tight_layout()
    plt.savefig(f"{directory_for_image}/plots_check_threshold/{limb_and_side.upper()}_{task_name}.png")
    plt.close()


#----------------------------------

def find_initial_k(signal, mu, sigma, sf, k_range=range(2, 20)):
    for k in k_range:
        threshold = mu + k * sigma
        labels_array, n_lbl = label(signal > threshold)
        for lbl in range(1, n_lbl + 1):
            idx = np.where(labels_array == lbl)[0]
            if idx.size >= int(0.5* sf): # ACHRUNG war vorher 1!!
                return k
    return k_range[-1]

def take_out_short_off_onset(onsets, offsets, min_time_period, sampling_freq):
    onsets_clean = onsets.copy()
    offsets_clean = offsets.copy()
    min_sample_period = min_time_period * sampling_freq

    if hasattr(onsets, 'tolist'):
        onsets_clean = onsets.tolist()
        offsets_clean = offsets.tolist()

    for i in range(len(offsets) - 2, -1, -1):
        time_between = onsets[i + 1] - offsets[i]
        if time_between <= min_sample_period:
            del onsets_clean[i + 1]
            del offsets_clean[i]

    return onsets_clean, offsets_clean


def interactive_multichannel_plot(limb_and_side, df, task_name, emg_columns, baseline_parts, sf):
    sync_times = df["Sync_Time (s)"]
    baseline_start = baseline_parts[f"{limb_and_side}_{task_name[:-10]}"][0]
    baseline_end = baseline_parts[f"{limb_and_side}_{task_name[:-10]}"][1]
    baseline_part = df.iloc[int(baseline_start * sf):int(baseline_end * sf)]

    if "arm" in limb_and_side.lower():
        channels = ["brachioradialis_L", "deltoideus_L", "SVM_L"] if "l" in limb_and_side.lower() \
            else ["brachioradialis_R", "deltoideus_R", "SVM_R"]
    elif "leg" in limb_and_side.lower():
        channels = ["tibialisAnterior_L", "SVM_R"] if "l" in limb_and_side.lower() \
            else ["tibialisAnterior_R", "SVM_R"]
    else:
        print("Unbekannte limb_and_side:", limb_and_side)
        return

    n_channels = len(channels)
    fig_height = 3.5 * n_channels
    fig, axs = plt.subplots(n_channels, 1, figsize=(12, fig_height))
    axs = np.atleast_1d(axs)
    plt.subplots_adjust(bottom=0.15 + 0.10 * n_channels)

    threshold_lines = []
    onset_lines = [[] for _ in channels]
    offset_lines = [[] for _ in channels]
    mu_sigma = []
    signals = []

    k_sliders = []
    dur_sliders = []
    pause_sliders = []

    for i, channel in enumerate(channels):
        if channel not in df.columns:
            axs[i].text(0.5, 0.5, f"{channel} not found", ha='center')
            continue

        signal = df[channel].values
        signals.append(signal)
        baseline_signal = baseline_part[channel]
        mu = baseline_signal.mean()
        sigma = baseline_signal.std()
        mu_sigma.append((mu, sigma))

        #initial_k = find_initial_k(signal, mu, sigma, sf)

        axs[i].plot(sync_times, signal, label=channel)
        axs[i].set_title(f"{channel}")
        axs[i].legend(loc='upper right')
        threshold_line, = axs[i].plot([], [], 'r--', label='Threshold')
        threshold_lines.append(threshold_line)

        # k-Slider
        ax_slider_k = plt.axes([0.10, 0.05 + i * 0.09, 0.25, 0.03])
        slider_k = Slider(ax_slider_k, f'{channel} k', 5, 75, valinit=5, valstep=0.5)
        k_sliders.append(slider_k)

        # Dauer-Slider
        ax_slider_dur = plt.axes([0.40, 0.05 + i * 0.09, 0.2, 0.03])
        slider_dur = Slider(ax_slider_dur, f'Dauer (s)', 0, 3.0, valinit=0, valstep=0.1)
        dur_sliders.append(slider_dur)

        # Pause-Slider
        ax_slider_pause = plt.axes([0.65, 0.05 + i * 0.09, 0.2, 0.03])
        slider_pause = Slider(ax_slider_pause, f'Pause (s)', 0, 3.0, valinit=0, valstep=0.05)
        pause_sliders.append(slider_pause)

    def update(val):
        for i, channel in enumerate(channels):
            if channel not in df.columns:
                continue

            signal = signals[i]
            mu, sigma = mu_sigma[i]
            print(f"for {channel} in {task_name} = baseline-part mean: {mu} and baseline-part sigma: {sigma}")
            k = k_sliders[i].val
            min_duration_s = dur_sliders[i].val
            min_pause_s = pause_sliders[i].val

            threshold = mu + k * sigma
            threshold_lines[i].set_data(sync_times, np.full_like(sync_times, threshold))

            labels_array, n_lbl = label(signal > threshold)
            valid = np.zeros_like(signal, dtype=bool)
            for lbl in range(1, n_lbl + 1):
                idx = np.where(labels_array == lbl)[0]
                if idx.size >= int(min_duration_s * sf):
                    valid[idx] = True

            onsets = np.where(np.diff(valid.astype(int)) == 1)[0] + 1
            offsets = np.where(np.diff(valid.astype(int)) == -1)[0] + 1

            onsets, offsets = take_out_short_off_onset(onsets, offsets, min_pause_s, sf)

            for l in onset_lines[i] + offset_lines[i]:
                l.remove()
            onset_lines[i].clear()
            offset_lines[i].clear()

            for x in onsets:
                l = axs[i].axvline(sync_times[x], ls="--", c="g")
                onset_lines[i].append(l)
            for x in offsets:
                l = axs[i].axvline(sync_times[x], ls="--", c="k")
                offset_lines[i].append(l)

        fig.canvas.draw_idle()

    for s in k_sliders + dur_sliders + pause_sliders:
        s.on_changed(update)

    update(None)
    plt.suptitle(f"{limb_and_side.upper()} – Task: {task_name}", fontsize=14)
    plt.show()


from matplotlib import gridspec
def simple_interactive_multichannel_plot(df, channels, baseline_parts, baseline_key, sf,
                                         save_OnOffsets=False, json_path=None, give_out_dict=False):
    """
    Interaktive Plots with sliders under the plots.
    with option to save plots or not

    df: input df
    channels: list of channels that should be plotted
    baseline_parts: dict -> {baseline_key: (start_s, end_s)}
    baseline_key: Key for baseline_parts
    sf: sampling frequency
    """
    sync_times = df["Sync_Time (s)"].to_numpy()
    baseline_start, baseline_end = baseline_parts[baseline_key]
    baseline_part = df.iloc[int(baseline_start * sf):int(baseline_end * sf)]

    results_seconds = {ch: [] for ch in channels}
    results_samples = {ch: [] for ch in channels}

    n = len(channels)
    fig = plt.figure(figsize=(14, 3.8 * n))
    gs = gridspec.GridSpec(2 * n, 1, height_ratios=[4, 1] * n, hspace=0.35)

    # Speicher
    ax_sig = []
    threshold_lines = []
    onset_lines = [[] for _ in channels]
    offset_lines = [[] for _ in channels]
    mu_sigma = []
    signals = []
    k_sliders, dur_sliders, pause_sliders = [], [], []

    # Helper: finding initial k
    def find_initial_k(signal, mu, sigma, sf):
        # simpler default: 20
        return 3

    for i, ch in enumerate(channels):
        ax = fig.add_subplot(gs[2 * i, 0], sharex=ax_sig[0] if i > 0 else None)
        ax_sig.append(ax)

        if ch not in df.columns:
            ax.text(0.5, 0.5, f"{ch} not found", ha="center", va="center", transform=ax.transAxes)
            mu_sigma.append((None, None))
            signals.append(None)
            threshold_lines.append(None)
        else:
            sig = df[ch].to_numpy()
            signals.append(sig)
            mu = baseline_part[ch].mean()
            sigma = baseline_part[ch].std()
            mu_sigma.append((mu, sigma))

            ax.plot(sync_times, sig, label=ch)
            ax.set_title(ch)
            ax.legend(loc="upper right")
            (thr_line,) = ax.plot([], [], "r--", label="Threshold")
            threshold_lines.append(thr_line)

        # Slider-Zeile unter dem Plot
        ax_ctrl = fig.add_subplot(gs[2 * i + 1, 0])
        ax_ctrl.set_axis_off()

        # Drei Slider-Achsen nebeneinander
        lefts = [0.08, 0.42, 0.72]
        widths = [0.28, 0.22, 0.22]
        # Slider-Achsen sind Unterachsen der ctrl-Achse
        ax_k = ax_ctrl.inset_axes([lefts[0], 0.10, widths[0], 0.8])
        ax_dur = ax_ctrl.inset_axes([lefts[1], 0.10, widths[1], 0.8])
        ax_pause = ax_ctrl.inset_axes([lefts[2], 0.10, widths[2], 0.8])

        init_k = 20 if mu_sigma[-1][0] is None else find_initial_k(signals[-1], *mu_sigma[-1], sf)
        s_k = Slider(ax_k, f'{ch} k', 1, 20, valinit=init_k, valstep=0.5)
        s_d = Slider(ax_dur, 'Dauer (s)', 0, 3.0, valinit=0, valstep=0.05)
        s_p = Slider(ax_pause, 'Pause (s)', 0, 3.0, valinit=0, valstep=0.05)
        k_sliders.append(s_k);
        dur_sliders.append(s_d);
        pause_sliders.append(s_p)

    def update(_):
        for i, ch in enumerate(channels):
            if ch not in df.columns:
                # Falls Kanal fehlt: Ergebnis leer halten
                results_seconds[ch] = []
                results_samples[ch] = []
                continue

            sig = signals[i]
            mu, sigma = mu_sigma[i]
            k = k_sliders[i].val
            min_duration_s = dur_sliders[i].val
            min_pause_s = pause_sliders[i].val

            thr = mu + k * sigma
            threshold_lines[i].set_data(sync_times, np.full_like(sync_times, thr))

            # Threshold-Maske
            mask = sig > thr

            # Erste Runs aus der *rohen* Maske
            labels_array, n_lbl = label(mask)
            runs = []
            for lbl_id in range(1, n_lbl + 1):
                idx = np.where(labels_array == lbl_id)[0]
                runs.append((idx[0], idx[-1] + 1))  # [start, end) in Samples

            # Kurze Lücken MERGEN (min_pause) – ohne Edges zu verändern
            min_gap = int(round(min_pause_s * sf))
            if runs:
                merged = [runs[0]]
                for s, e in runs[1:]:
                    prev_s, prev_e = merged[-1]
                    gap = s - prev_e
                    if gap <= min_gap:
                        # zusammenlegen
                        merged[-1] = (prev_s, e)
                    else:
                        merged.append((s, e))
            else:
                merged = []

            # Zu kurze Runs (nach dem Mergen) komplett verwerfen
            min_len = int(round(min_duration_s * sf))
            kept = [(s, e) for (s, e) in merged if (e - s) >= min_len]

            # Finale On-/Offsets aus den behaltenen Runs
            onsets = np.array([s for (s, e) in kept], dtype=int)
            offsets = np.array([e for (s, e) in kept], dtype=int)

            # Plot-Markierungen aktualisieren
            for l in onset_lines[i] + offset_lines[i]:
                try:
                    l.remove()
                except Exception:
                    pass
            onset_lines[i].clear();
            offset_lines[i].clear()

            for x in onsets:
                onset_lines[i].append(ax_sig[i].axvline(sync_times[x], ls="--", c="g"))
            for x in offsets:
                offset_lines[i].append(ax_sig[i].axvline(sync_times[x], ls="--", c="k"))

            # Ergebnisse (Sekunden-Paare) merken – letzter Slider-Stand
            on_sec = [float(sync_times[x]) for x in onsets]
            off_sec = [float(sync_times[x]) for x in offsets]
            pairs_seconds = move_funcs.new_on_offsets(on_sec, off_sec, end_index=float(sync_times[-1]))  # oder move_funcs.new_on_offsets(...)
            results_seconds[ch] = pairs_seconds
            on_sample = [int(x) for x in onsets]
            off_sample = [int(x) for x in offsets]
            pairs_samples = move_funcs.new_on_offsets(on_sample, off_sample, end_index=len(sig) - 1)
            results_samples[ch] = pairs_samples

        fig.canvas.draw_idle()

    for s in k_sliders + dur_sliders + pause_sliders:
        s.on_changed(update)

    def _on_close(_evt):
        if save_OnOffsets:
            if not json_path:
                print("[WARN] save_OnOffsets=True, aber json_path ist None – überspringe Speichern.")
                return
            payload = {ch: {"on_offsets": results_samples.get(ch, [])} for ch in channels}
            try:
                import json, os
                os.makedirs(os.path.dirname(json_path), exist_ok=True)
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                print(f"[OK] On-/Offset-Paare gespeichert: {json_path}")
            except Exception as e:
                print(f"[ERROR] Konnte JSON nicht schreiben ({json_path}): {e}")

    fig.canvas.mpl_connect("close_event", _on_close)  # NEW

    update(None)
    ax_sig[-1].set_xlabel("Zeit (s)")
    fig.suptitle(f"Channels: {', '.join(channels)}", fontsize=14, y=0.99)
    plt.show()

    return results_samples if give_out_dict else None


#--------------------- not working well ----------- #
def sliding_window_threshold_detection(
    signal: np.ndarray,
    sf: int,
    window_duration: float = 4.0,
    k: float = 1.3,
    min_duration: float = 0.5,
    min_pause: float = 0.3
):
    """
    Dynamische Schwellenwertbasierte Bewegungserkennung mit Sliding Window.

    Parameters:
        signal (np.ndarray): Sensorsignal (z. B. EMG, ACC).
        sf (int): Samplingrate in Hz.
        window_duration (float): Länge des Sliding Windows in Sekunden.
        k (float): Multiplikator für den Schwellenwert (Threshold = mean + k * std im Window).
        min_duration (float): Minimale Dauer eines Aktivitätssegments (s).
        min_pause (float): Minimale Pause zwischen zwei Segmenten (s).

    Returns:
        dict: Detektions-Ergebnis mit Onsets, Offsets und Schwellenkurve.
    """
    window_size = int(window_duration * sf)
    min_dur_samples = int(min_duration * sf)
    min_pause_samples = int(min_pause * sf)

    threshold_curve = np.full_like(signal, np.nan, dtype=float)

    for t in range(window_size, len(signal)):
        window = signal[t - window_size:t]
        mu = np.mean(window)
        sigma = np.std(window)
        threshold_curve[t] = mu + k * sigma

    valid = signal > threshold_curve
    valid[:window_size] = False

    labels_array, n_lbl = label(valid)
    cleaned = np.zeros_like(valid, dtype=bool)

    for lbl in range(1, n_lbl + 1):
        idx = np.where(labels_array == lbl)[0]
        if len(idx) >= min_dur_samples:
            cleaned[idx] = True

    onsets = np.where(np.diff(cleaned.astype(int)) == 1)[0] + 1
    offsets = np.where(np.diff(cleaned.astype(int)) == -1)[0] + 1

    if offsets.size > 0 and onsets.size > 0:
        if offsets[0] < onsets[0]:
            offsets = offsets[1:]
        if onsets.size > offsets.size:
            onsets = onsets[:-1]

    clean_onsets = [onsets[0]] if len(onsets) > 0 else []
    clean_offsets = []

    for i in range(len(offsets) - 1):
        if onsets[i + 1] - offsets[i] < min_pause_samples:
            continue
        else:
            clean_offsets.append(offsets[i])
            clean_onsets.append(onsets[i + 1])
    if len(offsets) > 0:
        clean_offsets.append(offsets[-1])

    return {
        "onsets": clean_onsets,
        "offsets": clean_offsets,
        "threshold_curve": threshold_curve
    }



def plot_sliding_window_detection(
    df, sync_times, limb_and_side, task_name, baseline_parts, sf,
    window_duration=4.0, k=1.3, min_duration=0.5, min_pause=0.3
):
    """
    Visualisiert Sliding-Window-basierte Schwellenwertdetektion für alle Kanäle eines bestimmten limbs.
    """

    if "arm" in limb_and_side.lower():
        channels = ["brachioradialis_L", "deltoideus_L", "SVM_L"] if "l" in limb_and_side.lower() \
            else ["brachioradialis_R", "deltoideus_R", "SVM_R"]
    elif "leg" in limb_and_side.lower():
        channels = ["tibialisAnterior_L", "SVM_R"] if "l" in limb_and_side.lower() \
            else ["tibialisAnterior_R", "SVM_R"]
    else:
        print("Unbekannte limb_and_side:", limb_and_side)
        return

    n_channels = len(channels)
    fig_height = 3.5 * n_channels
    fig, axs = plt.subplots(n_channels, 1, figsize=(12, fig_height))
    axs = np.atleast_1d(axs)
    plt.subplots_adjust(bottom=0.1, top=0.9)

    for i, channel in enumerate(channels):
        if channel not in df.columns:
            axs[i].text(0.5, 0.5, f"{channel} not found", ha='center')
            continue

        signal = df[channel].values
        result = sliding_window_threshold_detection(
            signal=signal,
            sf=sf,
            window_duration=window_duration,
            k=k,
            min_duration=min_duration,
            min_pause=min_pause
        )

        axs[i].plot(sync_times, signal, label=channel)
        axs[i].plot(sync_times, result["threshold_curve"], 'r--', label=f"Dynamic Threshold (k={k})")

        for x in result["onsets"]:
            axs[i].axvline(sync_times[x], ls="--", c="g", label="Onset" if x == result["onsets"][0] else "")
        for x in result["offsets"]:
            axs[i].axvline(sync_times[x], ls="--", c="k", label="Offset" if x == result["offsets"][0] else "")

        axs[i].set_title(f"{channel}")
        axs[i].legend(loc='upper right')

    plt.suptitle(f"{limb_and_side.upper()} – Task: {task_name}", fontsize=14)
    plt.show()



import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

def plot_with_behavior(
    df,
    behavioral_array,
    onoffset_array_delt,
    onoffset_array_brachio,
    onoffset_array_acc,
    *,
    save=False,
    save_path="../images/plot_brachio_delt_svm_behavior_R.png",
    dpi=300,
    figsize=(12, 9)
):
    """
    4 Subplots:
      1) brachioradialis_R (mit On-/Offset-Linien)
      2) deltoideus_R      (mit On-/Offset-Linien)
      3) SVM               (mit On-/Offset-Linien aus ACC, falls gewünscht)
      4) Behavioral states als Farbbalken

    Erwartete DF-Spalten (fix):
      - "Sync_Time (s)"
      - "brachioradialis_R"
      - "deltoideus_R"
      - "SVM"   (umbenannt von SVM_R)
    """

    # --- Checks ---
    #required_cols = ["Sync_Time (s)", "brachioradialis_L", "deltoideus_L", "SVM_R", "SVM_L"]
    required_cols = ["Sync_Time (s)", "brachioradialis_R", "deltoideus_R", "SVM"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Spalte '{c}' fehlt im DataFrame.")

    if behavioral_array is None or len(behavioral_array) < 1:
        raise ValueError("behavioral_array ist leer.")

    # --- Farben & Styles ---
    state_colors = {0: 'lightsteelblue', 1: 'darkseagreen', 2: 'orchid'}
    onset_color, offset_color = "gray", "silver"

    # Gridlines sicher aus
    plt.rcParams["axes.grid"] = False

    # --- Daten ---
    time_vals         = df["Sync_Time (s)"].to_numpy()
    brachioradialis_L = df["brachioradialis_R"].to_numpy()
    deltoideus_L      = df["deltoideus_R"].to_numpy()
    brachioradialis_R = df["brachioradialis_R"].to_numpy()
    deltoideus_R      = df["deltoideus_R"].to_numpy()
    #SVM_vals          = df["SVM_L"].to_numpy()
    SVM_vals            = df["SVM"].to_numpy()

    if time_vals[0] == time_vals[-1]:
        raise ValueError("Zeitachse hat keine Breite (alle Zeiten gleich).")

    # Grenzen passend zur Länge der behavioral-Samples
    time_boundaries = np.linspace(time_vals[0], time_vals[-1], len(behavioral_array) + 1)

    # --- Figure/Axes (kein Grid, sauberes Layout) ---
    fig, axs = plt.subplots(
        4, 1, sharex=True, figsize=figsize, constrained_layout=True,
        gridspec_kw={'height_ratios': [1, 1, 1, 0.45]}
    )

    # Helper: Behavioral-Hinterlegung
    def shade_behavior(ax, alpha=0.15, zorder=0):
        current_state = behavioral_array[0]
        start_idx = 0
        for i in range(1, len(behavioral_array)):
            if behavioral_array[i] != current_state:
                ax.axvspan(time_boundaries[start_idx], time_boundaries[i],
                           facecolor=state_colors.get(current_state, 'white'),
                           alpha=alpha, zorder=zorder, edgecolor="none",
                           linewidth=0, antialiased=False)
                current_state = behavioral_array[i]
                start_idx = i
        ax.axvspan(time_boundaries[start_idx], time_boundaries[-1],
                   facecolor=state_colors.get(current_state, 'white'),
                   alpha=alpha, zorder=zorder, edgecolor="none",
                           linewidth=0, antialiased=False)

    # Helper: sichere On-/Off-Linien (Index-Clamping)
    def draw_onoffset_lines(ax, pairs, xvals, on_color, off_color, zorder=3):
        n = len(xvals)
        for pair in (pairs or []):
            if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                continue
            on_idx  = int(np.clip(pair[0], 0, n - 1))
            off_idx = int(np.clip(pair[1], 0, n - 1))
            ax.axvline(xvals[on_idx],  ls="--", c=on_color,  zorder=zorder)
            ax.axvline(xvals[off_idx], ls="--", c=off_color, zorder=zorder)

    # --- Subplot 1: brachioradialis_R ---
    shade_behavior(axs[0], alpha=0.15, zorder=0)
    axs[0].plot(time_vals, brachioradialis_L, color='lightskyblue', zorder=2, label="M. brachioradialis L")
    draw_onoffset_lines(axs[0], onoffset_array_brachio, time_vals, onset_color, offset_color, zorder=3)
    axs[0].set_ylabel("Amplitude (µV)")
    axs[0].set_title("brachioradialis", pad=5)
    axs[0].spines[['top', "bottom"]].set_visible(False)
    #axs[0].tick_params(bottom=True, left=True)
    axs[0].set_yticks([-400, 0, 400])

    # --- Subplot 2: deltoideus_R ---
    shade_behavior(axs[1], alpha=0.15, zorder=0)
    axs[1].plot(time_vals, deltoideus_L, color="cornflowerblue", zorder=2, label="M. deltoideus anterior L")
    draw_onoffset_lines(axs[1], onoffset_array_delt, time_vals, onset_color, offset_color, zorder=3)
    axs[1].set_ylabel("Amplitude (µV)")
    axs[1].set_title("deltoideus", pad=5)
    axs[1].spines[['top', "bottom"]].set_visible(False)
    axs[1].tick_params(bottom=True, left=True)
    #sp = axs[1].spines["bottom"]
    #axs[1].set_visible(False)
    axs[1].set_yticks([-500, 0, 500, 1000])

    # --- Subplot 3: SVM ---
    shade_behavior(axs[2], alpha=0.15, zorder=0)
    axs[2].plot(time_vals, SVM_vals, color='steelblue', zorder=2, label="right hand ACC SVM")
    draw_onoffset_lines(axs[2], onoffset_array_acc, time_vals, onset_color, offset_color, zorder=3)
    axs[2].set_ylabel("Acceleration (g)")
    #axs[2].set_title("SVM_R", pad=5)
    axs[2].set_title("SVM", pad=5)
    axs[2].spines[['top', "bottom"]].set_visible(False)
    axs[2].tick_params(bottom=True, left=True, )
    axs[2].set_yticks([0, 0.25, 0.5])

    # --- Subplot 4: Behavioral als Balken ---
    shade_behavior(axs[3], alpha=0.7, zorder=0)
    axs[3].set_yticks([])
    axs[3].set_xlabel("Time (s)")
    axs[3].spines[['top', 'left']].set_visible(False)
    axs[3].spines['bottom'].set_visible(True)
    axs[3].set_xticks([0, 5, 10, 15, 20, 25, 30, 35])
    sp = axs[3].spines["bottom"]
    sp.set_linewidth(0.8)
    sp.set_color("black")
    sp.set_zorder(10)

    # --- Legenden (innen, hohe zorder) ---
    emg_elements = [
        Line2D([0], [0], color='lightskyblue', label="M. brachioradialis_L"),
        Line2D([0], [0], color='cornflowerblue', label="M. deltoideus anterior_L"),
        Line2D([0], [0], color='steelblue', label="SVM"),
        Line2D([0], [0], color=onset_color,  linestyle='--', label='emg onset'),
        Line2D([0], [0], color=offset_color, linestyle='--', label='emg offset')
    ]
    leg0 = axs[0].legend(handles=emg_elements, loc='upper right', frameon=False)
    leg0.set_zorder(4)

    behavior_patches = [
        Patch(facecolor=state_colors[0], label='rest'),
        Patch(facecolor=state_colors[1], label='movement'),
        Patch(facecolor=state_colors[2], label='suppression'),
    ]
    leg3 = axs[3].legend(handles=behavior_patches, loc='upper right', frameon=False)
    leg3.set_zorder(4)

    # --- Achsenformatierung: Ticks & Limits ---
    for ax in axs:
        ax.set_xlim(time_vals[0], time_vals[-1])
        # optional: weiße Hintergründe für Styles, die dunkel sind
        ax.set_facecolor("white")
        # Ticks alle 10 s (rundet an Datenbereich an)
        t0, t1 = float(time_vals[0]), float(time_vals[-1])
        start_tick = 10 * np.floor(t0 / 10.0)
        end_tick   = 10 * np.ceil(t1 / 10.0)
        ticks = np.arange(start_tick, end_tick + 1e-9, 10.0)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{int(t)}" for t in ticks])
        ax.set_xlim(0, 35)
        #ax.set_xlim(15, 95)
        #ax.set_xlim(10,300)
        ax.tick_params(axis='x', labelbottom=True)
        ax.grid(False)  # sicherheitshalber

    # --- Render & Save/Show ---
    if save:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.canvas.draw()  # stellt sicher, dass Layout/Legenden final sind
        fig.savefig(
            save_path,
            dpi=dpi,
            facecolor=fig.get_facecolor(),
            edgecolor='none'
        )
        plt.close(fig)
    else:
        plt.show()


def plot_with_behavior_leg(
    df,
    behavioral_array,
    onoffset_array_tibia,
    onoffset_array_acc_leg,
    *,
    save=False,
    save_path="../images/plot_brachio_delt_svm_behavior_R.png",
    dpi=300,
    figsize=(12, 8)
):
    # --- Checks ---
    required_cols = ["Sync_Time (s)", "SVM_R", "SVM_L", "tibialisAnterior_L"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Spalte '{c}' fehlt im DataFrame.")
    if behavioral_array is None or len(behavioral_array) < 1:
        raise ValueError("behavioral_array ist leer.")

    # --- Farben & Styles ---
    state_colors = {0: 'lightsteelblue', 1: 'darkseagreen', 2: 'orchid'}
    onset_color, offset_color = "gray", "silver"

    plt.rcParams["axes.grid"] = False

    # --- Daten ---
    time_vals            = df["Sync_Time (s)"].to_numpy()
    tibialisAnterior_L   = df["tibialisAnterior_L"].to_numpy()
    SVM_vals             = df["SVM_R"].to_numpy()
    if time_vals[0] == time_vals[-1]:
        raise ValueError("Zeitachse hat keine Breite (alle Zeiten gleich).")

    # Zeitgrenzen & 50s-Ticks
    x_left, x_right = 50, 300
    ticks_50 = np.arange(x_left, x_right + 1e-9, 50.0)

    # Grenzen passend zur Länge der behavioral-Samples
    time_boundaries = np.linspace(time_vals[0], time_vals[-1], len(behavioral_array) + 1)

    # --- Figure/Axes ---
    fig, axs = plt.subplots(
        3, 1, sharex=True, figsize=figsize, constrained_layout=True,
        gridspec_kw={'height_ratios': [1, 1, 0.45]}
    )

    # Helper: Behavioral-Hinterlegung (ohne weiße Kanten, kein Antialiasing)
    def shade_behavior(ax, alpha=0.15, zorder=0):
        current_state = behavioral_array[0]
        start_idx = 0
        for i in range(1, len(behavioral_array)):
            if behavioral_array[i] != current_state:
                ax.axvspan(time_boundaries[start_idx], time_boundaries[i],
                           facecolor=state_colors.get(current_state, 'white'),
                           alpha=alpha, zorder=zorder,
                           edgecolor='none', linewidth=0, antialiased=False)
                current_state = behavioral_array[i]
                start_idx = i
        ax.axvspan(time_boundaries[start_idx], time_boundaries[-1],
                   facecolor=state_colors.get(current_state, 'white'),
                   alpha=alpha, zorder=zorder,
                   edgecolor='none', linewidth=0, antialiased=False)

    # Helper: On-/Off-Linien
    def draw_onoffset_lines(ax, pairs, xvals, on_color, off_color, zorder=3):
        n = len(xvals)
        for pair in (pairs or []):
            if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                continue
            on_idx  = int(np.clip(pair[0], 0, n - 1))
            off_idx = int(np.clip(pair[1], 0, n - 1))
            ax.axvline(xvals[on_idx],  ls="--", c=on_color,  zorder=zorder)
            ax.axvline(xvals[off_idx], ls="--", c=off_color, zorder=zorder)

    # --- Subplot 1: tibialis_L ---
    shade_behavior(axs[0], alpha=0.2, zorder=0)
    axs[0].plot(time_vals, tibialisAnterior_L, color='slateblue', zorder=2)
    draw_onoffset_lines(axs[0], onoffset_array_tibia, time_vals, onset_color, offset_color, zorder=3)
    axs[0].set_ylabel("Amplitude (µV)")
    axs[0].set_title("left tibialisAnterior", pad=5)
    axs[0].spines[['right', 'top']].set_visible(False)
    sp = axs[0].spines["left"]; sp.set_visible(True); sp.set_linewidth(0.8); sp.set_color('black'); sp.set_zorder(10)

    # --- Subplot 2: SVM ---
    shade_behavior(axs[1], alpha=0.2, zorder=0)
    axs[1].plot(time_vals, SVM_vals, color='darkblue', zorder=2)
    draw_onoffset_lines(axs[1], onoffset_array_acc_leg, time_vals, onset_color, offset_color, zorder=3)
    axs[1].set_ylabel("Acceleration (g)")
    axs[1].set_title("SVM leg", pad=5)
    axs[1].spines[["bottom"]].set_visible(True)
    for side in ['bottom', 'left']:
        sp = axs[1].spines[side]; sp.set_visible(True); sp.set_linewidth(0.8); sp.set_color('black'); sp.set_zorder(10)

    # --- Subplot 3: Behavioral als Balken (mit harter Kanten-Deaktivierung) ---
    shade_behavior(axs[2], alpha=0.8, zorder=0)  # nutzt edgecolor='none', antialiased=False
    axs[2].set_yticks([])
    axs[2].set_xlabel("Time (s)")
    axs[2].spines[['bottom']].set_visible(True)
    sp = axs[2].spines["bottom"]; sp.set_visible(True); sp.set_linewidth(0.8); sp.set_color('black'); sp.set_zorder(10)

    # --- Achsenformatierung: Ticks & Limits ---
    for ax in axs:
        ax.set_xlim(x_left, x_right)
        ax.set_facecolor("white")
        ax.set_xticks(ticks_50)
        ax.set_xticklabels([f"{int(t)}" for t in ticks_50])
        ax.grid(False)

    # x-Labels sichtbar im mittleren & unteren Plot, oben ohne
    axs[0].tick_params(axis='x', labelbottom=False)
    axs[1].tick_params(axis='x', bottom=True, labelbottom=True)
    axs[2].tick_params(axis='x', bottom=True, labelbottom=True)

    # --- Render & Save/Show ---
    if save:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.canvas.draw()
        fig.savefig(
            save_path,
            dpi=dpi,
            bbox_inches='tight',
            facecolor=fig.get_facecolor(),
            edgecolor='none'
        )
        plt.close(fig)
    else:
        plt.show()


# ----------

def get_tkeo_emg_task_dfs(filepaths):
    """
    applies tkeo processing to emg data and smoothing/RMS to acc data
    :param filepaths: filepaths from the sub folder (needs to be from processed data files)
    :return: dict of dataframes --> one df per task (=per measurement)
    """
    tkeo_emg_dfs_dict = {}

    for filepath in filepaths:

        # === read in file and get out df === #
        processed_df, task_name = read_in_h5(filepath)
        print(f"now processing: {task_name}")
        print(f"columns: {processed_df.columns}")

        ### === preprocessing EMG to envelope === #
        emg_cols = processed_df.columns[
            processed_df.columns.str.contains('brachioradialis|deltoideus|tibialis', case=False, regex=True)].tolist()


        tkeo_emg_df = add_tkeo(processed_df, emg_cols)

        ### === preprocessing ACC with smoothing and RMS === #
        acc_cols = processed_df.columns[processed_df.columns.str.contains('hand|knee|SVM', case=False, regex=True)].tolist()

        smoothed_acc_df = savgol(tkeo_emg_df, acc_cols)
        ready_for_detec_df = root_mean_square(smoothed_acc_df, acc_cols, window_size=100)

        tkeo_emg_dfs_dict[f"{task_name[:-10]}"] = ready_for_detec_df

    return emg_cols, tkeo_emg_dfs_dict


def get_envelope_emg_task_dfs(filepaths):
    """
    applies envelope processing to emg data and smoothing/RMS to acc data
    :param filepaths: filepaths from the sub folder (needs to be from processed data files)
    :return: dict of dataframes --> one df per task (=per measurement)
    """
    envelope_emg_dfs_dict = {}

    for filepath in filepaths:
        # === read in file and get out df === #
        processed_df, task_name = read_in_h5(filepath)
        print(f"now processing: {task_name}")
        print(f"columns: {processed_df.columns}")

        ### === preprocessing EMG to envelope === #
        emg_cols = processed_df.columns[
            processed_df.columns.str.contains('brachioradialis|deltoideus|tibialis', case=False,
                                              regex=True)].tolist()

        rectified_emg_df = rectify(processed_df, emg_cols)
        enveloped_emg_df = envelope(rectified_emg_df, emg_cols, 3)

        ### === preprocessing ACC with smoothing and RMS === #
        acc_cols = processed_df.columns[
            processed_df.columns.str.contains('hand|knee|SVM', case=False, regex=True)].tolist()

        smoothed_acc_df = savgol(enveloped_emg_df, acc_cols)
        ready_for_detec_df = root_mean_square(smoothed_acc_df, acc_cols, window_size=100)


        envelope_emg_dfs_dict[f"{task_name[:-10]}"] = ready_for_detec_df

    return emg_cols, envelope_emg_dfs_dict



# ----------------------ML functions --------------

def crop_edges(df, samples=1000, crop_only_end=False):
    if crop_only_end:
        if len(df) > 2 * samples:
            return df.iloc[:-samples].reset_index(drop=True)
    else:
        if len(df) > 2 * samples:
            return df.iloc[samples:-samples].reset_index(drop=True)
    return df.copy()

def sliding_windows(df, window_size=250, step_size=125):
    windows = []
    for start in range(0, len(df) - window_size + 1, step_size):
        window = df.iloc[start:start + window_size]
        windows.append(window)
    return windows

def extract_features_from_window(window, emg_channels, acc_channels):
    features = {}

    # EMG features
    for ch in emg_channels:
        sig = window[ch].values

        features[f"{ch}_rms"] = np.sqrt(np.mean(sig**2))
        features[f"{ch}_mav"] = np.mean(abs(sig))
        features[f"{ch}_std"] = np.std(sig)
        features[f"{ch}_max"] = np.max(sig)

        mean = np.mean(sig)
        if "envelope" in ch and mean != 0:
            features[f"{ch}_coefVar"] = np.std(sig) / mean
        if not "envelope" in ch and not "tkeo" in ch:
            zero_crossings = np.where(np.diff(np.sign(sig)))[0]
            features[f"{ch}_zeroCrossings"] = len(zero_crossings)

            diff1 = np.diff(sig)
            diff2 = np.diff(diff1)
            ssc = np.sum((diff1[:-1] * diff1[1:] < 0) & (np.abs(diff2) > 1e-6))
            features[f"{ch}_slopeSignChanges"] = ssc

            waveform_length = np.sum(np.abs(np.diff(sig)))
            features[f"{ch}_waveformLength"] = waveform_length



    # ACC features
    for ch in acc_channels:
        sig = window[ch].values
        features[f"{ch}_rms"] = np.sqrt(np.mean(sig**2)) # rms of window
        features[f"{ch}_mean"] = np.mean(sig) # mean
        features[f"{ch}_std"] = np.std(sig) # std
        features[f"{ch}_max"] = np.max(sig) # max
        features[f"{ch}_min"] = np.min(sig) #min
        features[f"{ch}_range"] = (np.max(sig) - np.min(sig)) # range between min and max value

        mean = np.mean(sig)
        if mean != 0:
            features[f"{ch}_coefVar"] = np.std(sig) / np.mean(sig) # coefVar if mean is not zero

        # PSD
        freqs, psd = welch(sig, fs=1000, nperseg=len(sig))   # getting out freqs and corresponding psd

        features[f"{ch}_psd_peakfreq"] = freqs[np.argmax(psd)] # most dominant frequency
        features[f"{ch}_psd_power_2_5Hz"] = np.sum(psd[(freqs >= 2) & (freqs <= 5)]) # power in 2-5Hz range
        features[f"{ch}_psd_power_8_20Hz"] = np.sum(psd[(freqs >= 8) & (freqs <= 10)])  # power in 8-20Hz range

        low = features[f"{ch}_psd_power_2_5Hz"]
        high = features[f"{ch}_psd_power_8_20Hz"]
        features[f"{ch}_psd_power_ratio_high_low"] = high / (low + 1e-6) # ratio of high/low power


    return features



def window_labels_with_onoff_bias_logged(
    y_samples,            # 1D array mit {0=rest,1=move,2=suppr} pro Sample
    sf,                   # Samplingrate [Hz]
    window_size,          # Fensterlänge in SAMPLES (z.B. int(0.250*sf))
    overlap_value,        # z.B. 0.5 für 50% Overlap -> stride = window_size*overlap_value
    tau_high=0.80,        # konservativer Majority-Threshold
    tau_low=0.55,         # Borderline-Schwelle für Fallback-Regeln
    class_names=("rest", "move", "suppr"),
    dump_json_path=None   # optionaler Pfad: wenn gesetzt, wird ein JSON-Log geschrieben
):
    """
    Returns:
      starts: np.ndarray[int]  Startindizes je Fenster (Samples)
      y_win:  np.ndarray[int]  Fensterlabels in {0,1,2,-1} (-1 = unknown/abstain)
      props:  np.ndarray[float] Form (n_win, 3), Klassenanteile je Fenster
      log:    list[dict]        Protokoll-Infos für jedes Fenster (JSON-serialisierbar)
    """
    L = int(window_size)
    S = int(max(1, round(window_size * overlap_value)))
    N = int(len(y_samples))
    n_classes = len(class_names)
    assert n_classes == 3, "Diese Variante erwartet 3 Klassen: rest, move, suppr."
    assert set(np.unique(y_samples)).issubset({0,1,2}), "y_samples muss 0/1/2 enthalten."

    starts = np.arange(0, N - L + 1, S, dtype=int)
    y_win  = np.full(len(starts), -1, dtype=int)
    props  = np.zeros((len(starts), n_classes), dtype=float)
    log    = []

    # Binäre Non-Rest-Maske und Transition-Indices (rest<->nonrest)
    nonrest = (y_samples != 0).astype(np.int8)
    trans_idx = np.where(np.diff(nonrest) != 0)[0]  # i: nonrest[i] != nonrest[i+1]

    for j, s0 in enumerate(starts):
        s1 = s0 + L
        w = y_samples[s0:s1]

        # Klassenanteile im Fenster
        counts = np.bincount(w, minlength=n_classes)
        props_j = counts / L
        props[j] = props_j
        c_hat = int(np.argmax(props_j))
        mprop = float(props_j[c_hat])

        decision = None
        assigned = None

        # 1) Sicherer Majority-Fall
        if mprop >= tau_high:
            assigned = c_hat
            decision = "majority"
        else:
            # 2) Borderline: Transition(en) im Fenster?
            tr_in_win = trans_idx[(trans_idx >= s0) & (trans_idx < s1 - 1)]
            nearest_info = {"has_transition_in_window": False, "type": None,
                            "nearest_idx": None, "nearest_time_sec": None,
                            "before": None, "after": None}
            if tr_in_win.size > 0:
                nearest_info["has_transition_in_window"] = True
                center = (s0 + s1) // 2
                nearest = int(tr_in_win[np.argmin(np.abs(tr_in_win - center))])
                before = int(nonrest[nearest])
                after  = int(nonrest[nearest + 1])
                nearest_info.update({
                    "nearest_idx": nearest,
                    "nearest_time_sec": nearest / float(sf),
                    "before": before,
                    "after": after
                })

                if before == 0 and after == 1:
                    # ONSET -> rather take non-rest
                    nonrest_class = 1 + int(np.argmax(props_j[1:3])) # move or suppression
                    assigned = nonrest_class
                    decision = "onset_in_window"
                    nearest_info["type"] = "onset"
                elif before == 1 and after == 0:
                    # OFFSET -> rather take rest
                    assigned = 0
                    decision = "offset_in_window"
                    nearest_info["type"] = "offset"
            else:
                nearest_info = None  # no transition in window

            # 3) Wenn noch nichts entschieden: Kontext außerhalb betrachten
            if assigned is None:
                prev_nonrest = int(nonrest[s0 - 1]) if s0 > 0 else 0
                next_nonrest = int(nonrest[s1]) if s1 < N else int(nonrest[-1])

                # „kurz nach ONSET“ Heuristik
                if prev_nonrest == 0 and (props_j[1] + props_j[2] > 0):
                    assigned = 1 + int(np.argmax(props_j[1:3]))
                    decision = "context_onset"
                # „kurz vor/nach OFFSET“ Heuristik
                elif prev_nonrest == 1 and (props_j[0] > 0) and next_nonrest == 0:
                    assigned = 0
                    decision = "context_offset"
                else:
                    prev_nonrest, next_nonrest = prev_nonrest, next_nonrest

            # 4) Letzter Fallback
            if assigned is None:
                if mprop >= tau_low:
                    assigned = int(w[len(w)//2])  # Center-Rule
                    decision = "center_fallback"
                else:
                    assigned = -1
                    decision = "unknown"

        y_win[j] = assigned

        # -------- Logging --------
        entry = {
            "win_idx": int(j),
            "start_sample": int(s0),
            "end_sample": int(s1),
            "start_sec": s0 / float(sf),
            "end_sec": s1 / float(sf),
            "start_ms": int(round(1000.0 * s0 / float(sf))),
            "end_ms": int(round(1000.0 * s1 / float(sf))),
            "decision": decision,
            "assigned_label": int(assigned),
            "assigned_label_name": (
                class_names[assigned] if (assigned in {0,1,2}) else "unknown"
            ),
            "majority_class": int(c_hat),
            "majority_prop": float(mprop),
            "props": {
                class_names[c]: float(props_j[c]) for c in range(n_classes)
            },
        }

        # Zusatzdetails: nearest transition + Kontext
        if 'tr_in_win' in locals() and (tr_in_win.size > 0):
            # nearest_info schon gefüllt
            entry["transition"] = nearest_info
        else:
            entry["transition"] = {
                "has_transition_in_window": False,
                "type": None,
                "nearest_idx": None,
                "nearest_time_sec": None,
                "before": None,
                "after": None
            }

        # Kontextfelder
        # (nur sinnvoll, wenn nicht auskommentiert/oben gesetzt)
        if decision in {"context_onset", "context_offset"} or True:
            # robust: randbedingungen
            prev_nonrest = int(nonrest[s0 - 1]) if s0 > 0 else 0
            next_nonrest = int(nonrest[s1]) if s1 < N else int(nonrest[-1])
            entry["context"] = {
                "prev_nonrest": prev_nonrest,
                "next_nonrest": next_nonrest
            }

        log.append(entry)

    # Optional: JSON dump
    if dump_json_path is not None:
        def _to_py(obj):
            # macht NumPy-Typen JSON-serialisierbar
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            return obj

        with open(dump_json_path, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2, default=_to_py)

    return starts, y_win, props, log




def create_eval_df_from_continuous(df_raw, y_samples, window_size, overlap_value, sf, sub_id):
    # df_raw: dein kontinuierliches, vorverarbeitetes Signal-DF (nach add_tkeo_add_envelope & acc-smoothing)
    starts, y_win, props, decisions = window_labels_with_onoff_bias_logged(y_samples, sf, window_size, overlap_value)

    feature_rows = []
    for s0, yk, p in zip(starts, y_win, props):
        s1 = s0 + window_size
        window = df_raw.iloc[s0:s1]
        feats = extract_features_from_window(
            window,
            emg_channels=[c for c in df_raw.columns if ("brachioradialis" in c or "deltoideus" in c)],
            acc_channels=["SVM_smooth_rms"]
        )
        feats["sub_ID"] = sub_id
        feats["window_range_ms"] = f"{int(s0/sf*1000)}-{int(s1/sf*1000)}ms"
        feats["label"] = { -1: "unknown", 0: "rest", 1: "move", 2: "suppr" }[int(yk)]
        # optional: weiche Anteile zur Analyse
        feats["p_rest"], feats["p_move"], feats["p_suppr"] = float(p[0]), float(p[1]), float(p[2])
        feature_rows.append(feats)

    df_feats = pd.DataFrame(feature_rows)
    col_order = ["sub_ID", "label", "window_range_ms"] + [c for c in df_feats.columns if c not in ["sub_ID","label","window_range_ms"]]
    return df_feats[col_order]


