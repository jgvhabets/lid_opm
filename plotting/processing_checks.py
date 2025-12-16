

import matplotlib.pyplot as plt
import numpy as np
import os
from itertools import product
from mne.viz import plot_topomap
import math

from utils.load_utils import get_onedrive_path

def plot_ica_topomaps_Z(raw, ica, batch_size=10, colormap='RdBu_r'):
    """
    Plot ICA components as topomaps for selected good Z channels, displaying them in batches.
    """

    # Get good channels
    all_channels = raw.info['ch_names']
    bad_channels = raw.info['bads']
    good_channels = [ch for ch in all_channels if ch not in bad_channels]

    # Get Z channels
    Z_channels = [ch_name for ch_name in raw.ch_names if '_bz' in ch_name]

    # Keep only good Z channels
    Z_good_channels = [ch for ch in Z_channels if ch in good_channels]

    # Get ICA channel names
    ica_channels = ica.info['ch_names']

    # Indices of Z good channels in ICA
    Z_good_ica_picks = [ica_channels.index(ch) for ch in Z_good_channels]

    # Use only info (no data copy) for plotting
    raw_info_Z = raw.copy().pick_channels(Z_good_channels).info

    # Get ICA components
    ica_data = ica.get_components()
    num_components = ica.n_components_

    # Suppress interactive rendering
    plt.ioff()

    # Loop through components in batches
    for i in range(0, num_components, batch_size):
        num_subplots = min(batch_size, num_components - i)
        rows = int(math.ceil(num_subplots / 6))  # 6 columns
        cols = 6

        fig, axes = plt.subplots(rows, cols, figsize=(7, 12))
        axes = axes.ravel()

        for idx, comp in enumerate(range(i, min(i + batch_size, num_components))):
            comp_data = ica_data[Z_good_ica_picks, comp]
            ax = axes[idx]

            plot_topomap(comp_data, raw_info_Z, axes=ax, show=False,
                         size=3, cmap=colormap)

            ax.text(-0.2, 0.5, f'{comp}', transform=ax.transAxes,
                    fontsize=6, va='center', ha='right')

        # Hide unused axes
        for j in range(idx + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()

    # Render all figures at once
    plt.show()
    plt.ion()



def plot_emgacc_check_for_tasks(recRaw, SAVE=False, SHOW=True,):

    fig, axes = plt.subplots(nrows=len(recRaw.aux_task_epochs) // 2,
                             ncols=2,
                             figsize=(8, 8))
    axes = axes.flatten()
    axestypes = []

    PRE_MARKER_SEC = 1
    POST_MARKER_SEC = 2
    PRE_I_GAP = PRE_MARKER_SEC * recRaw.aux_sfreq
    POST_I_GAP = POST_MARKER_SEC * recRaw.aux_sfreq
    EPOCH_SIZE = PRE_I_GAP + POST_I_GAP  

    subplot_list = list(recRaw.aux_task_epochs.keys())

    for i, (AUX_TYPE, (tasktype, epochtimes)) in enumerate(
        product(['ACC', 'EMG'], recRaw.aux_task_epochs.items())
    ):
        i_ax = np.where([tasktype == t for t in subplot_list])[0][0]
        gotype, SIDE = tasktype.split('_')
        axestypes.append(gotype)
        
        AUX_CLASS = getattr(recRaw, AUX_TYPE).copy()
        side_sigs = [s for s in AUX_CLASS.ch_names if SIDE in s]


        for sig in side_sigs:
            epochs = []
            values = AUX_CLASS.copy().pick(sig).get_data().ravel()  # get 1d array of data
                
            for i0 in epochtimes:
                add_values = values[i0 - PRE_I_GAP:i0 + POST_I_GAP]
                if len(add_values) == EPOCH_SIZE: epochs.append(add_values)
                else: print(f'i: {i0} too short ({len(add_values)})')
            # pad last broken epoch with zeros to enable 2d-array
            try:
                epochs = np.array(epochs)
            except ValueError:
                if any(np.array([len(e) for e in epochs]) < EPOCH_SIZE):
                    i_short = np.argmin([len(e) for e in epochs])
                    temp = np.zeros(EPOCH_SIZE)
                    temp[:len(epochs[i_short])] = epochs[i_short]
                    epochs[i_short] = temp
                    epochs = np.array(epochs)
            meansig = np.nanmean(epochs, axis=0)
            n_epochs = epochs.shape[0]

            # PLOTTING
            axes[i_ax].plot(meansig, label=sig.replace(f'_{SIDE}', '_'),)
        
        
        # once per ax plotting
        axes[i_ax].set_title(f'{tasktype} (n = {n_epochs})')
        axes[i_ax].axvline(ymin=0, ymax=1, x=PRE_I_GAP,
                           color='green', lw=3, alpha=.3,)
        axes[i_ax].axvline(ymin=0, ymax=1, x=PRE_I_GAP + recRaw.aux_sfreq * 1,
                           color='gray', lw=3, alpha=.3,)  # at 1 sec
        if gotype == 'abort':
            axes[i_ax].axvline(ymin=0, ymax=1, x=PRE_I_GAP + recRaw.aux_sfreq * .35,
                               color='darkred', lw=3, alpha=.3,)
        

    # combine legends
    handles, labels = [], []

    for ax in axes:
        ax.set_ylim(-2, 6)
        ax.set_ylabel('z-scored signal (au)')
        ax.set_xlabel('time to trial-cue (sec)')
        ax.set_xticks([t * recRaw.aux_sfreq for t in [0, 1, 2, 3]])
        ax.set_xticklabels([-1, 0, 1, 2])

        # legend mng
        hs, ls = ax.get_legend_handles_labels()
        for h, l in zip(hs, ls):
            if l not in labels:
                handles.append(h)
                labels.append(l)

    # Single legend outside figure
    fig.legend(handles, labels, ncols=4,
               loc="upper center",
               bbox_to_anchor=(0.5, 1.05),
               frameon=False,)

    plt.tight_layout()

    if SAVE:
        figpath = os.path.join(get_onedrive_path('figures'),
                               'processing', 'behav_gonogocheck')
        fname = f'EmgAcc_tasks_sub{recRaw.sub}_{recRaw.task}_{recRaw.acq}'
        plt.savefig(os.path.join(figpath, fname), dpi=300, facecolor='w',
                    bbox_inches="tight",)

    if SHOW: plt.show()
    else: plt.close()



def plot_channels_comparison(
    time_0, time_1, raw_channels, filtered_channels, raw_labels, filtered_labels, colors, 
    rec_label, y_label, axis_label, sync_ylim, show_legend=True
):
    """
    Plot comparison between raw and filtered MEG channels in stacked subplots.
    
    Creates two stacked subplots comparing raw and filtered versions
    of the same channels with matching colors and labels.
    
    Args:
        time: Time vector for x-axis
        raw_channels: List of raw channel signals
        filtered_channels: List of filtered channel signals
        raw_labels: List of labels for raw channels
        filtered_labels: List of labels for filtered channels
        colors: List of colors for channel plotting
        rec_label: Recording label for titles
        y_label: Y-axis label (default: "Amplitude (pT)")
        axis_label: Axis component label (default: "X")
        sync_ylim (bool): Whether to synchronize the y-axis limits.
        show_legend (bool): Whether to display the legend.
    """


    n_raw = min(len(raw_channels), len(colors), len(raw_labels))
    n_filtered = min(len(filtered_channels), len(colors), len(filtered_labels))
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    # Raw
    for i in range(n_raw):
        axes[0].plot(time_0, raw_channels[i], color=colors[i], linewidth=0.6, label=raw_labels[i])
    axes[0].set_title(f'MEG {axis_label} Component - {rec_label} - Raw (Selected)')
    axes[0].set_ylabel(y_label)
    axes[0].grid(True, alpha=0.3)
    if show_legend:
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    # Filtered
    for i in range(n_filtered):
        axes[1].plot(time_1, filtered_channels[i], color=colors[i], linewidth=0.6, label=filtered_labels[i])
    axes[1].set_title(f'MEG {axis_label} Component - {rec_label} - Filtered (Selected)')
    axes[1].set_ylabel(y_label)
    axes[1].grid(True, alpha=0.3)
    if show_legend:
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    if sync_ylim:
        fig = plt.gcf()
        axes = fig.get_axes()
        if len(axes) >= 2:
            y_limits = axes[0].get_ylim()  # Get limits from raw data subplot
            axes[1].set_ylim(y_limits)    # Apply to filtered subplot
    
    plt.tight_layout()
    
    plt.subplots_adjust(top=0.95)
    
    plt.show()