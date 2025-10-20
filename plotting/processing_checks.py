

import matplotlib.pyplot as plt
import numpy as np
import os
from itertools import product

from utils.load_utils import get_onedrive_path

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



