

import matplotlib.pyplot as plt
import numpy as np
import os

from utils.load_utils import get_onedrive_path

def plot_emgacc_check_for_tasks(recRaw, SAVE=False, SHOW=True,):

    fig, axes = plt.subplots(nrows=len(recRaw.task_epochs) // 2,
                            ncols=2,
                            figsize=(8, 8))
    axes = axes.flatten()
    axestypes = []

    EPOCH_SIZE = 3000  # sfreq = 1000 Hz; 1sec-pre, 1sec-task, 1-sec-post

    for i, (tasktype, epochtimes) in enumerate(recRaw.task_epochs.items()):

        gotype, side = tasktype.split('_')
        axestypes.append(gotype)
        
        side_sigs = [s for s in recRaw.rel_aux_sigs if side in s]


        for sig in side_sigs:
            epochs = []
            for i0, _ in epochtimes:
                epochs.append(getattr(recRaw, sig)[i0:i0+EPOCH_SIZE])
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

            axes[i].plot(meansig, label=sig.replace(f'_{side}', '_'),)
        
        
        # once per ax plotting
        axes[i].set_title(tasktype)
        axes[i].axvline(ymin=0, ymax=1, x=1000, color='green', lw=3, alpha=.3,)
        axes[i].axvline(ymin=0, ymax=1, x=2000, color='gray', lw=3, alpha=.3,)
        if gotype == 'abort':
            axes[i].axvline(ymin=0, ymax=1, x=1350, color='orange', lw=3, alpha=.3,)
        

    # combine legends
    handles, labels = [], []

    for ax in axes:
        ax.set_ylim(-2, 6)
        ax.set_ylabel('z-scored signal (au)')
        ax.set_xlabel('time to trial-start (sec)')
        ax.set_xticks([0, 1000, 2000, 3000])
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
        fname = f'emgaccCheck_gonogo_sub{recRaw.sub}_{recRaw.task}_{recRaw.acq}'
        plt.savefig(os.path.join(figpath, fname), dpi=300, facecolor='w',
                    bbox_inches="tight",)

    if SHOW: plt.show()
    else: plt.close()



