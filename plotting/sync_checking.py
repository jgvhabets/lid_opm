
import numpy as np

import matplotlib.pyplot as plt


def plot_check_trigger_distances_AN_FL(
    SUB, SES, ACQ, TASK,
    AN_trig_times, FL_trigger_times,
):

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    ax.plot(np.diff(AN_trig_times), color='orange', alpha=.3, lw=5,
            label='antneuro triggers',)

    ax.plot(np.diff(FL_trigger_times), color='purple', lw=1,
            label='fieldline triggers',)
    
    ax.set_xlabel('triggers over time (count)')
    ax.set_ylabel('inter-trigger interval (s)')

    ax.set_title(f'Trigger Check: sub-{SUB}, ses-{SES} '
                 f'{ACQ}-{TASK}')
    
    ax.legend()

    plt.tight_layout()

    plt.show()