import mne
import os
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from mne.preprocessing import annotate_muscle_zscore
from utils_visgam.visgam_paths_local import opmsubjectsdirs
from utils_visgam.visgam_preprocessing import detect_power_outliers, add_gfp_and_scaled_rms_as_misc

# Set which dataset to preprocess
nsubject    = 48
datafolder  = opmsubjectsdirs[nsubject]
print(datafolder)

##### Read data
filename    = 'preprocessed_1.fif' # data from preprocessing_1
raw         = mne.io.read_raw(os.path.join(datafolder, filename), preload=True)

##### Start with detection of bad segements/channels
### -----Mark segments without triggers at the beggining/end as BAD_NOTRIG
### -----Mark segments with slow drifts as BAD_MOVEMENT
raw.plot(duration=60, decim=2, use_opengl=True) # for srate 500 a decimation=2 allows to see up to 125Hz

# Epoch data
events_full, event_id_all = mne.events_from_annotations(raw)
stim_event_id = {k: v for k, v in event_id_all.items() if 'STIMON' in k}
raw_epochs = mne.Epochs(raw, event_id=stim_event_id, tmin=-1.5, tmax=2.5, baseline=None, 
                        proj=False, detrend=1, picks='mag', reject_by_annotation=True,
                        preload=True, verbose=False) 

### ----- Compute PSD only from valid epochs and look for Bad Channels
n_fft = np.int32(raw_epochs.info['sfreq'] * 2)
raw_PSD = raw_epochs.compute_psd(method='welch', fmin=0.2, fmax=120, n_fft=n_fft)
raw_PSD.plot(picks='mag', exclude='bads')

# Obtain average power in X and Y
OPMX = [item for item in raw_epochs.ch_names if item.endswith("x")]
OPMY = [item for item in raw_epochs.ch_names if item.endswith("y")]
raw_PSDX = raw_epochs.compute_psd(method="welch", fmin=0.2, fmax=120, picks=OPMX, n_fft=n_fft)
raw_PSDY = raw_epochs.compute_psd(method="welch", fmin=0.2, fmax=120, picks=OPMY, n_fft=n_fft)

# Plot and detect outliers based on MAD
outlier_chs_X = detect_power_outliers(raw_PSDX.average(), axis_label='X')
outlier_chs_Y = detect_power_outliers(raw_PSDY.average(), axis_label='Y')
print(outlier_chs_X + outlier_chs_Y)

### ----- Look at every channel and click (or add to new_bads manually) to reject
### Remove only if channel has clearly lost contact or has strong artifacts. 
raw.plot(duration=60, n_channels=8, decim=2, use_opengl=True, scalings='auto') 

# Add outlier_chs_EEG only after inspecting all the recording
new_bads = outlier_chs_X + outlier_chs_Y
new_bads = []
raw.info['bads'].extend(new_bads)
print(raw.info['bads'])

### ----- Mark sections with muscular artifacts
# This will just annotate the data. Later we'll visually inspect and add/remove annotations if needed
raw_tmp = raw.copy().drop_channels(raw.info['bads'])

# Remove muscle artifacts (without outlier channels)
threshold_muscle = 3 # in sd
annotations_muscle, scores_muscle = annotate_muscle_zscore(
    raw_tmp, ch_type='mag', threshold=threshold_muscle, min_length_good=0.1,
    filter_freq=[110, 140])

tr = np.full(len(scores_muscle), threshold_muscle)
plt.plot(raw_tmp.times, scores_muscle,raw_tmp.times, tr)
plt.xlabel('Time (s)');plt.ylabel('Z score');plt.title('Putative muscular activity')
plt.show()
# Add annotations to raw
raw.set_annotations(raw.annotations + annotations_muscle)

# Visualize filtered data to look for remaining muscle artifacts (annotate as BAD_muscle)
# Focus on the time after STIMON
raw_muscleband = raw_tmp.copy().filter(110, 140)
raw_muscleband.set_annotations(raw.annotations + annotations_muscle)
# Add GFP and total RMS power to aid localizing muscle artifacts
raw_muscleband_gfp = add_gfp_and_scaled_rms_as_misc(raw_muscleband, 'mag', 0.2)
# emg: RMS total power/ misc: GFP
raw_muscle_epochs = mne.Epochs(raw_muscleband_gfp, event_id=stim_event_id, tmin=-0.5, tmax=1.5,
                        baseline=None, proj=False, picks='all', reject_by_annotation=False,
                        preload=True, verbose=False)

ann_per_ep = raw_muscle_epochs.get_annotations_per_epoch()
marked_idx = [
    i for i, lst in enumerate(ann_per_ep)
    if any(desc.startswith("BAD_muscle") for (_, _, desc) in lst)
]
print("Marked epochs:", marked_idx)
# Removed trials
raw_muscle_epochs[marked_idx].plot(n_epochs=10, scalings=dict(misc=5e-13, emg=1e-13), block=True, picks=['all'], use_opengl=True)

# All trials
raw_muscle_epochs.plot(n_epochs=10, scalings=dict(misc=5e-13, emg=1e-13), block=True, picks=['all'], use_opengl=True)

# Get all trials dropped by the user/ Add annotations at the start of the trial
user_dropped = [i for i, log in enumerate(raw_muscle_epochs.drop_log) if 'USER' in log]
print("Manually dropped trial indices:", user_dropped)

if user_dropped:
    # Get stimulus onset times (in seconds)
    bad_duration = 2
    sfreq = raw_muscle_epochs.info['sfreq']
    dropped_samples = events_full[user_dropped][:, 0]
    onsets = (dropped_samples / sfreq) -0.5
    durations = [bad_duration] * len(onsets)
    descriptions = ['BAD_trial_muscle'] * len(onsets)
    print('Check the following time points: ', onsets)
    # Create and add annotations
    annotations_user = mne.Annotations(onset=onsets, duration=durations, description=descriptions)
    raw.set_annotations(raw.annotations + annotations_user)


# Epoch data after removing annotations
del raw_epochs, raw_tmp
raw_epochs = mne.Epochs(raw, event_id=stim_event_id, tmin=-1.5, tmax=2.5,
                        baseline=None, proj=False, picks='all', detrend=1, reject_by_annotation=True,
                        preload=True, verbose=False)

print('Removed trials, out of 240:', 240-len(raw_epochs))
print('% Removed trials:', ((240-len(raw_epochs))/240)*100)

# Inspect epochs before saving and drop epochs if needed (avoid that)
raw_epochs.plot(butterfly=True, n_epochs=10, scalings='auto', block=True, picks='all', use_opengl=True)

# Get all trials dropped by the user
user_dropped = [i for i, log in enumerate(raw_epochs.drop_log) if 'USER' in log]
print("Manually dropped trial indices:", user_dropped)

if user_dropped:
    # Get stimulus onset times (in seconds)
    sfreq = raw_epochs.info['sfreq']
    dropped_samples = events_full[user_dropped][:, 0]
    onsets = dropped_samples / sfreq
    print('Check the following time points: ', onsets)

# Last check before saving
raw.plot(duration=60, decim=2, use_opengl=True) # for srate 500 a decimation=10 allows to see up to 50Hz

# Save fully annotated data
tmpfilename = 'preprocessed_2.fif'
raw.save(os.path.join(datafolder, tmpfilename), overwrite=True)
