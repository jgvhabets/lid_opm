# to explore variables %whos
# Script 1: Read data and downsample
import os
import mne
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
#cd "S:\AG\AG-Psychose\Daten\EEG\OPM\DFG-OPM-Projekt\[ANALYSEN]\OPMMEG\[VISGAM]\visualgamma_eeg_opm\Visgam_Code"
from visgampaths import opmsubjectsdirs

# Read raw data, notch filter, downsample, bandpass filter and save
for nsubject in range(len(opmsubjectsdirs)):
    # Set which dataset to preprocess
    datafolder = opmsubjectsdirs[nsubject]

    # Set folders and filenames
    datafile = 'Visgam_readin_MNE.mat'
    eventfile = 'Visgam_readin_MNE_events.mat'

    # Load fieldtrip data
    visgamdataset = mne.io.read_raw_fieldtrip(
        fname=os.path.join(datafolder, datafile),
        info=None,
        data_name='data'  
    ) 

    visgamdataset.set_channel_types({ch: 'mag' for ch in visgamdataset.ch_names})
    visgamdataset.notch_filter(freqs=50, picks='mag') # Apply notch filter 

    # Load event data
    events = loadmat(os.path.join(datafolder, eventfile))['mne_events'] # Load events 
    onsets_sec = events[:, 0] / visgamdataset.info['sfreq'] # Compute onset times in seconds
    descriptions_stim = [f"STIMON_{i+1:03d}" for i in range(len(onsets_sec))] # Annotate trials

    # Create baseline annotations 1.5 seconds before each STIMON
    baseline_offset = 1.5  # in seconds
    onsets_bl = onsets_sec - baseline_offset
    descriptions_bl = ["BL" for i in range(len(onsets_sec))]

    # Combine both annotations
    all_onsets = np.concatenate([onsets_bl, onsets_sec])
    all_descriptions = descriptions_bl + descriptions_stim
    all_durations = [0.0] * len(all_onsets)  # zero-duration markers

    # Create and assign annotations
    annotations = mne.Annotations(onset=all_onsets, duration=all_durations, description=all_descriptions)
    visgamdataset.set_annotations(annotations)

    # Downsample and start preprocessing
    desired_sfreq = 500
    current_sfreq = visgamdataset.info['sfreq']
    lowpass_freq = None
    #lowpass_freq = np.fix(desired_sfreq / 3.0)
    highpass_freq = 0.5

    # Band pass filter and save preprocessing 1 file
    raw_resampled = visgamdataset.copy().filter(l_freq=highpass_freq, h_freq=lowpass_freq)
    raw_resampled.resample(sfreq=desired_sfreq)

    # Save downsampled and filtered data
    tmpfilename = 'preprocessed_1.fif'
    raw_resampled.save(os.path.join(datafolder, tmpfilename), overwrite=True)

    del raw_resampled, visgamdataset
