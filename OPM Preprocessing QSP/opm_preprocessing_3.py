import mne
import os
import matplotlib.pyplot as plt
import numpy as np
from mne.preprocessing import ICA
from utils_visgam.visgam_paths_local import opmsubjectsdirs, eegsubjectsdirs
from utils_visgam.visgam_preprocessing import plot_ica_for_opm

# Set which dataset to preprocess
nsubject = 38
datafolder = opmsubjectsdirs[nsubject]; print(datafolder)

# Load data
filename    = 'preprocessed_2.fif' # data from preprocessing_2
raw_cleaned = mne.io.read_raw(os.path.join(datafolder, filename), preload=True)

# Visualize data
#raw_cleaned.plot(decim=2, use_opengl=True)

# Remove channel


# Process the raw data for ICA
raw_resmpl = raw_cleaned.copy().drop_channels(raw_cleaned.info['bads']).pick(['mag'])
raw_resmpl.filter(l_freq=5, h_freq=40)  
raw_resmpl.resample(250)  
raw_resmpl_all = raw_resmpl

#raw_resmpl_all.plot(decim=2, use_opengl=True)

n_components = raw_resmpl_all.info['nchan']
ica = ICA(method='infomax', n_components=n_components, random_state=96,
          verbose=False)

ica.fit(raw_resmpl_all, verbose=True, reject_by_annotation=True)
ica.plot_sources(raw_resmpl_all, title='ICA', use_opengl=True)

# Get components for plotting
plot_ica_for_opm(ica, raw_resmpl_all.info)

# Set the components by clicking or select them as ica.exclude
#ica.exclude = [3]

# Apply to (a copy of) the original data
raw_ica = raw_cleaned.copy()
ica.apply(raw_ica)
print(f"Excluded ICA components: {ica.exclude}")

# Visualize data and save
raw_ica.plot(decim=3, start=50, use_opengl=True, highpass=5) 

# Save cleaned data
tmpfilename = 'preprocessed_3_ica.fif'
raw_ica.save(os.path.join(datafolder, tmpfilename), overwrite=True)

# Save ICA solution too to visualize later which components have been removed if needed
ica.save(os.path.join(datafolder, 'ica_solution.fif'), overwrite=True)
raw_resmpl_all.save(os.path.join(datafolder, 'ica_raw_resmpl_all.fif'), overwrite=True)




################################# Plot PSD
sources = ica.get_sources(raw_resmpl_all).load_data().copy()
sources.set_channel_types({ch: 'mag' for ch in sources.ch_names})
# Now "data" includes mag, so no picks error
sources.plot_psd(fmin=5, fmax=40, picks='data', dB=True)

# Load ICA results from EEG to see eye related movements
# Load downsampled and filtered data
datafoldereeg = eegsubjectsdirs[nsubject]
filenameeeg = 'ica_raw_resmpl_all_eeg.fif'
raw_cleaned_eeg = mne.io.read_raw(os.path.join(datafoldereeg, filenameeeg), preload=True)
# Load ICA solution
filenameeeg = 'ica_solution_eeg.fif'
icaeeg = mne.preprocessing.read_ica(os.path.join(datafoldereeg, filenameeeg))
icaeeg.plot_sources(raw_cleaned_eeg, title='ICA', use_opengl=True)


import numpy as np

def quick_diagnostics(raw, label=""):
    # Work on consistent mags without marked bads
    r = raw.copy().drop_channels(raw.info['bads']).pick("mag")
    nchan = r.info['nchan']
    
    # Projectors (SSP): total vs active
    projs = r.info['projs']
    n_projs, n_active = len(projs), sum(p['active'] for p in projs)
    
    # Channel–channel correlation on "good" samples, in physical units (Tesla)
    X = r.get_data(reject_by_annotation="omit", units="T")  # shape: (n_chan, n_times)
    C = np.corrcoef(X)  # channel-by-channel corr
    np.fill_diagonal(C, 0.0)
    i, j = np.unravel_index(np.nanargmax(np.abs(C)), C.shape)
    maxcorr = C[i, j]
    ch_i, ch_j = r.ch_names[i], r.ch_names[j]

    # Header scaling diversity
    unique_cal_range = {(ch['cal'], ch['range']) for ch in r.info['chs']}
    
    # Overall amplitude scale (median abs signal in Tesla)
    med_amp_T = float(np.median(np.abs(X)))

    print(f"[{label}] nchan={nchan} | SSP active/total: {n_active}/{n_projs}")
    print(f"[{label}] max |corr|={abs(maxcorr):.6f} between {ch_i} and {ch_j} (sign={np.sign(maxcorr):.0f})")
    print(f"[{label}] unique (cal, range) pairs: {len(unique_cal_range)}")
    print(f"[{label}] median |signal| (Tesla): {med_amp_T:.3e}")

quick_diagnostics(raw_resmpl_all, "problem")


# Epoch data
events_full, event_id_all = mne.events_from_annotations(raw_resmpl_all)
stim_event_id = {k: v for k, v in event_id_all.items() if 'STIMON' in k}
raw_epochs = mne.Epochs(raw_resmpl_all, event_id=stim_event_id, tmin=-1.5, tmax=2.5, baseline=None, 
                        proj=False, detrend=1, picks='mag', reject_by_annotation=True,
                        preload=True, verbose=False) 

### ----- Compute PSD only from valid epochs and look for Bad Channels
n_fft = np.int32(raw_epochs.info['sfreq'] * 2)
raw_PSD = raw_epochs.compute_psd(method='welch', fmin=0.2, fmax=120, n_fft=n_fft)
raw_PSD.plot(picks='mag', exclude='bads')