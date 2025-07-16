"""
MEG Analysis Functions

This file contains utility functions for MEG data analysis including:
- MNE RawArray creation
- FastICA application  
- Channel norm calculations
- Power spectrum calculations

These functions support the main MEG analysis pipeline.
"""

import numpy as np
import mne
from sklearn.decomposition import FastICA
from mne.filter import filter_data


def apply_meg_filters(data, sfreq, l_freq: int = 1, h_freq: int = 100):
    """
    Apply bandpass and notch filters to MEG data using MNE.
    
    Args:
        data: Input MEG signal
        sfreq: Sampling frequency (Hz)
    Returns:
        filtered_data: Filtered MEG signal
    """
    # Bandpass filter (1-100 Hz)
    data_bandpass = filter_data(data, sfreq=sfreq, l_freq=l_freq, h_freq=h_freq, 
                              method='fir', verbose=False)
        
    # Apply notch filters (50 Hz and harmonics)
    filtered_data = data_bandpass
    for freq in [50, 100, 150]:
        filtered_data = mne.filter.notch_filter(
            filtered_data, 
            Fs=sfreq,  
            freqs=freq,
            verbose=False
        )
    

    return filtered_data


def create_meg_raw(channels, ch_names, sfreq):
    """
    Create an MNE RawArray from MEG channels.
    
    Args:
        channels: list of arrays (n_channels, n_times)
        ch_names: list of channel names
        sfreq: sampling frequency (default: 375 Hz)
        
    Returns:
        MNE RawArray object
    """
    n_channels = len(channels)
    ch_types = ['mag'] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    # Explicitly set the unit to Tesla for each channel
    data = np.array(channels)
    return mne.io.RawArray(data, info, verbose=False)

def apply_fastica_to_channels(channels, n_components=None, random_state=0, max_iter=1000):
    """
    Apply FastICA to a list of MEG channel arrays (shape: n_channels x n_times).
    
    Args:
        channels: list or np.ndarray, shape (n_channels, n_times)
        n_components: int or None, number of ICA components (default: n_channels)
        random_state: int, random seed for reproducibility (default: 0)
        max_iter: int, maximum iterations for FastICA convergence (default: 1000)
        
    Returns:
        ica_signals: np.ndarray, shape (n_channels, n_times), ICA components
        ica_model: fitted FastICA object
    """
    data = np.array(channels)
    if n_components is None:
        n_components = data.shape[0]
    ica = FastICA(n_components=n_components, random_state=random_state, max_iter=max_iter)
    ica_signals = ica.fit_transform(data.T).T  # Transpose to (n_times, n_channels), then back
    return ica_signals, ica


# Function that calculates the norm of each channel:
def calculate_channel_norms(X_channels, Y_channels, Z_channels):
    """
    Calculate the Euclidean norm for each sensor using its X, Y, Z components.
    
    Computes the vector magnitude for each sensor at each time point using:
    norm = sqrt(x² + y² + z²)
    
    Args:
        X_channels: List of X component arrays for each sensor
        Y_channels: List of Y component arrays for each sensor
        Z_channels: List of Z component arrays for each sensor
    
    Returns:
        List of norm arrays for each sensor, where each array contains
        the magnitude values for all time points
    """
    norms = []
    n_channels = len(X_channels)
    
    for i in range(n_channels):
        # Calculate norm for each time point: sqrt(x² + y² + z²)
        norm = np.sqrt(
            X_channels[i]**2 + 
            Y_channels[i]**2 + 
            Z_channels[i]**2
        )
        norms.append(norm)
    
    return norms

# Function to calculate the power spectrum of a signal:
def calculate_power_spectrum(signal):
    """
    Calculate FFT and power spectrum of a signal.
    
    Computes the Fast Fourier Transform and power spectral density
    of the input signal using a sampling rate based on the sfreq value.
    
    Args:
        signal: 1D array of signal values
        
    Returns:
        freqs: Array of positive frequency bins
        power: Array of power values (doubled to compensate for removing negative frequencies)
    """    # Apply FFT
    n = len(signal)
    fft = np.fft.fft(signal) / n  #Normalization
    freqs = np.fft.fftfreq(n, d=0.002667)  # 375 Hz sampling rate
    power = np.abs(fft) ** 2  # Power calculation
    
    # Get positive frequencies only and double the values to compensate for removing negatives
    pos_mask = freqs >= 0
    return freqs[pos_mask], 2 * power[pos_mask]

