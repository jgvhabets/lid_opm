"""
MEG-EMG Analysis Plotting Functions

This file contains all the plotting and visualization functions used in the MEG-EMG analysis project.
Functions include:
- Power spectra calculation and plotting
- Time-frequency spectrograms 
- Component comparison visualizations
- Raw MEG data plotting
- Vector norm calculations

These functions support the main analysis pipeline
"""


import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.filter import filter_data

# Function to calculate individual power spectra using windowed segments with overlap
def calculate_individual_power_spectra(signal, sfreq=375, window_length=1.0, overlap=0.5):
    """
    Calculate individual power spectra using windowed segments with overlap.
    
    Args:
        signal: Input signal array
        sfreq: Sampling frequency in Hz (default: 375 Hz)
        window_length: Length of each window in seconds (default: 1.0 sec)
        overlap: Overlap between windows as a fraction (default: 0.5 for 50%)
    
    Returns:
        freqs: Frequency bins
        all_psds: List of power spectral densities for each window
        window_times: Start time of each window for labeling
    """
    # Calculate window parameters in samples
    window_samples = int(window_length * sfreq)
    step_samples = int(window_samples * (1 - overlap))
    
    # Create windows
    n_samples = len(signal)
    n_windows = (n_samples - window_samples) // step_samples + 1
    
    # Initialize array to store PSDs and window times
    all_psds = []
    window_times = []
    
    # Process each window
    for i in range(n_windows):
        start_idx = i * step_samples
        end_idx = start_idx + window_samples
        
        # Store window start time (in seconds)
        window_times.append(start_idx / sfreq)
        
        # Extract window
        window = signal[start_idx:end_idx]
        
        # Apply Hanning window to reduce spectral leakage
        windowed_data = window * np.hanning(len(window))
        
        # Calculate FFT
        fft = np.fft.rfft(windowed_data)
        
        # Calculate PSD
        psd = np.abs(fft)**2 / (sfreq * window_samples)
        psd[1:-1] *= 2  # Multiply by 2 (except DC and Nyquist) to account for negative frequencies
        
        all_psds.append(psd)
    
    # Calculate frequency bins
    freqs = np.fft.rfftfreq(window_samples, d=1/sfreq)
    
    return freqs, all_psds, window_times


def plot_all_channel_power_spectra(channels, channel_names, title, sfreq=375, window_length=1.0, overlap=0.5, freq_range=(1, 100)):
    """
    Plot power spectra for all channels in a single figure.
    
    Args:
        channels: List of channel signals
        channel_names: List of channel names or numbers
        title: Plot title
        sfreq: Sampling frequency in Hz
        window_length: Length of each window in seconds
        overlap: Overlap between windows as a fraction
        freq_range: Tuple of (min_freq, max_freq) to display
    """
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Generate colormap for different channels
    colors = plt.cm.rainbow(np.linspace(0, 1, len(channels)))
    
    # For each channel
    for i, (channel, name) in enumerate(zip(channels, channel_names)):
        # Calculate power spectrum
        freqs, all_psds, _ = calculate_individual_power_spectra(channel, sfreq, window_length, overlap)
        
        # Convert list of PSDs to array
        psd_array = np.array(all_psds)
        
        # Average across windows
        avg_psd = np.mean(psd_array, axis=0)
        
        # Get frequency indices within our range
        freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        plot_freqs = freqs[freq_mask]
        plot_psd = avg_psd[freq_mask]
        
        # Plot this channel's power spectrum
        plt.semilogy(plot_freqs, plot_psd, color=colors[i], 
                    linewidth=1.5, alpha=0.8, label=f'Channel {name}')
    
    # Add labels and title
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (pT²/Hz)')
    plt.title(f'Power Spectra - {title}')
    plt.grid(True, alpha=0.3)
    plt.xlim(freq_range)
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    plt.tight_layout()
    plt.show()

    
def plot_meg_spectrogram(channel, channel_name, time_data, title, sfreq=375, 
                         window_length=1.0, overlap=0.5, freq_range=(1, 100)):
    """
    Plot a time-frequency spectrogram for a single MEG channel.
    
    Args:
        channel: Input signal array for a single channel
        channel_name: Name or number of the channel
        time_data: Time data for signal duration
        title: Plot title
        sfreq: Sampling frequency in Hz (default: 375 Hz)
        window_length: Length of each window in seconds (default: 1.0 sec)
        overlap: Overlap between windows as a fraction (default: 0.5 for 50%)
        freq_range: Tuple of (min_freq, max_freq) to display
    """
    # Calculate window parameters in samples
    window_samples = int(window_length * sfreq)
    step_samples = int(window_samples * (1 - overlap))
    
    # Calculate spectrogram using sliding windows
    freqs, all_psds, window_times = calculate_individual_power_spectra(channel, sfreq, 
                                                                      window_length, overlap)
    
    # Convert list of PSDs to array for plotting
    psd_array = np.array(all_psds)
    
    # Get frequency indices within our range
    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    plot_freqs = freqs[freq_mask]
    plot_psd = psd_array[:, freq_mask]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create log-scaled power values for better visualization
    with np.errstate(divide='ignore'):  # Suppress divide by zero warnings
        log_psd = 10 * np.log10(plot_psd)
    
    # Plot spectrogram using imshow
    im = ax.imshow(
        log_psd.T,  # Transpose to get time on x-axis, freq on y-axis
        aspect='auto',
        origin='lower',
        extent=[window_times[0], window_times[-1], plot_freqs[0], plot_freqs[-1]],
        cmap='viridis',
        interpolation='nearest'
    )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Power (dB)')
    
    # Add labels and title
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(f'Spectrogram - Channel {channel_name} - {title}')
    
    # Add grid
    ax.grid(False)
    
    plt.tight_layout()
    plt.show()
    
    return fig

# Function to create component subplot
def plot_component_comparison(channels_start, channels_last, component_name, channel_names, rec1_label, rec11_label):
    # Calculate grid layout
    n_channels = len(channel_names)
    n_cols = 5
    n_rows = (n_channels + n_cols - 1) // n_cols
    
    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 15), layout="constrained")
    axes = axes.flatten()
    
    for i in range(n_channels):
        # Plot both recordings on the same subplot
        axes[i].plot(time_start, channels_start[i], color='#1f77b4', 
                    label=rec1_label, linewidth=1.5, alpha=0.8)
        axes[i].plot(time_last, channels_last[i], color='#ff7f0e', 
                    label=rec11_label, linewidth=1.5, alpha=0.7)
        
        # Add title and labels
        axes[i].set_title(f'Channel {channel_names[i]}', fontsize=7)
        axes[i].grid(True, alpha=0.3)
        
        # Add legend only for first subplot
        if i == 0:
            axes[i].legend()
        
        # Add y-label for leftmost plots
        if i % n_cols == 0:
            axes[i].set_ylabel('Magnitude (pT)')
        
        # Add x-label for bottom plots
        if i >= n_channels - n_cols:
            axes[i].set_xlabel('Time (sec)')
    
    # Hide unused subplots
    for i in range(n_channels, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'{component_name} Component Comparison: {rec1_label} vs {rec11_label}', fontsize=8)
    plt.tight_layout()
    plt.show()

