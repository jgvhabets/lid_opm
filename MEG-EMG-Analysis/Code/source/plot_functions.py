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
def calculate_individual_power_spectra(signal, sfreq, window_length=1.0, overlap=0.5):
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


def plot_all_channel_power_spectra(channels, channel_names, title, sfreq, window_length=1.0, overlap=0.5, freq_range=(1, 100)):
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
        channel = np.asarray(channel)
        # Calculate power spectrum
        freqs, all_psds, _ = calculate_individual_power_spectra(
            channel, sfreq, window_length, overlap
        )
        psd_array = np.array(all_psds)
        avg_psd = np.mean(psd_array, axis=0)
        freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        plot_freqs = freqs[freq_mask]
        plot_psd = avg_psd[freq_mask]

        plt.semilogy(
            plot_freqs, plot_psd, color=colors[i],
            linewidth=1.5, alpha=0.8, label=f'Channel {name}'
        )

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (pT²/Hz)')
    plt.title(f'Power Spectra - {title}')
    plt.grid(True, alpha=0.3)
    plt.xlim(freq_range)
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    plt.tight_layout()


def plot_meg_2x3_grid(
    data_list_A,         # list of arrays (channels) for dataset A
    data_list_B,         # list of arrays (channels) for dataset B
    time_vector,         # time array (np.array or pd.Series)
    time_windows,        # list of 3 boolean masks
    channel_labels_A,    # list of channel labels for dataset A
    channel_labels_B,    # list of channel labels for dataset B
    time_labels,         # list of 3 time window labels
    colors_A,            # list/array of colors for dataset A
    colors_B,            # list/array of colors for dataset B
    suptitle_A,          # string for the first row title
    suptitle_B,          # string for the second row title
    fig_title            # string for the figure title
):
    """
    Plot a 2x3 grid: columns=time windows, rows=datasets (A, B).
    Each subplot overlays all channels for the selected time window.
    Only the first subplot of each row shows the legend with channel labels.
    Handles different numbers of channels in A and B.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=False)
    n_channels_A = len(data_list_A)
    n_channels_B = len(data_list_B)
    for col, time_label in enumerate(time_labels):
        t_mask = time_windows[col]
        # First row: Dataset A (overlay all channels)
        for ch_idx in range(n_channels_A):
            label = channel_labels_A[ch_idx] if (col == 0 and ch_idx < len(channel_labels_A)) else None
            axes[0, col].plot(
                time_vector[t_mask],
                data_list_A[ch_idx][t_mask],
                color=colors_A[ch_idx % len(colors_A)],
                alpha=0.7,
                label=label
            )
        axes[0, col].set_title(f"{time_label}")
        axes[0, col].set_ylabel(suptitle_A, fontsize=12)
        axes[0, col].grid(True, alpha=0.3)
        if col == 0:
            axes[0, col].legend(fontsize=8, loc='upper right')
        # Second row: Dataset B (overlay all channels)
        for ch_idx in range(n_channels_B):
            label = channel_labels_B[ch_idx] if (col == 0 and ch_idx < len(channel_labels_B)) else None
            axes[1, col].plot(
                time_vector[t_mask],
                data_list_B[ch_idx][t_mask],
                color=colors_B[ch_idx % len(colors_B)],
                alpha=0.7,
                label=label
            )
        axes[1, col].set_ylabel(suptitle_B, fontsize=12)
        axes[1, col].grid(True, alpha=0.3)
        if col == 0:
            axes[1, col].legend(fontsize=8, loc='upper right')
        axes[1, col].set_xlabel('Time (sec)')
    plt.suptitle(fig_title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()

def plot_channels_comparison(
    time, raw_channels, filtered_channels, raw_labels, filtered_labels, colors, 
    rec_label, y_label="Amplitude (pT)", axis_label="X"
):
    """
    Plot comparison between raw and filtered MEG channels in stacked subplots.
    
    Creates two vertically stacked subplots comparing raw and filtered versions
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
    """

    n_raw = min(len(raw_channels), len(colors), len(raw_labels))
    n_filtered = min(len(filtered_channels), len(colors), len(filtered_labels))
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    # Raw
    for i in range(n_raw):
        axes[0].plot(time, raw_channels[i], color=colors[i], linewidth=0.6, label=raw_labels[i])
    axes[0].set_title(f'MEG {axis_label} Component - {rec_label} - Raw (Selected)')
    axes[0].set_ylabel(y_label)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    # Filtered
    for i in range(n_filtered):
        axes[1].plot(time, filtered_channels[i], color=colors[i], linewidth=0.6, label=filtered_labels[i])
    axes[1].set_title(f'MEG {axis_label} Component - {rec_label} - Filtered (Selected)')
    axes[1].set_ylabel(y_label)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()


def plot_ica_max_amplitudes(ica_components, component_names=None, title="Max Amplitude of ICA Components"):
    """
    Plots a horizontal barplot of the max amplitude of each ICA component, with value annotations.

    Args:
        ica_components: Array of shape (n_components, n_samples)
        component_names: List of component names (default: None, auto-generated)
        title: Plot title (default: "Max Amplitude of ICA Components")
    """
    max_amplitudes = np.max(np.abs(ica_components), axis=1)
    n_components = ica_components.shape[0]
    if component_names is None:
        component_names = [f"ICA {i+1}" for i in range(n_components)]
    plt.figure(figsize=(10, 0.5 * n_components + 2))
    bars = plt.barh(component_names, max_amplitudes)
    plt.xlabel("Max Amplitude")
    plt.ylabel("ICA Component")
    plt.title(title)
    # Annotate each bar with its value
    for bar, value in zip(bars, max_amplitudes):
        plt.text(value, bar.get_y() + bar.get_height()/2, f"{value:.2e}", va='center', ha='left')
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt

def plot_power_spectrum_func(component, ax, sfreq, window_length=1.0, overlap=0.5, freq_range=(1, 100)):
    """
    Plot power spectrum of a single ICA component on given axis.
    
    Calculates and plots the power spectrum of an ICA component using
    windowed segments with overlap, displaying on a logarithmic scale.

        Args:
        component: Single ICA component signal array
        ax: Matplotlib axis object to plot on
        sfreq: Sampling frequency in Hz (default: 375)
        window_length: Window length in seconds (default: 1.0)
        overlap: Overlap fraction between windows (default: 0.5)
        freq_range: Frequency range tuple (min, max) in Hz (default: (1, 100))
    """
    freqs, all_psds, _ = calculate_individual_power_spectra(
        component, sfreq, window_length, overlap
    )
    psd_array = np.array(all_psds)
    avg_psd = np.mean(psd_array, axis=0)
    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    plot_freqs = freqs[freq_mask]
    plot_psd = avg_psd[freq_mask]
    ax.semilogy(plot_freqs, plot_psd, color='b', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (pT²/Hz)')
    ax.grid(True, alpha=0.3)

def plot_single_ica_power_spectrum(component, ax, sfreq, window_length=1.0, overlap=0.5, freq_range=(1, 100)):
    """
    Plot power spectrum of a single ICA component on given axis.
    
    Calculates and plots the power spectrum of an ICA component using
    windowed segments with overlap, displaying on a logarithmic scale.

    Args:
        component: Single ICA component signal array
        ax: Matplotlib axis object to plot on
        sfreq: Sampling frequency in Hz (required parameter)
        window_length: Window length in seconds (default: 1.0)
        overlap: Overlap fraction between windows (default: 0.5)
        freq_range: Frequency range tuple (min, max) in Hz (default: (1, 100))
    """
    freqs, all_psds, _ = calculate_individual_power_spectra(
        component, sfreq, window_length, overlap
    )
    psd_array = np.array(all_psds)
    avg_psd = np.mean(psd_array, axis=0)
    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    plot_freqs = freqs[freq_mask]
    plot_psd = avg_psd[freq_mask]
    ax.semilogy(plot_freqs, plot_psd, color='b', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (pT²/Hz)')
    ax.grid(True, alpha=0.3)



def plot_ica_power_spectra_grid(ica_components, plot_power_spectrum_func, component_names=None, title="ICA Power Spectra (4x4)"):
    """
    Plot power spectra for multiple ICA components in a 4x4 grid layout.

    Args:
        ica_components: Array of shape (n_components, n_samples)
        plot_power_spectrum_func: Function to plot power spectrum with signature (component, ax)
        component_names: List of component names (default: None, auto-generated)
        title: Figure title (default: "ICA Power Spectra (4x4)")
    """
    n_components = min(16, ica_components.shape[0])
    if component_names is None:
        component_names = [f"ICA {i+1}" for i in range(ica_components.shape[0])]
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    fig.suptitle(title)
    axes = axes.flatten()
    for i in range(16):
        ax = axes[i]
        if i < n_components:
            plot_power_spectrum_func(ica_components[i], ax)
            ax.set_title(component_names[i])
        else:
            ax.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_ica_components(ica_signals, time, axis_label, rec_label):
    
    """
    Plot ICA component time series in vertically stacked subplots.
    
    Creates a vertical stack of subplots, one for each ICA component,
    showing their time series evolution.
    
    Args:
        ica_signals: Array of shape (n_components, n_samples)
        time: Time vector for x-axis
        axis_label: Label for the axis/component type
        rec_label: Recording label for the title
    """
    n_components = ica_signals.shape[0]
    n_cols = 1
    n_rows = n_components
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 2.2 * n_rows), sharex=True)
    if n_components == 1:
        axes = [axes]
    for i in range(n_components):
        axes[i].plot(time, ica_signals[i])
        axes[i].set_ylabel(f'C{i+1}')
    fig.suptitle(f'ICA Components ({axis_label} axis) - {rec_label}', fontsize=10)
    plt.xlabel('Time (s)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()