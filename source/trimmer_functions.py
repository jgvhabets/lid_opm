import numpy as np
from scipy.signal import find_peaks

def trim_meg_from_trigger_minimum(raw_meg, trigger_channel_index, threshold_factor=0.5, min_distance_sec=0.1):
    """
    Trim MEG data to start from the first deep minimum of the trigger channel.
    
    Parameters:
    -----------
    raw_meg : mne.io.Raw
        The raw MEG data object
    trigger_channel_index : int,
        Index of the trigger channel (157th channel, 0-based index 156)
    threshold_factor : float, default=0.5
        Factor to determine "deep" minima (minima below threshold_factor * global_min)
    min_distance_sec : float, default=0.1
        Minimum distance between peaks in seconds
    
    Returns:
    --------
    trimmed_raw : mne.io.Raw
        MEG data trimmed to start from first deep minimum
    t_zero_idx : int
        Index of the first deep minimum in original data
    t_zero_time : float
        Time (in seconds) of the first deep minimum
    """
    
    # Extract trigger channel
    trigger_channel = raw_meg.get_data(picks=[trigger_channel_index])[0]
    sfreq = raw_meg.info["sfreq"]
    
    # Find all minima as peaks (inverted signal)
    min_distance_samples = int(sfreq * min_distance_sec)
    all_minima, properties = find_peaks(-trigger_channel, distance=min_distance_samples)
    
    # Get the global minimum value
    global_min = np.min(trigger_channel)
    
    # Only keep minima that are less than threshold_factor * global_min (deep enough)
    threshold = threshold_factor * global_min
    deep_minima = [idx for idx in all_minima if trigger_channel[idx] < threshold]
    
    if not deep_minima:
        raise RuntimeError(f"No deep minima found in trigger channel. "
                         f"Adjust threshold_factor (current: {threshold_factor}) or check data.")
    
    # Take the first deep minimum as t=0
    t_zero_idx = deep_minima[0]
    t_zero_time = t_zero_idx / sfreq
    
    print(f"First robust trigger minimum found:")
    print(f"  - Index: {t_zero_idx}")
    print(f"  - Time: {t_zero_time:.3f} s")
    print(f"  - Value: {trigger_channel[t_zero_idx]:.2f}")
    print(f"  - Threshold used: {threshold:.2f}")
    
    # Trim the raw MEG data from t_zero_idx onwards
    trimmed_raw = raw_meg.copy().crop(tmin=t_zero_time)
    
    return trimmed_raw, t_zero_idx, t_zero_time

def trim_meg_to_match_emg_duration(raw_meg, time_difference):
    """
    Trim MEG data to match EMG duration by removing time from the end.
    
    Parameters:
    -----------
    raw_meg : mne.io.Raw
        The raw MEG data object
    time_difference : float
        Time difference to remove from MEG (EMG_timeline - MEG_timeline)
        If positive, MEG is longer and needs trimming
        If negative, EMG is longer (no trimming needed)
    
    Returns:
    --------
    trimmed_raw : mne.io.Raw
        MEG data trimmed to match EMG duration
    actual_time_removed : float
        Actual time removed (may differ slightly due to sampling)
    """
    
    if time_difference <= 0:
        print(f"No trimming needed. Time difference: {time_difference:.3f} seconds")
        return raw_meg.copy(), 0.0
    
    # Calculate new end time
    original_duration = raw_meg.times[-1]
    new_end_time = original_duration - time_difference
    
    if new_end_time <= 0:
        raise ValueError(f"Cannot trim {time_difference:.3f}s from {original_duration:.3f}s MEG data")
    
    # Crop the MEG data from start (0) to new_end_time
    trimmed_raw = raw_meg.copy().crop(tmin=0, tmax=new_end_time)
    
    actual_time_removed = original_duration - trimmed_raw.times[-1]
    
    print(f"MEG data trimmed:")
    print(f"  - Original duration: {original_duration:.3f} s")
    print(f"  - New duration: {trimmed_raw.times[-1]:.3f} s")
    print(f"  - Time removed: {actual_time_removed:.3f} s")
    
    return trimmed_raw, actual_time_removed