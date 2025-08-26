"""
Configuration Management for MEG Preprocessing Pipeline

This module provides utilities for loading and managing subject-specific 
configuration files with support for multiple data types (source, raw, processed).
"""

import json
import numpy as np
import os
from typing import Dict, Any, List, Tuple
import mne
from mne.preprocessing import compute_proj_hfc
from MEG_analysis_functions import apply_meg_filters


def load_subject_config(subject_id: str, config_dir: str = '../configs', version: str = None) -> Dict[str, Any]:
    """
    Load configuration file for a specific subject and version.
    """
    config_file = os.path.join(config_dir, f'{subject_id}_config.json')
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in config file {config_file}: {e}")
    
    # If version is specified, select that version
    if version is not None:
        if version not in config:
            raise KeyError(f"Version '{version}' not found in config file {config_file}")
        config = config[version]
    
    return config

def load_and_display_config(subject_id: str, config_dir: str = '../configs', version: str = None, verbose: bool = True) -> Dict[str, Any]:
    """
    Convenience function to load configuration and display available options.
    """
    config = load_subject_config(subject_id, config_dir, version)
    display_available_options(config, verbose)
    return config


def display_available_options(config: Dict[str, Any], verbose: bool = True) -> None:
    """
    Display all available data types, setups, and conditions from configuration.
    """
    if verbose:
        print(f"Configuration loaded for {config.get('subject_id', 'Unknown')}")
        print("=" * 50)
        print("SELECTABLE PARAMETERS:")
        print(f"DATA_TYPES: {config['data_types']}")
        print(f"SETUPS: {config['setups']}")
        print("CONDITIONS:")
        for setup, conditions in config['conditions'].items():
            print(f"  {setup}: {conditions}")


def validate_file_selection(
    config: Dict[str, Any],
    data_type: str,
    setup: str,
    condition: str,
    time_point: int = None
) -> Tuple[str, Dict[str, str]]:
    """
    Validate selection and generate filename and path info.
    """
    # Validate data_type
    if data_type not in config['data_types']:
        raise ValueError(f"Data type '{data_type}' not found. Available: {config['data_types']}")
    
    # Validate setup
    if setup not in config['setups']:
        raise ValueError(f"Setup '{setup}' not found. Available: {config['setups']}")
    
    # Validate condition
    if condition not in config['conditions'][setup]:
        raise ValueError(f"Condition '{condition}' not found for {setup}. Available: {config['conditions'][setup]}")
    
    # Validate time_point if required by filename pattern
    filename_pattern = config.get("filename_pattern")
    if filename_pattern is not None and "{time}" in filename_pattern:
        if time_point is None:
            raise ValueError("time_point must be provided for this dataset.")
        if time_point not in config["time_points"]:
            raise ValueError(f"time_point '{time_point}' not found. Available: {config['time_points']}")
        filename = filename_pattern.format(
            setup=setup,
            condition=condition,
            time=time_point
        )
        # For sub-02, path info is just the data_path
        path_info = {
            'base_folder': config["data_path"]
        }
    else:
        # For sub-91 and similar, use file_structure from config
        file_structure = config['file_structure'][data_type]
        filename_pattern = file_structure['filename_pattern']
        filename = filename_pattern.format(
            subject_id=config['subject_id'],
            setup=setup,
            condition=condition
        )
        path_info = {
            'base_folder': file_structure['base_folder'],
            'subfolder': file_structure['subfolder'],
            'file_extension': file_structure['file_extension']
        }
    
    return filename, path_info



def preprocess_meg_data(raw, start_sfreq, target_sfreq, l_freq, h_freq, notch_freqs, hfc_components, verbose=True):
    """
    Apply standard MEG preprocessing pipeline: downsampling, HFC, then filtering.
    
    Args:
        raw (mne.io.Raw): Input MNE Raw object
        target_sfreq (int): Target sampling frequency in Hz - REQUIRED
        l_freq (int): Low frequency cutoff for bandpass filter in Hz - REQUIRED
        h_freq (int): High frequency cutoff for bandpass filter in Hz - REQUIRED
        notch_freqs (list): List of frequencies for notch filtering in Hz - REQUIRED
        hfc_components (int): Number of HFC components to compute - REQUIRED
        verbose (bool): Whether to print preprocessing information
        
    Returns:
        tuple: (raw_preprocessed, preprocessing_info)
    """

    # Make a copy of the raw data to avoid modifying the original:
    raw_copy = raw.copy()

    # Store original info
    original_sfreq = start_sfreq
    original_duration = raw_copy.times[-1]
    
    # STEP 1: DOWNSAMPLING
    if verbose:
        print("\n" + "="*50)
        print("STEP 1: DOWNSAMPLING")
        print("="*50)

    if raw_copy.info['sfreq'] != target_sfreq:
        raw_downsampled = raw_copy.resample(target_sfreq, verbose=False)
        if verbose:
            print(f"Downsampled from {original_sfreq} Hz to {target_sfreq} Hz")
    else:
        raw_downsampled = raw_copy.copy()
        if verbose:
            print(f"No downsampling needed (already at {target_sfreq} Hz)")
    
    # STEP 2: HOMOGENEOUS FIELD CORRECTION (HFC)
    if verbose:
        print("\n" + "="*50)
        print("STEP 2: HOMOGENEOUS FIELD CORRECTION (HFC)")
        print("="*50)
    
    try:
        # Compute HFC projectors
        proj_hfc = compute_proj_hfc(raw_downsampled.info, order=hfc_components)
        
        # Apply HFC projectors
        raw_hfc = raw_downsampled.copy()
        raw_hfc.add_proj(proj_hfc)
        raw_hfc.apply_proj()
        
        if verbose:
            print(f"Applied HFC with {len(proj_hfc)} components")
            
        
    except Exception as e:
        if verbose:
            print(f"HFC failed: {str(e)}")
            print("Continuing without HFC")
        raw_hfc = raw_downsampled.copy()
    
    # STEP 3: FILTERING (LINE NOISE REMOVAL)
    if verbose:
        print("\n" + "="*50)
        print("STEP 3: FILTERING (LINE NOISE REMOVAL)")
        print("="*50)

    hfc_data = raw_hfc.get_data()
    
    
    try:
        filtered_data = apply_meg_filters(
            data=hfc_data,
            sfreq=raw_hfc.info['sfreq'],
            l_freq=l_freq,
            h_freq=h_freq
        )
        
        raw_preprocessed = mne.io.RawArray(
            filtered_data,
            raw_hfc.info.copy(),
            verbose=False
        )
        
        if verbose:
            print(f"Applied bandpass filter: {l_freq}-{h_freq} Hz")
            print(f"Applied notch filters at: {notch_freqs} Hz")
        
    except Exception as e:
        if verbose:
            print(f"Filtering failed: {str(e)}")
        raw_preprocessed = raw_hfc
    
    return raw_preprocessed

def remove_ica_artifacts(preprocessed_channels, ica_signals, ica_model, artifact_components, verbose=True):
    """
    Remove specified ICA components (artifacts) from preprocessed MEG data.
    
    Args:
        preprocessed_channels (list): List of preprocessed MEG channel arrays
        ica_signals (np.array): ICA components (n_components, n_times)
        ica_model: Fitted ICA model from sklearn
        artifact_components (list): List of component indices to remove (1-based: ICA-1, ICA-2, etc.)
        verbose (bool): Print removal information
        
    Returns:
        list: MEG channels with artifacts removed
    """
    # Convert 1-based to 0-based indexing
    artifact_indices = [comp - 1 for comp in artifact_components]

    # Convert preprocessed channels to array format
    meg_data_array = np.array(preprocessed_channels)
    
    # Create a copy of ICA signals for modification
    ica_clean = ica_signals.copy()
        
    # Zero out the artifact components
    for idx, component_idx in enumerate(artifact_indices):
        if 0 <= component_idx < ica_clean.shape[0]:
            ica_clean[component_idx, :] = 0
            if verbose:
                print(f" ICA-{artifact_components[idx]} removed")
        else:
            print(f"Warning: ICA-{artifact_components[idx]} out of range (max: ICA-{ica_clean.shape[0]})")
    
    # Reconstruct the cleaned data using the inverse ICA transformation
    cleaned_data = ica_model.inverse_transform(ica_clean.T).T
    
    # Convert back to list format for consistency
    cleaned_channels = [cleaned_data[i] for i in range(cleaned_data.shape[0])]
    
    return cleaned_channels

def preview_file_header(file_path, num_lines=23):
    """
    Prints the first `num_lines` of the file at `file_path`.
    """
    if os.path.exists(file_path):
        print(f"\nPreview of the first {num_lines} lines in file:")
        with open(file_path, "r") as f:
            for i in range(num_lines):
                line = f.readline()
                if not line:
                    break
                print(line.strip())
    else:
        print("\nSelected file does not exist!")
        print(f"Checked path: {file_path}")


def extract_sample_interval(file_path):
    """
    Extracts the sample interval (Delta_X) from the header of a LabVIEW .lvm file.

    Args:
    file_path : str
        Path to the .lvm file.
    Returns:
    float
        The sample interval (Delta_X) in seconds.
    Raises:
    ValueError
        If Delta_X is not found in the file header.
    """
    with open(file_path, "r") as f:
        for line in f:
            if line.strip().startswith("Delta_X"):
                parts = line.strip().split('\t')
                for part in parts[1:]:
                    try:
                        return float(part)
                    except ValueError:
                        continue
                raise ValueError("No valid Delta_X value found in line.")
    raise ValueError("Delta_X not found in file header.")

def create_time_window_mask(time_array, start_time, window_duration):
    """
    Create a boolean mask and time window for a given time array.

    Parameters:
        time_array (np.ndarray): Array of time points.
        start_time (float): Start time of the window (in seconds).
        window_duration (float): Duration of the window (in seconds).

    Returns:
        mask (np.ndarray): Boolean mask for the time window.
        time_window (np.ndarray): Time points within the window.
    """
    mask = (time_array >= start_time) & (time_array <= start_time + window_duration)
    time_window = time_array[mask]
    return mask, time_window