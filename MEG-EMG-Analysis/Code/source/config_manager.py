"""
Configuration Management for MEG Preprocessing Pipeline

This module provides utilities for loading and managing subject-specific 
configuration files with support for multiple data types (source, raw, processed).
"""

import json
import os
from typing import Dict, Any, List, Tuple
import mne
from mne.preprocessing import compute_proj_hfc
from MEG_analysis_functions import apply_meg_filters


def load_subject_config(subject_id: str, config_dir: str = '../configs') -> Dict[str, Any]:
    """
    Load configuration file for a specific subject.
    """
    config_file = os.path.join(config_dir, f'{subject_id}_config.json')
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in config file {config_file}: {e}")
    
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


def validate_file_selection(config: Dict[str, Any], data_type: str, setup: str, condition: str) -> Tuple[str, Dict[str, str]]:
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
    
    # Generate filename using pattern
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


def load_and_display_config(subject_id: str, config_dir: str = '../configs', 
                           verbose: bool = True) -> Dict[str, Any]:
    """
    Convenience function to load configuration and display available options.
    """
    config = load_subject_config(subject_id, config_dir)
    display_available_options(config, verbose)
    return config

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