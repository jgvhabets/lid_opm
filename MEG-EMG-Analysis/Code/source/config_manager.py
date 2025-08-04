"""
Configuration Management for MEG Preprocessing Pipeline

This module provides utilities for loading and managing subject-specific 
configuration files with support for multiple data types (source, raw, processed).
"""

import json
import os
from typing import Dict, Any, List, Tuple


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