# MEG-EMG Analysis Pipeline

This repository contains the complete pipeline for analyzing OPM-MEG data.

## Project Structure

```
Code/
├── configs/                    # Configuration files (JSON)
│   └── sub-91_config.json     # Subject-specific parameters
├── source/                     # Core analysis functions
│   ├── find_paths.py          # Path management utilities
│   ├── MEG_analysis_functions.py  # MEG processing functions
│   ├── plot_functions.py      # Visualization utilities
│   └── trimmer_functions.py   # Data trimming functions
├── MEG-trimmer.py             # Main trimming workflow
├── MEG-h5-synchro.py          # MEG-EMG synchronization
├── MEG-Analysis.py            # Comprehensive MEG analysis
├── read-datas.py              # Data reading and preprocessing
└── README.md                  # This file
```

## Dependencies & Requirements

### Python Libraries
```bash
pip install mne pandas numpy matplotlib scipy scikit-learn h5py
```

### Required Packages:
- **MNE-Python** (>=1.0): Neurophysiological data analysis
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Plotting and visualization
- **scipy**: Scientific computing (signal processing)
- **scikit-learn**: Machine learning (FastICA)
- **h5py**: HDF5 file handling

### System Requirements:
- Python 3.8+
- macOS/Linux/Windows
- 8GB+ RAM (for large MEG datasets)
- OneDrive sync (for automatic path detection)

## Data Structure

The pipeline expects data organized as follows:
```
LID_MEG/
└── data/
    ├── source_data/        # Raw acquisition files
    │   └── sub-XX/
    │       └── OPM_data/   # .con files (MEG)
    ├── processed_data/     # Preprocessed files
    │   └── sub-XX/
    │       └── EMG_ACC_data/  # .h5 files (EMG/ACC)
    └── raw_data/          # Pipeline outputs
        └── sub-XX/
            └── OPM_data/  # Trimmed .fif files
```

## Module Descriptions

### `/configs/` - Configuration Files
JSON files containing all parameters, file paths, and settings to eliminate hardcoded values.

**Structure:**
- `dataset_info`: Subject and dataset metadata
- `file_paths`: Input/output file locations
- `recording_parameters`: Hardware settings (sampling rates, channels)
- `channel_mappings`: EMG, ACC, and MEG channel assignments
- `processing_settings`: Analysis control flags

**Usage:**
```python
config = load_config('sub-91')
trigger_index = config['recording_parameters']['meg']['trigger_channel_index']
```

### `/source/` - Core Functions

#### `find_paths.py`
Device-independent path management for OneDrive synced folders.

**Key Functions:**
- `get_onedrive_path(folder)`: Automatic OneDrive folder detection
- `get_available_subs(data, folder)`: List available subjects

#### `MEG_analysis_functions.py`
Core MEG processing utilities.

**Key Functions:**
- `apply_meg_filters()`: Bandpass and notch filtering
- `create_meg_raw()`: MNE RawArray creation
- `apply_fastica_to_channels()`: Independent Component Analysis
- `calculate_channel_norms()`: Vector magnitude calculation

#### `plot_functions.py`
Comprehensive visualization toolkit.

**Key Functions:**
- `plot_channels_comparison()`: Raw vs filtered comparisons
- `plot_meg_2x3_grid()`: Multi-window time series plots
- `plot_ica_components()`: ICA component visualization
- `plot_all_channel_power_spectra()`: Frequency domain analysis

#### `trimmer_functions.py`
Data synchronization and trimming utilities.

**Key Functions:**
- `trim_meg_from_trigger_minimum()`: Remove pre-recording artifacts
- `trim_meg_to_match_emg_duration()`: Temporal alignment

## Main Scripts

### `MEG-trimmer.py` - Data Synchronization
**Purpose:** Synchronizes MEG and EMG recordings by trimming MEG data to match EMG duration.

**Workflow:**
1. Load MEG (.con) and EMG (.h5) files
2. Find first trigger minimum to remove pre-recording artifacts
3. Trim MEG to match EMG duration for temporal alignment
4. Save synchronized data as .fif files


### `MEG-h5-synchro.py` - Advanced Synchronization
**Purpose:** Detailed MEG-EMG synchronization with ICA analysis.

**Features:**
- Trigger-based alignment
- Independent Component Analysis
- Artifact identification and removal
- Multi-channel visualization

### `MEG-Analysis.py` - Comprehensive Analysis
**Purpose:** Complete MEG analysis pipeline with filtering, ICA, and visualization.

**Features:**
- Multi-recording comparison
- Power spectral analysis
- ICA artifact removal
- Channel selection and exclusion

### `read-datas.py` - Data Import & Preprocessing
**Purpose:** Raw data import and initial preprocessing.

**Features:**
- .lvm and .con file reading
- Initial filtering and conversion
- Data synchronization
- Quality assessment plots

## Configuration System

The pipeline uses JSON configuration files to eliminate hardcoded values:


## Typical Workflow

1. **Data Preparation:**
   ```bash
   # Organize data in expected folder structure
   # Create subject-specific config file
   ```

2. **Data Synchronization:**
   ```python
   python MEG-trimmer.py
   # Output: Synchronized .fif files
   ```

3. **Analysis:**
   ```python
   python MEG-h5-synchro.py  # Advanced synchronization + ICA
   python MEG-Analysis.py    # Comprehensive analysis + ICA
   ```

4. **Visualization:**
   - Automatic plots generated during analysis
   - Power spectra, time series, ICA components
   - Raw vs filtered comparisons

## File Formats

**Input:**
- `.con`: Raw MEG data (KIT format)
- `.h5`: Processed EMG/ACC data (HDF5 format)
- `.lvm`: Raw MEG data (LabVIEW format)

**Output:**
- `.fif`: MNE-compatible MEG data
- `.png`: Analysis plots and figures

## Analysis Outputs

The pipeline generates:
- **Synchronized datasets** (.fif files)
- **Quality metrics** (timeline comparisons, data integrity checks)
- **Visualization plots** (time series, power spectra, ICA components)
- **Processed data** ready for downstream analysis

## Research Applications

This pipeline supports analysis of:
- **Movement disorders** (dystonia, dyskinesia)
- **Motor cortex activity** (MEG signals)
- **Signal synchronisation:** (MEG - EMG - ACC signals)


## Contact & Support

For issues with specific analyses or data formats, refer to:
- MNE-Python documentation: https://mne.tools/
- Configuration examples in `/configs/`
- Function docstrings in `/source/`

---

**Last Updated:** July 23, 2025  
**Version:** 1.0  
**Compatible with:** MNE-Python 1.0+, Python 3.8+