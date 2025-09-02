# MEG-EMG Analysis Pipeline

This repository contains the pipeline for analyzing OPM-MEG data, including preprocessing, synchronization, and visualization.

## Project Structure

```
Code/
├── configs/                    # Subject configuration files (JSON)
│   ├── sub-02_config.json
│   └── sub-91_config.json
├── source/                     # Core analysis modules
│   ├── config_manager.py
│   ├── find_paths.py
│   ├── MEG_analysis_functions.py
│   ├── plot_functions.py
│   └── trimmer_functions.py
├── notebook/                   # Jupyter notebooks for analysis
│   ├── sub-02_preprocessing.ipynb
│   └── sub-91_preprocessing.ipynb
└── README.md                   # This file
```

## Dependencies & Requirements

### Python Libraries
```bash
pip install mne pandas numpy matplotlib scipy scikit-learn h5py
```

### Required Packages:
- **MNE-Python** (>=1.8.0): Neurophysiological data analysis
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **matplotlib**: Visualization
- **scipy**: Signal processing
- **scikit-learn**: Machine learning (FastICA)
- **h5py**: HDF5 file handling

### System Requirements:
- Python 3.12+
- macOS/Linux/Windows
- 8GB+ RAM recommended

## Data Structure

Expected data organization:
```
LID_MEG/
└── data/
    ├── source_data/        # Raw acquisition files (.con, .lvm)
    ├── processed_data/     # Preprocessed files (.h5)
    └── raw_data/           # Pipeline outputs (.fif)
```

## Module Descriptions

### `/configs/` - Configuration Files
JSON files with subject-specific parameters, paths, and settings.

### `/source/` - Core Functions

- [`config_manager.py`](source/config_manager.py): Load and validate configuration, manage preprocessing pipeline.
- [`find_paths.py`](source/find_paths.py): Device-independent path management.
- [`MEG_analysis_functions.py`](source/MEG_analysis_functions.py): MEG processing (filtering, ICA, etc.).
- [`plot_functions.py`](source/plot_functions.py): Visualization utilities.
- [`trimmer_functions.py`](source/trimmer_functions.py): Data trimming and synchronization.

### `/notebook/` - Jupyter Notebooks

- [`sub-02_preprocessing.ipynb`](notebook/sub-02_preprocessing.ipynb): Preprocessing workflow for subject 02.
- [`sub-91_preprocessing.ipynb`](notebook/sub-91_preprocessing.ipynb): Preprocessing workflow for subject 91.

## Typical Workflow

1. **Configure subject parameters** in `/configs/`.
2. **Run preprocessing and analysis** using the notebooks in `/notebook/`.
3. **Use core functions** from `/source/` for filtering, ICA, trimming, and visualization.

## File Formats

**Input:**
- `.con`: Raw MEG data (KIT format)
- `.h5`: Processed EMG/ACC data (HDF5)
- `.lvm`: Raw MEG data (LabVIEW format)

**Output:**
- `.fif`: MNE-compatible MEG data
- `.png`: Plots and figures

## Analysis Outputs

- Synchronized and preprocessed datasets
- Quality metrics and integrity checks
- Visualization plots (time series, spectra, ICA components)
- Ready-to-analyze data for research

## Contact & Support

- MNE-Python documentation: https://mne.tools/
- Example configs in `/configs/`
- Function docstrings in `/source/`
