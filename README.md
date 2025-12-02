# lid_opm
Exploring cortical patterns of levodopa-induced dyskinesia using OPM-MEG.

## Overview

This repository contains a standardized and reusable pipeline for the preprocessing of OPM-MEG data. The analysis is conducted using Python within a Jupyter Notebook environment, leveraging the MNE-Python library.

The primary goal is to provide a robust workflow for cleaning OPM-MEG recordings. The pipeline is designed to be modular, using a main notebook that sources parameters from configuration files and utility functions from a `source` directory.

The current pipeline implements the following preprocessing steps:
1.  **Data Loading**: Loads raw OPM-MEG data in `.fif` format.
2.  **Dynamic Channel Selection**: Automatically identifies MEG channels based on naming conventions (e.g., `_by`, `_bz`).
3.  **Resampling**: Downsamples the data to a target sampling frequency.
4.  **Filtering**: Applies band-pass and notch filters to remove noise and power-line interference.
5.  **Artifact Suppression**: Uses Homogeneous Field Correction (HFC) to reduce external magnetic field interference.
6.  **Visualization**: Generates plots to compare raw and processed data, ensuring the quality of the preprocessing steps.

## Project Structure

The repository is organized as follows:

```
.
├── configs/
│   ├── config_sub95.json         # Subject-specific settings (paths, tasks)
│   └── preproc_settings_v1.json  # General preprocessing parameters (filters, HFC)
│
├── notebooks/
│   └── sub-95_preprocessing.ipynb  # Main notebook implementing the pipeline
│
├── source/
│   ├── find_paths.py             # Utility for locating data paths
│   └── plot_functions.py         # Functions for creating visualizations
│
└── README.md                     # This file
```

## Dependencies and Setup

### Recommended Environment
The pipeline was developed and tested with the following package versions:
-   **Python**: `3.12.7`
-   **mne**: `1.8.0`
-   **numpy**: `1.26.4`
-   **matplotlib**: (latest version recommended)
-   **jupyter**: (latest version recommended)

### Installation
It is highly recommended to use a dedicated `conda` environment.

**Option A: Install from an environment file**

If a `lidopm_env.yml` file is available, you can create the environment with:
```bash
conda create --name meg-env --file lidopm_env.yml
```
*(To generate this file from an existing environment, use: `conda env export > lidopm_env.yml`)*

**Option B: Manual Installation**

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd lid_opm
    ```

2.  **Create and activate a new Conda environment:**
    ```bash
    conda create --name meg-env python=3.12
    conda activate meg-env
    ```

3.  **Install the required packages using pip:**
    ```bash
    pip install numpy==1.26.4 mne==1.8.0 matplotlib jupyter
    ```

## How to Run the Pipeline

1.  Ensure your Conda environment is activated (`conda activate meg-env`).
2.  Launch Jupyter Lab or Jupyter Notebook:
    ```bash
    jupyter lab
    ```
3.  Navigate to the `notebooks/` directory and open `sub-95_preprocessing.ipynb`.
4.  Modify the `TASK` variable in the notebook to select the desired recording session.
5.  Run the cells sequentially to execute the entire preprocessing workflow.

## Branch Information

**Note:** This branch (`dev_preprocess_exploration`) is based on `dev_preprocess` and is dedicated to exploring new test datasets and testing preprocessing workflows.

## Research Context

### Go/No-Go Task
The experimental design and analysis methods are inspired by the following publications:
-   Cao et al., *Neurobiol of Dis* 2024, [https://doi.org/10.1016/j.nbd.2024.106689](https://doi.org/10.1016/j.nbd.2024.106689)
-   Kühn et al., *Brain* 2004, [DOI: 10.1093/brain/awh106](https://doi.org/10.1093/brain/awh106)
-   Alegre et al., *Exp Neurol* 2013, [https://doi.org/10.1016/j.expneurol.2012.08.027](https://doi.org/10.1016/j.expneurol.2012.08.027)