# lid_opm
Exploring cortical patterns of levodopa induced dyskinesia using OPM-MEG.


- EMG ACC analysis
This branch will have all of the code relating to the EMG-data collection, EMG-Analysis and also code for the Accelerometer. Excited to get started!

# Dependencies
`conda create -n lid_opm python numpy pandas scipy scikit-learn jupyter matplotlib`
additional packages:
`pip install mne h5io pymatreader`
`pip install pygame pylsl`  # for go/no-go task

package version:
- Python sys 3.12.7
- numpy 1.26.4
- mne 1.8.0


### Go/ No-Go Task
Inspired by methods used by Cao et al. Neurobiol of Dis 2024, https://doi.org/10.1016/j.nbd.2024.106689; Kühn et al. Brain 2004, DOI: 10.1093/brain/awh106; and Alegre et al. Exp Neurol 2013, https://doi.org/10.1016/j.expneurol.2012.08.027.