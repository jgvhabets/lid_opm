import math

import numpy as np
import mne
from scipy.signal import correlate
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

test = mne.io.read_raw_ant("C:/Users/User/Documents/bachelorarbeit/data/Accelerometer/"
                           "Test_Accelerometer_2025-04-22_14-22-47.cnt", preload=True)
full_array = test.get_data()

# taking only first 6 channels = ACC channels
ACC_array = full_array[0:6]

# notch filter and high-pass filtering
ACC_array = mne.filter.notch_filter(x=ACC_array[0:6], freqs=[50,100,150], Fs=1000)
low_filtered = mne.filter.filter_data(ACC_array, sfreq=1000, l_freq=1, h_freq=25)
low_filtered[5] *= -1

x_cha = low_filtered[5]
y_cha = low_filtered[0]
z_cha = low_filtered[1]
x_ant = low_filtered[2]
y_ant = low_filtered[3]
z_ant = low_filtered[4]

# euclidean norm:
norm_charite = np.sqrt(x_cha**2 + y_cha**2 + z_cha**2)
norm_ant = np.sqrt(x_ant**2 + y_ant**2 + z_ant**2)

# correlation coefficient
pairs = [[5,2, "X"], [0,3, "Y"], [1,4, "Z"]]
for pair in pairs:
    corr = np.corrcoef(low_filtered[pair[0]], low_filtered[pair[1]]) [0,1]
    print(f"Similarity for {pair[2]}: {corr:.2f}")

corr_norm = np.corrcoef(norm_charite, norm_ant) [0,1]
print(f"Similarity for Euclidean norm signals: {corr_norm:.2f}")

# plot for x-Axis Auto-corr & Cross-corr
ncc = correlate(low_filtered[5], low_filtered[2], mode='same', method='fft') / (np.linalg.norm(low_filtered[5]) * np.linalg.norm(low_filtered[2]))
lags = np.arange(-len(low_filtered[5])+1, len(low_filtered[2]))

plt.figure()
plt.plot(lags, correlate(low_filtered[5], low_filtered[5], mode='full'), label='Auto-corr: ACC-charite vs ACC-charite')
plt.plot(lags, correlate(low_filtered[5], low_filtered[2], mode='full'), label='Cross-corr: ACC-charite vs ACC-AntNeuro')
# plt.axvline(x=0, color='k', linestyle='--')
#plt.title("Cross-correlate ≈ Auto-correlate")
#plt.xlabel("lags")
#plt.ylabel("correlate")
#plt.legend()
#plt.savefig("Auto-Corr_Cross-corr.svg")
plt.show()


ncc = correlate(norm_charite, norm_ant, mode='same', method='fft') / (np.linalg.norm(norm_charite) * np.linalg.norm(norm_ant))
lags = np.arange(-len(norm_charite)+1, len(norm_ant))

plt.figure()
plt.plot(lags, correlate(norm_ant, norm_ant, mode='full'), label='Auto-corr: ACC-Ant vs ACC-Ant')
plt.plot(lags, correlate(norm_charite, norm_ant, mode='full'), label='Cross-corr: ACC-Ant vs ACC-Charite')
# plt.axvline(x=0, color='k', linestyle='--')
plt.title("Cross-correlate ≈ Auto-correlate")
plt.xlabel("lags")
plt.ylabel("correlate")
plt.legend()
#plt.savefig("Auto-Corr_Cross-corr.svg")
plt.show() # auschecken!!!!!


norm_charite = norm_charite / np.linalg.norm(norm_charite)
norm_ant = norm_ant / np.linalg.norm(norm_ant)


def normalized_cross_correlation(x, y, mode='full'):
    """
    Berechnet normalisierte Cross-Korrelation: Werte liegen zwischen -1 und +1.
    """
    corr = correlate(x, y, mode=mode)

    auto_x = correlate(x, x, mode=mode)
    auto_y = correlate(y, y, mode=mode)

    norm_factor = np.sqrt(auto_x * auto_y)
    norm_factor[norm_factor == 0] = np.nan

    return corr / norm_factor


# Normierte Korrelationen
auto_corr = normalized_cross_correlation(norm_ant, norm_ant)
cross_corr = normalized_cross_correlation(norm_charite, norm_ant)

lags = np.arange(-len(norm_ant) + 1, len(norm_ant))

# --------------------------
# plot
plt.figure(figsize=(10, 5))
plt.plot(lags, auto_corr, label='Auto-Corr: Ant vs Ant')
plt.plot(lags, cross_corr, label='Cross-Corr: Charité vs Ant')
plt.title('Normalisierte Auto- und Cross-Korrelation')
plt.xlabel('lags')
plt.ylabel('correlation coefficient')
plt.legend()
plt.grid()
plt.show() ## iwas ist komisch hier