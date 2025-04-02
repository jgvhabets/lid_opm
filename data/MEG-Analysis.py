#################
### LIBRARIES ###
#################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


#######################################################
#################
### FUNCTIONS ###
#################
#######################################################

# Function that calculates the norm of each channel;

def calculate_channel_norms(X_channels, Y_channels, Z_channels):
    """
    Calculate the Euclidean norm for each sensor using its X, Y, Z components.
    
    Args:
        X_channels: List of X component arrays for each sensor
        Y_channels: List of Y component arrays for each sensor
        Z_channels: List of Z component arrays for each sensor
    
    Returns:
        List of norm arrays for each sensor
    """
    norms = []
    n_channels = len(X_channels)
    
    for i in range(n_channels):
        # Calculate norm for each time point: sqrt(x² + y² + z²)
        norm = np.sqrt(
            X_channels[i]**2 + 
            Y_channels[i]**2 + 
            Z_channels[i]**2
        )
        norms.append(norm)
    
    return norms


#################
### MAIN CODE ###
#################

#######################################################
#######################################################


# READING THE FILE AND DATAFRAME CREATION:

#file rec1:
file_path_1 = "plfp65/plfp65_rec1_13.11.2024_12-51-13_array1.lvm"
df_start = pd.read_csv(file_path_1, header= 22, sep='\t')

# file rec11:
file_path_11 = "plfp65/plfp65_rec11_13.11.2024_14-18-30_array1.lvm"
df_last = pd.read_csv(file_path_11, header=22, sep='\t')

# It seems that the column "Comment" is composed by Nan values, so I decide to remove it from the frame
df_start = df_start.drop(columns=["Comment"])
df_last = df_last.drop(columns=["Comment"])

# ORGANIZING DATAS
X_channels_names = [col for col in df_start.columns if "X" in col]
Y_channels_names = [col for col in df_start.columns if "Y" in col]
Z_channels_names = [col for col in df_start.columns if "Z" in col]

X_extras = ['X_Value', 'MUX_Counter1', 'MUX_Counter2']
X_channels_names = [col for col in X_channels_names if col not in X_extras]



# Extracting the channels from startting and last files:

X_channels_start = [df_start[col].values for col in X_channels_names]
Y_channels_start = [df_start[col].values for col in Y_channels_names]
Z_channels_start = [df_start[col].values for col in Z_channels_names]

X_channels_last = [df_last[col].values for col in X_channels_names]
Y_channels_last = [df_last[col].values for col in Y_channels_names]
Z_channels_last = [df_last[col].values for col in Z_channels_names]

# Excluding channels filled with zeros in both files:
X_channels_start = X_channels_start[:20]
Y_channels_start = Y_channels_start[:20]
Z_channels_start = Z_channels_start[:20]

X_channels_last = X_channels_last[:20]
Y_channels_last = Y_channels_last[:20]
Z_channels_last = Z_channels_last[:20]

X_channels_names = X_channels_names[:20]  # Also trim the names to match

print('After excluding the empty channels ')
print('we are considering: ', len(X_channels_names), ' channels for each component')

# Assign a specific color to each component
component_colors = {
    'X': 'blue',     # Blue for X component
    'Y': 'green',    # Green for Y component
    'Z': 'red',      # Red for Z component
    'Norm': 'purple' # Purple for the norm component
}

# Pick a sensor
time_start = df_start["X_Value"]
time_last = df_last["X_Value"]

#########################################################################
# Now let's normalize the data and plot the channels:
print("\nCalculating channel norms...")
norms_start = calculate_channel_norms(X_channels_start, Y_channels_start, Z_channels_start)
norms_last = calculate_channel_norms(X_channels_last, Y_channels_last, Z_channels_last)

# Plotting the normalized channels:

# Create figure with subplots for norm comparisons
n_channels = len(X_channels_names)
n_cols = 5
n_rows = (n_channels + n_cols - 1) // n_cols

# Create figure
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 15))
axes = axes.flatten()

# Plot norms for each channel
for i in range(n_channels):
    # Calculate means
    #mean_start = np.mean(norms_start[i])
    #mean_last = np.mean(norms_last[i])
    # Plot both norms on the same subplot
    axes[i].plot(time_start, norms_start[i], color='#1f77b4', label='plfp65_rec1', linewidth=1.5, alpha=0.7)
    axes[i].plot(time_last, norms_last[i], color='#ff7f0e', label='plfp65_rec11', linewidth=1.5, alpha=0.7)

    # Plot horizontal lines for means
    #axes[i].axhline(y=mean_start, color="black", label='plfp65_rec1 mean', linewidth=2.5, linestyle='--',)
    #axes[i].axhline(y=mean_last, color="dimgray",label='plfp65_rec11 mean', linewidth=2.5, linestyle='--',)
    
    
    # Add title and labels
    axes[i].set_title(f'Channel {X_channels_names[i]}', fontsize=10)
    axes[i].grid(True, alpha=0.3)
    
    # Add legend only for the first subplot
    if i == 0:
        axes[i].legend()
    
    # Add y-label for leftmost subplots
    if i % n_cols == 0:
        axes[i].set_ylabel('Magnitude')
    
    # Add x-label for bottom subplots
    if i >= n_channels - n_cols:
        axes[i].set_xlabel('Time (sec)')

# Hide unused subplots
for i in range(n_channels, len(axes)):
    axes[i].set_visible(False)

plt.suptitle('Channel Norms Comparison: plfp65_rec1 vs plfp65_rec11', fontsize=14)
plt.tight_layout()
plt.show()
