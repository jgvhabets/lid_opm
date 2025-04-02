#################
### LIBRARIES ###
#################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re  # For regular expressions to extract numbers

#######################################################
#################
### FUNCTIONS ###
#################
#######################################################


# Function to check if a channel is filled with zeros
def check_zero(channel_list, channel_names):
    c = 0
    for i, channel in enumerate(channel_list):
        if np.all(channel == 0):  # Check if there are 0's channels
            c = c + 1
            print("The channel ", channel_names[i], " is filled with 0's ")
        else:
            continue
    print("There are ", c, " empty channels.")

# Function to save each component's channels as subplots with consistent color
def save_component_subplots(channels, channel_names, time, component, figsize=(15, 12), n_cols=5):
   
    # Create 'plot' folder if it doesn't exist
    if not os.path.exists('plot'):
        os.makedirs('plot')
        print('** Directory -plot- created **')

    n_channels = len(channels)
    n_rows = (n_channels + n_cols - 1) // n_cols  # Calculate number of rows needed
    
    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True)
    axes = axes.flatten()  # Flatten for easier indexing
    
    # Get the color for this component
    color = component_colors[component]
    
    # Plot each channel in its own subplot
    for i, (channel, name) in enumerate(zip(channels, channel_names)):
        axes[i].plot(time, channel, color=color, linewidth=1)
        axes[i].set_title(name, fontsize=10)
        axes[i].grid(True, alpha=0.3)
        
        # Add y-label only for leftmost plots
        if i % n_cols == 0:
            axes[i].set_ylabel('Amplitude')
    
    # Hide unused subplots
    for i in range(n_channels, len(axes)):
        axes[i].set_visible(False)
    
    # Add x-label only for bottom plots
    for i in range(n_cols * (n_rows-1), n_cols * n_rows):
        if i < len(axes) and axes[i].get_visible():
            axes[i].set_xlabel('Time (sec)')
    
    # Add overall title
    plt.suptitle(f"{component} Channels - Individual Visualization", fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    # Save the plot in the 'plot' folder with the component's name
    save_path = os.path.join('plot', f"{component}_channels.png")
    plt.savefig(save_path)

    # Close the plot to release memory
    plt.close()
    print('*** The ', component, ' subplots image is in the PLOT folder ***')

# Function to extract channel number from channel name (X1, X2, etc.)
def extract_channel_number(name):
    # Use regex to find digits in the name
    match = re.search(r'\d+', name)
    if match:
        return match.group()
    else:
        return ""  # Return empty string if no digits found

# Function to calculate and visualize the norm
def calculate_and_save_norm(X_channels, Y_channels, Z_channels, time, channel_names, figsize=(15, 12), n_cols=5):
    # Create 'plot' folder if it doesn't exist
    if not os.path.exists('plot'):
        os.makedirs('plot')
        print('** Directory -plot- created **')
    
    # Number of channels
    n_channels = len(X_channels)
    n_rows = (n_channels + n_cols - 1) // n_cols  # Calculate number of rows needed
    
    # Calculate norm for each channel
    norm_channels = []
    for i in range(n_channels):
        # Calculate the Euclidean norm (sqrt(x² + y² + z²)) for each time point
        norm = np.sqrt(X_channels[i]**2 + Y_channels[i]**2 + Z_channels[i]**2)
        norm_channels.append(norm)
    
    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True)
    axes = axes.flatten()  # Flatten for easier indexing
    
    # Plot each norm channel in its own subplot
    for i, (norm_channel, name) in enumerate(zip(norm_channels, channel_names[:n_channels])):
        # Extract channel number from name (X1, X2, etc.)
        channel_num = extract_channel_number(name)
        if not channel_num:  # If no number found, use index
            channel_num = str(i+1)
            
        norm_name = f"Norm{channel_num}"
        
        axes[i].plot(time, norm_channel, color='purple', linewidth=1)
        axes[i].set_title(norm_name, fontsize=10)
        axes[i].grid(True, alpha=0.3)
        
        # Add y-label only for leftmost plots
        if i % n_cols == 0:
            axes[i].set_ylabel('Magnitude')
    
    # Hide unused subplots
    for i in range(n_channels, len(axes)):
        axes[i].set_visible(False)
    
    # Add x-label only for bottom plots
    for i in range(n_cols * (n_rows-1), n_cols * n_rows):
        if i < len(axes) and axes[i].get_visible():
            axes[i].set_xlabel('Time (sec)')
    
    # Add overall title
    plt.suptitle("Vector Norm Channels - Individual Visualization", fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    # Save the plot
    save_path = os.path.join('plot', "Norm_channels.png")
    plt.savefig(save_path)

    # Close the plot to release memory
    plt.close()
    print('*** The Norm subplots image is in the PLOT folder ***')
    
    return norm_channels

# Function to plot combined components
def plot_combined_components(X_channel, Y_channel, Z_channel, norm_channel, channel_num, time):
    plt.figure(figsize=(12, 8))
    plt.plot(time, X_channel, color='blue', linewidth=1, label='X')
    plt.plot(time, Y_channel, color='green', linewidth=1, label='Y')
    plt.plot(time, Z_channel, color='red', linewidth=1, label='Z')
    plt.plot(time, norm_channel, color='purple', linewidth=1.5, label='Norm')
    
    plt.title(f"Channel {channel_num} - All Components", fontsize=12)
    plt.xlabel('Time (sec)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the plot
    if not os.path.exists('plot'):
        os.makedirs('plot')
    save_path = os.path.join('plot', f"Channel_{channel_num}_combined.png")
    plt.savefig(save_path)
    plt.close()
    print(f"*** Combined visualization for Channel {channel_num} saved in the PLOT folder ***")

###############################################################################################
###############################################################################################


#################
### MAIN CODE ###
#################

#######################################################
#######################################################


# READING THE FILE AND DATAFRAME CREATION 
file_path = "plfp65/plfp65_rec1_13.11.2024_12-51-13_array1.lvm"
df = pd.read_csv(file_path, header=22, sep='\t')

time = df['X_Value']
print(len(time))
print(time[len(time)-1])

exit()

# It seems that the column "Comment" is composed by Nan values, so I decide to remove it from the frame
df = df.drop(columns=["Comment"])

# ORGANIZING DATAS
X_channels_names = [col for col in df.columns if "X" in col]
Y_channels_names = [col for col in df.columns if "Y" in col]
Z_channels_names = [col for col in df.columns if "Z" in col]

X_extras = ['X_Value', 'MUX_Counter1', 'MUX_Counter2']
X_channels_names = [col for col in X_channels_names if col not in X_extras]

print('Filtered X channels:', len(X_channels_names))

X_channels = [df[col].values for col in X_channels_names]
Y_channels = [df[col].values for col in Y_channels_names]
Z_channels = [df[col].values for col in Z_channels_names]

# CLEANING THE CHANNELS
print("*********** X ***********")
check_zero(X_channels, X_channels_names)
print("*********** Y ***********")
check_zero(Y_channels, Y_channels_names)
print("*********** Z ***********")
check_zero(Z_channels, Z_channels_names)

# After using check_zero() with every list of channels i found out that in every component
# the channels from 21 to 64 are filled with 0s.

# Excluding channels filled with zeros
X_channels = X_channels[:20]
Y_channels = Y_channels[:20]
Z_channels = Z_channels[:20]

X_channels_names = X_channels_names[:20]  # Also trim the names to match

# Assign a specific color to each component
component_colors = {
    'X': 'blue',     # Blue for X component
    'Y': 'green',    # Green for Y component
    'Z': 'red',      # Red for Z component
    'Norm': 'purple' # Purple for the norm component
}

# Pick a sensor
time = df["X_Value"]

# Create the three separate subplot figures
save_component_subplots(X_channels, X_channels_names, time, "X")
save_component_subplots(Y_channels, Y_channels_names, time, "Y")
save_component_subplots(Z_channels, Z_channels_names, time, "Z")

# Calculate and visualize the norm
print("*********** CALCULATING NORMS ***********")
norm_channels = calculate_and_save_norm(X_channels, Y_channels, Z_channels, time, X_channels_names)

# Plot a combined visualization for the first channel as an example
channel_index = 0  # First channel (index 0)
# Extract channel number from the name
channel_num = extract_channel_number(X_channels_names[channel_index])
if not channel_num:
    channel_num = "1"  # Default to 1 if no number found

plot_combined_components(
    X_channels[channel_index],
    Y_channels[channel_index],
    Z_channels[channel_index],
    norm_channels[channel_index],
    channel_num,
    time
)

print(len(norm_channels))