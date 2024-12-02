import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

# Replace 'data.csv' with the path to your CSV file
data_file = '/Users/tairan/Downloads/rnai_side-viii_t4t5_batch/rnai_side-viii_t4t5_batch1/rnai_side-viii_t4t5_batch1_filtered_1.csv'

# Read the data
df = pd.read_csv(data_file, names=['Frame', 'Timestamp', 'Arena', 'x', 'y', 'Orientation', 'Stimulus', 'Frequency', 'Clockwise'])

# Map 'Stimulus' column to boolean: 'on' -> True, 'off' -> False
df['Stimulus'] = df['Stimulus'].map({'on': True, 'off': False})

# Shift the 'Stimulus' column to compare with the previous row
df['Stimulus_prev'] = df['Stimulus'].shift(1, fill_value=False)

# Detect Trial Starts: Transition from False (off) to True (on)
df['Trial_Start'] = (~df['Stimulus_prev']) & df['Stimulus']

# Detect Trial Ends: Transition from True (on) to False (off)
df['Trial_End'] = df['Stimulus_prev'] & (~df['Stimulus'])

# Assign trial numbers
trial_num = 0
trial_nums = []
for idx, row in df.iterrows():
    if row['Trial_Start']:
        trial_num += 1
    trial_nums.append(trial_num)
df['Trial_Num'] = trial_nums
print(len(df['Trial_Num']))

# Prepare lists to store trials
clockwise_trials = []
counter_clockwise_trials = []

# Process each trial
for trial in df['Trial_Num'].unique():
    if trial == 0:
        continue  # Skip data before the first trial starts
    # Get trial data
    trial_data = df[df['Trial_Num'] == trial]
    trial_start_idx = trial_data.index[0]
    trial_end_idx = trial_data.index[-1]
    # Get frames where 'Stimulus' is 'on', limit to 30 frames
    stim_on_data = trial_data[trial_data['Stimulus'] == True].iloc[:30]
    # Get 10 frames before the trial starts
    pre_stim_data = df.loc[max(0, trial_start_idx - 10):trial_start_idx - 1]
    # Combine data
    trial_df = pd.concat([pre_stim_data, stim_on_data])
    # Align trajectories at x=0 and y=0
    trial_df = trial_df.copy()
    trial_df.reset_index(drop=True, inplace=True)
    trial_df['x_aligned'] = trial_df['x'] - trial_df.iloc[9]['x']
    trial_df['y_aligned'] = trial_df['y'] - trial_df.iloc[9]['y']
    # Get 'Clockwise' value for this trial
    clockwise_value = trial_df.iloc[0]['Clockwise']
    if clockwise_value:
        clockwise_trials.append(trial_df)
    else:
        counter_clockwise_trials.append(trial_df)

# Function to plot trajectories with color-coded time information
def plot_trajectories(trials_list, title):
    fig, ax = plt.subplots(figsize=(10, 8))
    all_x = []
    all_y = []
    
    # Loop over each trial
    for trial_df in trials_list:
        # Filter out any rows with NaN in x_aligned or y_aligned
        trial_df = trial_df.dropna(subset=['x_aligned', 'y_aligned'])
        
        if trial_df.empty:
            continue  # Skip if the DataFrame is empty after dropping NaN rows
        
        x = trial_df['x_aligned'].values
        y = trial_df['y_aligned'].values
        
        # Ensure that we're plotting only valid data
        ax.plot(x, y, color='grey', alpha=0.5)
        
        all_x.append(x)
        all_y.append(y)
    
    if not all_x or not all_y:
        print("No valid trajectories to plot.")
        return
    
    # Compute the maximum length of the trajectories
    max_length = max(len(x) for x in all_x)
    
    # Pad trajectories to ensure equal length arrays
    padded_x = np.array([np.pad(x, (0, max_length - len(x)), 'edge') for x in all_x])
    padded_y = np.array([np.pad(y, (0, max_length - len(y)), 'edge') for y in all_y])
    
    # Compute the mean trajectory
    mean_x = np.nanmean(padded_x, axis=0)  # Use nanmean to handle potential NaNs
    mean_y = np.nanmean(padded_y, axis=0)
    
    # Check if there are still any NaN values in the result
    if np.isnan(mean_x).any() or np.isnan(mean_y).any():
        print("NaN values encountered in the mean calculation. Check input data.")
        return
    
    # Generate time information for coloring
    time_indices = np.linspace(0, 1, len(mean_x))  # Normalized time values (0 to 1)
    # Update to new colormap method
    cmap = plt.colormaps['viridis']  # Use the updated colormap retrieval
    
    # Plot the mean trajectory with color-coded time information
    for i in range(len(mean_x) - 1):
        ax.plot([mean_x[i], mean_x[i + 1]], [mean_y[i], mean_y[i + 1]], 
                 color=cmap(time_indices[i]), linewidth=2)
    
    # Add a color bar to indicate time progression
    norm = plt.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Needed for ScalarMappable
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Time Progression (Normalized)')
    
    # Finalize the plot
    ax.set_title(title)
    ax.set_xlabel('x position (aligned)')
    ax.set_ylabel('y position (aligned)')
    ax.grid(True)
    plt.show()

# Plot Clockwise trials
if clockwise_trials:
    plot_trajectories(clockwise_trials, 'Clockwise Trials')
else:
    print("No Clockwise trials found.")

# Plot Counter-Clockwise trials
if counter_clockwise_trials:
    plot_trajectories(counter_clockwise_trials, 'Counter-Clockwise Trials')
else:
    print("No Counter-Clockwise trials found.")
