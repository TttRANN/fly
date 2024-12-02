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
print(f"Total frames: {len(df['Trial_Num'])}")

# Prepare lists to store trials
clockwise_trials = []
counter_clockwise_trials = []

# Define pre-stimulation and post-stimulation frame counts
pre_stim_frames = 10  # Frames before stimulation starts
post_stim_frames = 30  # Frames after stimulation starts

# Process each trial
for trial in df['Trial_Num'].unique():
    if trial == 0:
        continue  # Skip data before the first trial starts
    # Get trial data
    trial_data = df[df['Trial_Num'] == trial]
    trial_start_idx = trial_data.index[0]
    trial_end_idx = trial_data.index[-1]
    # Get frames where 'Stimulus' is 'on', limit to post_stim_frames
    stim_on_data = trial_data[trial_data['Stimulus'] == True].iloc[:post_stim_frames]
    # Get pre_stim_frames before the trial starts
    pre_stim_data = df.loc[max(0, trial_start_idx - pre_stim_frames):trial_start_idx - 1]
    # Combine data
    trial_df = pd.concat([pre_stim_data, stim_on_data])
    # Align trajectories at stimulation start (x=0, y=0 at stim start)
    trial_df = trial_df.copy()
    trial_df.reset_index(drop=True, inplace=True)
    stim_start_idx = len(pre_stim_data)  # Index where stimulation starts in trial_df
    alignment_x = trial_df['x'].iloc[stim_start_idx]
    alignment_y = trial_df['y'].iloc[stim_start_idx]
    trial_df['x_aligned'] = trial_df['x'] - alignment_x
    trial_df['y_aligned'] = trial_df['y'] - alignment_y
    # Generate time array for this trial (relative to stim start)
    total_length = len(trial_df)
    time_array = np.arange(-stim_start_idx, total_length - stim_start_idx)
    # Flip trials that end with negative y_aligned
    if trial_df['y_aligned'].iloc[-1] > 0:
        # Get 'Clockwise' value for this trial
        clockwise_value = trial_df.iloc[stim_start_idx]['Clockwise']
        trial_dict = {
            'x_aligned': trial_df['x_aligned'].values,
            'y_aligned': trial_df['y_aligned'].values,
            'time': time_array
        }
        if clockwise_value:
            clockwise_trials.append(trial_dict)
        else:
            counter_clockwise_trials.append(trial_dict)
    elif trial_df['y_aligned'].iloc[-1] < 0:
        # Flip x_aligned and y_aligned
        trial_df['x_aligned'] = -trial_df['x_aligned']
        trial_df['y_aligned'] = -trial_df['y_aligned']
        # Get 'Clockwise' value for this trial
        clockwise_value = trial_df.iloc[stim_start_idx]['Clockwise']
        trial_dict = {
            'x_aligned': trial_df['x_aligned'].values,
            'y_aligned': trial_df['y_aligned'].values,
            'time': time_array
        }
        if clockwise_value:
            clockwise_trials.append(trial_dict)
        else:
            counter_clockwise_trials.append(trial_dict)
    # Else, if y_aligned.iloc[-1] == 0, you may decide what to do (include or exclude)

# Function to plot trajectories with color-coded time information
def plot_trajectories(trials_list, title):
    fig, ax = plt.subplots(figsize=(10, 8))
    all_x = []
    all_y = []
    all_time = []

    # Loop over each trial
    for trial_data in trials_list:
        x = trial_data['x_aligned']
        y = trial_data['y_aligned']
        time_array = trial_data['time']

        # Filter out any NaN values
        valid_indices = ~np.isnan(x) & ~np.isnan(y)
        x = x[valid_indices]
        y = y[valid_indices]
        time_array = time_array[valid_indices]

        if len(x) == 0:
            continue  # Skip if no valid data

        # Plot individual trajectories
        # Before stimulation: light semi-transparent red
        pre_stim_indices = time_array < 0
        ax.plot(x[pre_stim_indices], y[pre_stim_indices], color='red', alpha=0.15)

        # During stimulation: light semi-transparent green
        post_stim_indices = time_array >= 0
        ax.plot(x[post_stim_indices], y[post_stim_indices], color='green', alpha=0.15)

        all_x.append(x)
        all_y.append(y)
        all_time.append(time_array)

    if not all_x or not all_y:
        print("No valid trajectories to plot.")
        return

    # Compute the maximum length of the trajectories
    max_length = max(len(x) for x in all_x)

    # Pad trajectories and time arrays to ensure equal length arrays
    padded_x = np.array([np.pad(x, (0, max_length - len(x)), 'edge') for x in all_x])
    padded_y = np.array([np.pad(y, (0, max_length - len(y)), 'edge') for y in all_y])
    padded_time = np.array([np.pad(t, (0, max_length - len(t)), 'edge') for t in all_time])

    # Compute the mean trajectory and mean time
    mean_x = np.nanmean(padded_x, axis=0)
    mean_y = np.nanmean(padded_y, axis=0)
    mean_time = np.nanmean(padded_time, axis=0)

    # Check for NaNs
    if np.isnan(mean_x).any() or np.isnan(mean_y).any():
        print("NaN values encountered in the mean calculation. Check input data.")
        return

    # Normalize the time indices for color mapping
    min_time = mean_time[0]
    max_time = mean_time[-1]
    norm = plt.Normalize(vmin=min_time, vmax=max_time)
    cmap = plt.colormaps['copper']

    # Plot the mean trajectory with color-coded time information
    for i in range(len(mean_x) - 1):
        ax.plot([mean_x[i], mean_x[i + 1]], [mean_y[i], mean_y[i + 1]],
                color=cmap(norm(mean_time[i])), linewidth=2)

    # Add a color bar to indicate time progression
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Needed for ScalarMappable
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Time (Frames)')

    # Finalize the plot
    ax.set_title(title)
    ax.set_xlabel('x position (aligned)')
    ax.set_ylabel('y position (aligned)')
    ax.grid(True)
    plt.xlim(left=-50)
    plt.xlim(right=50)
    plt.ylim(top=50)
    plt.ylim(bottom=-50)
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
