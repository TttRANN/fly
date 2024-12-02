import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import os
import glob

# Function to correct OpenCV theta assignment
def correct_opencv_theta_assignment(df):
    """
    Corrects the 'theta' angle assignment from OpenCV tracking.
    """
    df['direction'] = df['direction'].astype(float)
    if df['direction'].sum() > 0:
        factor = -1
    else:
        factor = 1
    cumulative_theta = df['ori'].iloc[0]
    theta_corr = [cumulative_theta * factor]
    for difference in df['ori'].diff().fillna(0).iloc[1:]:
        if difference > 90:
            difference -= 180
        elif difference < -90:
            difference += 180
        cumulative_theta += difference
        theta_corr.append(cumulative_theta * factor)
    df = df.assign(theta_corr=theta_corr)
    return df

# Updated Function to align smoothed trajectory and set stimulus onset at (0, 0)
def align_smoothed_trajectory(trial_df, stim_start_idx, window_length=11, polyorder=3):
    """
    Smooths the trajectory using Savitzky-Golay filter and aligns the trajectory such that the fly's
    movement is along the positive y-axis, with the stimulus onset at (0, 0).
    """
    # Smooth the trajectory
    trial_df['x_smoothed'] = savgol_filter(trial_df['pos_x'], window_length=window_length, polyorder=polyorder)
    trial_df['y_smoothed'] = savgol_filter(trial_df['pos_y'], window_length=window_length, polyorder=polyorder)

    # Check if we have enough data to compute the orientation 2 frames before stimulation onset
    if stim_start_idx < 3:
        # Not enough data to compute alignment
        print(f"Trial {trial_df.index[0]}: Not enough data to compute alignment.")
        trial_df['x_aligned'] = trial_df['x_smoothed'] - trial_df['x_smoothed'].iloc[stim_start_idx]
        trial_df['y_aligned'] = trial_df['y_smoothed'] - trial_df['y_smoothed'].iloc[stim_start_idx]
        return trial_df

    # Compute displacement vector between frames
    dx = trial_df['x_smoothed'].iloc[stim_start_idx] - trial_df['x_smoothed'].iloc[stim_start_idx - 1]
    dy = trial_df['y_smoothed'].iloc[stim_start_idx] - trial_df['y_smoothed'].iloc[stim_start_idx - 1]

    # Calculate angle and rotation matrix
    if np.hypot(dx, dy) < 1e-4:
        angle_to_align = 0.0
    else:
        angle_to_align = np.degrees(np.arctan2(dy, dx))
    rotation_angle = 90 - angle_to_align
    rotation_matrix = np.array([
        [np.cos(np.radians(rotation_angle)), -np.sin(np.radians(rotation_angle))],
        [np.sin(np.radians(rotation_angle)),  np.cos(np.radians(rotation_angle))]
    ])

    # Apply rotation to smoothed coordinates
    coords = np.c_[trial_df['x_smoothed'], trial_df['y_smoothed']]
    rotated_coords = np.dot(coords, rotation_matrix.T)

    # Shift coordinates to make stimulus onset (0, 0)
    x_shift = rotated_coords[stim_start_idx, 0]
    y_shift = rotated_coords[stim_start_idx, 1]
    trial_df['x_aligned'] = rotated_coords[:, 0] - x_shift
    trial_df['y_aligned'] = rotated_coords[:, 1] - y_shift

    return trial_df

# Function to determine if a fly is responsive
def is_fly_responsive(trial_data, speed_threshold=1):
    """
    Determines if the fly is responsive during the trial based on its movement speed.
    """
    x = trial_data['x_aligned']
    y = trial_data['y_aligned']
    frame_number = trial_data['frame_number']  # Original frame numbers in the trial

    stim_start_idx = 10  # Set this to the actual stimulus onset index for your data
    total_length = len(frame_number)  # Total number of frames in the trial

    # Create a time_array centered on the stimulus onset (0 at stim_start_idx)
    time_array = np.arange(-stim_start_idx, total_length - stim_start_idx)

    # Get indices before stimulation
    pre_stim_indices = time_array < 0

    # Check if there are enough data points before stimulation
    if np.sum(pre_stim_indices) < 1:
        # Not enough data before stimulation
        total_length_pre = 0
    else:
        # Compute displacements before stimulation
        dx_pre = np.diff(x[pre_stim_indices])
        dy_pre = np.diff(y[pre_stim_indices])

        # Compute distances between consecutive points
        distances_pre = np.sqrt(dx_pre**2 + dy_pre**2)

        # Compute total length before stimulation
        total_length_pre = np.sum(distances_pre)

    # If total length before stimulation is smaller than 3, fly is not responsive
    if total_length_pre < 2:
        return False

    # Get indices during stimulation
    stim_indices = time_array >= 0

    if np.sum(stim_indices) < 2:
        # Not enough data during stimulation
        return False

    # Compute displacements during stimulation
    dx = np.diff(x[stim_indices])
    dy = np.diff(y[stim_indices])

    # Compute speed (distance per frame)
    speed = np.sqrt(dx**2 + dy**2)

    # Compute mean speed
    mean_speed = np.mean(speed)

    # Check if mean speed exceeds threshold
    return mean_speed >= speed_threshold

# Plotting function
def plot_trajectories(trials_list, title, frames_before_stim=5, frames_after_stim=15):
    """
    Plots the trajectories of trials, aligning them at the stimulus onset and smoothing the data.
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    all_x_smoothed = []
    all_y_smoothed = []
    all_time = []

    for trial_data in trials_list:
        x_aligned = trial_data['x_aligned']
        y_aligned = trial_data['y_aligned']
        time_array = trial_data.get('time_array', np.arange(len(x_aligned)))

        # Filter out NaN values
        valid_indices = ~np.isnan(x_aligned) & ~np.isnan(y_aligned)
        x_aligned = x_aligned[valid_indices]
        y_aligned = y_aligned[valid_indices]
        time_array = time_array[valid_indices]

        if len(x_aligned) == 0:
            continue  # Skip if no valid data




        # Define the range of frames to plot around the stimulus onset
        stim_start_index = np.where(time_array == 0)[0][0] if 0 in time_array else 10
        start_frame = max(0, stim_start_index - frames_before_stim)
        end_frame = min(len(time_array), stim_start_index + frames_after_stim + 1)

        # Extract only the frames within the specified range
        x_plot = x_aligned[start_frame:end_frame]
        y_plot = y_aligned[start_frame:end_frame]
        time_plot = time_array[start_frame:end_frame]

        # Plot individual trajectories
        pre_stim_indices = time_plot <= 0
        ax.plot(x_plot[pre_stim_indices], y_plot[pre_stim_indices], color='red', alpha=0.3)
        post_stim_indices = time_plot >= 0
        ax.plot(x_plot[post_stim_indices], y_plot[post_stim_indices], color='green', alpha=0.3)

        all_x_smoothed.append(x_plot)
        all_y_smoothed.append(y_plot)
        all_time.append(time_plot)

    if not all_x_smoothed or not all_y_smoothed:
        print("No valid smoothed trajectories to plot.")
        return

    # Compute the maximum length of the trajectories
    max_length = max(len(x) for x in all_x_smoothed)

    # Pad trajectories and time arrays to ensure equal length arrays
    padded_x_smoothed = np.array([np.pad(x, (0, max_length - len(x)), 'edge') for x in all_x_smoothed])
    padded_y_smoothed = np.array([np.pad(y, (0, max_length - len(y)), 'edge') for y in all_y_smoothed])
    padded_time = np.array([np.pad(t, (0, max_length - len(t)), 'edge') for t in all_time])

    # Compute the mean trajectory and mean time
    mean_x_smoothed = np.nanmean(padded_x_smoothed, axis=0)
    mean_y_smoothed = np.nanmean(padded_y_smoothed, axis=0)
    mean_time = np.nanmean(padded_time, axis=0)

    # Normalize the time indices for color mapping
    min_time = mean_time[0]
    max_time = mean_time[-1]
    norm = plt.Normalize(vmin=min_time, vmax=max_time)
    cmap = plt.cm.viridis

    # Plot the mean trajectory with color-coded time information
    for i in range(len(mean_x_smoothed) - 1):
        ax.plot([mean_x_smoothed[i], mean_x_smoothed[i + 1]], [mean_y_smoothed[i], mean_y_smoothed[i + 1]],
                color=cmap(norm(mean_time[i])), linewidth=2)

    # Add a color bar to indicate time progression
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Time (Aligned to Stimulus Onset)')

    # Finalize the plot
    ax.set_title(f'{title} (Smoothed Trajectories)')
    ax.set_xlabel('x position (aligned)')
    ax.set_ylabel('y position (aligned)')
    ax.grid(True)
    plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import os
import glob

# (Assuming the previously defined functions are here: correct_opencv_theta_assignment, align_smoothed_trajectory, is_fly_responsive, plot_trajectories)

# Main processing loop
folder_path = '/Users/tairan/Downloads/test_roshan/'  # Update this to your folder path

# Process each CSV file in the folder
for file_path in glob.glob(f"{folder_path}/*.csv"):
    file_name = os.path.basename(file_path)
    file_template_name = 'Users/tairan/Downloads/test_roshan/R42F06-Gal4_control_4_2024116Dark.csv'
    df = pd.read_csv(file_path, names=["pos_x", "pos_y", "ori", "timestamp", "frame_number", "video_frame", "direction"])
    df1 = pd.read_csv(file_template_name, names=["pos_x", "pos_y", "ori", "timestamp", "frame_number", "video_frame", "direction"])
    df = df.apply(pd.to_numeric, errors='coerce')
    df1 = df1.apply(pd.to_numeric, errors='coerce')
    trials_list = []  # Reset trials list for each file
    trials_list2 = []

    # Iterate through the data to find the start and end points of trials
    for i in range(len(df["direction"]) - 1):
        if df["direction"].iloc[i] == 0 and df["direction"].iloc[i+1] == 1.0:
            start_point = i
        elif df["direction"].iloc[i] == 1.0 and df["direction"].iloc[i+1] == 0:
            end_point = i
            df_trial = df.iloc[start_point:end_point+1].reset_index(drop=True)

            # Correct 'theta' and align the smoothed trajectory
            df_trial = correct_opencv_theta_assignment(df_trial)
            stim_start_idx = 10  # Set stimulus onset index appropriately
            df_trial = align_smoothed_trajectory(df_trial, stim_start_idx)

            # Define the stimulus onset index
            # stim_start_idx is already defined above

            # Step 1: Separate pre- and post-stimulus frames
            pre_stim_indices = np.arange(0, stim_start_idx)
            post_stim_indices = np.arange(stim_start_idx, len(df_trial))

            # Step 2: Calculate cumulative path length and straightness for pre-stimulus frames
            pre_stim_x = df_trial['x_aligned'].values[pre_stim_indices]
            pre_stim_y = df_trial['y_aligned'].values[pre_stim_indices]
            pre_stim_path_diffs = np.sqrt(np.diff(pre_stim_x)**2 + np.diff(pre_stim_y)**2)
            pre_stim_cumulative_path_length = np.insert(np.cumsum(pre_stim_path_diffs), 0, 0)
            pre_stim_y_aligned = df_trial['y_aligned'].values[pre_stim_indices]
            epsilon = 0
            pre_stim_straightness = pre_stim_y_aligned / (pre_stim_cumulative_path_length + epsilon)

            # Step 3: Calculate cumulative path length and straightness for post-stimulus frames
            post_stim_x = df_trial['x_aligned'].values[post_stim_indices]
            post_stim_y = df_trial['y_aligned'].values[post_stim_indices]
            post_stim_path_diffs = np.sqrt(np.diff(post_stim_x)**2 + np.diff(post_stim_y)**2)
            # Start cumulative path length from the last value of pre-stim cumulative path length
            if len(pre_stim_cumulative_path_length) > 0:
                start_cumulative_length = pre_stim_cumulative_path_length[-1]
            else:
                start_cumulative_length = 0
            post_stim_cumulative_path_length = np.insert(np.cumsum(post_stim_path_diffs), 0, start_cumulative_length)
            post_stim_y_aligned = df_trial['y_aligned'].values[post_stim_indices]
            
            post_stim_straightness = post_stim_y_aligned / (post_stim_cumulative_path_length + epsilon)
            post_stim_straightness[0] =0

            # Step 4: Concatenate pre- and post-stimulus straightness arrays
            straightness = np.concatenate([pre_stim_straightness, post_stim_straightness])

            # Add to df_trial for easy plotting
            df_trial['straightness'] = straightness
            df_trial['time_array'] = np.arange(-stim_start_idx, len(df_trial) - stim_start_idx)

            # Calculate total path length for the trial
            total_path_diffs = np.sqrt(np.diff(df_trial['x_aligned'].values)**2 + np.diff(df_trial['y_aligned'].values)**2)
            path_length = np.sum(total_path_diffs)

            # Create 'trial_data' dictionary
            trial_data = {
                'x_aligned': df_trial['x_aligned'].values,
                'y_aligned': df_trial['y_aligned'].values,
                'frame_number': df_trial['frame_number'].values,
                'path_length': path_length,
                'straightness': df_trial['straightness'].values,
                'time_array': df_trial['time_array'].values
            }

            # Append the trial data to the list
            trials_list.append(trial_data)

    # After processing all trials, plot straightness across trials
    # Select responsive trials
    speed_threshold = 0.5  # Define speed threshold
    responsive_trials = [trial for trial in trials_list if is_fly_responsive(trial, speed_threshold)]

    if responsive_trials:
        # Plot each trial's straightness in grey and the average in black
        plt.figure(figsize=(10, 6))

        # Plot each individual trial in grey
        for trial in responsive_trials:
            plt.plot(trial['time_array'], trial['straightness'], color='grey', alpha=0.5)

        # Calculate and plot the average straightness across trials
        # First, find the common time axis
        min_time = np.min([np.min(trial['time_array']) for trial in responsive_trials])
        max_time = np.max([np.max(trial['time_array']) for trial in responsive_trials])
        common_time_array = np.arange(min_time, max_time + 1)

        # Initialize a list to hold straightness arrays aligned to common_time_array
        straightness_matrix = []

        for trial in responsive_trials:
            # Initialize an array of length len(common_time_array) filled with NaNs
            straightness_array = np.full(len(common_time_array), np.nan)
            indices = trial['time_array'] - min_time  # Compute indices relative to common_time_array
            indices = indices.astype(int)
            # Ensure indices are within bounds
            valid_indices = (indices >= 0) & (indices < len(common_time_array))
            straightness_array[indices[valid_indices]] = trial['straightness'][valid_indices]
            straightness_matrix.append(straightness_array)

        # Convert to numpy array
        straightness_matrix = np.array(straightness_matrix)

        # Compute average straightness, ignoring NaNs
        average_straightness = np.nanmean(straightness_matrix, axis=0)

        # Plot the average straightness
        plt.plot(common_time_array, average_straightness, color='black', linewidth=2, label="Average Straightness")

        # Labeling the plot
        plt.xlabel("Time (frames relative to stimulus onset)")
        plt.ylabel("Straightness")
        plt.title("Straightness Across Trials")
        plt.legend()
        plt.ylim([-1, 1])
        plt.xlim([0, 100])

        plt.show()

        # Plot trajectories
        plot_trajectories(responsive_trials, f'{file_name}')
    else:
        print(f"No responsive trials found for {file_name}.")
