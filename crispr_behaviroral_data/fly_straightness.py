import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.signal import periodogram, find_peaks, savgol_filter
import os
import glob
from scipy.signal import savgol_filter
import glob
import os
import pandas as pd

# Function to align trajectory and set stimulus onset at (0, 0)
def align_trajectory(trial_df, stim_start_idx):
    # Check if we have enough data to compute the orientation 2 frames before stimulation onset
    if stim_start_idx < 3:
        # Not enough data to compute alignment
        print(f"Trial {trial_df['Trial_Num'].iloc[0]}: Not enough data to compute alignment.")
        # Shift so that the onset of the stimulation is at (0, 0)
        trial_df['x_aligned'] = trial_df['pos_x'] - trial_df['pos_x'].iloc[stim_start_idx]
        trial_df['y_aligned'] = trial_df['pos_y'] - trial_df['pos_y'].iloc[stim_start_idx]
        return trial_df

    # Compute displacement vector between frames (stim_start_idx - 3) and (stim_start_idx - 2)
    dx = trial_df['pos_x'].iloc[stim_start_idx ] - trial_df['pos_x'].iloc[stim_start_idx - 1]
    dy = trial_df['pos_y'].iloc[stim_start_idx ] - trial_df['pos_y'].iloc[stim_start_idx - 1]

    # If the fly hasn't moved, set angle to zero
    if np.hypot(dx, dy) < 1e-4:
        angle_to_align = 0.0
    else:
        # Compute angle of the displacement vector
        angle_to_align = np.degrees(np.arctan2(dy, dx))

    # Align such that the movement is along the positive y-axis
    rotation_angle = 90 - angle_to_align

    rotation_matrix = np.array([
        [np.cos(np.radians(rotation_angle)), -np.sin(np.radians(rotation_angle))],
        [np.sin(np.radians(rotation_angle)),  np.cos(np.radians(rotation_angle))]
    ])

    # Apply rotation to the original coordinates
    coords = np.c_[trial_df['pos_x'], trial_df['pos_y']]
    rotated_coords = np.dot(coords, rotation_matrix.T)

    # Shift coordinates so that the stimulus onset is at (0, 0)
    x_shift = rotated_coords[stim_start_idx, 0]
    y_shift = rotated_coords[stim_start_idx, 1]
    x_aligned = rotated_coords[:, 0] - x_shift
    y_aligned = rotated_coords[:, 1] - y_shift

    trial_df['x_aligned'] = x_aligned
    trial_df['y_aligned'] = y_aligned

    return trial_df

# Function to correct OpenCV theta assignment
def correct_opencv_theta_assignment(df):
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

# Plotting function (as provided in your original code)
def plot_trajectories(trials_list, title):
    # Set the frame limits relative to stimulus onset
    frames_before_stim = 10
    frames_after_stim = 240
    # Plot smoothed trajectories
    fig, ax = plt.subplots(figsize=(10, 8))
    all_x_smoothed = []
    all_y_smoothed = []
    all_time = []

    for trial_data in trials_list:
            # Align and retrieve x and y coordinates
        x_aligned = trial_data['x_aligned']
        y_aligned = trial_data['y_aligned']
        # Use smoothed data if available, otherwise fallback to original
        x_smoothed_aligned = x_aligned
        y_smoothed_aligned = y_aligned
        time_array = trial_data.get('time', np.arange(-10, len(x_smoothed_aligned) - 10))

        # Filter out NaN values
        valid_indices = ~np.isnan(x_smoothed_aligned) & ~np.isnan(y_smoothed_aligned)
        x_smoothed_aligned = x_smoothed_aligned[valid_indices]
        y_smoothed_aligned = y_smoothed_aligned[valid_indices]
        time_array = time_array[valid_indices]

        if len(x_smoothed_aligned) == 0:
            continue  # Skip if no valid data

        # Define the range of frames to plot around the stimulus onset
        stim_start_index = np.where(time_array == 0)[0][0] if 0 in time_array else 10
        start_frame = max(0, stim_start_index - frames_before_stim)
        end_frame = min(len(time_array), stim_start_index + frames_after_stim + 1)

        # Extract only the frames within the specified range
        x_smoothed_aligned = x_smoothed_aligned[start_frame:end_frame]
        y_smoothed_aligned = y_smoothed_aligned[start_frame:end_frame]
        time_array = time_array[start_frame:end_frame]

        # Plot individual trajectories
        pre_stim_indices = time_array <= 0
        ax.plot(x_smoothed_aligned[pre_stim_indices], y_smoothed_aligned[pre_stim_indices], color='red', alpha=0.3)

        post_stim_indices = time_array >= 0
        ax.plot(x_smoothed_aligned[post_stim_indices], y_smoothed_aligned[post_stim_indices], color='green', alpha=0.3)

        all_x_smoothed.append(x_smoothed_aligned)
        all_y_smoothed.append(y_smoothed_aligned)
        all_time.append(time_array)

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

def is_fly_responsive(trial_data, speed_threshold=7,stim_start_idx = 10):
    x = trial_data['x_aligned']
    y = trial_data['y_aligned']
    frame_number = trial_data['frame_number']  # Original frame numbers in the trial
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

    # If total length before stimulation is smaller than 5, fly is not responsive
    if total_length_pre < 3:
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

# Updated Function to align smoothed trajectory and set stimulus onset at (0, 0)
def align_smoothed_trajectory(trial_df, stim_start_idx, window_length=11, polyorder=3):
    # Smooth the trajectory using Savitzky-Golay filter
    trial_df['x_smoothed'] = savgol_filter(trial_df['pos_x'], window_length=window_length, polyorder=polyorder)
    trial_df['y_smoothed'] = savgol_filter(trial_df['pos_y'], window_length=window_length, polyorder=polyorder)

    # Check if we have enough data to compute the orientation 2 frames before stimulation onset
    if stim_start_idx < 3:
        # Not enough data to compute alignment
        print(f"Trial {trial_df['Trial_Num'].iloc[0]}: Not enough data to compute alignment.")
        trial_df['x_aligned'] = trial_df['x_smoothed'] - trial_df['x_smoothed'].iloc[stim_start_idx]
        trial_df['y_aligned'] = trial_df['y_smoothed'] - trial_df['y_smoothed'].iloc[stim_start_idx]
        return trial_df

    # Compute displacement vector between frames (stim_start_idx - 3) and (stim_start_idx)
    dx = trial_df['x_smoothed'].iloc[stim_start_idx] - trial_df['x_smoothed'].iloc[stim_start_idx - 3]
    dy = trial_df['y_smoothed'].iloc[stim_start_idx] - trial_df['y_smoothed'].iloc[stim_start_idx - 3]

    # Calculate angle and rotation matrix as before
    angle_to_align = 0.0 if np.hypot(dx, dy) < 1e-4 else np.degrees(np.arctan2(dy, dx))
    rotation_angle = 90 - angle_to_align
    rotation_matrix = np.array([
        [np.cos(np.radians(rotation_angle)), -np.sin(np.radians(rotation_angle))],
        [np.sin(np.radians(rotation_angle)), np.cos(np.radians(rotation_angle))]
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

# Function to calculate rolling straightness
def calculate_rolling_straightness(x, y, window_size, step_size=1):
    '''
    Calculate the straightness of the trajectory over a rolling window basis.
    Returns an array of straightness values.
    '''
    straightness_values = []
    n_points = len(x)
    for start_idx in range(0, n_points - window_size + 1, step_size):
        end_idx = start_idx + window_size
        x_window = x[start_idx:end_idx]
        y_window = y[start_idx:end_idx]
        
        # Net displacement D
        D = np.sqrt((x_window[-1] - x_window[0])**2 + (y_window[-1] - y_window[0])**2)
        
        # Total path length L
        dx = np.diff(x_window)
        dy = np.diff(y_window)
        distances = np.sqrt(dx**2 + dy**2)
        L = np.sum(distances)
        
        if L == 0:
            S = np.nan
        else:
            S = D / L
        straightness_values.append(S)
    return np.array(straightness_values)

# Main code to process the data and calculate straightness
folder_path = '/Users/tairan/Downloads/BehaviouralData/Flp_control_25'

# Main loop to apply smoothing, alignment, and responsiveness check
for file_path in glob.glob(f"{folder_path}/*.npy"):
    file_name = os.path.basename(file_path)
    data = np.load(file_path)
    print(data)
    df = pd.DataFrame(file_path, names=["pos_x", "pos_y", "ori", "timestamp", "frame_number", "video_frame", "direction"])
    df = df.apply(pd.to_numeric, errors='coerce')
    df['Trial_Num'] = np.nan  # Add a column for trial number
    
    trials_list = []  # Reset trials list for each file
    trial_num = 0

    # Process direction transitions from 0 to 1.0
    for i in range(len(df["direction"]) - 1):
        if df["direction"].iloc[i] == 0 and df["direction"].iloc[i+1] == 1.0:
            start_point = i
        elif df["direction"].iloc[i] == 1.0 and df["direction"].iloc[i+1] == 0:
            end_point = i
            df_trial = df.iloc[start_point:end_point+1].reset_index(drop=True)
            df_trial['Trial_Num'] = trial_num  # Assign trial number

            # Correct 'theta' and align the smoothed trajectory
            df_trial = correct_opencv_theta_assignment(df_trial)
            df_trial = align_smoothed_trajectory(df_trial, 10)

            # Create 'trial_data' dictionary
            trial_data = {
                'x_aligned': df_trial['x_aligned'].values,
                'y_aligned': df_trial['y_aligned'].values,
                'frame_number': df_trial['frame_number'].values,
            }

            trials_list.append(trial_data)
            trial_num += 1  # Increment trial number

    # Process direction transitions from 0 to -1.0
    for i in range(len(df["direction"]) - 1):
        if df["direction"].iloc[i] == 0 and df["direction"].iloc[i+1] == -1.0:
            start_point = i
        elif df["direction"].iloc[i] == -1.0 and df["direction"].iloc[i+1] == 0:
            end_point = i
            df_trial = df.iloc[start_point:end_point+1].reset_index(drop=True)
            df_trial['Trial_Num'] = trial_num  # Assign trial number

            # Correct 'theta' and align the smoothed trajectory
            df_trial = correct_opencv_theta_assignment(df_trial)
            df_trial = align_smoothed_trajectory(df_trial, 10)

            # Create 'trial_data' dictionary
            trial_data_1 = {
                'x_aligned': -df_trial['x_aligned'].values,
                'y_aligned': df_trial['y_aligned'].values,
                'frame_number': df_trial['frame_number'].values,
            }

            trials_list.append(trial_data_1)
            trial_num += 1  # Increment trial number

    # Initialize list to collect straightness values for the current file
    file_straightness_values = []

    # Filter responsive trials
    speed_threshold = 0.5
    responsive_trials = [trial for trial in trials_list if is_fly_responsive(trial, speed_threshold)]
    
    # Calculate straightness for responsive trials
    window_size = 20  # Set window size for rolling straightness
    step_size = 5   # Set step size for rolling straightness

    for trial in responsive_trials:
        x_aligned = trial['x_aligned']
        y_aligned = trial['y_aligned']
        straightness_values = calculate_rolling_straightness(x_aligned, y_aligned, window_size, step_size)
        # Filter out NaN values
        straightness_values = straightness_values[~np.isnan(straightness_values)]
        file_straightness_values.extend(straightness_values)

    if responsive_trials:
        plot_trajectories(responsive_trials, f'{file_name}')
    else:
        print(f"No responsive trials found for {file_name}.")

    # Plot the probability distribution of the straightness values for the current file
    if file_straightness_values:
        plt.figure(figsize=(8, 6))
        plt.hist(file_straightness_values, bins=100, density=True, alpha=0.7, color='blue')
        plt.title(f'Probability Distribution of Straightness for {file_name}')
        plt.xlabel('Straightness (D/L)')
        plt.ylabel('Probability Density')
        plt.grid(True)
        plt.show()
    else:
        print(f"No straightness values to plot for {file_name}.")
