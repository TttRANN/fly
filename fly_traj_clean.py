import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

def align_trajectory(trial_df, stim_start_idx):
    # Check if we have enough data to compute the orientation 2 frames before stimulation onset
    if stim_start_idx < 3:
        # Not enough data to compute alignment
        print(f"Trial {trial_df['Trial_Num'].iloc[0]}: Not enough data to compute alignment.")
        # Shift so that the onset of the stimulation is at (0, 0)
        trial_df['x_aligned'] = trial_df['x'] - trial_df['x'].iloc[stim_start_idx]
        trial_df['y_aligned'] = trial_df['y'] - trial_df['y'].iloc[stim_start_idx]
        return trial_df
    dx = trial_df['x'].iloc[stim_start_idx ] - trial_df['x'].iloc[stim_start_idx - 3]
    dy = trial_df['y'].iloc[stim_start_idx ] - trial_df['y'].iloc[stim_start_idx - 3]

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
    coords = np.c_[trial_df['x'], trial_df['y']]
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
    # Rectifies head-rear ambiguity from OpenCV
    if df['Clockwise'].sum() == 0:
        factor = -1
    else:
        factor = 1
    cumulative_theta = df['Orientation'].iloc[0]
    theta_corr = [cumulative_theta * factor]
    for difference in df['Orientation'].diff().fillna(0).iloc[1:]:
        if difference > 90:
            difference -= 180
        elif difference < -90:
            difference += 180
        cumulative_theta += difference
        theta_corr.append(cumulative_theta * factor)
    df = df.assign(theta_corr=theta_corr)
    return df



# Function to check if the fly is responsive based on speed
def is_fly_responsive(trial_data, speed_threshold=7):
    x = trial_data['x_aligned']
    y = trial_data['y_aligned']
    time = trial_data['time']

    # Get indices before stimulation
    pre_stim_indices = time < 0

    # Check if there are enough data points before stimulation
    if np.sum(pre_stim_indices) < 2:
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

    # If total length before stimulation is smaller than 1, fly is not responsive
    if total_length_pre < 5:
        return False

    # Get indices during stimulation
    stim_indices = time >= 0

    if np.sum(stim_indices) < 2:
        # Not enough data during stimulation
        return False

    # Compute displacements during stimulation
    dx = np.diff(x[stim_indices])
    dy = np.diff(y[stim_indices])

    # Compute speed (distance per frame)
    speed = np.sqrt(dx**2 + dy**2)*60

    # Compute mean speed
    mean_speed = np.mean(speed)

    # Check if mean speed exceeds threshold
    return mean_speed >= speed_threshold


import os
import glob

folder_path = '/Users/tairan/Downloads/testfor/cas9_plexa-male_batch1'
pre_stim_frames = 6  # Frames before stimulation starts
post_stim_frames = 30  # Frames after stimulation starts

for file_path in glob.glob(f"{folder_path}/*.csv"):
    # with open(file_path, 'r') as file:
        file_name = os.path.basename(file_path)
        data_file = file_path

        df = pd.read_csv(data_file, names=['Frame', 'Timestamp', 'Arena', 'x', 'y', 'Orientation', 'Stimulus', 'Frequency', 'Clockwise', 'n'])

        # Map 'Stimulus' column to boolean: 'on' -> True, 'off' -> False
        df['Stimulus'] = df['Stimulus'].map({'on': True, 'off': False})

        # Ensure 'Clockwise' is numeric
        df['Clockwise'] = df['Clockwise'].astype(int)

        # Detect Trial Starts and Ends
        df['Stimulus_prev'] = df['Stimulus'].shift(1, fill_value=False)
        df['Trial_Start'] = (~df['Stimulus_prev']) & df['Stimulus']
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
            trial_df = trial_df.copy()
            trial_df.reset_index(drop=True, inplace=True)
            stim_start_idx = len(pre_stim_data) 
            window_length = 15  
            polyorder = 2
            x_smoothed = savgol_filter(trial_df['x'].values, window_length, polyorder)
            y_smoothed = savgol_filter(trial_df['y'].values, window_length, polyorder)
            # Create a copy of trial_df for smoothed data
            trial_df_smoothed = trial_df.copy()
            trial_df_smoothed['x'] = x_smoothed
            trial_df_smoothed['y'] = y_smoothed
            # Use the align_trajectory function on original data
            trial_df = align_trajectory(trial_df, stim_start_idx)
            # Use the align_trajectory function on smoothed data
            trial_df_smoothed = align_trajectory(trial_df_smoothed, stim_start_idx)
            # Apply the correct_opencv_theta_assignment function
            trial_df = correct_opencv_theta_assignment(trial_df)
            # Generate time array for this trial (relative to stim start)
            total_length = len(trial_df)
            time_array = np.arange(-stim_start_idx, total_length - stim_start_idx)
            # Compute angular velocities
            theta_corr = trial_df['theta_corr'].values
            angular_velocities = np.diff(theta_corr, prepend=theta_corr[0]) * 60  # Convert to degrees per second
            # Compute accumulated angle as cumulative sum of angular velocities
            accumulated_angle = np.cumsum(np.diff(theta_corr, prepend=theta_corr[0]))
            angular_time_array = time_array
            # Apply smoothing to angular_velocity
            angular_velocity_smoothed = savgol_filter(angular_velocities, window_length, polyorder)
            # Get 'Clockwise' value for this trial
            clockwise_value = trial_df['Clockwise'].iloc[stim_start_idx]
            trial_dict = {
                'x_aligned': trial_df['x_aligned'].values,
                'y_aligned': trial_df['y_aligned'].values,
                'x_smoothed_aligned': trial_df_smoothed['x_aligned'].values,
                'y_smoothed_aligned': trial_df_smoothed['y_aligned'].values,
                'time': time_array,
                'theta_corr': theta_corr,
                'angular_velocity': angular_velocities,
                'angular_velocity_smoothed': angular_velocity_smoothed,
                'angular_time': angular_time_array,
                'accumulated_angle': accumulated_angle,
                'trial_number': trial
            }
            if clockwise_value:
                clockwise_trials.append(trial_dict)
            else:
                trial_dict = {
                'x_aligned': -trial_df['x_aligned'].values,
                'y_aligned': trial_df['y_aligned'].values,
                'x_smoothed_aligned': trial_df_smoothed['x_aligned'].values,
                'y_smoothed_aligned': trial_df_smoothed['y_aligned'].values,
                'time': time_array,
                'theta_corr': theta_corr,
                'angular_velocity': angular_velocities,
                'angular_velocity_smoothed': angular_velocity_smoothed,
                'angular_time': angular_time_array,
                'accumulated_angle': accumulated_angle,
                'trial_number': trial
            }
                counter_clockwise_trials.append(trial_dict)

        # Function to plot trajectories with color-coded time information (original and smoothed separately)
        def plot_trajectories(trials_list, title):
            # Plot original trajectories
            fig, ax = plt.subplots(figsize=(10, 8))
            all_x = []
            all_y = []
            all_time = []

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
                pre_stim_indices = time_array <= 0
                ax.plot(x[pre_stim_indices], y[pre_stim_indices], color='red', alpha=0.3)

                # During stimulation: light semi-transparent green
                post_stim_indices = time_array >= 0
                ax.plot(x[post_stim_indices], y[post_stim_indices], color='green', alpha=0.3)

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
            cmap = plt.cm.viridis

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
            ax.set_title(f'{title} (Original Trajectories)')
            ax.set_xlabel('x position (aligned)')
            ax.set_ylabel('y position (aligned)')
            ax.grid(True)
            plt.show()

            # Plot smoothed trajectories
            fig, ax = plt.subplots(figsize=(10, 8))
            all_x_smoothed = []
            all_y_smoothed = []
            all_time = []

            for trial_data in trials_list:
                x_smoothed_aligned = trial_data['x_smoothed_aligned']
                y_smoothed_aligned = trial_data['y_smoothed_aligned']
                time_array = trial_data['time']

                # Filter out any NaN values
                valid_indices = ~np.isnan(x_smoothed_aligned) & ~np.isnan(y_smoothed_aligned)
                x_smoothed_aligned = x_smoothed_aligned[valid_indices]
                y_smoothed_aligned = y_smoothed_aligned[valid_indices]
                time_array = time_array[valid_indices]

                if len(x_smoothed_aligned) == 0:
                    continue  # Skip if no valid data

                # Plot individual trajectories
                # Before stimulation: light semi-transparent red
                pre_stim_indices = time_array <= 0
                ax.plot(x_smoothed_aligned[pre_stim_indices], y_smoothed_aligned[pre_stim_indices], color='red', alpha=0.3)

                # During stimulation: light semi-transparent green
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

            # Check for NaNs
            if np.isnan(mean_x_smoothed).any() or np.isnan(mean_y_smoothed).any():
                print("NaN values encountered in the mean calculation. Check input data.")
                return

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
            sm.set_array([])  # Needed for ScalarMappable
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('Time (Frames)')

            # Finalize the plot
            ax.set_title(f'{title} (Smoothed Trajectories)')
            ax.set_xlabel('x position (aligned)')
            ax.set_ylabel('y position (aligned)')
            ax.grid(True)
            plt.show()

        # Filter responsive trials
        speed_threshold = 30 # Adjust the threshold as needed
        responsive_clockwise_trials = [trial for trial in clockwise_trials if is_fly_responsive(trial, speed_threshold)]
        responsive_counter_clockwise_trials = [trial for trial in counter_clockwise_trials if is_fly_responsive(trial, speed_threshold)]
        responsive_trials_clockwise_trials = np.concatenate((responsive_clockwise_trials, responsive_counter_clockwise_trials))


        # Plot Responsive Clockwise trials
        if responsive_clockwise_trials:
            a=file_name.split("_")[0:3]
            a = file_name.split("_")[0:4]
            plot_trajectories(responsive_clockwise_trials, f"(Responsive Clockwise Trials) {'_'.join(a)}")
        else:
            print("No responsive CW trials found.")

        # Plot Responsive Counter-Clockwise trials
        if responsive_counter_clockwise_trials:
           plot_trajectories(responsive_counter_clockwise_trials, f"(Responsive Counter_Clockwise Trials) {'_'.join(a)}")
        else:
            print("No responsive CCW trials found.")
