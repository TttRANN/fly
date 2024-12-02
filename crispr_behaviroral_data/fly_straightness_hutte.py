import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

def align_trajectory(trial_df, stim_start_idx,file_name):
    """
    Aligns the trajectory of the fly so that the movement before stimulation is along the positive y-axis,
    and shifts coordinates so that the stimulus onset is at (0, 0).
    
    Parameters:
    - trial_df: pandas DataFrame containing 'x' and 'y' columns.
    - stim_start_idx: Index where stimulation starts in the trial DataFrame.

    Returns:
    - trial_df: pandas DataFrame with added 'x_aligned' and 'y_aligned' columns.

    """

    # Check if we have enough data to compute the orientation
    if stim_start_idx < 3:
        print(f"Trial {trial_df['Trial_Num'].iloc[0]}: Not enough data to compute alignment.")
        trial_df['x_aligned'] = trial_df['x'] - trial_df['x'].iloc[stim_start_idx]
        trial_df['y_aligned'] = trial_df['y'] - trial_df['y'].iloc[stim_start_idx]
        return trial_df

    # Compute displacement vector between frames
    try:
        dx = trial_df['x'].iloc[stim_start_idx] - trial_df['x'].iloc[stim_start_idx - 3]
        dy = trial_df['y'].iloc[stim_start_idx] - trial_df['y'].iloc[stim_start_idx - 3]
    except:
        print("opss")
        print(file_name)
        return False

    
    

    # If the fly hasn't moved, set angle to zero
    if np.hypot(dx, dy) < 1e-4:
        angle_to_align = 0.0
    else:
        angle_to_align = np.degrees(np.arctan2(dy, dx))

    # Align movement along the positive y-axis
    rotation_angle = 90 - angle_to_align
    rotation_matrix = np.array([
        [np.cos(np.radians(rotation_angle)), -np.sin(np.radians(rotation_angle))],
        [np.sin(np.radians(rotation_angle)),  np.cos(np.radians(rotation_angle))]
    ])

    # Apply rotation
    coords = np.c_[trial_df['x'], trial_df['y']]
    rotated_coords = np.dot(coords, rotation_matrix.T)

    # Shift coordinates
    x_shift = rotated_coords[stim_start_idx, 0]
    y_shift = rotated_coords[stim_start_idx, 1]
    trial_df['x_aligned'] = rotated_coords[:, 0] - x_shift
    trial_df['y_aligned'] = rotated_coords[:, 1] - y_shift

    return trial_df

def correct_opencv_theta_assignment(df):
    # Debugging print statement
    print(f"Inside correct_opencv_theta_assignment, type of df: {type(df)}")
    
    # Ensure df is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a DataFrame, but got a different type.")

    # Check if the DataFrame is empty
    if df.empty:
        raise ValueError("trial_df is empty.")
    
    # Ensure required columns are present
    required_columns = {'Clockwise', 'Orientation'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"trial_df is missing required columns. Found columns: {df.columns}")

    # Proceed with the main logic
    factor = -1 if df['Clockwise'].sum() == 0 else 1
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



def is_fly_responsive(trial_data, speed_threshold=7):
    """
    Checks if the fly is responsive based on speed.

    Parameters:
    - trial_data: Dictionary containing trial data with 'x_aligned', 'y_aligned', and 'time'.
    - speed_threshold: Speed threshold to determine responsiveness.

    Returns:
    - Boolean indicating if the fly is responsive.
    """
    x = trial_data['x_aligned']
    y = trial_data['y_aligned']
    time = trial_data['time']

    # Indices before stimulation
    pre_stim_indices = time < 0

    if np.sum(pre_stim_indices) < 2:
        total_length_pre = 0
    else:
        dx_pre = np.diff(x[pre_stim_indices])
        dy_pre = np.diff(y[pre_stim_indices])
        distances_pre = np.sqrt(dx_pre**2 + dy_pre**2)
        total_length_pre = np.sum(distances_pre)

    if total_length_pre < 5:
        return False

    stim_indices = time >= 0
    if np.sum(stim_indices) < 2:
        return False

    dx = np.diff(x[stim_indices])
    dy = np.diff(y[stim_indices])
    speed = np.sqrt(dx**2 + dy**2)
    mean_speed = np.mean(speed)

    return mean_speed >= speed_threshold

def plot_trajectories(trials_list, title):
    """
    Plots trajectories with color-coded time information (original and smoothed separately).
    
    Parameters:
    - trials_list: List of dictionaries containing trial data.
    - title: Title for the plot.
    """
    # Plot original trajectories
    fig, ax = plt.subplots(figsize=(10, 8))
    all_x = []
    all_y = []
    all_time = []

    for trial_data in trials_list:
        x = trial_data['x_aligned']
        y = trial_data['y_aligned']
        time_array = trial_data['time']

        valid_indices = ~np.isnan(x) & ~np.isnan(y)
        x = x[valid_indices]
        y = y[valid_indices]
        time_array = time_array[valid_indices]

        if len(x) == 0:
            continue

        pre_stim_indices = time_array <= 0
        ax.plot(x[pre_stim_indices], y[pre_stim_indices], color='red', alpha=0.3)
        post_stim_indices = time_array >= 0
        ax.plot(x[post_stim_indices], y[post_stim_indices], color='green', alpha=0.3)

        all_x.append(x)
        all_y.append(y)
        all_time.append(time_array)

    if not all_x or not all_y:
        print("No valid trajectories to plot.")
        return

    max_length = max(len(x) for x in all_x)
    padded_x = np.array([np.pad(x, (0, max_length - len(x)), 'edge') for x in all_x])
    padded_y = np.array([np.pad(y, (0, max_length - len(y)), 'edge') for y in all_y])
    padded_time = np.array([np.pad(t, (0, max_length - len(t)), 'edge') for t in all_time])

    mean_x = np.nanmean(padded_x, axis=0)
    mean_y = np.nanmean(padded_y, axis=0)
    mean_time = np.nanmean(padded_time, axis=0)

    if np.isnan(mean_x).any() or np.isnan(mean_y).any():
        print("NaN values encountered in the mean calculation.")
        return

    norm = plt.Normalize(vmin=mean_time[0], vmax=mean_time[-1])
    cmap = plt.cm.viridis

    for i in range(len(mean_x) - 1):
        ax.plot([mean_x[i], mean_x[i + 1]], [mean_y[i], mean_y[i + 1]],
                color=cmap(norm(mean_time[i])), linewidth=2)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Time (Frames)')

    ax.set_title(f'{title} (Original Trajectories)')
    ax.set_xlabel('x position (aligned)')
    ax.set_ylabel('y position (aligned)')
    ax.grid(True)
    # plt.show()
    plt.savefig(f"/Users/tairan/Downloads/{title}_ori_traj.png")

    # Plot smoothed trajectories
    fig, ax = plt.subplots(figsize=(10, 8))
    all_x_smoothed = []
    all_y_smoothed = []
    all_time = []

    for trial_data in trials_list:
        x_smoothed_aligned = trial_data['x_smoothed_aligned']
        y_smoothed_aligned = trial_data['y_smoothed_aligned']
        time_array = trial_data['time']

        valid_indices = ~np.isnan(x_smoothed_aligned) & ~np.isnan(y_smoothed_aligned)
        x_smoothed_aligned = x_smoothed_aligned[valid_indices]
        y_smoothed_aligned = y_smoothed_aligned[valid_indices]
        time_array = time_array[valid_indices]

        if len(x_smoothed_aligned) == 0:
            continue

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

    max_length = max(len(x) for x in all_x_smoothed)
    padded_x_smoothed = np.array([np.pad(x, (0, max_length - len(x)), 'edge') for x in all_x_smoothed])
    padded_y_smoothed = np.array([np.pad(y, (0, max_length - len(y)), 'edge') for y in all_y_smoothed])
    padded_time = np.array([np.pad(t, (0, max_length - len(t)), 'edge') for t in all_time])

    mean_x_smoothed = np.nanmean(padded_x_smoothed, axis=0)
    mean_y_smoothed = np.nanmean(padded_y_smoothed, axis=0)
    mean_time = np.nanmean(padded_time, axis=0)

    if np.isnan(mean_x_smoothed).any() or np.isnan(mean_y_smoothed).any():
        print("NaN values encountered in the mean calculation.")
        return

    norm = plt.Normalize(vmin=mean_time[0], vmax=mean_time[-1])
    cmap = plt.cm.viridis

    for i in range(len(mean_x_smoothed) - 1):
        ax.plot([mean_x_smoothed[i], mean_x_smoothed[i + 1]], [mean_y_smoothed[i], mean_y_smoothed[i + 1]],
                color=cmap(norm(mean_time[i])), linewidth=2)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Time (Frames)')

    ax.set_title(f'{title} (Smoothed Trajectories)')
    ax.set_xlabel('x position (aligned)')
    ax.set_ylabel('y position (aligned)')
    ax.grid(True)
    # plt.show()
    plt.xlim([-200,200])
    plt.ylim([-200,200])
    plt.savefig(f"/Users/tairan/Downloads/{title}_smoothed_traj.png")
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

def main():
    folder_path = '/Users/tairan/Downloads/plexa'  # Update this path as needed

    for file_path in glob.glob(os.path.join(folder_path, "*.csv")):
        file_name = os.path.basename(file_path)
        print(f"Processing file: {file_name}")

        df = pd.read_csv(
            file_path,
            names=['Frame', 'Timestamp', 'Arena', 'x', 'y', 'Orientation', 'Stimulus', 'Frequency', 'Clockwise', 'n'],
            skiprows=1  # Skip header row if present
        )

        # Convert columns to appropriate data types
        df['x'] = pd.to_numeric(df['x'], errors='coerce')
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        df['Orientation'] = pd.to_numeric(df['Orientation'], errors='coerce')
        df = df.dropna(subset=['x', 'y', 'Orientation'])

        df['Stimulus'] = df['Stimulus'].map({'on': True, 'off': False})
        df['Clockwise'] = df['Clockwise'].astype(int)

        df['Stimulus_prev'] = df['Stimulus'].shift(1, fill_value=False)
        df['Trial_Start'] = (~df['Stimulus_prev']) & df['Stimulus']
        df['Trial_End'] = df['Stimulus_prev'] & (~df['Stimulus'])

        trial_num = 0
        trial_nums = []
        for idx, row in df.iterrows():
            if row['Trial_Start']:
                trial_num += 1
            trial_nums.append(trial_num)
        df['Trial_Num'] = trial_nums
        print(f"Total frames: {len(df)}")

        clockwise_trials = []
        counter_clockwise_trials = []

        pre_stim_frames = 6
        post_stim_frames = 300

        for trial in df['Trial_Num'].unique():
            if trial == 0:
                continue
            trial_data = df[df['Trial_Num'] == trial]
            trial_start_idx = trial_data.index[0]
            trial_end_idx = trial_data.index[-1]
            stim_on_data = trial_data[trial_data['Stimulus'] == True].iloc[:post_stim_frames]
            pre_stim_data = df.loc[max(0, trial_start_idx - pre_stim_frames):trial_start_idx - 1]
            trial_df = pd.concat([pre_stim_data, stim_on_data]).copy()
            trial_df.reset_index(drop=True, inplace=True)
            stim_start_idx = len(pre_stim_data)

            # Check if trial_df has enough data points
            data_length = len(trial_df)
            if data_length < 3:
                print(f"Trial {trial}: Not enough data to compute alignment.")
                continue  # Skip this trial

            # Dynamically adjust window_length
            polyorder = 2
            window_length = min(15, data_length)
            if window_length % 2 == 0:
                window_length -= 1  # Ensure window_length is odd
            if window_length <= polyorder:
                window_length = polyorder + 1
                if window_length % 2 == 0:
                    window_length += 1

            if window_length > data_length:
                print(f"Trial {trial}: Not enough data points for Savitzky-Golay filter.")
                continue  # Skip this trial

            x_smoothed = savgol_filter(trial_df['x'].values, window_length, polyorder)
            y_smoothed = savgol_filter(trial_df['y'].values, window_length, polyorder)

            trial_df_smoothed = trial_df.copy()
            trial_df_smoothed['x'] = x_smoothed
            trial_df_smoothed['y'] = y_smoothed

            trial_df = correct_opencv_theta_assignment(trial_df)
            trial_df = align_trajectory(trial_df, stim_start_idx, file_name)
            trial_df_smoothed = align_trajectory(trial_df_smoothed, stim_start_idx, file_name)

            total_length = len(trial_df)
            time_array = np.arange(-stim_start_idx, total_length - stim_start_idx)
            clockwise_value = trial_df['Clockwise'].iloc[stim_start_idx]
            trial_dict = {
                'x_aligned': trial_df['x_aligned'].values,
                'y_aligned': trial_df['y_aligned'].values,
                'x_smoothed_aligned': trial_df_smoothed['x_aligned'].values,
                'y_smoothed_aligned': trial_df_smoothed['y_aligned'].values,
                'time': time_array,
                'trial_number': trial
            }
            if clockwise_value:
                clockwise_trials.append(trial_dict)
            else:
                trial_dict['x_aligned'] = -trial_dict['x_aligned']
                trial_dict['x_smoothed_aligned'] = -trial_dict['x_smoothed_aligned']
                counter_clockwise_trials.append(trial_dict)



        speed_threshold = 0.3
        responsive_clockwise_trials = [trial for trial in clockwise_trials if is_fly_responsive(trial, speed_threshold)]
        responsive_counter_clockwise_trials = [trial for trial in counter_clockwise_trials if is_fly_responsive(trial, speed_threshold)]
        responsive_trials = responsive_clockwise_trials + responsive_counter_clockwise_trials
            # Calculate straightness for responsive trials
        window_size = 20  # Set window size for rolling straightness
        step_size = 2   # Set step size for rolling straightness
        file_straightness_values=[]
        for trial in responsive_trials:
            x_aligned = trial['x_aligned']
            y_aligned = trial['y_aligned']
            straightness_values = calculate_rolling_straightness(x_aligned, y_aligned, window_size, step_size)
            # Filter out NaN values
            straightness_values = straightness_values[~np.isnan(straightness_values)]
            file_straightness_values.extend(straightness_values)

        if responsive_trials:
            plot_trajectories(responsive_trials, f'{file_name}')
                    # Plot the probability distribution of the straightness values for the current file
            if file_straightness_values:
                plt.figure(figsize=(8, 6))
                plt.hist(file_straightness_values, bins=100, density=True, alpha=0.7, color='blue')
                plt.title(f'Probability Distribution of Straightness for {file_name}')
                plt.xlabel('Straightness (D/L)')
                plt.ylabel('Probability Density')
                plt.grid(True)
                plt.ylim([0,6])
                # plt.show()
                plt.savefig(f"/Users/tairan/Downloads/{file_name}_straightness.png")
        else:
            print(f"No responsive trials found for {file_name}.")





if __name__ == "__main__":
    main()
