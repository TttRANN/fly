import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths, periodogram, savgol_filter
import pywt
import os
import glob

# Function to align the trajectory based on the movement direction before stimulation
def align_trajectory(trial_df, stim_start_idx):
    # Check if we have enough data to compute the orientation 3 frames before stimulation onset
    if stim_start_idx < 3:
        # Not enough data to compute alignment
        print(f"Trial {trial_df['Trial_Num'].iloc[0]}: Not enough data to compute alignment.")
        # Shift so that the onset of the stimulation is at (0, 0)
        trial_df['x_aligned'] = trial_df['x'] - trial_df['x'].iloc[stim_start_idx]
        trial_df['y_aligned'] = trial_df['y'] - trial_df['y'].iloc[stim_start_idx]
        return trial_df

    # Compute displacement vector between frames (stim_start_idx - 3) and (stim_start_idx)
    dx = trial_df['x'].iloc[stim_start_idx] - trial_df['x'].iloc[stim_start_idx - 3]
    dy = trial_df['y'].iloc[stim_start_idx] - trial_df['y'].iloc[stim_start_idx - 3]

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

# Function to perform stationary wavelet transform denoising
def swt_denoising(data, wavelet='bior2.6', level=7):
    if data.size == 0:
        return np.array([])
    max_level = pywt.swt_max_level(len(data))
    level = min(level, max_level)
    if level == 0:
        print("Data length is too short for the specified wavelet level.")
        return data
    freq = pywt.scale2frequency(wavelet, 2 ** np.arange(0, level)[::-1]) / 0.016
    swt_coeffs = pywt.swt(data, wavelet, level=level)
    index = np.argwhere((freq > 5) & (freq < 60)).flatten()
    # Zero out all coefficients except the ones in the desired frequency range
    swt_filtered = []
    for i, coeff in enumerate(swt_coeffs):
        if i in index:
            swt_filtered.append(coeff)
        else:
            swt_filtered.append((np.zeros_like(coeff[0]), np.zeros_like(coeff[1])))
    # Reconstruct signal
    re_data = pywt.iswt(swt_filtered, wavelet)
    return re_data



# Function to detect saccades using peak detection
def detect_saccades(angular_speed, threshold, width=[5, 30]):
    height = threshold
    peaks, _ = find_peaks(angular_speed, height=height, distance=5, prominence=0.8 * height, wlen=30, width=width, rel_height=1)
    peaks_neg, _ = find_peaks(-angular_speed, height=height, distance=5, prominence=0.8 * height, wlen=30, width=width, rel_height=1)
    # Merge peaks
    all_peaks = np.concatenate((peaks, peaks_neg))
    all_peak_values = np.concatenate((angular_speed[peaks], angular_speed[peaks_neg]))
    # Sort the peaks
    sorted_indices = np.argsort(all_peaks)
    all_peaks = all_peaks[sorted_indices]
    all_peak_values = all_peak_values[sorted_indices]
    return all_peaks, all_peak_values

# Function to detect saccades using continuous wavelet transform
def detect_saccades_cwt(angular_speed, threshold):
    # Denoise signal
    denoised_signal = swt_denoising(angular_speed, level=5)
    # Detect peaks in denoised signal
    cwt_peaks, cwt_peak_values = detect_saccades(denoised_signal, threshold=threshold)
    # Detect peaks in original signal
    peaks, peak_values = detect_saccades(angular_speed, threshold=threshold, width=[5, 30])
    # Match peaks between denoised and original signal
    final_peaks = []
    final_peak_values = []
    i, j = 0, 0
    while i < len(cwt_peaks) and j < len(peaks):
        if abs(cwt_peaks[i] - peaks[j]) <= 3:
            final_peaks.append(peaks[j])
            final_peak_values.append(peak_values[j])
            i += 1
            j += 1
        elif cwt_peaks[i] < peaks[j] - 3:
            i += 1
        else:
            j += 1
    return np.array(final_peaks), np.array(final_peak_values), denoised_signal

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
        distances_pre = np.sqrt(dx_pre ** 2 + dy_pre ** 2)

        # Compute total length before stimulation
        total_length_pre = np.sum(distances_pre)

    # If total length before stimulation is smaller than threshold, fly is not responsive
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
    speed = np.sqrt(dx ** 2 + dy ** 2)

    # Compute mean speed
    mean_speed = np.mean(speed)

    # Check if mean speed exceeds threshold
    return mean_speed >= speed_threshold

# Function to calculate rolling straightness (not used in the main code but included as per original code)
def calculate_rolling_straightness(x, y, window_size, step_size=1):
    straightness_values = []
    n_points = len(x)
    for start_idx in range(0, n_points - window_size + 1, step_size):
        end_idx = start_idx + window_size
        x_window = x[start_idx:end_idx]
        y_window = y[start_idx:end_idx]

        # Net displacement D
        D = np.sqrt((x_window[-1] - x_window[0]) ** 2 + (y_window[-1] - y_window[0]) ** 2)

        # Total path length L
        dx = np.diff(x_window)
        dy = np.diff(y_window)
        distances = np.sqrt(dx ** 2 + dy ** 2)
        L = np.sum(distances)

        if L == 0:
            S = np.nan
        else:
            S = D / L
        straightness_values.append(S)
    return np.array(straightness_values)

# Main processing code
folder_path = '/Users/tairan/Downloads/plexa'  # Update this path to your data folder

for file_path in glob.glob(f"{folder_path}/*.csv"):
    file_name = os.path.basename(file_path)
    data_file = file_path

    # Read the data
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

    # Prepare list to store trial numbers with mean PSD less than 1
    low_psd_trial_numbers = []

    # Define pre-stimulation and post-stimulation frame counts
    pre_stim_frames = 6  # Frames before stimulation starts
    post_stim_frames = 300  # Frames after stimulation starts

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
        stim_start_idx = len(pre_stim_data)  # Index where stimulation starts in trial_df

        # Apply smoothing to x and y before alignment
        window_length = 15  # Must be odd and greater than polyorder
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
        angular_velocity_smoothed = savgol_filter(angular_velocities, window_length=15, polyorder=2)

        # Compute threshold for saccade detection
        threshold = 4 * np.median(np.abs(angular_velocities) / 0.6745)

        # Detect saccades using cwt
        final_peaks, final_peak_values, reconstructed_signal = detect_saccades_cwt(angular_velocities, threshold)

        # Compute periodogram
        sampling_rate = 60  # Adjust if necessary
        freqs, psd = periodogram(angular_velocities, fs=sampling_rate)

        # Calculate mean PSD in the 10 Hz to 25 Hz range
        freq_mask = (freqs >= 10) & (freqs <= 25)
        mean_psd = np.mean(psd[freq_mask])

        # If mean PSD is less than 1, store the trial number
        if mean_psd < 1:
            low_psd_trial_numbers.append(trial)

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
            'saccade_times': final_peaks,
            'saccade_values': final_peak_values,
            'reconstructed_signal': reconstructed_signal,
            'freqs': freqs,
            'psd': psd,
            'mean_psd_10_25Hz': mean_psd,
            'trial_number': trial
        }

        if clockwise_value:
            clockwise_trials.append(trial_dict)
        else:
            # For counter-clockwise, invert x_aligned and x_smoothed_aligned
            trial_dict['x_aligned'] = -trial_dict['x_aligned']
            trial_dict['x_smoothed_aligned'] = -trial_dict['x_smoothed_aligned']
            counter_clockwise_trials.append(trial_dict)

    # Function to plot accumulated angles
    def plot_accumulated_angles(trials_list, title):
        plt.figure(figsize=(10, 6))
        all_accumulated_angles = []
        all_times = []
        for trial_data in trials_list:
            accumulated_angle = trial_data['accumulated_angle']
            time = trial_data['time']
            # Check for NaNs in accumulated_angle
            if np.isnan(accumulated_angle).any():
                print(f"NaNs found in accumulated_angle for trial {trial_data['trial_number']}")
                continue  # Skip this trial

            plt.plot(time, accumulated_angle, color='grey', alpha=0.5)
            all_accumulated_angles.append(accumulated_angle)
            all_times.append(time)
        if not all_accumulated_angles:
            print("No accumulated angles to plot.")
            return
        # Pad accumulated angles to match lengths
        max_length = max(len(aa) for aa in all_accumulated_angles)
        padded_accumulated_angles = np.array([
            np.pad(aa, (0, max_length - len(aa)), 'edge') for aa in all_accumulated_angles
        ])
        # Check for NaNs after padding
        if np.isnan(padded_accumulated_angles).any():
            print("NaNs found in padded_accumulated_angles")
        mean_accumulated_angle = np.nanmean(padded_accumulated_angles, axis=0)
        mean_time = np.linspace(all_times[0][0], all_times[0][-1], max_length)
        plt.plot(mean_time, mean_accumulated_angle, color='black', linewidth=2)
        plt.title(f'Accumulated Angle - {title}')
        plt.xlabel('Time (Frames)')
        plt.ylabel('Accumulated Angle (degrees)')
        plt.grid(True)
        plt.show()

    # Function to plot angular velocities (original and smoothed separately)
    def plot_angular_velocities(trials_list, title):
        # Plot original angular velocities
        plt.figure(figsize=(10, 6))
        all_angular_velocities = []
        all_times = []
        for trial_data in trials_list:
            angular_velocity = trial_data['angular_velocity']
            time = trial_data['angular_time']
            plt.plot(time, angular_velocity, color='grey', alpha=0.3)
            all_angular_velocities.append(angular_velocity)
            all_times.append(time)
        if not all_angular_velocities:
            print("No angular velocities to plot.")
            return
        # Pad angular velocities to match lengths
        max_length = max(len(av) for av in all_angular_velocities)
        padded_angular_velocities = np.array([np.pad(av, (0, max_length - len(av)), 'edge') for av in all_angular_velocities])
        mean_angular_velocity = np.nanmean(padded_angular_velocities, axis=0)
        mean_time = np.linspace(all_times[0][0], all_times[0][-1], max_length)
        plt.plot(mean_time, mean_angular_velocity, color='black', linewidth=2, label='Mean Original')
        plt.ylim(-1000, 1000)
        plt.title(f'Angular Velocity (Original) - {title}')
        plt.xlabel('Time (Frames)')
        plt.ylabel('Angular Velocity (degrees/second)')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot smoothed angular velocities
        plt.figure(figsize=(10, 6))
        all_angular_velocities_smoothed = []
        all_times = []
        for trial_data in trials_list:
            angular_velocity_smoothed = trial_data['angular_velocity_smoothed']
            time = trial_data['angular_time']
            plt.plot(time, angular_velocity_smoothed, color='grey', alpha=0.3)
            all_angular_velocities_smoothed.append(angular_velocity_smoothed)
            all_times.append(time)
        # Pad angular velocities to match lengths
        padded_angular_velocities_smoothed = np.array([np.pad(av, (0, max_length - len(av)), 'edge') for av in all_angular_velocities_smoothed])
        mean_angular_velocity_smoothed = np.nanmean(padded_angular_velocities_smoothed, axis=0)
        plt.plot(mean_time, mean_angular_velocity_smoothed, color='red', linewidth=2, label='Mean Smoothed')
        plt.ylim(-1000, 1000)
        plt.title(f'Angular Velocity (Smoothed) - {title}')
        plt.xlabel('Time (Frames)')
        plt.ylabel('Angular Velocity (degrees/second)')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Function to plot saccades detected
    def plot_saccades(trials_list, title):
        plt.figure(figsize=(10, 6))
        for trial_data in trials_list:
            reconstructed_signal = trial_data['reconstructed_signal']
            time = trial_data['angular_time']
            saccade_times = trial_data['saccade_times'].astype(int)  # Ensure integer indices

            # Handle cases where saccade_times might be empty
            if saccade_times.size == 0:
                continue

            # Remove any indices that are out of bounds
            valid_indices = (saccade_times >= 0) & (saccade_times < len(time))

            saccade_times = saccade_times[valid_indices]

            # Plot the reconstructed (denoised) signal
            plt.plot(time, reconstructed_signal, color='grey', alpha=0.5)

            # Plot saccade times
            plt.scatter(time[saccade_times], reconstructed_signal[saccade_times], color='red', s=10)

        plt.title(f'Saccades Detected - {title}')
        plt.xlabel('Time (Frames)')
        plt.ylabel('Reconstructed Angular Velocity')
        plt.grid(True)
        plt.show()


    # Function to plot periodogram of angular velocities
    def plot_angular_velocity_periodogram(trials_list, title, sampling_rate=60):
        plt.figure(figsize=(10, 6))
        all_freqs = []
        all_psd = []

        for trial_data in trials_list:
            freqs = trial_data['freqs']
            psd = trial_data['psd']
            plt.semilogy(freqs, psd, color='grey', alpha=0.5)
            all_freqs.append(freqs)
            all_psd.append(psd)

        if not all_psd:
            print("No periodograms to plot.")
            return

        # Interpolate PSDs to a common frequency grid
        common_freqs = np.linspace(0, max(freqs), 500)
        interpolated_psd = []
        for freqs_i, psd_i in zip(all_freqs, all_psd):
            psd_interp = np.interp(common_freqs, freqs_i, psd_i)
            interpolated_psd.append(psd_interp)

        mean_psd = np.nanmean(interpolated_psd, axis=0)

        plt.semilogy(common_freqs, mean_psd, color='black', linewidth=2)
        plt.title(f'Periodogram of Angular Velocity - {title}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density')
        plt.grid(True)
        plt.show()

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

    # Filter responsive trials
    speed_threshold = 1  # Adjust the threshold as needed
    responsive_clockwise_trials = [trial for trial in clockwise_trials if is_fly_responsive(trial, speed_threshold)]
    responsive_counter_clockwise_trials = [trial for trial in counter_clockwise_trials if is_fly_responsive(trial, speed_threshold)]

    # Plot Responsive Clockwise trials
    if responsive_clockwise_trials:
        a = file_name.split("_")[0:4]
        plot_trajectories(responsive_clockwise_trials[2:3], f"(Responsive Clockwise Trials) {'_'.join(a)}")
        plot_angular_velocities(responsive_clockwise_trials[2:3], f"(Responsive Clockwise Trials) {'_'.join(a)}")
        plot_accumulated_angles(responsive_clockwise_trials[2:3], f"(Responsive Clockwise Trials) {'_'.join(a)}")
        plot_angular_velocity_periodogram(responsive_clockwise_trials[2:3], f"(Responsive Clockwise Trials) {'_'.join(a)}")
        plot_saccades(responsive_clockwise_trials[2:3], f"(Responsive Clockwise Trials) {'_'.join(a)}")
    else:
        print("No responsive Clockwise trials found.")

    # # Plot Responsive Counter-Clockwise trials
    # if responsive_counter_clockwise_trials:
    #     a = file_name.split("_")[0:4]
    #     plot_trajectories(responsive_counter_clockwise_trials, f"(Responsive Counter-Clockwise Trials) {'_'.join(a)}")
    #     plot_angular_velocities(responsive_counter_clockwise_trials, f"(Responsive Counter-Clockwise Trials) {'_'.join(a)}")
    #     plot_accumulated_angles(responsive_counter_clockwise_trials, f"(Responsive Counter-Clockwise Trials) {'_'.join(a)}")
    #     plot_angular_velocity_periodogram(responsive_counter_clockwise_trials, f"(Responsive Counter-Clockwise Trials) {'_'.join(a)}")
    #     plot_saccades(responsive_counter_clockwise_trials, f"(Responsive Counter-Clockwise Trials) {'_'.join(a)}")
    # else:
    #     print("No responsive Counter-Clockwise trials found.")
