import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.signal import periodogram, find_peaks
import random

# Define the align_trajectory function
def align_trajectory(trial_df, stim_start_idx):
    # Select pre-stimulus data
    pre_stim_data = trial_df.iloc[:stim_start_idx]
    if len(pre_stim_data) < 2:
        # Not enough data to compute alignment
        print(f"Trial {trial_df['Trial_Num'].iloc[0]}: Not enough pre-stimulus data to compute alignment.")
        trial_df['x_aligned'] = trial_df['x'] - trial_df['x'].iloc[stim_start_idx]
        trial_df['y_aligned'] = trial_df['y'] - trial_df['y'].iloc[stim_start_idx]
        return trial_df

    # Compute displacements
    dx = pre_stim_data['x'].diff().values[1:]
    dy = pre_stim_data['y'].diff().values[1:]

    # Compute average displacement vector
    avg_dx = np.mean(dx)
    avg_dy = np.mean(dy)

    # If the fly hasn't moved before the stimulus, set angle to zero
    if np.hypot(avg_dx, avg_dy) < 1e-6:
        angle_to_align = 0.0
    else:
        # Compute angle of average displacement
        angle_to_align = np.degrees(np.arctan2(avg_dy, avg_dx))

    # Align such that the average motion is along the positive y-axis
    rotation_angle = 90 - angle_to_align

    rotation_matrix = np.array([
        [np.cos(np.radians(rotation_angle)), -np.sin(np.radians(rotation_angle))],
        [np.sin(np.radians(rotation_angle)),  np.cos(np.radians(rotation_angle))]
    ])

    # Apply rotation and translation
    x_shifted = trial_df['x'] - trial_df['x'].iloc[stim_start_idx]
    y_shifted = trial_df['y'] - trial_df['y'].iloc[stim_start_idx]
    aligned_coords = np.dot(np.c_[x_shifted, y_shifted], rotation_matrix.T)
    trial_df['x_aligned'], trial_df['y_aligned'] = aligned_coords[:, 0], aligned_coords[:, 1]
    return trial_df



# Rest of your imports and code
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.signal import periodogram, find_peaks
import random

# Replace 'data.csv' with the path to your CSV file
data_file = '/Users/tairan/Downloads/rnai_beat-iv_t4t5_batch1_filtered_2.csv'

# Read the data
df = pd.read_csv(data_file, names=['Frame', 'Timestamp', 'Arena', 'x', 'y', 'Orientation', 'Stimulus', 'Frequency', 'Clockwise','n'])

# Map 'Stimulus' column to boolean: 'on' -> True, 'off' -> False
df['Stimulus'] = df['Stimulus'].map({'on': True, 'off': False})

# Ensure 'Clockwise' is numeric (assuming it contains numeric values)
df['Clockwise'] = df['Clockwise'].astype(int)

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

# Prepare list to store trial numbers with mean PSD less than 1
low_psd_trial_numbers = []

# Define pre-stimulation and post-stimulation frame counts
pre_stim_frames = 10  # Frames before stimulation starts
post_stim_frames = 50  # Frames after stimulation starts

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

# Function to detect saccades using wavelet transform
def detect_saccades(angular_velocity, sampling_rate=30):
    # Compute the stationary wavelet transform (SWT)
    wavelet = 'bior2.6'
    max_level = pywt.swt_max_level(len(angular_velocity))
    if max_level < 1:
        # Handle cases where the data length is too short
        print(f"Data length ({len(angular_velocity)}) is too short for SWT decomposition.")
        return {
            'saccade_times': np.array([]),
            'saccade_properties': {},
            'reconstructed_signal': np.zeros_like(angular_velocity)
        }
    level = max_level
    coeffs = pywt.swt(angular_velocity, wavelet, level=level)
    
    # Reconstruct signal in 10–20 Hz band
    freq_bands = [(sampling_rate / 2) / (2 ** i) for i in range(level)]
    target_levels = [i for i, fb in enumerate(freq_bands) if 10 <= fb <= 20]
    
    # Initialize list to hold coefficients for reconstruction
    coeffs_recon = []
    for i, (cA, cD) in enumerate(coeffs):
        if i in target_levels:
            # Keep the detail coefficients (cD) in the target levels
            coeffs_recon.append((np.zeros_like(cA), cD))
        else:
            # Zero out coefficients in other levels
            coeffs_recon.append((np.zeros_like(cA), np.zeros_like(cD)))
    
    # Reconstruct the signal from the selected coefficients
    reconstructed_signal = pywt.iswt(coeffs_recon, wavelet)
    
    # Detect peaks in the reconstructed signal
    min_width_samples = int((5 / 1000) * sampling_rate)
    max_width_samples = int((250 / 1000) * sampling_rate)
    
    # Find peaks with minimum height and width constraints
    peaks, properties = find_peaks(
        np.abs(reconstructed_signal),
        height=200,
        width=(min_width_samples, max_width_samples)
    )
    
    saccade_info = {
        'saccade_times': peaks,
        'saccade_properties': properties,
        'reconstructed_signal': reconstructed_signal
    }
    return saccade_info

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

    # Use the align_trajectory function
    trial_df = align_trajectory(trial_df, stim_start_idx)

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

    # Detect saccades in angular velocity
    sampling_rate = 60  # Adjust if your data has a different sampling rate
    saccade_info = detect_saccades(angular_velocities, sampling_rate=sampling_rate)
    saccade_times = saccade_info['saccade_times']
    reconstructed_signal = saccade_info['reconstructed_signal']

    # Compute periodogram of angular velocities
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
        'time': time_array,
        'theta_corr': theta_corr,
        'angular_velocity': angular_velocities,
        'angular_time': angular_time_array,
        'accumulated_angle': accumulated_angle,
        'saccade_times': saccade_times,
        'reconstructed_signal': reconstructed_signal,
        'freqs': freqs,
        'psd': psd,
        'mean_psd_10_25Hz': mean_psd,
        'trial_number': trial
    }
    if clockwise_value:
        clockwise_trials.append(trial_dict)
    else:
        counter_clockwise_trials.append(trial_dict)

# Print the trial numbers with mean PSD less than 1 in 10-25 Hz range
print("Trials with mean PSD less than 1 in 10-25 Hz range:")
print(low_psd_trial_numbers)

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

# Function to plot angular velocities
def plot_angular_velocities(trials_list, title):
    plt.figure(figsize=(10, 6))
    all_angular_velocities = []
    all_times = []
    for trial_data in trials_list:
        angular_velocity = trial_data['angular_velocity']
        time = trial_data['angular_time']
        plt.plot(time, angular_velocity, color='grey', alpha=0.5)
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
    plt.plot(mean_time, mean_angular_velocity, color='black', linewidth=2)
    plt.ylim(-400,400)
    plt.title(f'Angular Velocity - {title}')
    plt.xlabel('Time (Frames)')
    plt.ylabel('Angular Velocity (degrees/second)')
    plt.grid(True)
    plt.show()


# Function to plot saccades
def plot_saccades(trials_list, title):
    plt.figure(figsize=(10, 6))
    for trial_data in trials_list:
        reconstructed_signal = trial_data['reconstructed_signal']
        time = trial_data['angular_time']
        saccade_times = trial_data['saccade_times']
        plt.plot(time, reconstructed_signal, color='grey', alpha=0.5)
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
        pre_stim_indices = time_array <= 0
        ax.plot(x[pre_stim_indices], y[pre_stim_indices], color='red', alpha=0.3)

        # # During stimulation: light semi-transparent green
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
    ax.set_title(title)
    ax.set_xlabel('x position (aligned)')
    ax.set_ylabel('y position (aligned)')
    ax.grid(True)
    plt.show()

# Plot Clockwise trials
if clockwise_trials:
    plot_trajectories(clockwise_trials[26:30], 'Clockwise Trials')
    print(clockwise_trials[26:27])
    plot_angular_velocities(clockwise_trials, 'Clockwise Trials')
    plot_accumulated_angles(clockwise_trials, 'Clockwise Trials')
    plot_angular_velocity_periodogram(clockwise_trials, 'Clockwise Trials')
    plot_saccades(clockwise_trials, 'Clockwise Trials')
else:
    print("No Clockwise trials found.")

# Plot Counter-Clockwise trials
if counter_clockwise_trials:
    plot_trajectories(counter_clockwise_trials, 'Counter-Clockwise Trials')
    plot_angular_velocities(counter_clockwise_trials, 'Counter-Clockwise Trials')
    plot_accumulated_angles(counter_clockwise_trials, 'Counter-Clockwise Trials')
    plot_angular_velocity_periodogram(counter_clockwise_trials, 'Counter-Clockwise Trials')
    plot_saccades(counter_clockwise_trials, 'Counter-Clockwise Trials')
else:
    print("No Counter-Clockwise trials found.")

print("\nClockwise Trials with mean PSD less than 1 in 10-25 Hz range:")
for trial_data in clockwise_trials:
    if trial_data['mean_psd_10_25Hz'] < 1:
        print(trial_data['trial_number'])

print("\nCounter-Clockwise Trials with mean PSD less than 1 in 10-25 Hz range:")
for trial_data in counter_clockwise_trials:
    if trial_data['mean_psd_10_25Hz'] < 1:
        print(trial_data['trial_number'])
