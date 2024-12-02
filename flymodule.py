import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.signal import periodogram, find_peaks, savgol_filter
import os
import matplotlib.pyplot as plt
import glob
import pingouin as pg
import seaborn as sns  # Import seaborn for plotting
import os
import glob
import numpy as np
import matplotlib.pyplot as plt


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



# Plotting function (as provided in your original code)
def plot_trajectories(trials_list, title):

    # Set the frame limits relative to stimulus onset
    frames_before_stim = 239
    frames_after_stim = 1
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
        time_array = trial_data.get('time', np.arange(-frames_before_stim, len(x_smoothed_aligned) - frames_before_stim))
        # time_array = trial_data.get('time', np.arange(0, len(x_smoothed_aligned) ))

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
        # pre_stim_indices = time_array <= 0
        pre_stim_indices = time_array <= 0
        ax.plot(x_smoothed_aligned[pre_stim_indices], y_smoothed_aligned[pre_stim_indices], color='red', alpha=0.3)

        # post_stim_indices = time_array >= 0
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
    plt.savefig(f"/Users/tywentking/Downloads/roshan3/{title}_traj.png")



# Updated Function to align smoothed trajectory and set stimulus onset at (0, 0)
def align_smoothed_trajectory(trial_df, stim_start_idx, window_length=11, polyorder=3):
    # Smooth the trajectory using Savitzky-Golay filter
    if len(trial_df['pos_x']) < window_length or len(trial_df['pos_y']) < window_length:
        print(f"Trial {trial_df['Trial_Num'].iloc[0]}: Skipping trajectory due to insufficient data.")
        return None 
    else:

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
def align_smoothed_trajectory_prestim(trial_df, stim_start_idx, window_length=11, polyorder=3):
    # Only use data up to stimulus onset
    pre_stim_df = trial_df.iloc[:stim_start_idx+1].copy().reset_index(drop=True)
    
    # Check if we have enough data to process
    min_length = max(window_length, 4)
    if len(pre_stim_df) < min_length:
        print(f"Trial {trial_df['Trial_Num'].iloc[0]}: Skipping trajectory due to insufficient pre-stim data.")
        return None
    
    # Smooth the trajectory using Savitzky-Golay filter
    pre_stim_df['x_smoothed'] = savgol_filter(pre_stim_df['pos_x'], window_length=window_length, polyorder=polyorder)
    pre_stim_df['y_smoothed'] = savgol_filter(pre_stim_df['pos_y'], window_length=window_length, polyorder=polyorder)
    
    # Compute displacement vector between frames
    idx2 = len(pre_stim_df) - 1  # Last frame index
    idx1 = idx2 - 3              # 3 frames before the last
    if idx1 < 0:
        idx1 = 0  # Ensure idx1 is not negative
    
    dx = pre_stim_df['x_smoothed'].iloc[idx2] - pre_stim_df['x_smoothed'].iloc[idx1]
    dy = pre_stim_df['y_smoothed'].iloc[idx2] - pre_stim_df['y_smoothed'].iloc[idx1]
    
    # Calculate angle and rotation matrix
    angle_to_align = 0.0 if np.hypot(dx, dy) < 1e-4 else np.degrees(np.arctan2(dy, dx))
    rotation_angle = 90 - angle_to_align
    rotation_matrix = np.array([
        [np.cos(np.radians(rotation_angle)), -np.sin(np.radians(rotation_angle))],
        [np.sin(np.radians(rotation_angle)), np.cos(np.radians(rotation_angle))]
    ])
    
    # Apply rotation to smoothed coordinates
    coords = np.c_[pre_stim_df['x_smoothed'], pre_stim_df['y_smoothed']]
    rotated_coords = np.dot(coords, rotation_matrix.T)
    
    # Shift coordinates to make stimulus onset at (0, 0)
    x_shift = rotated_coords[-1, 0]  # Last point is at stim_start_idx
    y_shift = rotated_coords[-1, 1]
    pre_stim_df['x_aligned'] = rotated_coords[:, 0] - x_shift
    pre_stim_df['y_aligned'] = rotated_coords[:, 1] - y_shift
    
    return pre_stim_df

# Function to perform stationary wavelet transform denoising
class Fly_dynamics:
    def __init__(self, df,frame_rate=60):
        self.df = df
        self.frame_rate= frame_rate
        
    # def correct_opencv_theta_assignment(self,df_trial):
    def correct_opencv_theta_assignment(self, df):
        factor = 1
        # Get the index of the first valid 'ori' value
        idx = df['ori'].first_valid_index()
        if idx is None:
            # Handle the case where 'ori' has no valid entries
            df['theta_corr'] = np.nan
            return df
        cumulative_theta = df['ori'].loc[idx]
        theta_corr = [cumulative_theta * factor]
        # Iterate over the indices starting from the first valid index
        for i in range(idx + 1, len(df)):
            curr_ori = df['ori'].iloc[i]
            prev_ori = df['ori'].iloc[i - 1]
            if pd.isna(curr_ori) or pd.isna(prev_ori):
                difference = 0
            else:
                difference = curr_ori - prev_ori
                if difference > 90:
                    difference -= 180
                elif difference < -90:
                    difference += 180
            cumulative_theta += difference
            theta_corr.append(cumulative_theta * factor)
        # Pad the beginning of theta_corr with NaNs to match the length of df
        theta_corr = [np.nan] * idx + theta_corr
        df['theta_corr'] = theta_corr
        # print(df)
        return df

    def angular_speed(self, df_trial,window_length=22, polyorder=3, smoothing = True):
        # Calculate angular speed (difference of theta_corFabsr, scaled by frame rate)
        angular_speed = np.diff(df_trial["theta_corr"]) * self.frame_rate
        if smoothing:
            # Smooth the angular speed using a Savitzky-Golay filter
            smoothed_angular_speed = savgol_filter(angular_speed, window_length, polyorder)
        
        # Pad the result to match the DataFrame's length
            smoothed_angular_speed = np.append(smoothed_angular_speed, np.nan)  # Add NaN for the last missing value
        
        # Add the smoothed angular speed as a new column
            df_trial["angular_speed"]=smoothed_angular_speed
        else:
            df_trial["angular_speed"]=angular_speed
        
        return df_trial  
    def fwd_speed(self, df_trial,window_length=22, polyorder=3, smoothing = True):
                # Compute displacements during stimulation
        dx = np.diff(df_trial['x_aligned'])
        dy = np.diff(df_trial['y_aligned'])
        frame_number = df_trial['frame_number']
        total_length = len(frame_number)
        if total_length < 2:
            return False
        fwd_speed = np.sqrt(dx**2 + dy**2)*self.frame_rate
        if smoothing:
            smoothed_fwd_speed=savgol_filter(fwd_speed, window_length, polyorder)
            smoothed_fwd_speed = np.append(smoothed_fwd_speed, np.nan)
            df_trial["fwd_speed"]=smoothed_fwd_speed
        else:
            df_trial["fwd_speed"]=fwd_speed
        return df_trial
    def is_fly_responsive_dual(self, trial, v_threshold=10, angular_threshold=20,
                            check_angular_only=True, check_fwd_v_only=False, check_both=False):
        frame_number = trial['frame_number']
        # print(trial['frame_number'])
        total_length = len(frame_number)
        if total_length < 2:
            return False
        
        # Compute angular speed and update df_trial
        self.angular_speed(trial,smoothing=True)
        mean_angular_speed = np.nanmean(trial['angular_speed'])
        # print(mean_angular_speed)
        
        # Compute forward speed and update df_trial
        self.fwd_speed(trial,smoothing=True)
        mean_fwd_speed = np.nanmean(trial['fwd_speed'])
        
        if check_angular_only:
            # print(f"check_angular_only: {mean_angular_speed >= angular_threshold}")
            return abs(mean_angular_speed) >= angular_threshold
        elif check_fwd_v_only:
            # print(f"check_fwd_v_only: {mean_fwd_speed >= v_threshold}")
            return abs(mean_fwd_speed) >= v_threshold
        elif check_both:
            # print(f"check_both:")
            both=(mean_angular_speed >= angular_threshold) and (mean_fwd_speed >= v_threshold)
            # print(f"check_both:{both}")
            return both
    def pre_stim_state(self, trial, v_threshold=7, angular_threshold=20,
                            check_angular_only=True, check_fwd_v_only=False, check_both=False):
        frame_number = trial['frame_number']
        total_length = len(frame_number)
        if total_length < 2:
            return False
        
        # Compute angular speed and update df_trial
        self.angular_speed(trial,smoothing=True)
        mean_angular_speed = np.nanmean(trial['angular_speed'])
        # print(mean_angular_speed)
        
        # Compute forward speed and update df_trial
        self.fwd_speed(trial,smoothing=True)
        mean_fwd_speed = np.nanmean(trial['fwd_speed'])
        
        if check_angular_only:
            # print(f"check_angular_only: {mean_angular_speed >= angular_threshold}")
            return abs(mean_angular_speed) >= angular_threshold
        elif check_fwd_v_only:
            # print(f"check_fwd_v_only: {mean_fwd_speed >= v_threshold}")
            return abs(mean_fwd_speed) >= v_threshold
        elif check_both:
            # print(f"check_both:")
            both=(mean_angular_speed >= angular_threshold) and (mean_fwd_speed >= v_threshold)
            # print(f"check_both:{both}")
            return both

        
    def calculate_rolling_straightness(self, df_trial,window_size, step_size=1):

        '''
        Calculate the straightness of the trajectory over a rolling window basis.
        Returns an array of straightness values.
        '''
        x = df_trial['pos_x']
        y = df_trial['pos_y']

        n_points = len(x)
        for start_idx in range(0, n_points - window_size + 1, step_size):
            end_idx = start_idx + window_size
            x_window = x[start_idx:end_idx]
            y_window = y[start_idx:end_idx]
            
            # Net displacement D
            D = np.sqrt((x_window[-1] - x_window[0])**2 + (y_window[-1] - y_window[0])**2)
            straightness_values = []
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


    def swt_denoising(self, data, wavelet='bior2.6', level=7):
        
        if data.size == 0:
            return data
        max_level = pywt.swt_max_level(len(data))
        if max_level == 0:
            # print("Data length is too short for any level of SWT decomposition.")
            return data
        level = min(level, max_level)
        # self.plot_swt_scalogram(self,data,wavelet)
        freq = pywt.scale2frequency(wavelet, 2 ** np.arange(0, level)[::-1]) / 0.016
        # print(f"freq:{freq[0:5]}")
        swt_coeffs = pywt.swt(data, wavelet, level=level)
        index = np.argwhere((freq > 10) & (freq < 20)).flatten()
        swt_filtered = []
        for i, coeff in enumerate(swt_coeffs):
            if i in index:
                swt_filtered.append(coeff)
            else:
                swt_filtered.append((np.zeros_like(coeff[0]), np.zeros_like(coeff[1])))
        re_data = pywt.iswt(swt_filtered, wavelet)
        return re_data

            # Function to detect saccades using peak detection
    def detect_saccades(self, angular_speed, threshold, width=[5, 30]):
        height = threshold
        peaks_syn, _ = find_peaks(
            angular_speed,
            height=height,
            distance=5,
            prominence=0.8 * height,
            wlen=30,
            width=width,
            rel_height=1,
        )
        peaks_anti, _ = find_peaks(
            -angular_speed,
            height=height,
            distance=5,
            prominence=0.8 * height,
            wlen=30,
            width=width,
            rel_height=1,
        )
        # Merge peaks
        all_peaks = np.concatenate((peaks_syn, peaks_anti))
        all_peak_values = np.concatenate((angular_speed[peaks_syn], angular_speed[peaks_anti]))
        # Sort the peaks
        sorted_indices = np.argsort(all_peaks)
        all_peaks = all_peaks[sorted_indices]
        all_peak_values = all_peak_values[sorted_indices]
        return all_peaks, all_peak_values, peaks_syn, peaks_anti

    def detect_saccades_cwt(self, df_trial,threshold):
        angular_speed = df_trial["angular_speed"]
        denoised_signal = self.swt_denoising(angular_speed, level=5)
        # threshold = 4 * np.median(np.abs(angular_speed) / 0.6745)
        
        # Now pass arrays to detect_saccades
        cwt_peaks, cwt_peak_values, _, _ = self.detect_saccades(denoised_signal, threshold)
        peaks, peak_values, peaks_syn, peaks_anti = self.detect_saccades(
            angular_speed, threshold=threshold, width=[5, 30]
        )
    # Rest of your matching peaks code...

        
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
        
        # Create sets for peaks_syn and peaks_anti for faster lookup
        set_peaks_syn = set(peaks_syn)
        set_peaks_anti = set(peaks_anti)
        
        # Separate final_peaks into final_peaks_syn and final_peaks_anti
        final_peaks_syn = []
        final_peaks_anti = []
        for peak in final_peaks:
            if peak in set_peaks_syn:
                final_peaks_syn.append(peak)
            elif peak in set_peaks_anti:
                final_peaks_anti.append(peak)
        
        return (
            np.array(final_peaks),
            np.array(final_peak_values),
            denoised_signal,
            np.array(final_peaks_syn),
            np.array(final_peaks_anti),
        )
    def process_direction_transitions(self, df_stimulus, frequencies):
            # Initialize 'Trial_Num' and 'frequency_stim' columns in df_trial
            self.df['Trial_Num'] = np.nan
            self.df['frequency_stim'] = np.nan
            trials_list = []
            trials_by_frequency = {freq: [] for freq in frequencies}
            trial_num = 1  # Initialize trial number

            # Process direction transitions for 0 to 1.0 and 0 to -1.0
            for direction_value in [1.0, -1.0]:
                i = 0
                while i < len(self.df["direction"]) - 1:
                    if self.df["direction"].iloc[i] == 0 and self.df["direction"].iloc[i + 1] == direction_value:
                    # if self.df["direction"].iloc[i] == direction_value and self.df["direction"].iloc[i + 1] == 0:
                        print("aaaaaaaaaaaaaaaaaaaaaaaaa")


                        start_point = i
                        # Find the end point for this trial
                        i += 1
                        while i < len(self.df["direction"]) - 1 and self.df["direction"].iloc[i] == direction_value:
                            i += 1
                        end_point = i

                        # Extract the trial DataFrame
                        df_trial = self.df.iloc[start_point:end_point].copy().reset_index(drop=True)
                        # Correct 'theta' and align the smoothed trajectory
                        # df_trial = self.correct_opencv_theta_assignment(df_trial)
                        df_trial = align_smoothed_trajectory(df_trial, 10)

                        # Skip the trial if alignment failed (None returned)
                        if df_trial is None:
                            print(f"Skipping trial {trial_num} due to insufficient data.")
                            trial_num += 1
                            continue

                        # Get frequency values
                        frequency_values = df_stimulus[df_stimulus["trial_number"] == trial_num]["temporal_frequency"].values
                        frequency_values = [int(float(value)) for value in frequency_values]

                        if frequency_values:
                            freq = frequency_values[0]
                        else:
                            freq = None

                        if freq not in frequencies:
                            trial_num += 1
                            continue  # Skip frequencies not in our list

                        # Get the indices in self.df for this trial
                        indices = self.df.index[start_point:end_point]

                        # Assign 'Trial_Num' and 'frequency_stim' to self.df
                        self.df.loc[indices, 'Trial_Num'] = trial_num
                        self.df.loc[indices, 'frequency_stim'] = freq
                        # angular_speed = np.diff(df_trial["theta_corr"]) * 60
                        # Also assign to df_trial for consistency
                        df_trial['Trial_Num'] = trial_num
                        df_trial['frequency_stim'] = freq

                        # Create 'trial_data' dictionary
                        trial_data = {
                            'x_aligned': df_trial['x_aligned'].values,
                            'y_aligned': df_trial['y_aligned'].values,
                            'frame_number': df_trial['frame_number'].values,
                            'theta_corr': df_trial['theta_corr'].values,
                            'trial_number': np.array(trial_num),
                            'frequency_stim': freq,
                            # "angular_speed": angular_speed
                        }

                        # Adjust x_aligned for -1.0 direction
                        if direction_value == -1.0:
                            trial_data['x_aligned'] = -trial_data['x_aligned']
                            trial_data['theta_corr'] = -trial_data['theta_corr']

                        # Append to trials_list and trials_by_frequency
                        trials_list.append(trial_data)
                        trials_by_frequency[freq].append(trial_data)

                        trial_num += 1  # Increment trial number
                    else:
                        i += 1  # Move to the next index

            # Return self.df with the new columns, along with other outputs
            return self.df, trials_list, trials_by_frequency









    # def process_direction_transitions(self, df_stimulus, frequencies, stim=True):
    #     """
    #     Processes direction transitions in the DataFrame to identify trials based on stimulus conditions.

    #     Parameters:
    #     - df_stimulus (pd.DataFrame): DataFrame containing stimulus conditions.
    #     - frequencies (list): List of frequencies to consider.
    #     - stim (bool): Flag indicating the direction of processing.

    #     Returns:
    #     - pd.DataFrame: Updated DataFrame with 'Trial_Num' and 'frequency_stim' columns.
    #     - list: List of trial data dictionaries.
    #     - dict: Dictionary categorizing trials by frequency.
    #     """
    #     # Initialize 'Trial_Num' and 'frequency_stim' columns in self.df
    #     self.df['Trial_Num'] = np.nan
    #     self.df['frequency_stim'] = np.nan

    #     trials_list = []
    #     trials_by_frequency = {freq: [] for freq in frequencies}
    #     trial_num = -1  # Initialize trial number

    #     # Process direction transitions based on the 'stim' flag
    #     for direction_value in [1.0, -1.0]:
    #         i = 0

    #         if stim:
    #             condition = (self.df["direction"].iloc[i] == 0) and (self.df["direction"].iloc[i + 1] == direction_value)
    #         else:
    #             condition = (self.df["direction"].iloc[i] == direction_value) and (self.df["direction"].iloc[i + 1] == 0)
    #         while i < len(self.df["direction"]) - 1:

    #             if condition:
    #                 start_point = i
    #                 # Find the end point for this trial
    #                 i += 1
    #                 while i < len(self.df["direction"]) - 1 and (
    #                     (self.df["direction"].iloc[i] == direction_value if stim else self.df["direction"].iloc[i] == 0)
    #                 ):
    #                     i += 1
    #                 end_point = i

    #                 # Extract the trial DataFrame
    #                 df_trial = self.df.iloc[start_point:end_point].copy().reset_index(drop=True)
    #                 df_trial = align_smoothed_trajectory(df_trial, 10)

    #                 # Skip the trial if alignment failed (None returned)
    #                 if df_trial is None:
    #                     print(f"Skipping trial {trial_num} due to insufficient data.")
    #                     trial_num += 1
    #                     continue

    #                 # Get frequency values for the current trial
    #                 frequency_values = df_stimulus[df_stimulus["trial_number"] == trial_num]["temporal_frequency"].values
    #                 frequency_values = [int(float(value)) for value in frequency_values]

    #                 if frequency_values:
    #                     freq = frequency_values[0]
    #                 else:
    #                     freq = None

    #                 if freq not in frequencies:
    #                     print(f"Skipping trial {trial_num} due to frequency {freq} not in specified frequencies.")
    #                     trial_num += 1
    #                     continue  # Skip frequencies not in our list

    #                 # Get the indices in self.df for this trial
    #                 indices = self.df.index[start_point:end_point]

    #                 # Assign 'Trial_Num' and 'frequency_stim' to self.df
    #                 self.df.loc[indices, 'Trial_Num'] = trial_num
    #                 self.df.loc[indices, 'frequency_stim'] = freq

    #                 # Calculate angular_speed
    #                 if 'theta_corr' in df_trial.columns:
    #                     angular_speed = np.diff(df_trial["theta_corr"]) * 60
    #                 else:
    #                     angular_speed = np.array([])  # Handle missing 'theta_corr' gracefully

    #                 # Create 'trial_data' dictionary
    #                 trial_data = {
    #                     'x_aligned': df_trial['x_aligned'].values if 'x_aligned' in df_trial.columns else np.array([]),
    #                     'y_aligned': df_trial['y_aligned'].values if 'y_aligned' in df_trial.columns else np.array([]),
    #                     'frame_number': df_trial['frame_number'].values if 'frame_number' in df_trial.columns else np.array([]),
    #                     'theta_corr': df_trial['theta_corr'].values if 'theta_corr' in df_trial.columns else np.array([]),
    #                     'trial_number': trial_num,
    #                     'frequency_stim': freq,
    #                     'angular_speed': angular_speed
    #                 }

    #                 # Adjust x_aligned and angular_speed for -1.0 direction
    #                 if direction_value == -1.0:
    #                     trial_data['x_aligned'] = -trial_data['x_aligned']
    #                     trial_data['angular_speed'] = -trial_data['angular_speed']

    #                 # Append to trials_list and trials_by_frequency
    #                 trials_list.append(trial_data)
    #                 trials_by_frequency[freq].append(trial_data)
    #                 print(f"Processed trial {trial_num} with frequency {freq}.")
    #                 trial_num += 1

    #             else:
    #                 i += 1  # Move to the next index if condition is not met

    #     # Return self.df with the new columns, along with other outputs
    #     return self.df, trials_list, trials_by_frequency











def plot_saccade(responsive_trials,fly,file_name,freq):
    # Initialize saccade occurrence matrices per trial and frame
    # Calculate straightness for responsive trials
    freq_straightness_values = []
    fly
    angular_speeds=[]
    peak_index = []
    peak_index_syn = []
    peak_index_anti = []
    for trial in responsive_trials:
        df_trial = pd.DataFrame(trial)
        angular_speed = np.diff(trial["theta_corr"]) * 60
        # angular_speeds2.append(np.diff(trial["theta_corr"]) * 60)
        angular_speeds.append(fly.swt_denoising(angular_speed, level=5))

        threshold_saccades = 4 * np.median(np.abs(angular_speed) / 0.6745)
        
        if len(df_trial["theta_corr"]) < 10:  # Adjust the threshold as needed
            print(f"Skipping trial {trial['trial_number']} due to insufficient data length.")
            continue
        # print("---------------------------------")
        final_peaks, _, _, final_peaks_syn, final_peaks_anti = fly.detect_saccades_cwt(df_trial,threshold_saccades)

        peak_index.append(final_peaks)
        peak_index_syn.append(final_peaks_syn)
        peak_index_anti.append(final_peaks_anti)
        num_responsive_trials = len(responsive_trials)
        max_frame_length = max(len(trial['theta_corr']) for trial in responsive_trials)
        saccade_matrix_syn = np.zeros((num_responsive_trials, max_frame_length))
        saccade_matrix_anti = np.zeros((num_responsive_trials, max_frame_length))
    
    # For each trial, mark saccade occurrences in the matrices
    for idx, (trial, peaks_syn_in_trial, peaks_anti_in_trial) in enumerate(zip(responsive_trials, peak_index_syn, peak_index_anti)):
        trial_length = len(trial['theta_corr'])
        for i in peaks_syn_in_trial:
            # Expand peaks by ±3 frames
            start = max(0, i - 15)
            end = min(trial_length, i + 15 + 1)
            saccade_matrix_syn[idx, start:end] = 1
        for i in peaks_anti_in_trial:
            start = max(0, i - 15)
            end = min(trial_length, i + 15 + 1)
            saccade_matrix_anti[idx, start:end] = 1
    
    # Create boolean matrices
    saccade_matrix_syn_bool = saccade_matrix_syn == 1
    saccade_matrix_anti_bool = saccade_matrix_anti == 1
    # print(saccade_matrix_syn_bool)

    both = saccade_matrix_syn_bool & saccade_matrix_anti_bool
    syn_only = saccade_matrix_syn_bool 
    anti_only = saccade_matrix_anti_bool
    
    # Counts per frame
    counts_both = np.sum(both, axis=0)
    counts_syn_only = np.sum(syn_only, axis=0)
    counts_anti_only = np.sum(anti_only, axis=0)
    
    # Fractions per frame
    fraction_syn_only = counts_syn_only / num_responsive_trials
    fraction_anti_only = counts_anti_only / num_responsive_trials
    
    # Plot the fractions per frame
    plt.figure(figsize=(10, 6), dpi=1500)
    frame_numbers = np.arange(max_frame_length)

    # Prepare the fractions in the correct stacking order
    fractions = [fraction_syn_only,fraction_anti_only , 1-fraction_anti_only-fraction_syn_only]

    # Plot using stackplot
    plt.stackplot(frame_numbers, fractions, labels=['Syn Saccades', 'Anti Saccades', 'Smooth Turning'], baseline='zero',colors=['blue', 'red', 'green'],alpha=0.5)
    # plt.ylim(0, 1)  
    plt.xlabel('Frame Number')
    plt.ylabel('Fraction of Trials')
    plt.title(f'Fraction of Trials with Saccades at Each Frame for {file_name} at {freq} Hz')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"/Users/tywentking/Downloads/roshan3/{file_name}_freq_{freq}_saccade_fraction_anti_syn.png")
    plt.close()
