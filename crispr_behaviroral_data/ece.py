import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Define the root directory
root_directory = '/Users/tairan/Downloads/ece'

# Define pre-stimulus and post-stimulus durations (number of samples before and after the stimulation)
pre_stimulus_duration = 200  # Frames before stimulation starts
post_stimulus_duration = 200  # Frames after stimulation ends

# Define the tolerance for grouping stimulation durations (in frames)
duration_tolerance = 5  # Adjust as needed

# Walk through the directory to find and process CSV files
for root, dirs, files in os.walk(root_directory):
    for file in files:
        if file.endswith('.csv'):
            # Initialize a dictionary to store data grouped by stimulation duration
            grouped_trials = {}

            # Read each CSV file
            data = pd.read_csv(os.path.join(root, file))

            # Extract the relevant columns
            # Adjust column indices as per your CSV structure
            column1 = data.iloc[:, 0]  # Column 1
            column2 = data.iloc[:, 1]  # Column 2
            column3 = data.iloc[:, 2]  # Column 3 (Assumed to be the same as dataOri)
            dataOri = column3  # Assuming Column 3 has the original data for angular speed
            stimStatus = data.iloc[:, 4]  # Assuming 5th column has the stimulus status

            # Calculate the angular speed difference
            angularSpeed = dataOri.diff().fillna(0)  # Fill NaN with 0 for the first difference

            # Identify transitions in stimStatus
            transitions = stimStatus.diff().fillna(0)
            start_indices = transitions[transitions == 1].index.tolist()
            end_indices = transitions[transitions == -1].index.tolist()

            # Adjust for cases where stimStatus starts or ends with 1
            if stimStatus.iloc[0] == 1:
                start_indices = [stimStatus.index[0]] + start_indices
            if stimStatus.iloc[-1] == 1:
                end_indices = end_indices + [stimStatus.index[-1]]

            # Ensure the number of start and end indices match
            num_trials = min(len(start_indices), len(end_indices))
            print(f'Processing file: {file}, number of trials: {num_trials}')

            # Process each trial
            for i in range(num_trials):
                start_idx = start_indices[i]
                end_idx = end_indices[i]

                # Adjust start and end indices to include pre- and post-stimulus data
                adjusted_start_idx = max(start_idx - pre_stimulus_duration, 0)
                adjusted_end_idx = min(end_idx + post_stimulus_duration, len(dataOri) - 1)

                # Calculate the stimulation duration
                stim_duration = end_idx - start_idx

                # Extract data for the condition check
                trial_column1 = column1.loc[adjusted_start_idx:adjusted_end_idx].values
                trial_column2 = column2.loc[adjusted_start_idx:adjusted_end_idx].values
                trial_column3 = column3.loc[adjusted_start_idx:adjusted_end_idx].values

                # Compute differences between consecutive rows for each column
                diff_col1 = np.diff(trial_column1)
                diff_col2 = np.diff(trial_column2)
                diff_col3 = np.diff(trial_column3)

                # Determine stimulation period within the trial
                stim_start_within_trial = start_idx - adjusted_start_idx
                stim_end_within_trial = stim_start_within_trial + stim_duration

                # Ensure indices are within the trial length
                stim_start_within_trial = max(stim_start_within_trial, 0)
                stim_end_within_trial = min(stim_end_within_trial, len(trial_column1) - 1)

                # Extract stimulation period differences
                stim_diff_col1 = diff_col1[stim_start_within_trial:stim_end_within_trial]
                stim_diff_col2 = diff_col2[stim_start_within_trial:stim_end_within_trial]
                stim_diff_col3 = diff_col3[stim_start_within_trial:stim_end_within_trial]

                # New Condition: Check for three consecutive zero differences in all three columns
                zero_diff = (
                    (stim_diff_col1 == 0) &
                    (stim_diff_col2 == 0) &
                    (stim_diff_col3 == 0)
                ).astype(int)  # Convert boolean to integer

                window_size = 3
                kernel = np.ones(window_size, dtype=int)
                convolved = np.convolve(zero_diff, kernel, mode='valid')

                has_three_consecutive_zeros = np.any(convolved == window_size)

                if has_three_consecutive_zeros:
                    print(f"Trial {i+1} dropped due to three consecutive zero differences in columns 1, 2, and 3 during stimulation.")
                    continue  # Skip to the next trial

                # Extract angular speed data for heatmap
                trial_data = angularSpeed.loc[adjusted_start_idx:adjusted_end_idx].values

                # Extract original data for accumulated angle calculation
                trial_data_ori = dataOri.loc[adjusted_start_idx:adjusted_end_idx].values

                # Recalculate angular speed within the trial
                angularSpeed_trial = np.diff(trial_data_ori, prepend=trial_data_ori[0])

                # Compute the accumulated angle
                accumulated_angle = trial_data_ori[0] + np.cumsum(angularSpeed_trial)

                # Group trials by stimulation duration with tolerance
                # Find existing group within tolerance
                grouped = False
                for key in grouped_trials.keys():
                    if abs(stim_duration - key) <= duration_tolerance:
                        grouped_trials[key]['heatmap_data'].append(trial_data)
                        grouped_trials[key]['accumulated_angle_data'].append(accumulated_angle)
                        grouped_trials[key]['stim_durations'].append(stim_duration)
                        grouped = True
                        break
                if not grouped:
                    grouped_trials[stim_duration] = {
                        'heatmap_data': [trial_data],
                        'accumulated_angle_data': [accumulated_angle],
                        'stim_durations': [stim_duration],
                    }

            # Check if there are any trials left after grouping
            if not grouped_trials:
                print(f"No valid trials left in file: {file} after applying the condition.")
                continue  # Skip to the next file

            # Now, process each group separately
            for key_duration, data_dict in grouped_trials.items():
                heatmap_data = data_dict['heatmap_data']
                accumulated_angle_data = data_dict['accumulated_angle_data']
                stim_durations_in_group = data_dict['stim_durations']

                # Calculate average stimulation duration for the group
                avg_stim_duration = int(np.mean(stim_durations_in_group))

                print(f"Processing group with average stimulation duration: {avg_stim_duration} frames, number of trials: {len(heatmap_data)}")

                # Find the maximum trial length in this group
                max_length = max(len(trial) for trial in heatmap_data)

                # Pad each trial to the maximum length for heatmap
                padded_trials = []
                for trial in heatmap_data:
                    padded_trial = np.pad(trial, (0, max_length - len(trial)), 'constant', constant_values=np.nan)
                    padded_trials.append(padded_trial)

                # Convert to a 2D numpy array for heatmap
                trial_matrix = np.array(padded_trials)

                # Plot the heatmap for the current group
                plt.figure(figsize=(12, 8))
                ax = sns.heatmap(trial_matrix, cmap='crest', yticklabels=True, xticklabels=True)

                # Draw vertical lines at stimulation start and end
                stim_start_line = pre_stimulus_duration
                stim_end_line = stim_start_line + avg_stim_duration

                plt.axvline(x=stim_start_line, color='black', linestyle='--', label='Stimulus Start')
                plt.axvline(x=stim_end_line, color='black', linestyle='--', label='Stimulus End')

                # Adjust x and y ticks if needed
                x_ticks = ax.get_xticks()
                y_ticks = ax.get_yticks()
                ax.set_xticks(x_ticks[::10])  # Adjust step size as needed
                ax.set_yticks(y_ticks[::2])
                plt.xlabel('Time (samples)')
                plt.ylabel('Trial')
                plt.title(f'Angular Speed Heatmap for {file}\nAverage Stimulation Duration: {avg_stim_duration} frames')
                plt.legend()
                plt.show()

                # Pad accumulated angle trials to the maximum length
                padded_accumulated_trials = []
                for trial in accumulated_angle_data:
                    padded_trial = np.pad(trial, (0, max_length - len(trial)), 'constant', constant_values=np.nan)
                    padded_accumulated_trials.append(padded_trial)

                # Convert to a 2D numpy array
                accumulated_trial_matrix = np.array(padded_accumulated_trials)

                # Plot individual accumulated angles in gray
                plt.figure(figsize=(12, 8))
                for trial in accumulated_trial_matrix:
                    plt.plot(trial, color='gray', alpha=0.5)

                # Compute and plot the average accumulated angle in red
                mean_accumulated_angle = np.nanmean(accumulated_trial_matrix, axis=0)
                plt.plot(mean_accumulated_angle, color='red', linewidth=2, label='Average Accumulated Angle')

                # Draw vertical lines at stimulation start and end
                plt.axvline(x=stim_start_line, color='black', linestyle='--', label='Stimulus Start')
                plt.axvline(x=stim_end_line, color='black', linestyle='--', label='Stimulus End')
                plt.xlabel('Time (samples)')
                plt.ylabel('Accumulated Angle')
                plt.title(f'Accumulated Angle for {file}\nAverage Stimulation Duration: {avg_stim_duration} frames')
                plt.legend()
                plt.show()
