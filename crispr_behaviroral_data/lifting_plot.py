import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import seaborn as sns
from scipy.stats import ranksums

# Path to the directory containing the CSV files
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import seaborn as sns
from scipy.stats import ranksums

# Path to the directory containing the CSV files
root_directory = '/Users/tairan/Downloads/results_rnai_'  # Replace with the correct path

# Dictionary to group files and accumulate percentages
grouped_data = defaultdict(list)

# Walk through all CSV files
for subdir, _, files in os.walk(root_directory):
    for file in files:
        if file.endswith('.csv'):
            file_path = os.path.join(subdir, file)
            
            # Load the CSV file with headers
            df = pd.read_csv(file_path)  # Automatically reads headers


            
            # Drop NaN values from the 'Small Count' column
            df_non_nan = df['Small Count'].dropna()  # Keep only non-NaN values
            
            # Calculate the total number of non-NaN rows
            total_rows = len(df_non_nan)
            
            # Calculate the number of non-zero values in the 'Small Count' column
            non_zero_count = df_non_nan.astype(bool).sum()  # Count non-zero values
            
            # Calculate the percentage of non-zero values
            percentage_non_zero = ((non_zero_count) / total_rows) * 100 if total_rows > 0 else 0
            
            # Extract condition information from the filename (ignoring batch number)
            parts = file.split('_')
            condition = '_'.join(parts[1:3])  # e.g., 'gilt1-29C'
            
            # Append the percentage to the corresponding group
            grouped_data[condition].append(percentage_non_zero)

# Filter out groups where NaN values exceed 40%
filtered_grouped_data = {}

for group, percentages in grouped_data.items():
    # Count the number of NaN values
    nan_count = sum(pd.isna(percentages))
    total_trials = len(percentages)
    
    if nan_count / total_trials <= 0.99:  # Keep groups where NaNs are 40% or less
        filtered_grouped_data[group] = [p for p in percentages if not pd.isna(p)]

# Perform Wilcoxon Rank-Sum test between control and other groups
control_group = 'rnai_gal4-control'  # Replace with the actual name of your control group if needed
p_values = {}

# Ensure we have control group data
if control_group in filtered_grouped_data:
    control_data = filtered_grouped_data[control_group]
    for group, percentages in filtered_grouped_data.items():
        if group != control_group:
            # Perform the Wilcoxon Rank-Sum Test
            _, p_value = ranksums(control_data, percentages)
            p_values[group] = p_value

# Set a significance level (e.g., 0.05)
significance_level = 0.05

# Flattening the filtered_grouped_data into a format suitable for the swarmplot and violin plot
flat_group = []
flat_percentage = []

# Custom sorting function to always sort non-off first, then -off, then control
def custom_sort_key(group):
    if 'rnai_gal4-control' in group.lower():
        return (4, group)  # Control group always last
    elif '29C-off' in group:
        return (3,group)
    elif '29C' in group:
        return (2,group)
    elif '-off' in group:
        return (1, group)  # -off groups second
    else:
        return (0, group)  # Non-off groups first

# Filter data to remove values below 25th percentile and above 75th percentile
for group, percentages in filtered_grouped_data.items():
    if len(percentages) > 0:
        lower_percentile = np.percentile(percentages, 0)
        upper_percentile = np.percentile(percentages, 100)
        
        # Filter the percentages between the 25th and 75th percentiles
        filtered_percentages = [p for p in percentages if lower_percentile <= p <= upper_percentile]
        
        flat_group.extend([group] * len(filtered_percentages))  # Repeat group name for each percentage
        flat_percentage.extend(filtered_percentages)  # Extend the percentage list

# Now apply custom sorting to the flat_group and flat_percentage lists
sorted_data = sorted(zip(flat_group, flat_percentage), key=lambda x: custom_sort_key(x[0]))

# Unzipping the sorted data back into separate lists
sorted_flat_group, sorted_flat_percentage = zip(*sorted_data)

# Print the sorted group names to verify the order
sorted_group_names = sorted(set(sorted_flat_group), key=custom_sort_key)
print("Sorted group names in order:")
for group in sorted_group_names:
    print(group)

# Filter out groups with fewer than 4 data points
final_flat_group = []
final_flat_percentage = []

for group in sorted_group_names:
    group_percentages = [p for g, p in zip(sorted_flat_group, sorted_flat_percentage) if g == group]
    if len(group_percentages) >= 2:  # Only iclude groups with 4 or more data points
        final_flat_group.extend([group] * len(group_percentages))
        final_flat_percentage.extend(group_percentages)

# Now plot the violin plot, box plot, and beeswarm plot with the sorted bins
if len(final_flat_group) > 0:
    plt.figure(figsize=(14, 8))
    
    # Violin plot
    sns.violinplot(x=final_flat_group, y=final_flat_percentage, inner=None, color='lightgray')
    
    # Box plot (overlaid on the violin plot)
    sns.boxplot(x=final_flat_group, y=final_flat_percentage, width=0.1, showcaps=False, 
                boxprops={'facecolor': 'None', 'edgecolor': 'blue', 'linewidth': 2},
                showfliers=True, whiskerprops={'linewidth': 2}, 
                medianprops={'color': 'blue', 'linewidth': 2})

    # Highlight the "control" group in red on the swarmplot
    for group in set(final_flat_group):
        color = 'red' if 'control' in group.lower() else 'black'
        if group in p_values and p_values[group] < significance_level:
            color = 'green'
        sns.swarmplot(x=[g for g in final_flat_group if g == group], 
                      y=[p for g, p in zip(final_flat_group, final_flat_percentage) if g == group], 
                      color=color, alpha=0.7, size=6)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=12)

    # Add labels and title
    plt.xlabel('Condition Name', fontsize=14)
    plt.ylabel('Percentage of Non-Zero Values', fontsize=14)
    # plt.title('Filtered Violin, Box, and Beeswarm Plot of Percentages by Condition (25th to 75th Percentile)', fontsize=16)

    # Add a grid for readability
    plt.grid(True, linestyle='--', alpha=0.7)

    # Set y-axis limit
    plt.ylim(0, 40)

    # Show the plot
    plt.tight_layout()
    plt.show()
else:
    print("No groups with 4 or more data points remain after filtering.")

# Print p-values for each group compared to the control
for group, p_value in p_values.items():
    print(f'Group {group} p-value: {p_value:.4f} {"(significant)" if p_value < significance_level else ""}')
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

# Set Seaborn style for better aesthetics
sns.set(style="whitegrid")

# Path to the directory containing the CSV files
# root_directory = '/Users/tairan/Downloads/results_v1_c1'  # Replace with the correct path

# Dictionary to store merged intervals per base group
grouped_merged_frames = defaultdict(list)

# Function to handle splitting and converting frame strings to lists of integers
def parse_frame_column(frame_column):
    frames = []
    for frame in frame_column.dropna():  # Drop NaN values
        # If there are comma-separated values, split and convert them to integers
        if isinstance(frame, str):
            frames.extend([int(f.strip()) for f in frame.split(',')])
        else:
            frames.append(int(frame))  # If it's a single integer value, just append it
    return frames

# Function to merge intervals if gaps between stop and start are <= 30 frames
def merge_intervals(start_frames, stop_frames):
    if not start_frames or not stop_frames:
        return []
    
    # Sort the intervals by start_frames
    sorted_intervals = sorted(zip(start_frames, stop_frames), key=lambda x: x[0])
    
    merged_intervals = []
    current_start, current_stop = sorted_intervals[0]
    
    for next_start, next_stop in sorted_intervals[1:]:
        if next_start - current_stop <= 30:
            # Merge the intervals
            current_stop = max(current_stop, next_stop)
        else:
            merged_intervals.append((current_start, current_stop))
            current_start, current_stop = next_start, next_stop
    
    # Append the last interval
    merged_intervals.append((current_start, current_stop))
    
    return merged_intervals

# Function to extract base group name by removing '-off' suffix if present
def extract_base_group(file_name):
    parts = file_name.split('_')
    if len(parts) < 3:
        print(f"Warning: Filename '{file_name}' does not have enough parts to extract group name.")
        return None
    # Extract the relevant parts (adjust indices based on your filename structure)
    group_name = '_'.join(parts[1:3])  # Example: 'gilt1_off' or 'gilt1'
    # Remove '-off' suffix if present
    base_group = group_name.lower().replace('-off', '').strip()
    return base_group

# Walk through all CSV files and group them by base group name
for subdir, _, files in os.walk(root_directory):
    for file in files:
        if file.endswith('.csv'):
            file_path = os.path.join(subdir, file)
            
            # Load the CSV file with headers
            try:
                df = pd.read_csv(file_path)  # Automatically reads headers
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue  # Skip this file if there's an error
            
            # Check if required columns exist
            required_columns = ['Small Count', 'Start Frames', 'Stop Frames']
            if not all(col in df.columns for col in required_columns):
                print(f"Skipping file: {file} (required columns missing)")
                continue
            
            # Filter out rows where 'Small Count' is NaN
            df_filtered = df.dropna(subset=['Small Count'])
            
            # Parse the 'Start Frames' and 'Stop Frames' columns after filtering
            start_frames = parse_frame_column(df_filtered['Start Frames'])
            stop_frames = parse_frame_column(df_filtered['Stop Frames'])
            
            # Skip files with no valid start or stop frames
            if not start_frames or not stop_frames:
                print(f"Skipping file: {file} (no valid start or stop frames)")
                continue
            
            # Extract base group name from filename
            base_group = extract_base_group(file)
            if not base_group:
                print(f"Skipping file: {file} (unable to extract base group name)")
                continue
            
            # Check if the file is an 'off' condition or not
            if '-off' in file.lower():
                # If it's an 'off' file, use the frames as they are
                merged_intervals = merge_intervals(start_frames, stop_frames)
            else:
                # If it's a non-'off' file, adjust the start and stop frames by adding 120
                adjusted_start = [start + 120 for start in start_frames]
                adjusted_stop = [stop + 120 for stop in stop_frames]
                merged_intervals = merge_intervals(adjusted_start, adjusted_stop)
            
            # Append the merged intervals to the corresponding base group
            grouped_merged_frames[base_group].extend(merged_intervals)

# Determine the maximum frame number across all groups
max_frame = 0
for intervals in grouped_merged_frames.values():
    for start, stop in intervals:
        if stop > max_frame:
            max_frame = stop

# Create a matrix where each row represents a base group and columns represent frames
# Initialize the DataFrame with zeros
heatmap_data = pd.DataFrame(0, index=grouped_merged_frames.keys(), columns=np.arange(max_frame + 1))

# Fill in the matrix: set cells to the count of overlapping events
for group, intervals in grouped_merged_frames.items():
    for start, stop in intervals:
        # Ensure start and stop are within the frame range
        start = max(start, 0)
        stop = min(stop, max_frame)
        # Increment the cell value by 1 for each event overlapping the frame
        heatmap_data.loc[group, start:stop] += 1  # Increment by 1 for each event

# Plot the heatmap
plt.figure(figsize=(20, max(5, len(heatmap_data) * 0.5)))  # Adjust height based on number of groups

# Choose a different colormap, e.g., 'viridis', 'magma', 'inferno', 'plasma', 'cividis', etc.
# Alternatively, use a custom colormap for better distinction
cmap = sns.color_palette("light:b", as_cmap=True)  # Changed from "Blues" to "viridis"

sns.heatmap(
    heatmap_data,
    cmap=cmap,
    cbar=True,
    linewidths=0,            # Removed grid lines by setting linewidths to 0
    linecolor='white',       # Optional: color won't matter since linewidths=0
    yticklabels=True,
    vmin=1,                  # Set the lower bound of the color scale
    vmax=10           # Set the upper bound of the color scale
)
# Add a vertical line at x = 120
plt.axvline(x=120, color='red', linestyle='--', lw=2)  # Red dashed line at x=120

# Adjust x-axis ticks if necessary
# plt.xticks(rotation=90, fontsize=10)
# plt.yticks(rotation=0, fontsize=10)

# Add labels and title
plt.xlabel('Frame Index', fontsize=14)
plt.ylabel('Group Name', fontsize=14)
# plt.title('Heatmap of Merged Events per Group (Start to Stop Frames)', fontsize=16)

# Adjust x-axis ticks if necessary
plt.xticks(rotation=90, fontsize=10)
plt.yticks(rotation=0, fontsize=10)

plt.tight_layout()
plt.show()
