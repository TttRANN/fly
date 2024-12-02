import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

# Set Seaborn style for better aesthetics
sns.set(style="whitegrid")

def merge_intervals(start_frames, stop_frames, max_gap=5):
    """
    Merges intervals where the gap between consecutive intervals is less than or equal to max_gap.
    
    Parameters:
    - start_frames (list of int): Start frames of the intervals.
    - stop_frames (list of int): Stop frames of the intervals.
    - max_gap (int): Maximum allowed gap between intervals to consider merging.
    
    Returns:
    - merged_intervals (list of tuples): List of merged (start, stop) tuples.
    """
    if not start_frames or not stop_frames:
        return []
    
    # Sort the intervals by start_frames
    sorted_intervals = sorted(zip(start_frames, stop_frames), key=lambda x: x[0])
    
    merged_intervals = []
    current_start, current_stop = sorted_intervals[0]
    
    for next_start, next_stop in sorted_intervals[1:]:
        if next_start - current_stop <= max_gap:
            # Merge the intervals
            current_stop = max(current_stop, next_stop)
        else:
            merged_intervals.append((current_start, current_stop))
            current_start, current_stop = next_start, next_stop
    
    # Append the last interval
    merged_intervals.append((current_start, current_stop))
    
    return merged_intervals

def parse_frame_entry(frame_entry):
    """
    Parses a single frame entry which can be a single integer, a comma-separated string of integers, or a list.
    Returns a list of integers.
    """
    if pd.isna(frame_entry):
        return []
    elif isinstance(frame_entry, list):
        # If the entry is a list (e.g., from JSON-like strings)
        return [int(f) for f in frame_entry]
    elif isinstance(frame_entry, str):
        # Remove any surrounding brackets or whitespace
        frame_entry = frame_entry.strip('[]').strip()
        if not frame_entry:
            return []
        # Split by comma and convert to integers
        return [int(f.strip()) for f in frame_entry.split(',')]
    else:
        # If it's a single number (int or float), convert to int
        return [int(frame_entry)]

import os

def extract_base_group(file_name):
    parts = file_name.lower().replace('-off', '').replace('.csv', '').split('_')
    print(f"{parts[2]}_{parts[5]}_{parts[6]}")
    return f"{parts[2]}_{parts[5]}_{parts[6]}"


import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd

def generate_heatmap(heatmap_data, file_pair_name, output_dir, num_frames=360, num_rows_const=100):
    """
    Generates and saves a heatmap with constant rows and columns from the provided data.
    
    Parameters:
    - heatmap_data (pd.DataFrame): DataFrame containing the heatmap data.
    - file_pair_name (str): Name to be used for the heatmap image (usually the base group name).
    - output_dir (str): Directory where the heatmap image will be saved.
    - num_frames (int): Constant number of frames (columns) to be displayed in the heatmap.
    - num_rows_const (int): Constant number of rows to be displayed in the heatmap.
    """
    # Determine the current number of rows and columns in the input data
    num_rows, num_cols = heatmap_data.shape
    
    # If the number of columns is less than num_frames, pad with zeros
    if num_cols < num_frames:
        padding_cols = pd.DataFrame(0, index=heatmap_data.index, columns=range(num_cols, num_frames))
        heatmap_data = pd.concat([heatmap_data, padding_cols], axis=1)
    
    # If the number of columns is greater than num_frames, trim excess columns
    elif num_cols > num_frames:
        heatmap_data = heatmap_data.iloc[:, :num_frames]
    
    # If the number of rows is less than num_rows_const, pad with zeros
    if num_rows < num_rows_const:
        padding_rows = pd.DataFrame(0, index=range(num_rows, num_rows_const), columns=heatmap_data.columns)
        heatmap_data = pd.concat([heatmap_data, padding_rows], axis=0)
    
    # If the number of rows is greater than num_rows_const, trim excess rows
    elif num_rows > num_rows_const:
        heatmap_data = heatmap_data.iloc[:num_rows_const, :]
    
    # Set figure size based on the number of rows
    plt.figure(figsize=(50, max(2, num_rows_const * 0.25)))
    
    # Choose an appropriate colormap
    cmap = sns.color_palette("Blues", as_cmap=True)
    
    # Plot the heatmap
    ax = sns.heatmap(
        heatmap_data,
        cmap=cmap,
        cbar=True,
        linewidths=0.5,  # This controls the grid thickness
        linecolor='gray',  # This controls the grid color
        yticklabels=True,
        vmin=0,
        vmax=1
    )
    
    # Add grid lines manually
    ax.grid(True, which='both', axis='both', linestyle='-', color='gray', linewidth=0.5)

    # Set the aspect ratio of the heatmap to control row height
    ax.set_aspect(aspect='auto')
    
    # Add labels and title
    plt.xlabel('Frame Index', fontsize=50)
    plt.ylabel('Row Index', fontsize=50)
    plt.title(f'Heatmap for {file_pair_name}', fontsize=50)
    
    # Adjust x and y ticks
    plt.xticks(rotation=90, fontsize=20)
    plt.yticks(rotation=0, fontsize=20)
    plt.axvline(x=120, color='red', linestyle='--', lw=2)  # Red dashed line at x=120

    plt.tight_layout()
    
    # Save the heatmap as a PNG image
    heatmap_filename = f'heatmap_{file_pair_name}.png'
    heatmap_path = os.path.join(output_dir, heatmap_filename)
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to {heatmap_path}")


def process_file_pair(off_file_path, on_file_path, base_group, output_dir):
    """
    Processes a pair of '-off' and non-off CSV files to generate a combined heatmap.
    
    Parameters:
    - off_file_path (str): Path to the '-off' CSV file.
    - on_file_path (str): Path to the non-off CSV file.
    - base_group (str): Base group name for labeling.
    - output_dir (str): Directory where the heatmap image will be saved.
    """
    # Initialize lists to hold merged intervals for each row
    merged_intervals_off = []
    merged_intervals_on = []
    
    # Process '-off' file
    if off_file_path:
        try:
            df_off = pd.read_csv(off_file_path)
        except Exception as e:
            print(f"Error reading '{off_file_path}': {e}")
            return
        
        # Check for required columns
        required_columns = ['Video Filename', 'Small Count', 'Start Frames', 'Stop Frames']
        if not all(col in df_off.columns for col in required_columns):
            print(f"Skipping file: {off_file_path} (required columns missing)")
            return
        
        # Filter out rows where 'Small Count' is NaN
        df_off_filtered = df_off.dropna(subset=['Small Count'])
        
        # Iterate over each row to parse and merge intervals
        for idx, row in df_off_filtered.iterrows():
            start_entry = row['Start Frames']
            stop_entry = row['Stop Frames']
            
            start_frames = parse_frame_entry(start_entry)
            stop_frames = parse_frame_entry(stop_entry)
            
            # Ensure that start_frames and stop_frames have the same length
            if len(start_frames) != len(stop_frames):
                print(f"Row {idx} in '{off_file_path}' has mismatched Start and Stop Frames.")
                merged_intervals_off.append([])  # Append empty list for this row
                continue
            
            # Merge intervals
            merged = merge_intervals(start_frames, stop_frames, max_gap=5)
            merged_intervals_off.append(merged)
    
    # Process non-off file
    if on_file_path:
        try:
            df_on = pd.read_csv(on_file_path)
        except Exception as e:
            print(f"Error reading '{on_file_path}': {e}")
            return
        
        # Check for required columns
        required_columns = ['Video Filename', 'Small Count', 'Start Frames', 'Stop Frames']
        if not all(col in df_on.columns for col in required_columns):
            print(f"Skipping file: {on_file_path} (required columns missing)")
            return
        
        # Filter out rows where 'Small Count' is NaN
        df_on_filtered = df_on.dropna(subset=['Small Count'])
        
        # Iterate over each row to parse, shift, and merge intervals
        for idx, row in df_on_filtered.iterrows():
            start_entry = row['Start Frames']
            stop_entry = row['Stop Frames']
            
            start_frames = parse_frame_entry(start_entry)
            stop_frames = parse_frame_entry(stop_entry)
            
            # Ensure that start_frames and stop_frames have the same length
            if len(start_frames) != len(stop_frames):
                print(f"Row {idx} in '{on_file_path}' has mismatched Start and Stop Frames.")
                merged_intervals_on.append([])  # Append empty list for this row
                continue
            
            # Shift frames by 120
            shifted_start = [start + 120 for start in start_frames]
            shifted_stop = [stop + 120 for stop in stop_frames]
            
            # Merge intervals
            merged = merge_intervals(shifted_start, shifted_stop, max_gap=5)
            merged_intervals_on.append(merged)
    
    # Combine merged intervals from both files
    combined_merged_intervals = []
    max_num_rows = max(len(merged_intervals_off), len(merged_intervals_on))
    
    for i in range(max_num_rows):
        # Get intervals from '-off' file if available
        intervals_off = merged_intervals_off[i] if i < len(merged_intervals_off) else []
        # Get intervals from non-off file if available
        intervals_on = merged_intervals_on[i] if i < len(merged_intervals_on) else []
        # Combine both intervals
        combined_intervals = intervals_off + intervals_on
        combined_merged_intervals.append(combined_intervals)
    
    # Determine the maximum frame number to define the heatmap's x-axis
    max_frame = 0
    for intervals in combined_merged_intervals:
        for start, stop in intervals:
            if stop > max_frame:
                max_frame = stop
    
    # Handle cases where there are no intervals
    if max_frame == 0 and not any(combined_merged_intervals):
        print(f"No valid intervals in file pair '{base_group}'. Setting default max_frame to 120.")
        max_frame = 120  # Default frame range
    
    # Create a binary matrix: rows are video segments, columns are frames
    num_rows = len(combined_merged_intervals)
    heatmap_matrix = np.zeros((num_rows, max_frame + 1), dtype=int)
    
    # Populate the matrix using combined intervals
    for row_idx, intervals in enumerate(combined_merged_intervals):
        for start, stop in intervals:
            # Ensure start and stop are within valid range
            start = max(start, 0)
            stop = min(stop, max_frame)
            if start > stop:
                continue  # Skip invalid intervals
            # Set the corresponding frames to 1
            heatmap_matrix[row_idx, start:stop + 1] = 1
    
    # Convert the matrix to a DataFrame for easier plotting with seaborn
    heatmap_df = pd.DataFrame(heatmap_matrix, 
                              index=range(num_rows),  # Row indices
                              columns=np.arange(max_frame + 1))
    
    # Generate and save the heatmap
    generate_heatmap(heatmap_df, base_group, output_dir)

def main():
    # Path to the directory containing the CSV files
    root_directory = '/Users/tairan/Downloads/results_rnai_'  # Replace with the correct path
    
    # Directory to save heatmaps
    output_dir = os.path.join(root_directory, 'heatmaps')
    os.makedirs(output_dir, exist_ok=True)  # Create if not exists
    
    # Dictionary to map base group to 'off' and 'on' files
    base_group_files = defaultdict(dict)
    
    # First, collect all files and group by base group
    for subdir, _, files in os.walk(root_directory):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(subdir, file)
                base_group = extract_base_group(file)
                if not base_group:
                    print(f"Skipping file: {file} (unable to extract base group)")
                    continue
                if '-off' in file.lower():
                    base_group_files[base_group]['off'] = file_path
                else:
                    base_group_files[base_group]['on'] = file_path
    
    # Now, process each base group
    for base_group, files_dict in base_group_files.items():
        off_file = files_dict.get('off', None)
        on_file = files_dict.get('on', None)
        
        if off_file and on_file:
            print(f"Processing pair for base group: {base_group}")
            process_file_pair(off_file, on_file, base_group, output_dir)
        elif off_file and not on_file:
            print(f"Processing only '-off' file for base group: {base_group}")
            process_file_pair(off_file, None, base_group, output_dir)
        elif on_file and not off_file:
            print(f"Processing only 'on' file for base group: {base_group}")
            process_file_pair(None, on_file, base_group, output_dir)
        else:
            print(f"No valid files found for base group: {base_group}")

if __name__ == "__main__":
    main()
