import pandas as pd
import matplotlib.pyplot as plt
import re

def extract_time_from_segment(segment_name):
    # Extract the start and end time from the segment name using regex
    match = re.search(r'segment_(\d+)_(\d+)\.mp4', segment_name)
    if match:
        start_time = int(match.group(1))
        end_time = int(match.group(2))
        return start_time, end_time
    else:
        raise ValueError(f"Invalid segment name format: {segment_name}")

def plot_speed_angular_speed(df, start_time, end_time):
    # Filter the DataFrame based on the start and end time
    filtered_df = df[(df.iloc[:, 0] >= start_time) & (df.iloc[:, 0] <= end_time)]

    # Extract the necessary columns
    time = filtered_df.iloc[:, 0].values
    x_pos = filtered_df.iloc[:, 3].values
    y_pos = filtered_df.iloc[:, 4].values
    angle = filtered_df.iloc[:, 5].values

    # Calculate the speed (Euclidean distance) between consecutive positions
    speed = ((x_pos[1:] - x_pos[:-1])**2 + (y_pos[1:] - y_pos[:-1])**2)**0.5
    # Calculate the angular speed between consecutive angles
    angular_speed = abs(angle[1:] - angle[:-1])

    # Plot the speed and angular speed
    plt.figure(figsize=(14, 7))

    plt.subplot(2, 1, 1)
    plt.plot(time[1:], speed, label='Speed')
    plt.xlabel('Time')
    plt.ylabel('Speed')
    plt.title(f'Speed over Time for segment {start_time}_{end_time}')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(time[1:], angular_speed, label='Angular Speed', color='orange')
    plt.xlabel('Time')
    plt.ylabel('Angular Speed')
    plt.title(f'Angular Speed over Time for segment {start_time}_{end_time}')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

def process_segments(file_path, segment_names):
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Process each segment
    for segment in segment_names:
        start_time, end_time = extract_time_from_segment(segment)
        plot_speed_angular_speed(df, start_time, end_time)

# Example usage:

file_path = '/Users/tairan/Downloads/29c/rnai_BEAT-IV-29C_t4t5_batch1/rnai_BEAT-IV-29C_t4t5_batch1_filtered_2.csv'
segment_names = [
    # 'segment_020970_020999.mp4',
    # 'segment_021000_021029.mp4',
    # 'segment_021030_021059.mp4',
    # 'segment_021060_021089.mp4',
    # 'segment_021090_021119.mp4'
    'segment_2220_2460.mp4'
]

process_segments(file_path, segment_names)


