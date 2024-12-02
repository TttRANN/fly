import os
import re
import cv2
import pandas as pd

def process_video_segments(csv_path, video_path, save_dir):
    """
    Processes a video based on segment information from a CSV file.
    Extracts segments where Var5 transitions from 0 to 1 and 1 to 0,
    and saves each segment as a separate video file.

    Args:
        csv_path (str): Path to the CSV file containing segment data.
        video_path (str): Path to the video file to process.
        save_dir (str): Directory where the extracted segments will be saved.
    """
    # Load the CSV data
    print(f"Loading CSV data from {csv_path}...")
    try:
        data = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return

    # Assign appropriate column names based on the number of columns
    expected_columns = 6
    if data.shape[1] == expected_columns:
        data.columns = ['Var1', 'Var2', 'Var3', 'Var4', 'Var5', 'Var6']
    else:
        print(f"Unexpected number of columns: {data.shape[1]} in file {csv_path}")
        return

    # Ensure Var1 is of integer type (assuming it represents frame numbers)
    try:
        data['Var1'] = data['Var1'].astype(int)
    except ValueError:
        print(f"Error: 'Var1' column in {csv_path} contains non-integer values.")
        return

    # Open the video file
    print(f"Opening video file {video_path}...")
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return

    # Retrieve video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video properties: FPS={fps}, Width={frame_width}, Height={frame_height}, Total Frames={total_frames}")

    # Identify segments based on Var5 transitions
    print(f"Identifying segments in {csv_path}...")
    segments = []
    start_frame = None

    for i in range(len(data) - 1):
        current_state = data.iloc[i]['Var5']
        next_state = data.iloc[i + 1]['Var5']

        if current_state == 0 and next_state == 1:
            start_frame = data.iloc[i]['Var1']
            print(f"Transition to 'on' detected at frame {start_frame}")

        elif current_state == 1 and next_state == 0:
            end_frame = data.iloc[i]['Var1']
            if start_frame is not None:
                segments.append((start_frame, end_frame))
                print(f"Transition to 'off' detected at frame {end_frame}, segment: {start_frame} to {end_frame}")
                start_frame = None

    # Handle case where video ends while still in 'on' state
    if start_frame is not None:
        segments.append((start_frame, total_frames))
        print(f"Video ended while 'on'. Segment: {start_frame} to {total_frames}")

    if not segments:
        print(f"No segments found in {csv_path}.")
        video.release()
        return

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Process and save each segment
    for idx, (start_frame, end_frame) in enumerate(segments, start=1):
        # Validate frame numbers
        if start_frame >= total_frames or end_frame > total_frames:
            print(f"Invalid frame range for segment {idx}: {start_frame} to {end_frame}. Skipping.")
            continue

        # Set the video position to the start frame
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Define the output video path
        segment_filename = f'segment_{start_frame}_{end_frame}.avi'
        segment_path = os.path.join(save_dir, segment_filename)

        # Initialize the video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Using 'XVID' codec for better compatibility
        out = cv2.VideoWriter(segment_path, fourcc, fps, (frame_width, frame_height))

        print(f"Saving segment {idx}: frames {start_frame} to {end_frame} to {segment_path}...")

        # Write frames to the output video
        for frame_idx in range(start_frame, end_frame + 1):
            ret, frame = video.read()
            if not ret:
                print(f"Warning: Failed to read frame {frame_idx}. Ending segment {idx}.")
                break
            out.write(frame)

        # Release the video writer
        out.release()
        print(f"Segment {idx} saved successfully.")

    # Release the video capture object
    video.release()
    print(f"All segments from {csv_path} have been processed and saved to {save_dir}.\n")


def process_segments_for_all_files(parent_directory):
    """
    Traverses the parent directory to find all CSV files matching a specific pattern,
    identifies corresponding video files, and processes the segments.

    Args:
        parent_directory (str): The root directory to start searching for CSV and video files.
    """
    # Define the regex pattern to match CSV filenames (e.g., HS_31_2024922_.csv)
    csv_pattern = re.compile(r'^HS_(\d+)_(\d+)_.csv$', re.IGNORECASE)

    # Traverse the directory tree
    for root, dirs, files in os.walk(parent_directory):
        for file in files:
            if file.lower().endswith('.csv'):
                csv_match = csv_pattern.match(file)
                if csv_match:
                    filter_number_1, filter_number_2 = csv_match.groups()
                    csv_path = os.path.join(root, file)
                    print(f"Processing CSV: {csv_path}")

                    # Dynamically determine the corresponding video filename based on CSV
                    # Adjust the pattern below according to your actual video naming convention
                    video_filename_pattern = f'HS_{filter_number_1}_{filter_number_2}_.avi'
                    video_path = os.path.join(root, video_filename_pattern)

                    if os.path.exists(video_path):
                        save_dir = os.path.join(root, 'segments')
                        process_video_segments(csv_path, video_path, save_dir)
                        print(f'Processed CSV: {csv_path} for video {video_path}')
                        print(f'Save Directory: {save_dir}\n')
                    else:
                        print(f"No matching video file found for CSV: {csv_path}. Expected video filename: {video_filename_pattern}. Skipping.\n")
                else:
                    print(f"Filename {file} does not match expected pattern. Skipping.\n")


def main():
    # Define the parent directory containing CSV and video files
    parent_directory = '/home/twang/Downloads/test101'

    if not os.path.isdir(parent_directory):
        print(f"Error: The directory {parent_directory} does not exist.")
        return

    # Start processing
    process_segments_for_all_files(parent_directory)


if __name__ == '__main__':
    main()
