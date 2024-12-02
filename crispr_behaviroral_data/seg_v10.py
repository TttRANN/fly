import os
import re
import cv2
import pandas as pd

def process_video_segments(csv_path, video_path, save_dir_base):
    # Load the CSV data
    print(f"Loading CSV data from {csv_path}...")
    data = pd.read_csv(csv_path)
    
    # Determine the number of columns and assign appropriate column names
    if data.shape[1] == 9:
        # If there are 9 columns
        data.columns = ['Var1', 'Var2', 'Var3', 'Var4', 'Var5', 'Var6', 'Var7', 'Var8', 'Var9']
    elif data.shape[1] == 10:
        # If there are 10 columns
        data.columns = ['Var1', 'Var2', 'Var3', 'Var4', 'Var5', 'Var6', 'Var7', 'Var8', 'Var9', 'Var10']
        
    else:
        print(f"Unexpected number of columns: {data.shape[1]} in file {csv_path}")
        return  # Exit the function if the number of columns is unexpected
    data['Var1'] = data['Var1'].astype(int)
    
    # Open the video stream
    print(f"Opening video file {video_path}...")
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return
    
    # Get video properties
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video properties: FPS={fps}, Width={frame_width}, Height={frame_height}")
    
    # Initialize segment tracking
    start_frame = None
    end_frame = None
    segments = []
    
    # Identify segments
    print(f"Identifying segments in {csv_path}...")
    for i in range(len(data) - 1):
        current_state = data.iloc[i]['Var7']
        next_state = data.iloc[i + 1]['Var7']
        
        if current_state == 'off' and next_state == 'on':
            start_frame = data.iloc[i]['Var1']
            print(f"Transition to 'on' detected at frame {start_frame}")
    
        elif current_state == 'on' and next_state == 'off':
            end_frame = data.iloc[i]['Var1']
            if start_frame is not None:
                # Get Var9 value for the segment
                var9_values = data[(data['Var1'] >= start_frame) & (data['Var1'] <= end_frame)]['Var9']
                var9_value = var9_values.mode()[0]  # Most common value in Var9 during the segment
                segments.append((start_frame, end_frame, var9_value))
                print(f"Transition to 'off' detected at frame {end_frame}, segment: {start_frame} to {end_frame}, Var9: {var9_value}")
                start_frame = None
    
    # Process each segment and save based on Var9 value
    for segment in segments:
        start_frame, end_frame, var9_value = segment
        
        # Determine the folder based on Var9
        if var9_value in [True, 'True', 'true']:
            var9_folder = 'clockwise'
        else:
            var9_folder = 'anticlockwise'
        save_dir = os.path.join(save_dir_base, var9_folder)
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Processing last two seconds of segment {segment}...")
        
        segment_duration = end_frame - start_frame
        frames_to_save = segment_duration  # Last two seconds or the entire segment if shorter
        save_start_frame = start_frame  # Start saving from last two seconds
        
        # Set video to the frame where the last two seconds start
        video.set(cv2.CAP_PROP_POS_FRAMES, save_start_frame)
        
        # Create a video writer to save the segment
        segment_filename = f'segment_{start_frame}_{end_frame}_last_2_seconds.mp4'
        segment_path = os.path.join(save_dir, segment_filename)
        out = cv2.VideoWriter(segment_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        
        # Write the frames for the last two seconds
        for frame_idx in range(save_start_frame, end_frame + 1):
            ret, frame = video.read()
            if ret:
                out.write(frame)
            else:
                print(f"Failed to read frame {frame_idx}")
                break
        
        out.release()
        print(f"Segment from frame {save_start_frame} to {end_frame} has been saved to {segment_path}.")
    
    video.release()

import os
import re

def process_segments_for_all_files(parent_directory):
    # Find all CSV files that match the pattern
    for root, dirs, files in os.walk(parent_directory):
        for file in files:
            if file.endswith('.csv'):
                csv_path = os.path.join(root, file)
                print(f"Processing CSV: {csv_path}")
                
                # Updated regex pattern to match various filename conventions
                segment_number_match = re.search(
                    r'rnai_(.*)_t4t5_batch(\d+)_filtered_(\d+)\.csv|rnai_(.*)_batch(\d+)_filtered_(\d+)\.csv', 
                    file, 
                    re.IGNORECASE
                )
                
                if segment_number_match:
                    if segment_number_match.group(1):
                        base_name = segment_number_match.group(1)
                        batch_number = segment_number_match.group(2)
                        filter_number = segment_number_match.group(3)
                    else:
                        base_name = segment_number_match.group(4)
                        batch_number = segment_number_match.group(5)
                        filter_number = segment_number_match.group(6)
    
                    # Try matching different video filename patterns
                    video_filenames = [
                        f'rnai_{base_name}_t4t5_batch{batch_number}_arena_{filter_number}_resized.mp4',
                        f'rnai_{base_name}_batch{batch_number}_arena_{filter_number}_resized.mp4'
                    ]
    
                    video_path = None
                    for video_filename in video_filenames:
                        potential_path = os.path.join(root, video_filename)
                        if os.path.exists(potential_path):
                            video_path = potential_path
                            break
    
                    if video_path:
                        save_dir_base = os.path.join(
                            root, 
                            f'seg{batch_number}_filtered_{filter_number}_arena_{filter_number}_l2sec'
                        )
                        process_video_segments(csv_path, video_path, save_dir_base)
                        print(f'Processed CSV: {csv_path} for arena {filter_number}')
                        print(f'Save Directory Base: {save_dir_base}')
                    else:
                        print(f"No matching video file found for CSV: {csv_path}, skipping.")
                else:
                    print(f"Filename {file} does not match expected pattern.")

# Call the function to process all files
parent_directory = '/Users/tairan/Downloads/29c'
process_segments_for_all_files(parent_directory)
