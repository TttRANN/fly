# import cv2
# import pandas as pd
# import os

# def process_video_segments(csv_path, video_path, save_dir):
#     # Load the CSV data
#     print("Loading CSV data...")
#     data = pd.read_csv(csv_path)

#     # Assign column names
#     # data.columns = ['Var1', 'Var2', 'Var3', 'Var4', 'Var5', 'Var6', 'Var7', 'Var8', 'Var9', 'Var10','Var11','Var12','Var13','Var14','Var15','Var16','Var17']
#     data.columns = ['Var1', 'Var2', 'Var3', 'Var4', 'Var5', 'Var6', 'Var7', 'Var8', 'Var9','Var10']
#     # Convert 'Var1' to integer
#     data['Var1'] = data['Var1'].astype(int)

#     # Open the video stream
#     print("Opening video file...")
#     video = cv2.VideoCapture(video_path)
#     if not video.isOpened():
#         print("Error: Could not open video.")
#         return

#     # Get video properties
#     fps = int(video.get(cv2.CAP_PROP_FPS))
#     frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     print(f"Video properties: FPS={fps}, Width={frame_width}, Height={frame_height}")

#     # Initialize segment tracking
#     start_frame = None
#     end_frame = None
#     segments = []

#     # Identify segments
#     print("Identifying segments...")
#     for i in range(len(data) - 1):
#         current_state = data.iloc[i]['Var7']
#         next_state = data.iloc[i + 1]['Var7']
        
#         if current_state == 'off' and next_state == 'on':
#             start_frame = data.iloc[i]['Var1']
#             print(f"Transition to 'on' detected at frame {start_frame}")

#         # Detect transition from "on" to "off"
#         elif current_state == 'on' and next_state == 'off':
#             end_frame = data.iloc[i]['Var1']
#             if start_frame is not None:
#                 segments.append((start_frame, end_frame))
#                 print(f"Transition to 'off' detected at frame {end_frame}, segment: {start_frame} to {end_frame}")
#                 start_frame = None
#     # Process each segment
# # Process each segment
#     os.makedirs(save_dir, exist_ok=True)
#     for segment in segments:
#         print(f"Processing segment {segment}...")
#         frame_position = float(segment[0])  # Explicit conversion to float
#         if video.set(cv2.CAP_PROP_POS_FRAMES, frame_position):
#             segment_path = os.path.join(save_dir, f'segment_{segment[0]}_{segment[1]}.mp4')
#             out = cv2.VideoWriter(segment_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
            
#             for frame_idx in range(segment[0], segment[1] + 1):
#                 ret, frame = video.read()
#                 if ret:
#                     out.write(frame)
#                 else:
#                     print(f"Failed to read frame {frame_idx}")
#                     break
#             out.release()
#             print(f"Segment from frame {segment[0]} to {segment[1]} has been saved to {segment_path}.")
#         else:
#             print(f"Failed to set video to frame {frame_position}")
    


# import os
# import re

# # Parent directory containing the files
# parent_directory = '/Users/tairan/Downloads/rnai_fas3-29C_t4t5_batch3'

# # Function to process each segment based on the csv, video, and save directory
# def process_segments_for_all_files(parent_directory):
#     # Find all CSV files in the parent directory
#     csv_files = [f for f in os.listdir(parent_directory) if f.endswith('.csv')]
    
#     for csv_file in csv_files:
#         # Extract the segment number from the CSV filename
#         segment_number_match = re.search(r'rnai_fas3-29C_t4t5_batch3_(\d+)\.csv', csv_file)
#         segment_number = segment_number_match.group(1) if segment_number_match else 'unknown'
        
#         # Construct the corresponding video file path
#         video_filename = f'rnai_fas3-29C_t4t5_batch3_{segment_number}_resized.mp4'
#         video_path = os.path.join(parent_directory, video_filename)
        
#         # Construct the save directory path
#         save_dir = os.path.join(parent_directory, f'seg{segment_number}')
        
#         # Ensure save directory exists
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
        
#         # Construct full CSV path
#         csv_path = os.path.join(parent_directory, csv_file)
        
#         # Check if the video file exists before processing
#         if os.path.exists(video_path):
#             # Process the video segments
#             process_video_segments(csv_path, video_path, save_dir)
            
#             print(f'Processed segment {segment_number}:')
#             print(f'CSV Path: {csv_path}')
#             print(f'Video Path: {video_path}')
#             print(f'Save Directory: {save_dir}')
#         else:
#             print(f"Video file {video_filename} not found for segment {segment_number}, skipping.")

# # Call the function to process all files
# process_segments_for_all_files(parent_directory)


import os
import re
import cv2
import pandas as pd

def process_video_segments(csv_path, video_path, save_dir):
    # Load the CSV data
    print(f"Loading CSV data from {csv_path}...")
    data = pd.read_csv(csv_path)
    a = 0
    
    # Determine the number of columns and assign appropriate column names
    if data.shape[1] == 9:
        # If there are 9 columns
        data.columns = ['Var1', 'Var2', 'Var3', 'Var4', 'Var5', 'Var6', 'Var7', 'Var8', 'Var9']
        a="DataShapeOne"
    elif data.shape[1] == 10:
        # If there are 10 columns
        data.columns = ['Var1', 'Var2', 'Var3', 'Var4', 'Var5', 'Var6', 'Var7', 'Var8', 'Var9', 'Var10']
        a="DataShapeTwo"
    elif data.shape[1] == 7:
        data.columns = ["pos_x","pos_y","ori","timestamp","frame_number","video_frame","direction"]
        a="DataShapeThree"
    else:
        print(f"Unexpected number of columns: {data.shape[1]} in file {csv_path}")
        return  # Exit the function if the number of columns is unexpected
    # data['Var1'] = data['Var1'].astype(int)

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
        if a == "DataShapeOne" or a == "DataShapeTwo":
            current_state = data.iloc[i]['Var7']
            next_state = data.iloc[i + 1]['Var7']
            
            if current_state == 'off' and next_state == 'on':
                start_frame = data.iloc[i]['Var1']
                print(f"Transition to 'on' detected at frame {start_frame}")

            elif current_state == 'on' and next_state == 'off':
                end_frame = data.iloc[i]['Var1']
                if start_frame is not None:
                    segments.append((start_frame, end_frame))
                    print(f"Transition to 'off' detected at frame {end_frame}, segment: {start_frame} to {end_frame}")
                    start_frame = None
        else:
            current_state = data.iloc[i]["direction"]
            next_state = data.iloc[i+1]["direction"]
            OnSituation1 = current_state == 0 and next_state == -1
            OnSituation2 = current_state == 0 and next_state == 1
            OffSituation1 = current_state == -1 and next_state == 0
            OffSituation2 = current_state == 1 and next_state == 0
            if OnSituation1 or OnSituation2:
                start_frame = data.iloc[i]['video_frame']
                print(f"Transition to 'on' detected at frame {start_frame}")

            elif OffSituation1 or OffSituation2:
                end_frame = data.iloc[i]['video_frame']
                if start_frame is not None:
                    segments.append((start_frame, end_frame))
                    print(f"Transition to 'off' detected at frame {end_frame}, segment: {start_frame} to {end_frame}")
                    start_frame = None


    

    # Process each segment
    os.makedirs(save_dir, exist_ok=True)
    for segment in segments:
        print(f"Processing segment {segment}...")
        frame_position = int(segment[0])
        if video.set(cv2.CAP_PROP_POS_FRAMES, frame_position):
            # segment_path = os.path.join(save_dir, f'segment_{segment[0]}_{segment[1]}.mp4')
            segment_path = os.path.join(save_dir, 'Cas9_R42F06-Gal4_control_1_2024116Dark.avi')
            out = cv2.VideoWriter(segment_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
            
            for frame_idx in range(int(segment[0]), int(segment[1] + 1)):
                ret, frame = video.read()
                if ret:
                    out.write(frame)
                else:
                    print(f"Failed to read frame {frame_idx}")
                    break
            out.release()
            print(f"Segment from frame {segment[0]} to {segment[1]} has been saved to {segment_path}.")
        else:
            print(f"Failed to set video to frame {frame_position}")

import os
import re

# def process_segments_for_all_files(parent_directory):
#     # Find all CSV files that match the pattern
#     for root, dirs, files in os.walk(parent_directory):
#         for file in files:
#             if file.endswith('.csv'):
#                 csv_path = os.path.join(root, file)
#                 print(f"Processing CSV: {csv_path}")
                
#                 # Updated regex pattern to match various filename conventions
#                 # segment_number_match = re.search(r'rnai_(.*)_t4t5_batch(\d+)_filtered_(\d+)\.csv|rnai_(.*)_batch(\d+)_filtered_(\d+)\.csv', file, re.IGNORECASE)
#                 segment_number_match=re.search(r'Cas9_(.*)_t4t5_batch(\d+)_filtered_(\d+)\.csv|rnai_(.*)_batch(\d+)_filtered_(\d+)\.csv', file, re.IGNORECASE)
#                 if segment_number_match:
#                     if segment_number_match.group(1):
#                         base_name = segment_number_match.group(1)
#                         batch_number = segment_number_match.group(2)
#                         filter_number = segment_number_match.group(3)
#                     else:
#                         base_name = segment_number_match.group(4)
#                         batch_number = segment_number_match.group(5)
#                         filter_number = segment_number_match.group(6)

#                     # Try matching different video filename patterns
#                     video_filenames = [
#                         f'rnai_{base_name}_t4t5_batch{batch_number}_arena_{filter_number}_resized.mp4',
#                         f'rnai_{base_name}_batch{batch_number}_arena_{filter_number}_resized.mp4'
#                     ]

#                     video_path = None
#                     for video_filename in video_filenames:
#                         potential_path = os.path.join(root, video_filename)
#                         if os.path.exists(potential_path):
#                             video_path = potential_path
#                             break

#                     if video_path:
#                         save_dir = os.path.join(root, f'seg{batch_number}_filtered_{filter_number}_arena_{filter_number}')
#                         process_video_segments(csv_path, video_path, save_dir)
#                         print(f'Processed CSV: {csv_path} for arena {filter_number}')
#                         print(f'Save Directory: {save_dir}')
#                     else:
#                         print(f"No matching video file found for CSV: {csv_path}, skipping.")
#                 else:
#                     print(f"Filename {file} does not match expected pattern.")

# Call the function to process all files
parent_directory = '/Users/tairan/Downloads/segment/Cas9_R42F06-Gal4_control_1/Cas9_R42F06-Gal4_control_1_2024116Dark.csv'
process_video_segments(parent_directory, '/Users/tairan/Downloads/segment/Cas9_R42F06-Gal4_control_1/Cas9_R42F06-Gal4_control_1_2024116Dark.avi', '/Users/tairan/Downloads/segment/Cas9_R42F06-Gal4_control_1')

# process_segments_for_all_files(parent_directory)

# # Call the function to process all files
# parent_directory = '/Users/tairan/Downloads/rnai_seg2/wrongname'
# process_segments_for_all_files(parent_directory)



