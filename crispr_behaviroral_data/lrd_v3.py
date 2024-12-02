import cv2
import numpy as np
import os
import re
import csv
import time

def apply_circular_mask(frame, radius, center):
    """Apply a circular mask to a frame."""
    mask = np.zeros_like(frame)
    cv2.circle(mask, center, radius, (255, 255, 255), thickness=-1)
    masked_frame = cv2.bitwise_and(frame, mask)
    return masked_frame

def process_video(input_video_path, output_video_path):
    time1 = time.time()
    try:
        # Load the video
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {input_video_path}")
            return "NaN"

        frame_count = 0
        background_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        background_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = cap.get(cv2.CAP_PROP_FPS)
        center = (background_width // 2, background_height // 2)
        radius = int(min(background_width, background_height) * 0.45)
        consecutive_invalid_frames = 0

        # Initialize video writer to save the output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (background_width, background_height))

        # Process the first 100 frames with the circular mask
        while frame_count < 241 and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            masked_frame = apply_circular_mask(frame, radius, center)
            gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
            
            frame_count += 1

        # Reset the frame counter for the main processing loop
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_count = 0
        small, last_small_contour_frame = 0, -30
        contour_areas = []

        # Continue processing the video without the mask
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Otsu's thresholding
            ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            valid_contours = [c for c in contours if 50 < cv2.contourArea(c) < 10000]

            if valid_contours:
                biggest_contour = max(valid_contours, key=cv2.contourArea)
                ellipse = cv2.fitEllipse(biggest_contour)
                contour_areas.append(cv2.contourArea(biggest_contour))

                # Get the centroid of the ellipse
                centroid = (int(ellipse[0][0]), int(ellipse[0][1]))

                # Check if the centroid is outside the valid boundaries
                if centroid[0] < 0.1 * background_width or centroid[0] > 0.9 * background_width or \
                   centroid[1] < 0.1 * background_height or centroid[1] > 0.9 * background_height:
                    consecutive_invalid_frames += 1
                    print(f"Invalid frame: centroid {centroid} is outside the frame at frame {frame_count}")

                    if consecutive_invalid_frames >= 10:
                        cap.release()
                        out.release()
                        return "NaN"  # Return "NaN" if 10 consecutive frames are invalid
                else:
                    consecutive_invalid_frames = 0  # Reset the counter if a valid frame is found

                # Draw the ellipse on the frame
                cv2.ellipse(frame, ellipse, (25, 25, 0), 1)

                avg_largest_contour_area = np.mean(contour_areas)
                if cv2.contourArea(biggest_contour) < avg_largest_contour_area / 1.3:
                    if frame_count - last_small_contour_frame >= 30:
                        last_small_contour_frame = frame_count
                        small += 1
            else:
                contour_areas.append(0)
                consecutive_invalid_frames = 0  # Reset if no valid contours are found

            # Write the frame with ellipse overlay to the output video
            out.write(frame)

            frame_count += 1

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print(f"Processing completed for video: {input_video_path}")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("No flies detected, skipping this video.")
        return "NaN"  # Return "NaN" in case of an exception

    time2 = time.time()
    print(time2 - time1)

    return small

# ... (Rest of the script remains the same)

import os
import re
import csv

# Set the parent directory
parent_dic = 'rnai_issue'
parent_directory = f'/Users/tairan/Downloads/{parent_dic}'

# Function to extract numerical value from filename for sorting
def extract_number(filename):
    match = re.search(r'segment_(\d+)_', filename)
    return int(match.group(1)) if match else float('inf')

# Loop through each 'rnai_*' subfolder
for rnai_folder in os.listdir(parent_directory):
    rnai_folder_path = os.path.join(parent_directory, rnai_folder)
    
    if os.path.isdir(rnai_folder_path):
        print(f"Processing folder: {rnai_folder}")
        
        # Find all 'seg*_filtered_*_arena_*' folders in the current 'rnai_*' folder
        seg_folders = [f for f in os.listdir(rnai_folder_path) 
                       if os.path.isdir(os.path.join(rnai_folder_path, f)) and re.match(r'seg\d+_filtered_\d+_arena_\d+', f)]
        
        for seg_folder in seg_folders:
            # Extract the segment number and arena number from the folder name
            seg_number_match = re.search(r'seg(\d+)_filtered_(\d+)_arena_(\d+)_l2sec', seg_folder)
            if seg_number_match:
                seg_number = seg_number_match.group(1)
                filtered_number = seg_number_match.group(2)
                arena_number = seg_number_match.group(3)
            else:
                print(f"Skipping folder {seg_folder}, pattern not matched.")
                continue
            
            input_folder = os.path.join(rnai_folder_path, seg_folder)
            
            # Sort videos by the numerical value in their names
            sorted_videos = sorted([f for f in os.listdir(input_folder) if f.endswith('.mp4')], key=extract_number)
            
            videos_with_small_contours = 0
            small_contour_counts = {}
            csv_filename = os.path.join(rnai_folder_path, f'results_{rnai_folder}_seg{seg_number}_arena{arena_number}.csv')
            
            for video_filename in sorted_videos:
                input_video_path = os.path.join(input_folder, video_filename)
                output_video_path = os.path.join(input_folder, f'output_{video_filename}')
                
                # Call the function to process the video
                small_contour_count = process_video(input_video_path, output_video_path)
                
                # Append results to a CSV file
                with open(csv_filename, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([f'processed_{video_filename}', small_contour_count])
                
                if small_contour_count != "NaN":
                    videos_with_small_contours += 1
                
                small_contour_counts[video_filename] = small_contour_count  

            # Print summary for the current segment folder
            print(f'Processing complete for folder: {seg_folder}')
            print(f'Number of videos with at least one lifting behavior: {videos_with_small_contours}')
            print('Occurrences for each video:')
            for video, count in small_contour_counts.items():
                print(f'{video}: {count}')
