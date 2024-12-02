import cv2
import numpy as np
import os
import re

import csv
import time


def process_video(input_video_path):
    # time1=time.time()
    try:
        # Load the video
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {input_video_path}")
            return

        small, last_small_contour_frame, frame_count = 0, -30, 0
        contour_areas, valid_contours = [], []


        # Get video dimensions and FPS
        background_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        background_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Calculate the radius for the circular mask
        mask_radius = int((background_width // 2) * 0.99)
        center = (background_width // 2, background_height // 2)

        # Initialize the mask with zeros
        mask = np.zeros((background_height, background_width), dtype=np.uint8)
        cv2.circle(mask, center, mask_radius, 255, thickness=-1)


        # Initialize a list to store frame numbers
        frame_numbers = []

        contour_below_threshold_count = 0

        # Loop through the video frame by frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Apply the mask to the current frame
            masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

            # Convert the frame to grayscale
            gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)

            # Apply a binary threshold to the frame
            ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
           

            # Find contours in the thresholded frame
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            valid_contours = [c for c in contours if 50 < cv2.contourArea(c) < 10000]

            if valid_contours:
                biggest_contour = max(valid_contours, key=cv2.contourArea)
                ellipse = cv2.fitEllipse(biggest_contour)
                contour_areas.append(cv2.contourArea(biggest_contour))
                cv2.ellipse(masked_frame, ellipse, (25, 25, 0), 1)
                avg_largest_contour_area = np.mean(contour_areas)
                if cv2.contourArea(biggest_contour) < avg_largest_contour_area / 1.2:
                    if frame_count - last_small_contour_frame >= 30:
                        contour_below_threshold_count += 1
                        last_small_contour_frame = frame_count
                        small += 1
            else:
                contour_areas.append(0)
  


            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

        print(f"Processing completed for video: {input_video_path}")


    except Exception as e:
        print(f"An error occurred: {e}")
        print("No flies detected, skipping this video.")
    # time2=time.time()
    # print(time2-time1)

    return small

results = []
parent_dic='rnai_fas3-29C_t4t5_batch3'

parent_directory = f'/Users/tairan/Downloads/{parent_dic}'
time1=time.time()
# Function to extract numerical value from filename for sorting
def extract_number(filename):
    match = re.search(r'segment_(\d+)_', filename)
    return int(match.group(1)) if match else float('inf')

# Find all folders that contain "seg" in their name
seg_folders = [f for f in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, f)) and "seg" in f]

for seg_folder in seg_folders:
    # Extract the segment number from the folder name
    seg_number_match = re.search(r'seg(\d+)', seg_folder)
    seg_number = seg_number_match.group(1) if seg_number_match else 'unknown'

    # Set input and output folders
    input_folder = os.path.join(parent_directory, seg_folder)


    # Sort the video files by segment number
    sorted_videos = sorted([f for f in os.listdir(input_folder) if f.endswith('.mp4')], key=extract_number)



    # Initialize counters
    videos_with_small_contours = 0
    small_contour_counts = {}

    # Define the CSV filename using the segment number
    csv_filename = os.path.join(parent_directory, f'results_{parent_dic}_{seg_number}.csv')

    # Process each video in the folder
    for video_filename in sorted_videos:
        input_video_path = os.path.join(input_folder, video_filename)
   
        
        # Process the video and get the contour count
        small_contour_count = process_video(input_video_path)
        
        # Save results to the corresponding CSV file
        with open(csv_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([f'processed_{video_filename}', small_contour_count])
        
        # Update contour count tracking
        if small_contour_count:
            videos_with_small_contours += 1
        small_contour_counts[video_filename] = small_contour_count  

    # Print the results for the current folder
    print(f'Processing complete for folder: {seg_folder}')
    print(f'Number of videos with at least one lifting behavior: {videos_with_small_contours}')
    print('Occurrences for each video:')
    for video, count in small_contour_counts.items():
        print(f'{video}: {count}')
time2=time.time()
print(time2-time1)