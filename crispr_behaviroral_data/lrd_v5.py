import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

def process_video(input_video_path, output_animation_path, fps):
    time1 = time.time()
    try:
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
        contour_areas = []
        contour_areas_over_time = []  # Initialize the list to store contour areas over time
        small, last_small_contour_frame = 0, -30

        # Process the video frames with the circular mask applied
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Apply the circular mask to the frame
            masked_frame = apply_circular_mask(frame, radius, center)
            gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
            
            blur = cv2.GaussianBlur(gray, (9,9), 0)
            thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,23,3)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            valid_contours = [c for c in contours if 10 < cv2.contourArea(c) < 10000]

            if valid_contours:
                biggest_contour = max(valid_contours, key=cv2.contourArea)
                ellipse = cv2.fitEllipse(biggest_contour)
                contour_areas.append(cv2.contourArea(biggest_contour))
                contour_areas_over_time.append(cv2.contourArea(biggest_contour))  # Store the contour area

                # Get the centroid of the ellipse
                centroid = (int(ellipse[0][0]), int(ellipse[0][1]))

                # Check if the centroid is outside the valid boundaries
                if centroid[0] < 0.1 * background_width or centroid[0] > 0.9 * background_width or \
                   centroid[1] < 0.1 * background_height or centroid[1] > 0.9 * background_height:
                    consecutive_invalid_frames += 1
                    print(f"Invalid frame: centroid {centroid} is outside the frame at frame {frame_count}")

                    if consecutive_invalid_frames >= 5:
                        cap.release()
                        return "NaN"  # Return "NaN" if 10 consecutive frames are invalid
                else:
                    consecutive_invalid_frames = 0  # Reset the counter if a valid frame is found

                # Draw the ellipse on the frame
                cv2.ellipse(frame, ellipse, (0,255,255), 2)

                avg_largest_contour_area = np.mean(contour_areas)
                if cv2.contourArea(biggest_contour) < avg_largest_contour_area / 1.3:
     
                    if frame_count - last_small_contour_frame >= 30:
                        last_small_contour_frame = frame_count
                        small += 1
            else:
                contour_areas.append(0)
                contour_areas_over_time.append(0)  # Store 0 if no valid contours are found
                consecutive_invalid_frames = 0  # Reset if no valid contours are found

            frame_count += 1

        print(f"Processing completed for video: {input_video_path}")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("No flies detected, skipping this video.")
        return "NaN"  

    time2 = time.time()
    print(time2 - time1)

    # Create the animation
    fig, ax = plt.subplots()
    ax.set_xlim(0, frame_count)
    ax.set_ylim(0, max(contour_areas_over_time) if contour_areas_over_time else 10000)
    line, = ax.plot([], [], lw=2, label='Contour Area')
    avg_line, = ax.plot([], [], lw=2, label='Average Contour Area', color='r')

    def init():
        line.set_data([], [])
        avg_line.set_data([], [])
        return line, avg_line

    def update(frame):
        xdata = list(range(frame + 1))
        ydata = contour_areas_over_time[:frame + 1]
        avg_ydata = [np.mean(contour_areas_over_time[:i+1]) for i in range(frame + 1)]
        line.set_data(xdata, ydata)
        avg_line.set_data(xdata, avg_ydata)
        return line, avg_line

    ani = animation.FuncAnimation(fig, update, frames=len(contour_areas_over_time),
                                  init_func=init, blit=True, repeat=False)

    # Save the animation as an MP4 file
    ani.save(output_animation_path, writer=animation.FFMpegWriter(fps=fps))

    plt.close(fig)  # Close the plot to avoid displaying it

    return small


import cv2
import numpy as np


def apply_circular_mask(frame, radius, center):
    """Apply a circular mask to a frame, setting everything outside the circle to white."""
    mask = np.zeros_like(frame)
    cv2.circle(mask, center, radius, (255, 255, 255), thickness=-1)
    inverted_mask = cv2.bitwise_not(mask)
    masked_frame = cv2.bitwise_and(frame, mask)
    masked_frame += inverted_mask
    return masked_frame





def combine_videos(input_video_path, animation_video_path, output_combined_video_path, mask_radius, center):
    cap1 = cv2.VideoCapture(input_video_path)
    cap2 = cv2.VideoCapture(animation_video_path)
    
    # Font settings for overlaying text
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (30, 50)
    fontScale = 1
    color = (255, 255, 0)  # Yellow color in BGR
    thickness = 2
    frame_count = 0
    background_height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    background_width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    center = (background_width // 2, background_height // 2)
    radius = int(min(background_width, background_height) * 0.45)
    consecutive_invalid_frames = 0

    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Could not open one or both video files.")
        return

    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap1.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_combined_video_path, fourcc, fps, (width * 2, height))

    contour_areas = []
    last_small_contour_frame = -30
    frame_count = 0
    
    # Declare small outside the loop to retain its value across frames
    small = 0

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        masked_frame = apply_circular_mask(frame1, mask_radius, center)

        gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 23, 3)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  
        valid_contours = [c for c in contours if 10 < cv2.contourArea(c) < 10000]

        if valid_contours:
            biggest_contour = max(valid_contours, key=cv2.contourArea)
            ellipse = cv2.fitEllipse(biggest_contour)
            contour_areas.append(cv2.contourArea(biggest_contour))

            avg_largest_contour_area = np.mean(contour_areas)
            if cv2.contourArea(biggest_contour) < avg_largest_contour_area / 1.3:
                if frame_count - last_small_contour_frame >= 30:
                    last_small_contour_frame = frame_count
                    small += 1  # Increment small
                # Get the centroid of the ellipse
                centroid = (int(ellipse[0][0]), int(ellipse[0][1]))

                # Check if the centroid is outside the valid boundaries
                if centroid[0] < 0.1 * background_width or centroid[0] > 0.9 * background_width or \
                   centroid[1] < 0.1 * background_height or centroid[1] > 0.9 * background_height:
                    consecutive_invalid_frames += 1
                    print(f"Invalid frame: centroid {centroid} is outside the frame at frame {frame_count}")

                    if consecutive_invalid_frames >= 10:
                        cap.release()
                        return "NaN"  # Return "NaN" if 10 consecutive frames are invalid
                else:
                    consecutive_invalid_frames = 0  # Reset the counter if a valid frame is found
                    

        # Convert threshold image to color so we can put text on it
        thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        # Update the text with the current value of `small`
        cv2.putText(thresh_color, str(small), org, font, fontScale, color, thickness, cv2.LINE_AA)

        frame2 = cv2.resize(frame2, (width, height))

        combined_frame = np.hstack((thresh_color, frame2))
        out.write(combined_frame)

        frame_count += 1

    cap1.release()
    cap2.release()
    out.release()
    print(f"Combined video saved as: {output_combined_video_path}")

import os
import re
import cv2
import csv

# Set the parent directory
parent_dic = 'rnai_seg4'
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
                       if os.path.isdir(os.path.join(rnai_folder_path, f)) and re.match(r'1seg\d+_filtered_\d+_arena_\d+', f)]
        
        for seg_folder in seg_folders:
            # Extract the segment number and arena number from the folder name
            seg_number_match = re.search(r'seg(\d+)_filtered_(\d+)_arena_(\d+)', seg_folder)
            if seg_number_match:
                seg_number = seg_number_match.group(1)
                filtered_number = seg_number_match.group(2)
                arena_number = seg_number_match.group(3)
            else:
                print(f"Skipping folder {seg_folder}, pattern not matched.")
                continue
            
            input_folder = os.path.join(rnai_folder_path, seg_folder)
            
            # Create an output folder for the combined videos
            output_folder = os.path.join(input_folder, 'output_videos')
            os.makedirs(output_folder, exist_ok=True)
            
            # Sort videos by the numerical value in their names
            sorted_videos = sorted([f for f in os.listdir(input_folder) if f.endswith('.mp4')], key=extract_number)
            
            videos_with_small_contours = 0
            small_contour_counts = {}
            csv_filename = os.path.join(rnai_folder_path, f'results_{rnai_folder}_seg{seg_number}_arena{arena_number}.csv')
            
            for video_filename in sorted_videos:
                input_video_path = os.path.join(input_folder, video_filename)
                
                # Set the output animation path
                output_animation_path = os.path.join(output_folder, video_filename.replace('.mp4', '_animation.mp4'))
                
                # Set the output combined video path
                output_combined_video_path = os.path.join(output_folder, video_filename.replace('.mp4', '_combined.mp4'))
                
                # Get the frame size and center for masking
                cap = cv2.VideoCapture(input_video_path)
                if cap.isOpened():
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    center = (width // 2, height // 2)
                    mask_radius = int((width // 2) * 0.9)
                cap.release()
                
                # Call the function to process the video and save the animation
                small_contour_count = process_video(input_video_path, output_animation_path, fps=30)
                print(small_contour_count)
                # Combine the masked original video and the contour area animation into a single video
                combine_videos(input_video_path, output_animation_path, output_combined_video_path, mask_radius, center)
                
                # Append results to a CSV file
                with open(csv_filename, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([f'processed_{video_filename}', small_contour_count])
                
                if small_contour_count:
                    videos_with_small_contours += 1
                
                small_contour_counts[video_filename] = small_contour_count  

           # Print summary for the current segment folder
            print(f'Processing complete for folder: {seg_folder}')
            print(f'Number of videos with at least one lifting behavior: {videos_with_small_contours}')
            print('Occurrences for each video:')
            for video, count in small_contour_counts.items():
                print(f'{video}: {count}')
