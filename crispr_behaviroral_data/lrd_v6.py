
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time


import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation





def apply_circular_mask(frame, radius, center):
    """Apply a circular mask to a frame, setting everything outside the circle to white."""
    mask = np.zeros_like(frame)
    cv2.circle(mask, center, radius, (255, 255, 255), thickness=-1)
    inverted_mask = cv2.bitwise_not(mask)
    masked_frame = cv2.bitwise_and(frame, mask)
    masked_frame += inverted_mask
    return masked_frame


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
        small, last_small_contour_frame = 0, -30

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            masked_frame = apply_circular_mask(frame, radius, center)
            gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
            
            blur = cv2.GaussianBlur(gray, (9, 9), 0)
            thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 23, 3)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            valid_contours = [c for c in contours if 10 < cv2.contourArea(c) < 10000]

            if valid_contours:
                biggest_contour = max(valid_contours, key=cv2.contourArea)
                ellipse = cv2.fitEllipse(biggest_contour)
                contour_areas.append(cv2.contourArea(biggest_contour))

                centroid = (int(ellipse[0][0]), int(ellipse[0][1]))

                if centroid[0] < 0.1 * background_width or centroid[0] > 0.8 * background_width or \
                   centroid[1] < 0.1 * background_height or centroid[1] > 0.8 * background_height:
                    consecutive_invalid_frames += 1
                    print(f"Invalid frame: centroid {centroid} is outside the frame at frame {frame_count}")

                    if consecutive_invalid_frames >= 10:
                        cap.release()
                        return "NaN"
                else:
                    consecutive_invalid_frames = 0

                cv2.ellipse(frame, ellipse, (0, 255, 255), 2)

                avg_largest_contour_area = np.mean(contour_areas)
                if cv2.contourArea(biggest_contour) < avg_largest_contour_area / 1.3:
                    if frame_count - last_small_contour_frame >= 30:
                        last_small_contour_frame = frame_count
                        small += 1

            frame_count += 1

        cap.release()
        print(f"Processing completed for video: {input_video_path}")

    except Exception as e:
        print(f"An error occurred: {e}")
        return "NaN"

    time2 = time.time()
    print(f"Processing time: {time2 - time1} seconds")

    return small  # Return only the small count

def combine_videos(input_video_path, animation_video_path, output_combined_video_path, mask_radius, center, output_animation_path):
    # First, process the input video to get the small count
    small_count = process_video(input_video_path, output_animation_path, fps=30)

    if small_count == "NaN":
        print("Processing failed. Exiting.")
        return

    cap1 = cv2.VideoCapture(input_video_path)
    cap2 = cv2.VideoCapture(animation_video_path)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (30, 50)
    fontScale = 1
    color = (255, 255, 0)
    thickness = 2

    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Could not open one or both video files.")
        return

    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap1.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_combined_video_path, fourcc, fps, (width * 2, height))

    frame_count = 0

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        masked_frame = apply_circular_mask(frame1, mask_radius, center)

        thresh_color = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
        thresh_color = cv2.cvtColor(thresh_color, cv2.COLOR_GRAY2BGR)

        # Display the small count on the thresholded frame
        cv2.putText(thresh_color, f"Small: {small_count}", org, font, fontScale, color, thickness, cv2.LINE_AA)

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
parent_dic = 'rnai_seg3'
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
            
            # Create or clear the CSV file before starting processing
            with open(csv_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Video Filename', 'Small Contour Count'])

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
                    mask_radius = int((width // 2) * 0.8)
                cap.release()
                
                # Call the function to process the video and save the animation
                small_contour_count = process_video(input_video_path, output_animation_path, fps=30)

                # Combine the masked original video and the contour area animation into a single video
                combine_videos(input_video_path, output_animation_path, output_combined_video_path, mask_radius, center, output_animation_path)
                print(small_contour_count)
                # Append results to the CSV file
                with open(csv_filename, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([video_filename, small_contour_count])
                    print(type(small_contour_count))
                if  type(small_contour_count) == 'int' and small_contour_count > 0 :
                    videos_with_small_contours += 1
                else:
                    videos_with_small_contours =0
                
                small_contour_counts[video_filename] = small_contour_count  

            # Print summary for the current segment folder
            print(f'Processing complete for folder: {seg_folder}')
            print(f'Number of videos with at least one lifting behavior: {videos_with_small_contours}')
            print('Occurrences for each video:')
            for video, count in small_contour_counts.items():import os
import re
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time


def apply_circular_mask(frame, radius, center):
    """Apply a circular mask to a frame, setting everything outside the circle to white."""
    mask = np.zeros_like(frame)
    cv2.circle(mask, center, radius, (255, 255, 255), thickness=-1)
    inverted_mask = cv2.bitwise_not(mask)
    masked_frame = cv2.bitwise_and(frame, mask)
    masked_frame += inverted_mask
    return masked_frame


import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy.stats import zscore

def process_video(video_path, output_path, background, mask, max_frames=400, max_contour_area=3000, max_movement=50, frame_window=30):
    # Load the background image
    background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    background_height, background_width = background_gray.shape
    # Open the video stream
    video = cv2.VideoCapture(video_path)

    # Get the video's frames per second (fps)
    fps = int(video.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object for MP4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (background_width * 2, background_height))

    # Initialize a list to store the fly positions
    fly_positions = []
    area = np.array([])
    frame_count = 0

    # Counter for contours below the threshold
    contour_below_threshold_count = 0
    contour_found_in_video = False

    # Track the last frame where a small contour was detected
    last_small_contour_frame = -frame_window

    # Create a figure for the plot
    fig, ax = plt.subplots()
    canvas = FigureCanvas(fig)

    # List to store the areas of small contours
    small_contour_areas = []

    while True:
        ret, frame = video.read()
        if not ret or frame_count >= max_frames:
            break

        # Resize the current frame to match the background image dimensions
        frame = cv2.resize(frame, (background_width, background_height))

        # Convert the current frame to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply the circular mask to the current frame
        frame_gray_masked = cv2.bitwise_and(frame_gray, frame_gray, mask=mask)

        # Compute the absolute difference between the masked current frame and the masked background image
        background_diff = cv2.absdiff(frame_gray_masked, cv2.bitwise_and(background_gray, background_gray, mask=mask))

        # Threshold the difference image
        _, thresholded = cv2.threshold(background_diff, 15, 255, cv2.THRESH_BINARY)

        # Apply morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
        thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Filter contours by area
            valid_contours = [contour for contour in contours if cv2.contourArea(contour) < max_contour_area]

            if valid_contours:
                # Get the largest contour from the valid contours
                largest_contour = max(valid_contours, key=cv2.contourArea)

                # Calculate contour areas
                contour_areas = [cv2.contourArea(cnt) for cnt in valid_contours]
                
                # Normalize and find z-scores
                z_scores = zscore(contour_areas)
                
                # Define a threshold for small outliers
                z_threshold = -2  # Example threshold, can be adjusted
                
                # Identify small outliers
                small_outliers = [area for area, z in zip(contour_areas, z_scores) if z < z_threshold]
                small_contour_areas.extend(small_outliers)  # Store small contour areas
                
                # Count the small contours in this frame
                small_contour_count_in_frame = len(small_outliers)

                if small_contour_count_in_frame > 0:
                    contour_below_threshold_count += small_contour_count_in_frame
                    contour_found_in_video = True

                # Fit an ellipse to the largest contour
                if len(largest_contour) >= 5:
                    ellipse = cv2.fitEllipse(largest_contour)

                    # Get the center of the ellipse
                    cX, cY = int(ellipse[0][0]), int(ellipse[0][1])

                    # Only update the fly position if the movement is within the allowed range
                    if not fly_positions or np.linalg.norm(np.array([cX, cY]) - np.array(fly_positions[-1])) < max_movement:
                        fly_positions.append((cX, cY))

                        # Draw the ellipse on the frame
                        cv2.ellipse(frame, ellipse, (0, 255, 0), 2)

                        # Draw the current position in green
                        cv2.circle(frame, (cX, cY), 5, (0, 255, 0), -1)

                        area = np.append(area, cv2.contourArea(largest_contour))

            # Draw the trajectory in red on the original frame
            for i in range(1, len(fly_positions)):
                cv2.line(frame, fly_positions[i - 1], fly_positions[i], (0, 0, 255), 2)

            # Update the plot
            ax.clear()
            ax.plot(area)
            ax.set_title('Contour Area Over Time')
            ax.set_xlabel('Frame')
            ax.set_ylabel('Area')

            # Render the plot to a canvas
            canvas.draw()
            plot_image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
            plot_image = plot_image.reshape(canvas.get_width_height()[::-1] + (4,))

            # Resize the plot image to match the video frame height
            plot_image = cv2.resize(plot_image, (background_width, background_height))

            # Ensure both frame and plot_image have the same number of channels
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Combine the video frame and the plot image side by side
            combined_frame = np.hstack((frame_rgb, plot_image))

            # Display the contour count on the frame
            cv2.putText(combined_frame, f'Contours Counts: {contour_below_threshold_count}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 0), 2, cv2.LINE_AA)

            # Write the combined frame to the output video
            out.write(cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR))

            frame_count += 1

    # Release resources
    video.release()
    out.release()
    cv2.destroyAllWindows()

    return contour_below_threshold_count, small_contour_areas


    
def combine_videos(input_video_path, animation_video_path, output_combined_video_path, mask_radius, center, output_animation_path):
    # First, process the input video to get the small count and generate the animation
    small_count, small_counts = process_video(input_video_path, output_animation_path, fps=30)
    print(type(small_counts))

    if small_count == "NaN":
        print("Processing failed. Exiting.")
        return

    cap1 = cv2.VideoCapture(input_video_path)
    cap2 = cv2.VideoCapture(animation_video_path)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (30, 50)
    fontScale = 1
    color = (255, 255, 0)
    thickness = 2

    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Could not open one or both video files.")
        return

    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap1.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_combined_video_path, fourcc, fps, (width * 2, height))

    frame_count = 0
    current_small_display = small_counts[0] if small_counts else 0  # Start with the first small count

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        masked_frame = apply_circular_mask(frame1, mask_radius, center)

        thresh_color = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
        thresh_color = cv2.cvtColor(thresh_color, cv2.COLOR_GRAY2BGR)

        # Update the current small display according to the frame count
        if frame_count < len(small_counts):
            current_small_display = small_counts[frame_count]

        # Display the current small count on the thresholded frame
        cv2.putText(thresh_color, f"Small: {current_small_display}", org, font, fontScale, color, thickness, cv2.LINE_AA)

        frame2 = cv2.resize(frame2, (width, height))

        combined_frame = np.hstack((thresh_color, frame2))
        out.write(combined_frame)

        frame_count += 1

    cap1.release()
    cap2.release()
    out.release()
    print(f"Combined video saved as: {output_combined_video_path}")



# Set the parent directory
parent_dic = 'rnai_seg3'
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
            
            # Create or clear the CSV file before starting processing
            with open(csv_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Video Filename', 'Small Contour Count'])

# Inside the loop where you're processing each video
            # Inside the loop where you're processing each video
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
                    mask_radius = int((width // 2) * 0.8)
                cap.release()
                
                # Call the function to process the video and save the animation
                small_contour_count = process_video(input_video_path, output_animation_path, fps=30)

                # Combine the masked original video and the contour area animation into a single video
                combine_videos(input_video_path, output_animation_path, output_combined_video_path, mask_radius, center, output_animation_path)
                
                # Append results to the CSV file
                with open(csv_filename, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([video_filename, small_contour_count])
                
                # Correctly compare the small count
                if small_contour_count > 0:
                    videos_with_small_contours += 1
                
                small_contour_counts[video_filename] = small_contour_count

            # Print summary for the current segment folder
            print(f'Processing complete for folder: {seg_folder}')
            print(f'Number of videos with at least one lifting behavior: {videos_with_small_contours}')
            print('Occurrences for each video:')
            for video, count in small_contour_counts.items():
                print(f'{video}: {count}')
