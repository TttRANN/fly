import cv2
import numpy as np
import os
import csv

def apply_circular_mask(frame, radius, center):
    """Apply a circular mask to a frame."""
    mask = np.zeros_like(frame)
    cv2.circle(mask, center, radius, (255, 255, 255), thickness=-1)
    masked_frame = cv2.bitwise_and(frame, mask)
    return masked_frame

def calculate_curvature(centroids):
    curvatures = []
    for i in range(1, len(centroids) - 1):
        # Get the previous, current, and next points
        p1 = np.array(centroids[i-1])
        p2 = np.array(centroids[i])
        p3 = np.array(centroids[i+1])
        
        # Calculate vectors between the points
        v1 = p1 - p2
        v2 = p3 - p2

        # Check norms to avoid division by zero
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            curvatures.append(0)
            continue
        
        # Normalize vectors
        v1_norm = v1 / norm_v1
        v2_norm = v2 / norm_v2
        
        # Calculate the angle between the vectors
        dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        angle = np.arccos(dot_product)
        
        # Calculate the distance between p1 and p3
        distance = np.linalg.norm(p3 - p1)
        
        # Curvature is defined as angle / distance
        if distance > 0:
            curvature = angle / distance
        else:
            curvature = 0
        
        curvatures.append(curvature)
    
    return curvatures

def draw_curvature_graph_on_canvas(curvatures, canvas_size, current_frame_index):
    canvas = np.ones(canvas_size, dtype=np.uint8) * 255  # white background
    
    if len(curvatures) < 2:
        return canvas

    # Normalize curvature values to fit within the canvas height
    max_curvature = max(curvatures)
    min_curvature = min(curvatures)
    if max_curvature == min_curvature:
        normalized_curvatures = [0.5 * canvas_size[1]] * len(curvatures)
    else:
        normalized_curvatures = [
            int(canvas_size[1] - ((curv - min_curvature) / (max_curvature - min_curvature) * (canvas_size[1] - 60)) - 30)
            if not np.isnan(curv) else 0  # Ensure no NaN values
            for curv in curvatures
        ]

    # Draw x and y axes with increased margin
    margin = 60  # Increased margin for better readability
    cv2.line(canvas, (margin, 30), (canvas_size[1] - margin, 30), (0, 0, 0), 2)  # X-axis at the top
    cv2.line(canvas, (margin, 30), (margin, canvas_size[0] - 30), (0, 0, 0), 2)  # Y-axis on the left

    # Label the y-axis with correct orientation and spacing
    y_ticks = 5
    for i in range(y_ticks + 1):
        y_pos = int(margin + (canvas_size[0] - 2 * margin) * i / y_ticks)
        curv_value = round(min_curvature + (max_curvature - min_curvature) * (y_ticks - i) / y_ticks, 2)
        cv2.putText(canvas, f'{curv_value}', (5, y_pos + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Label the x-axis (frame indices) with more space and clear orientation
    x_ticks = 10
    for i in range(x_ticks + 1):
        x_pos = int(margin + (canvas_size[1] - margin - 30) * i / x_ticks)
        frame_label = int(i * len(curvatures) / x_ticks)
        cv2.putText(canvas, f'{frame_label}', (x_pos, canvas_size[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Draw the graph on the canvas
    for i in range(1, len(normalized_curvatures)):
        x1 = int(margin + (i - 1) / len(curvatures) * (canvas_size[1] - margin - 30))
        y1 = int(normalized_curvatures[i - 1])  # Convert to int
        x2 = int(margin + i / len(curvatures) * (canvas_size[1] - margin - 30))
        y2 = int(normalized_curvatures[i])  # Convert to int
        
        color = (0, 0, 255) if i <= current_frame_index else (200, 200, 200)  # Red for past, grey for future

        # Ensure x1, y1, x2, y2 are within valid range and are integers
        if 0 <= x1 < canvas_size[1] and 0 <= y1 < canvas_size[0] and \
           0 <= x2 < canvas_size[1] and 0 <= y2 < canvas_size[0]:
            cv2.line(canvas, (x1, y1), (x2, y2), color, 2)
    
    return canvas

def track_fly_and_calculate_curvature(input_video_path, output_video_path):
    # Load the video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {input_video_path}")
        return None

    frame_count = 0
    background_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    background_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    center = (background_width // 2, background_height // 2)
    radius = int(min(background_width, background_height) * 0.45)
    consecutive_invalid_frames = 0

    # Canvas size for the curvature plot
    canvas_width = 400  # Increased width for the x-axis
    canvas_size = (background_height, canvas_width, 3)  # Height same as video, width for the graph
    
    # Initialize video writer to save the output with additional space for the curvature plot
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (background_width + canvas_width, background_height))

    contour_areas = []
    centroids = []

    # Process the video frames with the circular mask applied
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Apply the circular mask to the frame
        masked_frame = apply_circular_mask(frame, radius, center)
        gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Otsu's thresholding on the masked grayscale frame
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        valid_contours = [c for c in contours if 50 < cv2.contourArea(c) < 10000]

        if valid_contours:
            biggest_contour = max(valid_contours, key=cv2.contourArea)
            ellipse = cv2.fitEllipse(biggest_contour)
            contour_areas.append(cv2.contourArea(biggest_contour))

            # Get the centroid of the ellipse
            centroid = (int(ellipse[0][0]), int(ellipse[0][1]))
            centroids.append(centroid)

            # Check if the centroid is outside the valid boundaries
            if centroid[0] < 0.1 * background_width or centroid[0] > 0.9 * background_width or \
               centroid[1] < 0.1 * background_height or centroid[1] > 0.9 * background_height:
                consecutive_invalid_frames += 1
                print(f"Invalid frame: centroid {centroid} is outside the frame at frame {frame_count}")

                if consecutive_invalid_frames >= 10:
                    cap.release()
                    out.release()
                    return None  # Return None if 10 consecutive frames are invalid
            else:
                consecutive_invalid_frames = 0  # Reset the counter if a valid frame is found

            # Draw the ellipse on the frame
            cv2.ellipse(frame, ellipse, (25, 25, 0), 1)
            
            # Calculate and draw curvature graph
            curvatures = calculate_curvature(centroids)
            canvas = draw_curvature_graph_on_canvas(curvatures, canvas_size, frame_count)
            combined_frame = np.hstack((frame, canvas))
            out.write(combined_frame)
        
        frame_count += 1

    # Release the video objects
    cap.release()
    out.release()

    if len(centroids) > 2:
        return curvatures
    else:
        return None



def process_videos_in_folder(folder_path, output_folder_path, csv_output_path):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder_path, exist_ok=True)

    # Prepare the CSV file
    csv_file = open(csv_output_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Video', 'Frame Index', 'Curvature'])

    # Process each video in the folder
    for video_file in os.listdir(folder_path):
        if video_file.endswith('.mp4'):
            input_video_path = os.path.join(folder_path, video_file)
            output_video_path = os.path.join(output_folder_path, f"processed_{video_file}")
            
            # Process the video and calculate curvatures
            curvatures = track_fly_and_calculate_curvature(input_video_path, output_video_path)
            
            if curvatures is not None:
                for frame_index, curvature in enumerate(curvatures):
                    csv_writer.writerow([video_file, frame_index, curvature])
                print(f"Processed video: {video_file}")
            else:
                print(f"Could not process video: {video_file}")

    # Close the CSV file
    csv_file.close()
import numpy as np

import numpy as np

def classify_segmented_behavior(curvatures, segment_length=30, high_curvature_threshold=0.5, low_variability_threshold=0.2):
    behaviors = []
    num_segments = len(curvatures) // segment_length

    for i in range(num_segments):
        segment_curvatures = curvatures[i * segment_length:(i + 1) * segment_length]
        mean_curvature = np.mean(segment_curvatures)
        curvature_variability = np.std(segment_curvatures)

        if mean_curvature > high_curvature_threshold and curvature_variability < low_variability_threshold:
            behaviors.append("Spinning")
        elif mean_curvature < high_curvature_threshold and curvature_variability > low_variability_threshold:
            behaviors.append("Drifting")
        else:
            behaviors.append("Other")

    return behaviors

# Example usage
# curvatures = track_fly_and_calculate_curvature(input_video_path, output_video_path)



# Example usage
input_video_path = '/Users/tairan/Downloads/29c/rnai_DIP-alpha-29C_t4t5_batch1/seg1_filtered_3_arena_3/segment_14340_14580.mp4'
output_video_path = '/Users/tairan/Downloads/processed_videos/'
curvatures = track_fly_and_calculate_curvature(input_video_path, output_video_path)

if curvatures is not None:
    behavior = classify_segmented_behavior(curvatures)
    print(f"Classified behavior: {behavior}")
else:
    print("No curvature data available to classify behavior.")

# # Usage
# folder_path = '/Users/tairan/Downloads/29c/rnai_DIP-alpha-29C_t4t5_batch1/seg1_filtered_3_arena_3/'
# output_folder_path = '/Users/tairan/Downloads/processed_videos/'
# csv_output_path = '/Users/tairan/Downloads/curvature_data.csv'

# process_videos_in_folder(folder_path, output_folder_path, csv_output_path)





