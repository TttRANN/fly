import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

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
        _, thresholded = cv2.threshold(background_diff,15, 255, cv2.THRESH_BINARY)

        # Apply morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
        thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Filter contours by area
            valid_contours = [contour for contour in contours if cv2.contourArea(contour) < max_contour_area]
            largest_contour = max(valid_contours, key=cv2.contourArea)
            if valid_contours:
                # Count the contours below the threshold size
                if cv2.contourArea(largest_contour) < 400:
                        # Only increment the count if no small contour has been detected in the last 20 frames
                    if frame_count - last_small_contour_frame >= frame_window:
                        contour_below_threshold_count += 1
                        contour_found_in_video = True
                        last_small_contour_frame = frame_count

                # Find the largest contour by area among valid contours
                largest_contour = max(valid_contours, key=cv2.contourArea)

                # Get the bounding box coordinates
                x, y, w, h = cv2.boundingRect(largest_contour)
                # Calculate moments for the centroid
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = 0, 0

                # Only update the fly position if the movement is within the allowed range
                if not fly_positions or np.linalg.norm(np.array([cX, cY]) - np.array(fly_positions[-1])) < max_movement:
                    fly_positions.append((cX, cY))

                    # Draw the rectangle on the frame
                    cv2.drawContours(frame, largest_contour, -1, (0, 255, 0), 3)
                    # print(contours)
                    area = np.append(area, cv2.contourArea(largest_contour))

                    # Draw the current position in green
                    cv2.circle(frame, (cX, cY), 5, (0, 255, 0), -1)

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
        plot_image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        plot_image = plot_image.reshape(canvas.get_width_height()[::-1] + (3,))

        # Resize the plot image to match the video frame height
        plot_image = cv2.resize(plot_image, (background_width, background_height))

        # Combine the video frame and the plot image side by side
        combined_frame = np.hstack((frame, plot_image))

        # Display the contour count on the frame
        cv2.putText(combined_frame, f'Contours < 400: {contour_below_threshold_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 0), 2, cv2.LINE_AA)

        # Write the combined frame to the output video
        out.write(combined_frame)

        frame_count += 1

    # Release resources
    video.release()
    out.release()
    cv2.destroyAllWindows()

    return contour_found_in_video, contour_below_threshold_count

# Set the path to the folder containing the videos
input_folder = '/Users/tairan/Downloads/rnai_gilt1-batch3/seg3'
output_folder = '/Users/tairan/Downloads/rnai_gilt1-batch3/b3output3_4'

# Load the static background image
background = cv2.imread('/Users/tairan/Downloads/WTR.jpg')
background_height, background_width, _ = background.shape

# Calculate the radius for the circular mask (10% smaller than half the width)
mask_radius = int((background_width // 2) * 0.95)
center = (background_width // 2, background_height // 2)

# Initialize the mask with zeros (same dimensions as the background image)
mask = np.zeros((background_height, background_width), dtype=np.uint8)

# Draw a filled circle on the mask
cv2.circle(mask, center, mask_radius, 255, thickness=-1)

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Initialize counters
videos_with_small_contours = 0
small_contour_counts = {}

# Process each video in the folder
for video_filename in os.listdir(input_folder):
    if video_filename.endswith('.mp4'):  # Adjust if there are other video formats
        input_video_path = os.path.join(input_folder, video_filename)
        output_video_path = os.path.join(output_folder, f'processed_{video_filename}')
        found_small_contour, small_contour_count = process_video(input_video_path, output_video_path, background, mask)
        
        if found_small_contour:
            videos_with_small_contours += 1
        small_contour_counts[video_filename] = small_contour_count

# Print the results
print(f'Number of videos with at least one contour < 400: {videos_with_small_contours}')
print('Occurrences for each video:')
for video, count in small_contour_counts.items():
    print(f'{video}: {count}')

print("Processing complete.")