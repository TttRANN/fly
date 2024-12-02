import cv2
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import csv

def process_video(input_video_path, output_video_path):
    try:
        # Load the video
        angle1=[]
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {input_video_path}")
            return

        small, last_small_contour_frame, frame_count = 0, -30, 0
        contour_areas, valid_contours = [], []
        centroids = []
        centroids1 = []

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

        # Prepare to write the output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (background_width * 2, background_height))

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
            ret1, thresh1 = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)

            # Find contours in the thresholded frame
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours1, _ = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            valid_contours = [c for c in contours if 50 < cv2.contourArea(c) < 10000]
            valid_contours1 = [c for c in contours1 if 50 < cv2.contourArea(c) < 10000]

            if valid_contours:
                biggest_contour = max(valid_contours, key=cv2.contourArea)
                ellipse = cv2.fitEllipse(biggest_contour)
                # cv2.drawContours(masked_frame, [biggest_contour], -1, (0, 255, 0), 2)
                contour_areas.append(cv2.contourArea(biggest_contour))
                cv2.ellipse(masked_frame, ellipse, (25, 25, 0), 1)

                # Get the centroid of the ellipse
                centroid = (int(ellipse[0][0]), int(ellipse[0][1]))
                centroids.append(centroid)


                avg_largest_contour_area = np.mean(contour_areas)
                if cv2.contourArea(biggest_contour) < avg_largest_contour_area / 1.2:
                    if frame_count - last_small_contour_frame >= 30:
                        contour_below_threshold_count += 1
                        last_small_contour_frame = frame_count
                        small += 1
            else:
                contour_areas.append(0)
                centroids.append(None)

            if valid_contours1:
                biggest_contour1 = max(valid_contours1, key=cv2.contourArea)
                # cv2.drawContours(masked_frame, [biggest_contour1], -1, (0, 255, 0), 2)
                # biggest_contour = max(valid_contours, key=cv2.contourArea)
                ellipse1 = cv2.fitEllipse(biggest_contour1)
                # cv2.drawContours(masked_frame, [biggest_contour], -1, (0, 255, 0), 2)
                # contour_areas.append(cv2.contourArea(biggest_contour1))
                cv2.ellipse(masked_frame, ellipse1, (25, 25, 0), 1)

                # Get the centroid of the ellipse
                centroid1 = (int(ellipse1[0][0]), int(ellipse1[0][1]))
                centroids1.append(centroid1)

 

            frame_numbers.append(frame_count)
            if centroids1 and centroids:  # Ensure both centroids are available
                        # for i in range(min(len(centroids), len(centroids1))):
                            if centroids[-1] is not None and centroids1[-1] is not None:
                                # Calculate the direction vector
                                dx = centroids1[-1][0] - centroids[-1][0]
                                dy = centroids1[-1][1] - centroids[-1][1]
                                
                                # Scale the vector to make the arrow longer
                                end_point = (int(centroids1[-1][0] + 5 * dx), 
                                            int(centroids1[-1][1] + 5 * dy))
                                
                                # Draw the arrow indicating the orientation
                                cv2.arrowedLine(masked_frame, end_point, centroids[-1], (0, 0, 255), 2)
                                x=[centroids[-1][0]-centroids1[-1][0]]
                                y=[centroids[-1][1]-centroids1[-1][1]]

                                angle=np.arctan2(y, x) * 180 / np.pi
                                # angle = np.arctan2(y, x) * 180 / np.pi
                                angle = (angle + 90) % 360

                                angle1.append(angle)

        
            # Plot the contour areas
            # fig, ax = plt.subplots(figsize=(300, 100))
            fig, ax = plt.subplots()
            canvas = FigureCanvas(fig)
            ax.plot(angle1, label="Contour Area")
            ax.set_xlabel('Frame Number')
            ax.set_ylabel('Contour Area')
            ax.set_title('Contour Area vs Frame Number')
            ax.legend()
            ax.grid(True)

            # Render the plot to a canvas
            canvas = FigureCanvas(fig)
            canvas.draw()
            plot_image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            plot_image = plot_image.reshape(canvas.get_width_height()[::-1] + (3,))
            plot_image = cv2.cvtColor(plot_image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

            # Resize the plot image to match the video frame height
            plot_image = cv2.resize(plot_image, (background_width, background_height))

            # Combine the video frame and the plot image side by side
            combined_frame = np.hstack((masked_frame, plot_image))
            cv2.putText(combined_frame, f'Contours Counts: {contour_below_threshold_count}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 0), 2, cv2.LINE_AA)

            # Write the combined frame to the output video
            out.write(combined_frame)

            # Clear the plot to avoid overlap in the next frame
            plt.close(fig)

            frame_count += 1

            # Exit loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release video capture and writer
        cap.release()
        out.release()
        print(f"Processing completed for video: {input_video_path}")

        # Save the centroids to a CSV file
        with open('centroids.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Frame Number', 'Centroid X', 'Centroid Y'])
            for i, centroid in enumerate(centroids):
                if centroid is not None:
                    writer.writerow([i, centroid[0], centroid[1]])
                else:
                    writer.writerow([i, 'None', 'None'])
        with open('/Users/tairan/Downloads/control_empty-gal4_no-enhancer/results_angle1.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['frame','Angle'])
                for i,angle in enumerate(angle1):
                
                    writer.writerow([i,-angle])
                print(angle1)

    except Exception as e:
        print(f"An error occurred: {e}")
        print("No flies detected, skipping this video.")

    return small

results = []

# Main script to process all videos in a folder
input_folder = '/Users/tairan/Downloads/9/seg0'
output_folder = '/Users/tairan/Downloads/9/outputnew1'
# Function to extract numerical value from filename for sorting
def extract_number(filename):
    match = re.search(r'segment_(\d+)_', filename)
    return int(match.group(1)) if match else float('inf')

# Sort the video files by segment number
sorted_videos = sorted([f for f in os.listdir(input_folder) if f.endswith('.mp4')], key=extract_number)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)



# Initialize counters
videos_with_small_contours = 0
small_contour_counts = {}

# Process each video in the folder




for video_filename in os.listdir(input_folder):
    if video_filename.endswith('.mp4'):  # Adjust if there are other video formats
        input_video_path = os.path.join(input_folder, video_filename)
        output_video_path = os.path.join(output_folder, f'processed_{video_filename}')
        
        # Process the video and get the contour count
        small_contour_count = process_video(input_video_path, output_video_path)
        
        # Save results to CSV
        with open('/Users/tairan/Downloads/9/results.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Trial','Count'])
            writer.writerow([f'processed_{video_filename}', small_contour_count])
        
        # Update contour count tracking
        if small_contour_count:
            videos_with_small_contours += 1
        small_contour_counts[video_filename] = small_contour_count  

# Print the results
print(f'Number of videos with at least one lifting behavior: {videos_with_small_contours}')
print('Occurrences for each video:')
for video, count in small_contour_counts.items():
    print(f'{video}: {count}')

print("Processing complete.")


