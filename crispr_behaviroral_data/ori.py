import cv2
import numpy as np

def draw_orientation(frame, centroid, orientation, major_axis_length, color=(0, 0, 255), thickness=2):
    """
    Draws an arrow on the frame along the long axis of the ellipse to represent the fly's orientation.
    
    Parameters:
    - frame: The image frame on which to draw.
    - centroid: A tuple (x, y) representing the centroid of the fly.
    - orientation: The angle of the major axis of the ellipse, in degrees.
    - major_axis_length: Length of the major axis of the ellipse.
    - color: The color of the arrow in BGR format (default is red).
    - thickness: The thickness of the arrow line (default is 2).
    """
    # Add 90 degrees to the orientation angle
    corrected_orientation = orientation + 90

    # Convert corrected orientation angle from degrees to radians
    angle_rad = np.deg2rad(corrected_orientation)
    
    # Calculate the endpoint of the arrow based on the corrected orientation angle and major axis length
    x2 = int(centroid[0] + (major_axis_length / 2) * np.cos(angle_rad))
    y2 = int(centroid[1] + (major_axis_length / 2) * np.sin(angle_rad))
    
    # Draw the arrowed line from the centroid to the calculated endpoint
    cv2.arrowedLine(frame, centroid, (x2, y2), color, thickness)

def process_video(input_video_path, output_video_path):
    try:
        # Load the video
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {input_video_path}")
            return

        small, last_small_contour_frame, frame_count = 0, -30, 0
        contour_areas, valid_contours = [], []
        previous_orientation = None  # To track the orientation of the previous frame

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
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (background_width, background_height))

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

            # Filter and draw the contours
            valid_contours = [c for c in contours if 50 < cv2.contourArea(c) < 10000]

            if valid_contours:
                # Assume the largest contour corresponds to the fly
                largest_contour = max(valid_contours, key=cv2.contourArea)

                # Fit an ellipse to the largest contour
                ellipse = cv2.fitEllipse(largest_contour)

                # Draw the ellipse on the frame
                cv2.ellipse(masked_frame, ellipse, (0, 255, 0), 2)

                # Get the centroid, orientation, and major axis length from the ellipse
                centroid = (int(ellipse[0][0]), int(ellipse[0][1]))
                orientation = ellipse[2]
                major_axis_length = max(ellipse[1])  # ellipse[1] gives (major_axis_length, minor_axis_length)

                # If this is not the first frame, check the angle difference
                if previous_orientation is not None:
                    angle_difference = abs(orientation - previous_orientation)
                    if angle_difference > 100:
                        print(f"Large angle change detected: {angle_difference} degrees at frame {frame_count}")
                        # Skip this frame or adjust the angle as needed
                        previous_orientation = orientation
                        continue

                # Draw the orientation arrow along the long axis of the ellipse
                draw_orientation(masked_frame, centroid, orientation, major_axis_length)

                # Update the previous orientation
                previous_orientation = orientation

            # Write the processed frame to the output video
            out.write(masked_frame)

            frame_count += 1

            # Exit loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release video capture and writer
        cap.release()
        out.release()
        print(f"Processing completed for video: {input_video_path}")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("No flies detected, skipping this video.")

    return small

# Example usage
input_folder = '/Users/tairan/Downloads/rnai_fas3_t4t5_batch3/seg0/segment_540_780.mp4'
output_folder = '/Users/tairan/Downloads/outputnew2.mp4'

# Call the function to process the video
process_video(input_folder, output_folder)
