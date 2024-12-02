import cv2
import numpy as np
import os

def process_video(input_video_path, output_dir, frames_per_image):
    try:
        # Load the video
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {input_video_path}")
            return

        # Initialize variables
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

        # Get the total number of frames in the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_group = []
        image_count = 0

        for frame_count in range(total_frames):
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
                # cv2.ellipse(masked_frame, ellipse, (25, 25, 0), 5)

            # Add the current frame to the group
            frame_group.append(masked_frame)

            # Check if we've reached the desired number of frames to merge
            if len(frame_group) == frames_per_image:
                # Merge frames in the group
                merged_frame = np.mean(frame_group, axis=0).astype(np.uint8)

                # Save the merged frame as an image
                output_image_path = os.path.join(output_dir, f"merged_frame_{image_count}.png")
                cv2.imwrite(output_image_path, merged_frame)
                image_count += 1

                # Clear the frame group for the next set of frames
                frame_group.clear()

        # If any frames remain in the group after the loop, process them as well
        if frame_group:
            merged_frame = np.mean(frame_group, axis=0).astype(np.uint8)
            output_image_path = os.path.join(output_dir, f"merged_frame_{image_count}.png")
            cv2.imwrite(output_image_path, merged_frame)

        cap.release()
        print("Processing complete.")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Example usage:
input_video_path = '/Users/tairan/Downloads/segment_179_419.mp4'
output_dir = '/Users/tairan/Downloads/output.mp4'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process the video with a specific number of frames per image
frames_per_image = 10
process_video(input_video_path, output_dir, frames_per_image)
