import cv2
import numpy as np
import os

def process_video(input_video_path, output_dir):
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

        # Get the total number of frames in the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Loop through the video frame by frame
        for i in range(6):
            denominator = pow(2, i)
            frame_frag = 15
            
            # Initialize a list to store frames for merging
            frame_group = []
            group_count = 0

            for j in range(denominator):
                start_frame = j * frame_frag
                end_frame = (j + 1) * frame_frag

                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

                for _ in range(start_frame, end_frame):
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

                # Merge frames in the group after processing all frames within the current range
                if frame_group:
                    merged_frame = np.mean(frame_group, axis=0).astype(np.uint8)

                    # Save the merged frame as an image
                    output_image_path = os.path.join(output_dir, f"iteration_{i}_merged_frame_{j}.png")
                    cv2.imwrite(output_image_path, merged_frame)
                    group_count=group_count+1

                    # Clear the frame group
                    frame_group.clear()

            print(f"Iteration {i+1} complete. {group_count} images saved to {output_dir}")

        cap.release()
        print("Processing complete.")
    

    except Exception as e:
        print(f"An error occurred: {str(e)}")



# Example usage:
input_video_path = '/Users/tairan/Downloads/segment_179_419.mp4'
output_dir = '/Users/tairan/Downloads/output.mp4'

# # Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

process_video(input_video_path, output_dir)


