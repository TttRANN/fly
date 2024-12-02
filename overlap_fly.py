

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def process_video(input_video_path, output_dir, start_frame, end_frame, frame_step):
    try:
        # Load the video
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {input_video_path}")
            return

        # Get video dimensions
        background_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        background_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        # Initialize background image (e.g., a blank image)
        background_frame = np.zeros((background_height, background_width, 3), dtype=np.uint8)
        composite_image = cv2.cvtColor(background_frame, cv2.COLOR_BGR2RGBA)

        # Create an alpha channel for transparency
        composite_image[:, :, 3] = 0  # Set initial alpha to 0 (fully transparent)

        # Calculate the radius for the circular mask
        mask_radius = int((background_width // 2) * 0.99)
        center = (background_width // 2, background_height // 2)

        # Initialize the mask with zeros
        mask = np.zeros((background_height, background_width), dtype=np.uint8)
        cv2.circle(mask, center, mask_radius, 255, thickness=-1)

        # Ensure the end frame does not exceed the total number of frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        end_frame = min(end_frame, total_frames)

        # Initialize list to store fly positions (optional, for plotting trajectory)
        fly_positions = []

        # Loop through the specified frame range with the given step
        for frame_number in range(start_frame, end_frame, frame_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret:
                print(f"Frame {frame_number} could not be read.")
                continue

            # Apply the mask to the current frame
            masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

            # Convert the frame to grayscale
            gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)

            # Apply a binary threshold to the frame
            ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

            # Find contours in the thresholded frame
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Filter contours based on area
            valid_contours = [c for c in contours if 50 < cv2.contourArea(c) < 10000]

            if valid_contours:
                # Assuming the largest contour is the fly
                biggest_contour = max(valid_contours, key=cv2.contourArea)

                # Ensure the contour has at least 5 points to fit an ellipse
                if len(biggest_contour) >= 5:
                    # Fit an ellipse to the fly's contour
                    ellipse = cv2.fitEllipse(biggest_contour)

                    # Create a mask for the fly based on the fitted ellipse
                    fly_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    cv2.ellipse(fly_mask, ellipse, 255, thickness=-1)

                    # Extract the fly image using the mask
                    fly_image = cv2.bitwise_and(frame, frame, mask=fly_mask)

                    # Crop the fly image to the bounding rectangle of the ellipse
                    x, y, w, h = cv2.boundingRect(biggest_contour)
                    fly_image_cropped = fly_image[y:y+h, x:x+w]
                    fly_mask_cropped = fly_mask[y:y+h, x:x+w]
                    fly_image = cv2.resize(fly_image, (10, 10))

                    # Convert the cropped fly image to RGBA
                    fly_image_rgba = cv2.cvtColor(fly_image_cropped, cv2.COLOR_BGR2RGBA)

                    # Set the alpha channel based on the mask
                    fly_image_rgba[:, :, 3] = fly_mask_cropped

                    # Overlay the fly image onto the composite image
                    overlay_image(composite_image, fly_image_rgba, x, y)

                    # Store the center of the ellipse for trajectory plotting (optional)
                    cX, cY = int(ellipse[0][0]), int(ellipse[0][1])
                    fly_positions.append((cX, cY))
                else:
                    print(f"Contour in frame {frame_number} has less than 5 points; cannot fit ellipse.")
            else:
                print(f"No valid contours found in frame {frame_number}.")

        cap.release()

        # Plot the composite image
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(composite_image, cv2.COLOR_BGRA2RGBA))
        plt.axis('off')
        plt.title(f"Fly Trajectory from Frame {start_frame} to {end_frame} (Step: {frame_step})")


        if fly_positions:
                

                x_positions, y_positions = zip(*fly_positions)
                dx = np.diff(x_positions)
                dy = np.diff(y_positions)
                # plt.figure(dpi=1500)
                # print(np.size(dx))
                # print(np.size(x_positions))
                plt.plot(x_positions[:6], y_positions[:6], color='white', linewidth=2)
                print(np.size(x_positions))
                plt.plot(x_positions[5:], y_positions[5:], color='red', linewidth=2)
    #             plt.quiver(x_positions[:6], y_positions[:6],  # Starting points of arrows
    #     dx[0:6], dy[0:6],  # Direction vectors (differences between points)
    #     angles='xy', scale_units='xy', scale=1, color='white', width=0.005
    # )
    #             plt.quiver(x_positions[8:30], y_positions[8:30],  # Starting points of arrows
    #     dx[7:29], dy[7:29],  # Direction vectors (differences between points)
    #     angles='xy', scale_units='xy', scale=1, color='red', width=0.005
    # )
    #             plt.quiver(x_positions[31:-1], y_positions[31:-1],  # Starting points of arrows
    #     dx[30:-1], dy[30:-1],  # Direction vectors (differences between points)
    #     angles='xy', scale_units='xy', scale=1, color='white', width=0.005
    # )

        # Save the figure
        output_image_path = os.path.join(output_dir, f"fly_trajectory_{start_frame}_to_{end_frame}_step_{frame_step}.png")
        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
        plt.show()

        print(f"Fly trajectory plotted and saved to {output_image_path}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

def overlay_image(background, overlay, x, y):
    """
    Overlay an RGBA image onto a background image at position (x, y) with alpha blending.
    """
    bg_height, bg_width = background.shape[:2]
    overlay_height, overlay_width = overlay.shape[:2]

    # Ensure the overlay does not go out of bounds
    if x + overlay_width > bg_width:
        overlay_width = bg_width - x
        overlay = overlay[:, :overlay_width]
    if y + overlay_height > bg_height:
        overlay_height = bg_height - y
        overlay = overlay[:overlay_height]

    # Extract the regions of interest from the background
    bg_roi = background[y:y+overlay_height, x:x+overlay_width].copy()

    # Convert to float for blending
    overlay_img = overlay[:, :, :3].astype(float)
    overlay_alpha = overlay[:, :, 3].astype(float) / 255.0
    bg_img = bg_roi[:, :, :3].astype(float)
    bg_alpha = bg_roi[:, :, 3].astype(float) / 255.0

    # Compute the alpha blending
    combined_alpha = overlay_alpha + bg_alpha * (1 - overlay_alpha)
    # Avoid division by zero
    combined_alpha[combined_alpha == 0] = 1e-6

    combined_color = (overlay_img * overlay_alpha[..., None] + bg_img * bg_alpha[..., None] * (1 - overlay_alpha[..., None])) / combined_alpha[..., None]

    # Update the background ROI with blended result
    bg_roi[:, :, :3] = combined_color.astype(np.uint8)
    bg_roi[:, :, 3] = (combined_alpha * 255).astype(np.uint8)

    # Replace the ROI on the background image
    background[y:y+overlay_height, x:x+overlay_width] = bg_roi

# Main function to run the script
if __name__ == "__main__":
    # Input video path
    # input_video_path = "path_to_your_video.mp4"
    input_video_path = '/Users/tairan/Downloads/testfor/cas9_plexa-male_batch1/seg1_filtered_0_arena_0_l2sec/segment_20159_20399_last_2_seconds.mp4'
  # Replace with your video file path

    # Output directory to save the results
    output_dir = '/Users/tairan/Downloads/output1' # Replace with your desired output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Parameters for processing
    start_frame = 0          # Starting frame number
    end_frame = 300      # Ending frame number
    frame_step = 30          # Process every 'frame_step' frames

    # Call the process_video function
    process_video(input_video_path, output_dir, start_frame, end_frame, frame_step)
