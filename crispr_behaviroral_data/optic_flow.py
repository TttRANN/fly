import cv2
import numpy as np

# Step 1: Load Video
video_file = '/Users/tairan/Downloads/rnai_gilt-29C_t4t5_batch3/seg1/segment_2820_3060.mp4'
video_reader = cv2.VideoCapture(video_file)

# Get video dimensions and FPS
background_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
background_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video_reader.get(cv2.CAP_PROP_FPS))

# Calculate the radius for the circular mask
mask_radius = int((background_width / 2) * 0.8)
center = (background_width // 2, background_height // 2)

# Initialize the mask with zeros
mask = np.zeros((background_height, background_width), dtype=np.uint8)
cv2.circle(mask, center, mask_radius, 255, -1)  # Draw a filled circle in the mask

# Step 2: Create Optical Flow Object (Farneback Method)
# Parameters for Farneback's optical flow
flow_params = dict(
    pyr_scale=0.5, levels=3, winsize=15, iterations=3,
    poly_n=5, poly_sigma=1.2, flags=0
)

# Step 3: Set up Video Writer to save output video
output_video_file = '/Users/tairan/Downloads/output_segment_2820_3060_with_flow_vectors.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video_file, fourcc, 60.0, (background_width, background_height))

# Step 4: Loop Through Video Frames and Compute Optical Flow
ret, prev_frame = video_reader.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_gray = cv2.bitwise_and(prev_gray, prev_gray, mask=mask)  # Apply mask to the first frame

while True:
    ret, frame = video_reader.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    masked_frame = cv2.bitwise_and(frame_gray, frame_gray, mask=mask)  # Apply the mask to the grayscale frame

    # Compute optical flow using Farneback's method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, masked_frame, None, **flow_params)

    # Create an empty image for drawing the vector field
    flow_vectors = np.zeros_like(frame)

    # Step size for drawing arrows
    step = 5

    for y in range(0, flow.shape[0], step):
        for x in range(0, flow.shape[1], step):
            fx, fy = flow[y, x]
            cv2.arrowedLine(flow_vectors, (x, y), (int(x + fx), int(y + fy)), (255, 255, 255), 1, tipLength=0.5)

    # Combine original frame with the vector field
    combined_frame = cv2.addWeighted(frame, 0.8, flow_vectors, 0.5, 0)

    # Write the frame with optical flow vector field to the video
    video_writer.write(combined_frame)

    # Update previous frame
    prev_gray = masked_frame.copy()

# Step 5: Close the Video Writer and Release the Video Reader
video_writer.release()
video_reader.release()
