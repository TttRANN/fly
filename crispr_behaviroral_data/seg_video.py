import cv2
import numpy as np
import os

def segment_video(video_path, output_dir, segment_length=16):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize variables
    segment_index = 0
    frame_count = 0
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Apply the circular mask to the frame
        frame = apply_circular_mask(frame, width, height)

        frames.append(frame)
        frame_count += 1

        # When we reach the segment length, save the segment as a new video
        if frame_count == segment_length:
            segment_filename = os.path.join(output_dir, f"segment_{segment_index:04d}.mp4")
            save_video_segment(segment_filename, frames, fps, width, height)
            segment_index += 1
            frame_count = 0
            frames = []

    # Save any remaining frames as the last segment
    if frames:
        segment_filename = os.path.join(output_dir, f"segment_{segment_index:04d}.mp4")
        save_video_segment(segment_filename, frames, fps, width, height)

    cap.release()
    print("Video segmentation completed.")

def apply_circular_mask(frame, width, height):
    # Create a mask with a black background
    mask = np.zeros((height, width, 3), dtype=np.uint8)

    # Define the center and radius of the circle (90% of the smallest dimension)
    center = (width // 2, height // 2)
    radius = int(min(width, height) * 0.45)

    # Draw a white filled circle in the center of the mask
    cv2.circle(mask, center, radius, (255, 255, 255), -1)

    # Apply the mask: The inside of the circle remains the original frame, the outside becomes black
    frame = cv2.bitwise_and(frame, mask)

    return frame

def save_video_segment(filename, frames, fps, width, height):
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()

# Example usage
video_path = '/Users/tairan/Downloads/29c/rnai_BEAT-IV-29C_t4t5_batch1/seg1_filtered_2_arena_2/segment_180_420.mp4'
# output_video_path = '/Users/tairan/Downloads/processed_videos/'
# video_path = '/Users/tairan/Downloads/segment_240_480.mp4'  # Replace with your video path
output_dir = '/Users/tairan/Downloads/segment'  # Replace with your output directory

segment_video(video_path, output_dir, segment_length=30)
