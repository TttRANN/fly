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

    # Extract the starting frame number from the output folder name
    output_folder_name = os.path.basename(output_dir)
    start_frame_number = int(output_folder_name.split('_')[1])

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
            if len(frames) >= 10:  # Only save if segment has 10 or more frames
                end_frame_number = start_frame_number + frame_count - 1
                segment_filename = os.path.join(output_dir, f"segment_{start_frame_number:06d}_{end_frame_number:06d}.mp4")
                save_video_segment(segment_filename, frames, fps, width, height)
            segment_index += 1
            frame_count = 0
            frames = []
            start_frame_number = end_frame_number + 1

    # Save any remaining frames as the last segment, if it meets the minimum frame requirement
    if frames and len(frames) >= 10:
        end_frame_number = start_frame_number + len(frames) - 1
        segment_filename = os.path.join(output_dir, f"segment_{start_frame_number:06d}_{end_frame_number:06d}.mp4")
        save_video_segment(segment_filename, frames, fps, width, height)

    cap.release()
    print(f"Video segmentation completed for {video_path}.")

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

def process_all_videos_in_folder(folder_path, output_base_dir, segment_length=16):
    # Get all video files in the folder
    video_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]

    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        output_dir = os.path.join(output_base_dir, os.path.splitext(video_file)[0])

        print(f"Processing video: {video_path}")
        segment_video(video_path, output_dir, segment_length)

# Example usage
folder_path = '/Users/tairan/Downloads/29c/rnai_SIDE-VIII-29C_t4t5_batch2/seg2_filtered_3_arena_3'  # Replace with your folder path
output_base_dir = '/Users/tairan/Downloads/processed_videos_rnai_SIDE-VIII-29C_t4t5_batch2_seg2_filtered_3_arena_3'  # Replace with your output base directory
segment_length = 30  # Adjust the segment length as needed

process_all_videos_in_folder(folder_path, output_base_dir, segment_length)
