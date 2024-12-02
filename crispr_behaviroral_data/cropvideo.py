import os
import cv2

def save_last_two_seconds(input_video_path, output_video_path):
    """Extracts and saves the last two seconds of the input video."""
    
    # Open the video
    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        print(f"Error: Failed to open video {input_video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Width of the video
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Height of the video

    # Calculate the number of frames to save for the last two seconds
    frames_to_save = fps * 2  # Number of frames in the last 2 seconds
    start_frame = max(0, total_frames - frames_to_save)  # Starting frame for the last two seconds

    # Set the video capture to the start frame of the last two seconds
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Set up the video writer for the output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Loop over frames and write them to the output video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    print(f"Saved last two seconds of video to: {output_video_path}")


def process_all_videos_in_subfolders(input_dir, output_dir):
    """Processes all videos in subfolders and saves the last two seconds to the output directory."""
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Walk through all subdirectories and files
    for subdir, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.mp4'):  # Check if the file is a video (you can add more formats if needed)
                input_video_path = os.path.join(subdir, file)

                # Create a corresponding subfolder structure in the output directory
                relative_subdir = os.path.relpath(subdir, input_dir)
                output_subfolder = os.path.join(output_dir, relative_subdir)
                if not os.path.exists(output_subfolder):
                    os.makedirs(output_subfolder)

                # Output video path in the new directory
                output_video_path = os.path.join(output_subfolder, file.replace('.mp4', '_last_2_seconds.mp4'))

                # Process the video and save the last two seconds
                save_last_two_seconds(input_video_path, output_video_path)


# Example usage:
input_dir = "/Users/tairan/Downloads/test111/rnai_cg9394_t4t5_batch2/1seg2_filtered_1_arena_1"
output_dir = "/Users/tairan/Downloads/processed_videos"

process_all_videos_in_subfolders(input_dir, output_dir)
