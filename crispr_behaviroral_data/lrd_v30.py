import os
import re
import cv2
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def apply_circular_mask(frame, radius, center):
    """Apply a circular mask to a frame, setting everything outside the circle to black."""
    # Ensure that center coordinates and radius are integers
    center = (int(center[0]), int(center[1]))
    radius = int(radius)

    # Create a mask with the same dimensions as the frame
    mask = np.zeros_like(frame)

    # Draw a white circle on the mask
    cv2.circle(mask, center, radius, (255, 255, 255), thickness=-1)

    # Apply the mask to the frame
    masked_frame = cv2.bitwise_and(frame, mask)

    return masked_frame

def process_video(input_video_path, output_animation_path, fps):
    start_time = time.time()

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {input_video_path}")
        return [], [], [], [], []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    background_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    background_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    center = (background_width // 2, background_height // 2)
    radius = int(min(background_width, background_height) * 0.5)
    radius1 = int(min(background_width, background_height) * 0.4)

    # Initialize the video writer to save the processed video
    output_video_path = output_animation_path.replace('_animation.mp4', '_processed.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (background_width, background_height))

    axis_ratios_over_time = []  # List to store axis ratios over time
    small_list = []
    small = 0
    consecutive_small_frames = 0
    axis_ratios = []

    small_start_frames = []
    small_stop_frames = []
    small_centroid_x = []
    small_centroid_y = []
    current_small_centroid_x = []
    current_small_centroid_y = []
    small_condition_active = False

    frame_count = 0
    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            print("End of video reached unexpectedly.")
            break

        masked_frame = apply_circular_mask(frame, radius, center)
        gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 23, 3)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        valid_contours = [c for c in contours if 50 < cv2.contourArea(c) < 200]

        if valid_contours:
            biggest_contour = max(valid_contours, key=cv2.contourArea)

            if len(biggest_contour) < 5:
                print('Contour has fewer than 5 points')
                small = 'NaN'
                small_list.append('NaN')
                break

            ellipse = cv2.fitEllipse(biggest_contour)
            cv2.ellipse(masked_frame, ellipse, (0, 255, 0), 2)

            centroid = (int(ellipse[0][0]), int(ellipse[0][1]))
            distance_from_center = np.sqrt(
                (centroid[0] - center[0]) ** 2 + (centroid[1] - center[1]) ** 2)

            # Extract the axes lengths
            axes_lengths = ellipse[1]
            major_axis = max(axes_lengths)
            minor_axis = min(axes_lengths)

            # Calculate the ratio of the major axis to the minor axis
            axis_ratio = major_axis / minor_axis
            axis_ratios.append(axis_ratio)
            axis_ratios_over_time.append(axis_ratio)
            avg_axis_ratio = np.mean(axis_ratios)

            print(f"Frame {frame_count}:")
            print(f"Major axis: {major_axis}")
            print(f"Minor axis: {minor_axis}")
            print(f"Axis ratio: {axis_ratio}")
            print(f"Average axis ratio: {avg_axis_ratio}")

            if distance_from_center > radius1:
                print('Centroid outside of the radius')
                small = 'NaN'
                small_list.append('NaN')
                break

            # --- Draw the major and minor axes on the masked_frame ---
            # Get the center, axes lengths, and angle from the ellipse
            (x_center, y_center) = ellipse[0]
            (major_axis_length, minor_axis_length) = ellipse[1]
            angle = ellipse[2]

            # Convert angle from degrees to radians
            angle_rad = np.deg2rad(angle)

            # Calculate end points of the major axis
            dx_major = (major_axis_length / 2) * np.cos(angle_rad)
            dy_major = (major_axis_length / 2) * np.sin(angle_rad)
            p1_major = (int(x_center - dx_major), int(y_center - dy_major))
            p2_major = (int(x_center + dx_major), int(y_center + dy_major))
            cv2.line(masked_frame, p1_major, p2_major, (255, 0, 0), 2)  # Blue color for major axis

            # Calculate end points of the minor axis
            angle_rad_perp = angle_rad + np.pi / 2
            dx_minor = (minor_axis_length / 2) * np.cos(angle_rad_perp)
            dy_minor = (minor_axis_length / 2) * np.sin(angle_rad_perp)
            p1_minor = (int(x_center - dx_minor), int(y_center - dy_minor))
            p2_minor = (int(x_center + dx_minor), int(y_center + dy_minor))
            cv2.line(masked_frame, p1_minor, p2_minor, (0, 0, 255), 2)  # Red color for minor axis
            # --- End of drawing axes ---

            # If the axis ratio is smaller than a threshold, mark it as a "small condition"
            if axis_ratio < avg_axis_ratio / 1.2:
                consecutive_small_frames += 1
                axis_ratios.pop(-1)  # Remove the current ratio
                avg_axis_ratio = np.mean(axis_ratios) if axis_ratios else axis_ratio
                axis_ratios.append(axis_ratio)  # Re-add the current ratio

                if consecutive_small_frames > 1 and not small_condition_active:
                    # Start a new small condition
                    small_start_frames.append(frame_count - 1)
                    small_condition_active = True
                    small += 1  # Increment the small condition count
                    print(f"small_condition_active set to True at frame {frame_count}")

                current_small_centroid_x.append(centroid[0])
                current_small_centroid_y.append(centroid[1])

            else:
                if small_condition_active and consecutive_small_frames >= 5:
                    print(f"Triggered small condition logic at frame {frame_count}")

                    # Append the centroids of this small condition
                    small_centroid_x.append(np.mean(current_small_centroid_x))
                    small_centroid_y.append(np.mean(current_small_centroid_y))

                    current_small_centroid_x = []
                    current_small_centroid_y = []

                if small_condition_active:
                    small_stop_frames.append(frame_count)
                    small_condition_active = False
                    print(f"small_condition_active set to False at frame {frame_count}")

                consecutive_small_frames = 0

            # Append the current `small` value for this frame (active or not)
            small_list.append(small)

        frame_count += 1

        cv2.imshow("Ellipse Fitting", masked_frame)
        out.write(masked_frame)  # Save the frame with axes drawn

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if small_condition_active:
        small_stop_frames.append(total_frames - 1)
        small_centroid_x.append(np.mean(current_small_centroid_x))
        small_centroid_y.append(np.mean(current_small_centroid_y))
        print(f"small_condition_active ended at the last frame: {total_frames - 1}")

    # Ensure small_list matches the length of the video
    while len(small_list) < total_frames:
        small_list.append(small)

    cap.release()
    out.release()  # Release the video writer
    cv2.destroyAllWindows()

    print(f"Processing completed for video: {input_video_path} in {time.time() - start_time} seconds.")
    print(f"Small condition start frames: {small_start_frames}")
    print(f"Small condition stop frames: {small_stop_frames}")
    print(f"Small condition count: {small}")

    # Create the animation (as before)
    create_animation(axis_ratios_over_time, output_animation_path, fps, small_start_frames, small_stop_frames)

    return small_list, small_start_frames, small_stop_frames, small_centroid_x, small_centroid_y

def create_animation(axis_ratios_over_time, output_animation_path, fps, small_start_frames, small_stop_frames):
    """Create an animation for the axis ratios over time with shaded areas for the 'small' condition."""
    frame_count = len(axis_ratios_over_time)
    
    fig, ax = plt.subplots()
    ax.set_xlim(0, frame_count)
    ax.set_ylim(0, max(axis_ratios_over_time) * 1.1 if axis_ratios_over_time else 10)
    
    line, = ax.plot([], [], lw=2, label='Axis Ratio')
    avg_line, = ax.plot([], [], lw=2, label='Average Axis Ratio', color='r')

    # Shade the areas where the small condition is satisfied
    for start, stop in zip(small_start_frames, small_stop_frames):
        if stop - start >= 1:  # Only shade if at least 1 consecutive frame meets the small condition
            ax.axvspan(start, stop, facecolor='gray', alpha=0.3, label="Small Condition Active")

    ax.legend()

    def init():
        line.set_data([], [])
        avg_line.set_data([], [])
        return line, avg_line

    def update(frame):
        xdata = list(range(frame + 1))
        ydata = axis_ratios_over_time[:frame + 1]
        avg_ydata = [np.mean(axis_ratios_over_time[:i+1]) for i in range(frame + 1)]
        line.set_data(xdata, ydata)
        avg_line.set_data(xdata, avg_ydata)
        return line, avg_line

    ani = animation.FuncAnimation(fig, update, frames=frame_count, init_func=init, blit=True, repeat=False)
    ani.save(output_animation_path, writer=animation.FFMpegWriter(fps=fps))
    plt.close(fig)

def combine_videos(input_video_path, animation_video_path, output_combined_video_path, mask_radius, center, small_list):
    """Combine the processed video with axis drawing and the animation video side by side, overlaying the 'small' value for each frame."""
    cap1 = cv2.VideoCapture(input_video_path)
    cap2 = cv2.VideoCapture(animation_video_path)

    # Check if the videos were opened correctly
    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Could not open one or both video files.")
        return

    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)

    # Ensure both videos have the same frame rate (fps)
    if fps1 != fps2:
        print(f"Warning: FPS mismatch between videos! FPS1: {fps1}, FPS2: {fps2}")
    fps = min(fps1, fps2)  # Use the lower frame rate to avoid issues

    # Get the frame count of both videos
    frame_count1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = min(frame_count1, frame_count2)  # Use the minimum to avoid mismatches

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_combined_video_path, fourcc, fps, (width * 2, height))

    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (30, 50)  # Position to display the small value on the video
    fontScale = 1
    color = (255, 255, 0)  # Yellow color for text
    thickness = 2

    frame_idx = 0  # To keep track of the current frame

    # Loop through all frames, ensuring you process as many frames as exist in both videos
    while frame_idx < frame_count:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            print("End of one of the videos reached.")
            break

        # Resize the animation frame to match the video dimensions
        frame2 = cv2.resize(frame2, (width, height))

        # Combine the two frames side by side
        combined_frame = np.hstack((frame1, frame2))

        # Get the current value from small_list for this frame
        if frame_idx < len(small_list):
            small_value = small_list[frame_idx] if isinstance(small_list[frame_idx], (int, float)) else 0
        else:
            small_value = 0  # If small_list is shorter than the frame count, default to 0

        # Overlay the 'small' value (num of flipping) for the current frame
        cv2.putText(combined_frame, f"Num of Flipping: {small_value}", org, font, fontScale, color, thickness, cv2.LINE_AA)

        # Write the combined frame to the output video
        out.write(combined_frame)

        frame_idx += 1  # Increment frame index to move to the next frame

    cap1.release()
    cap2.release()
    out.release()
    print(f"Combined video saved as: {output_combined_video_path}")

def extract_number(filename):
    """Extract numerical value from filename for sorting."""
    match = re.search(r'segment_(\d+)_', filename)
    return int(match.group(1)) if match else float('inf')

def process_folder(parent_directory, rnai_folder, seg_folder, fps=30):
    """Process the videos within a specific segment folder."""
    input_folder = os.path.join(parent_directory, rnai_folder, seg_folder)
    output_folder = os.path.join(input_folder, 'output_videos')
    os.makedirs(output_folder, exist_ok=True)

    sorted_videos = sorted([f for f in os.listdir(input_folder) if f.endswith('.mp4')], key=extract_number)
    small_contour_counts = {}

    for video_filename in sorted_videos:
        input_video_path = os.path.join(input_folder, video_filename)
        output_animation_path = os.path.join(output_folder, video_filename.replace('.mp4', '_animation.mp4'))
        output_combined_video_path = os.path.join(output_folder, video_filename.replace('.mp4', '_combined.mp4'))

        cap = cv2.VideoCapture(input_video_path)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            center = (width // 2, height // 2)
            mask_radius = int((width // 2))
        cap.release()

        # Run the process_video function and retrieve small_list, start, and stop frames
        small_list, small_start_frames, small_stop_frames, small_x, small_y = process_video(input_video_path, output_animation_path, fps)

        # Get the path to the processed video
        processed_video_path = output_animation_path.replace('_animation.mp4', '_processed.mp4')

        # Combine videos with small_list using the processed video
        combine_videos(processed_video_path, output_animation_path, output_combined_video_path, mask_radius, center, small_list)

        # Create a structured dictionary entry for this video
        small_contour_counts[video_filename] = {
            'small_count': small_list,  # List of small count over frames
            'small_start_frames': small_start_frames,  # List of start frames
            'small_stop_frames': small_stop_frames,
            'small_x': small_x,
            'small_y': small_y  # List of stop frames
        }

    return small_contour_counts

def main(parent_directory):
    """Main function to process each RNAi folder and segment folder."""
    # Record the start time of the main function
    start_time = time.time()

    for rnai_folder in os.listdir(parent_directory):
        rnai_folder_path = os.path.join(parent_directory, rnai_folder)
        
        # Skip if the path is not a directory
        if not os.path.isdir(rnai_folder_path):
            continue

        # Find segment folders that match the pattern
        seg_folders = [f for f in os.listdir(rnai_folder_path)
                       if os.path.isdir(os.path.join(rnai_folder_path, f)) and re.match(r'seg\d+_filtered_\d+_arena_\d+$', f)]

        for seg_folder in seg_folders:
            seg_number_match = re.search(r'seg(\d+)_filtered_(\d+)_arena_(\d+)', seg_folder)
            
            # Skip if the folder name pattern does not match
            if not seg_number_match:
                print(f"Skipping folder {seg_folder}, pattern not matched.")
                continue

            # Create CSV output file to store results for this segment
            csv_filename = os.path.join(rnai_folder_path, f'results_{rnai_folder}_seg{seg_number_match.group(1)}_arena{seg_number_match.group(3)}.csv')
            small_contour_counts = process_folder(parent_directory, rnai_folder, seg_folder)

            # Writing the results to CSV
            try:
                with open(csv_filename, 'a', newline='') as f:
                    writer = csv.writer(f)
                    
                    # Write the header only once
                    writer.writerow(['Video Filename', 'Small Count', 'Start Frames', 'Stop Frames', 'Small X', 'Small Y'])
                    
                    for video_filename, small_data in small_contour_counts.items():
                        # Assuming small_data contains small_count, start and stop frames, etc.
                        small_count = small_data.get('small_count', ['NaN'])[-1]  # Get last small_count or 'NaN'
                        small_start_frames = small_data.get('small_start_frames', [])  # Get start frames or empty list
                        small_stop_frames = small_data.get('small_stop_frames', [])
                        small_x = small_data.get('small_x', [])
                        small_y = small_data.get('small_y', [])

                        # Convert lists to comma-separated values for easy readability in CSV
                        start_frames_str = ",".join(map(str, small_start_frames))  # e.g., "100,200"
                        stop_frames_str = ",".join(map(str, small_stop_frames))    # e.g., "150,250"
                        small_x_str = ",".join(map(str, small_x))
                        small_y_str = ",".join(map(str, small_y))

                        # Write small_count, and start/stop frames as a comma-separated string
                        writer.writerow([video_filename, small_count, start_frames_str, stop_frames_str, small_x_str, small_y_str])

            except Exception as e:
                print(f"Error writing to CSV file {csv_filename}: {e}")

            print(f'Processing complete for folder: {seg_folder}')

    # Record the end time of the main function
    end_time = time.time()

    # Calculate and print the total execution time
    total_time = end_time - start_time
    print(f"Total execution time: {total_time:.2f} seconds")




if __name__ == "__main__":
    parent_directory = '/Users/tairan/Downloads/gilt1'


    main(parent_directory)



