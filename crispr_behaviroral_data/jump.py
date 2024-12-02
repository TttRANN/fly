import os
import re
import cv2
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def apply_circular_mask(frame, radius, center):
    """Apply a circular mask to a frame, setting everything outside the circle to white."""
    mask = np.zeros_like(frame)
    cv2.circle(mask, center, radius, (255, 255, 255), thickness=-1)
    masked_frame = cv2.bitwise_and(frame, mask)
    return masked_frame

def process_video(input_video_path, output_animation_path, fps):
    start_time = time.time()

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {input_video_path}")
        return [], [], [], [], [], [], [], [], []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    background_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    background_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    center = (background_width // 2, background_height // 2)
    radius = int(min(background_width, background_height) * 0.5)
    radius1 = int(min(background_width, background_height) * 0.4)

    contour_areas_over_time = []
    small_list = []
    small = 0
    consecutive_small_frames = 0
    contour_areas = []

    small_start_frames = []
    small_stop_frames = []
    small_centroid_x = []
    small_centroid_y = []
    current_small_centroid_x = []
    current_small_centroid_y = []
    small_condition_active = False

    # Variables for jump detection
    jump_threshold = 10  # Adjust this threshold as needed
    min_jump_frames = 2  # Minimum consecutive frames to qualify as a jump
    centroid_x_list = []
    centroid_y_list = []
    jump_list = []
    jump_start_frames = []
    jump_stop_frames = []
    jump_condition_active = False
    consecutive_jump_frames = 0

    # Variables for displacement tracking
    displacements = []

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
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 23, 3
        )
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        valid_contours = [c for c in contours if 50 < cv2.contourArea(c) < 200]

        if valid_contours:
            biggest_contour = max(valid_contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(biggest_contour)
            contour_areas.append(contour_area)
            contour_areas_over_time.append(contour_area)
            avg_largest_contour_area = np.mean(contour_areas)

            if len(biggest_contour) < 5:
                print('Contour has fewer than 5 points')
                small = 'NaN'
                small_list.append('NaN')
                break

            ellipse = cv2.fitEllipse(biggest_contour)
            cv2.ellipse(masked_frame, ellipse, (0, 255, 0), 2)

            centroid = (int(ellipse[0][0]), int(ellipse[0][1]))
            distance_from_center = np.sqrt(
                (centroid[0] - center[0]) ** 2 + (centroid[1] - center[1]) ** 2
            )
            (ellipse_center, axes, angle) = ellipse

            # # The major axis is the largest dimension in 'axes'
            # major_axis = max(axes)
            # minor_axis = min(axes)
            # print(f"Major axis: {major_axis}")
            # print(f"Minor axis: {minor_axis}")

            if distance_from_center > radius1:
                print('Centroid outside of the radius')
                small = 'NaN'
                small_list.append('NaN')
                break

            # Append centroid positions for jump detection
            centroid_x_list.append(centroid[0])
            centroid_y_list.append(centroid[1])

            # Displacement calculation
            if len(centroid_x_list) > 1:
                displacement = np.sqrt(
                    (centroid_x_list[-1] - centroid_x_list[-2]) ** 2 +
                    (centroid_y_list[-1] - centroid_y_list[-2]) ** 2
                )
            else:
                displacement = 0  # For the first frame
            displacements.append(displacement)

            # Jump detection logic
            if displacement > jump_threshold:
                consecutive_jump_frames += 1
                if not jump_condition_active:
                    jump_condition_active = True
                    jump_start_frames.append(frame_count - 1)
                    # print(f"Jump started at frame {frame_count - 1}")
            else:
                if jump_condition_active:
                    if consecutive_jump_frames >= min_jump_frames:
                        jump_stop_frames.append(frame_count - 1)
                        # print(f"Jump detected from frame {jump_start_frames[-1]} to {jump_stop_frames[-1]}")
                    else:
                        # Remove start frame if jump is too short
                        jump_start_frames.pop(-1)
                    jump_condition_active = False
                consecutive_jump_frames = 0

            # Record jump status for the current frame
            jump_list.append(1 if jump_condition_active else 0)

            # Small condition logic
            if contour_area < avg_largest_contour_area / 1.2:
                consecutive_small_frames += 1
                contour_areas.pop(-1)
                avg_largest_contour_area = np.mean(contour_areas)
                contour_areas.append(avg_largest_contour_area)

                if consecutive_small_frames > 4 and not small_condition_active:
                    # Start a new small condition
                    small_start_frames.append(frame_count - 1)
                    small_condition_active = True
                    small += 1  # Increment the small condition count immediately when the condition starts
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

        else:
            # If no valid contours are found
            if centroid_x_list:
                centroid_x_list.append(centroid_x_list[-1])
                centroid_y_list.append(centroid_y_list[-1])
            else:
                centroid_x_list.append(0)
                centroid_y_list.append(0)
            displacements.append(0)
            jump_list.append(0)
            small_list.append(small)

        frame_count += 1

        # cv2.imshow("Ellipse Fitting", masked_frame)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # Handle jump condition at the end of the video
    if jump_condition_active:
        if consecutive_jump_frames >= min_jump_frames:
            jump_stop_frames.append(frame_count - 1)
            # print(f"Jump condition ended at the last frame: {frame_count - 1}")
        else:
            jump_start_frames.pop(-1)

    if small_condition_active:
        small_stop_frames.append(total_frames - 1)
        small_centroid_x.append(np.mean(current_small_centroid_x))
        small_centroid_y.append(np.mean(current_small_centroid_y))
        # print(f"small_condition_active ended at the last frame: {total_frames - 1}")

    # Ensure lists match the length of the video
    while len(small_list) < total_frames:
        small_list.append(small)
    while len(jump_list) < total_frames:
        jump_list.append(0)
    while len(displacements) < total_frames:
        displacements.append(0)

    cap.release()
    cv2.destroyAllWindows()

    # print(f"Processing completed for video: {input_video_path} in {time.time() - start_time} seconds.")
    # print(f"Small condition start frames: {small_start_frames}")
    # print(f"Small condition stop frames: {small_stop_frames}")
    # print(f"Small condition count: {small}")
    # print(f"Jump start frames: {jump_start_frames}")
    # print(f"Jump stop frames: {jump_stop_frames}")

    create_animation(contour_areas_over_time, output_animation_path, fps, small_start_frames, small_stop_frames,displacements)

    return small_list, small_start_frames, small_stop_frames, small_centroid_x, small_centroid_y, jump_list, jump_start_frames, jump_stop_frames, displacements

def create_animation(contour_areas_over_time, output_animation_path, fps, small_start_frames, small_stop_frames, displacements):
    """Create an animation for the contour areas and displacement over time with shaded areas for the 'small' condition."""
    frame_count = len(contour_areas_over_time)

    fig, ax1 = plt.subplots()

    # Plot contour areas on the left y-axis
    ax1.set_xlim(0, frame_count)
    ax1.set_ylim(0, max(contour_areas_over_time) if contour_areas_over_time else 10000)
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Contour Area', color='b')

    line_contour, = ax1.plot([], [], lw=2, label='Contour Area', color='b')
    avg_line, = ax1.plot([], [], lw=2, label='Average Contour Area', color='r')

    # Shade the areas where the small condition is satisfied
    for start, stop in zip(small_start_frames, small_stop_frames):
        if stop - start >= 1:  # Adjust as needed
            ax1.axvspan(start, stop, facecolor='gray', alpha=0.3, label="Small Condition Active")

    # Create another axis for displacement on the right y-axis
    ax2 = ax1.twinx()
    ax2.set_ylim(0, max(displacements) if displacements else 1)
    ax2.set_ylabel('Displacement', color='g')
    line_displacement, = ax2.plot([], [], lw=2, label='Displacement', color='g')

    def init():
        line_contour.set_data([], [])
        avg_line.set_data([], [])
        line_displacement.set_data([], [])
        return line_contour, avg_line, line_displacement

    def update(frame):
        xdata = list(range(frame + 1))
        ydata_contour = contour_areas_over_time[:frame + 1]
        avg_ydata = [np.mean(contour_areas_over_time[:i+1]) for i in range(frame + 1)]
        ydata_displacement = displacements[:frame + 1]

        line_contour.set_data(xdata, ydata_contour)
        avg_line.set_data(xdata, avg_ydata)
        line_displacement.set_data(xdata, ydata_displacement)
        return line_contour, avg_line, line_displacement

    ani = animation.FuncAnimation(fig, update, frames=frame_count, init_func=init, blit=True, repeat=False)
    ani.save(output_animation_path, writer=animation.FFMpegWriter(fps=fps))
    plt.close(fig)


def combine_videos(input_video_path, animation_video_path, output_combined_video_path, mask_radius, center, small_list, jump_list, displacements):
    """Combine the original video with contour animation video side by side and overlay the 'small', 'jump', and 'displacement' values for each frame."""
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
    org_small = (30, 50)    # Position to display the small value on the video
    org_jump = (30, 20)    # Position to display the jump value on the video
    org_displacement = (30, 150)  # Position to display the displacement value
    fontScale = 0.5
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

        # Apply the circular mask on the first video
        masked_frame = apply_circular_mask(frame1, mask_radius, center)

        # Resize the animation frame to match the video dimensions
        frame2 = cv2.resize(frame2, (width, height))

        # Combine the two frames side by side
        combined_frame = np.hstack((masked_frame, frame2))

        # Get the current value from small_list and jump_list for this frame
        if frame_idx < len(small_list):
            small_value = small_list[frame_idx] if isinstance(small_list[frame_idx], (int, float)) else 0
        else:
            small_value = 0  # If small_list is shorter than the frame count, default to 0

        if frame_idx < len(jump_list):
            jump_value = jump_list[frame_idx]
        else:
            jump_value = 0  # If jump_list is shorter than the frame count, default to 0

        # Get the displacement value for the current frame
        if frame_idx < len(displacements):
            displacement_value = displacements[frame_idx]
        else:
            displacement_value = 0

        # Overlay the 'small', 'jump', and 'displacement' values for the current frame
        cv2.putText(combined_frame, f"Num of Flipping: {small_value}", org_small, font, fontScale, color, thickness, cv2.LINE_AA)
        cv2.putText(combined_frame, f"Jumping: {'Yes' if jump_value else 'No'}", org_jump, font, fontScale, color, thickness, cv2.LINE_AA)
        # cv2.putText(combined_frame, f"Displacement: {displacement_value:.2f}", org_displacement, font, fontScale, color, thickness, cv2.LINE_AA)

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

        # Run the process_video function and retrieve outputs
        (
            small_list,
            small_start_frames,
            small_stop_frames,
            small_x,
            small_y,
            jump_list,
            jump_start_frames,
            jump_stop_frames,
            displacements
        ) = process_video(input_video_path, output_animation_path, fps)

        # Combine videos with small_list, jump_list, and displacements
        combine_videos(
            input_video_path,
            output_animation_path,
            output_combined_video_path,
            mask_radius,
            center,
            small_list,
            jump_list,
            displacements
        )

        # Create a structured dictionary entry for this video
        small_contour_counts[video_filename] = {
            'small_count': small_list,  # List of small count over frames
            'small_start_frames': small_start_frames,  # List of start frames
            'small_stop_frames': small_stop_frames,
            'small_x': small_x,
            'small_y': small_y,
            'jump_list': jump_list,
            'jump_start_frames': jump_start_frames,
            'jump_stop_frames': jump_stop_frames,
            'displacements': displacements
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
                    writer.writerow([
                        'Video Filename',
                        'Small Count',
                        'Start Frames',
                        'Stop Frames',
                        'Small X',
                        'Small Y',
                        'Jump Count',
                        'Jump Start Frames',
                        'Jump Stop Frames',
                        'Average Displacement'  # Added Average Displacement header
                    ])

                    for video_filename, data in small_contour_counts.items():
                        # Extract data
                        small_count = data.get('small_count', ['NaN'])[-1]  # Get last small_count or 'NaN'
                        small_start_frames = data.get('small_start_frames', [])  # Get start frames or empty list
                        small_stop_frames = data.get('small_stop_frames', [])
                        small_x = data.get('small_x', [])
                        small_y = data.get('small_y', [])
                        jump_start_frames = data.get('jump_start_frames', [])
                        jump_stop_frames = data.get('jump_stop_frames', [])
                        displacements = data.get('displacements', [])

                        # Calculate jump_count
                        jump_count = len(jump_start_frames)  # Number of jumps detected

                        # Calculate average displacement
                        if displacements:
                            average_displacement = np.mean(displacements)
                        else:
                            average_displacement = 0

                        # Convert lists to comma-separated strings
                        start_frames_str = ",".join(map(str, small_start_frames))
                        stop_frames_str = ",".join(map(str, small_stop_frames))
                        small_x_str = ",".join(map(str, small_x))
                        small_y_str = ",".join(map(str, small_y))
                        jump_start_frames_str = ",".join(map(str, jump_start_frames))
                        jump_stop_frames_str = ",".join(map(str, jump_stop_frames))

                        # Write the data to CSV
                        writer.writerow([
                            video_filename,
                            small_count,
                            start_frames_str,
                            stop_frames_str,
                            small_x_str,
                            small_y_str,
                            jump_count,
                            jump_start_frames_str,
                            jump_stop_frames_str,
                            f"{average_displacement:.2f}"  # Include average displacement
                        ])

            except Exception as e:
                print(f"Error writing to CSV file {csv_filename}: {e}")

            print(f'Processing complete for folder: {seg_folder}')

    # Record the end time of the main function
    end_time = time.time()

    # Calculate and print the total execution time
    total_time = end_time - start_time
    print(f"Total execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    parent_directory = '/Users/tairan/Downloads/rnai_issue'
    main(parent_directory)
