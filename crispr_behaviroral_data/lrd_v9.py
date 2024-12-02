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
    # inverted_mask = cv2.bitwise_not(mask)
    masked_frame = cv2.bitwise_and(frame, mask)
    # masked_frame += inverted_mask
    return masked_frame

def process_video(input_video_path, output_animation_path, fps):
    """Process a video to detect contours, fit an ellipse, and generate an animation of contour areas over time."""
    start_time = time.time()

    # Open video capture
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {input_video_path}")
        return [], [], [], [], []

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames in the video
    background_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    background_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    center = (background_width // 2, background_height // 2)
    radius = int(min(background_width, background_height) * 0.45)

    # Initialize lists and variables
    contour_areas_over_time = []
    small_list = []
    small = 0
    consecutive_small_frames = 0
    contour_areas = []

    small_start_frames = []
    small_stop_frames = []
    small_centroid_x = []
    small_centroid_y = []
    small_condition_active = False

    frame_count = 0  # Start from frame 0
    while frame_count < total_frames:  # Ensure we process all frames up to total_frames
        ret, frame = cap.read()
        if not ret:
            print("End of video reached unexpectedly.")
            break  # End of video, stop processing

        # Apply circular mask and preprocess the frame
        masked_frame = apply_circular_mask(frame, radius, center)
        gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 23, 3)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Filter valid contours by area
        valid_contours = [c for c in contours if 50 < cv2.contourArea(c) < 400]

        if valid_contours:
            biggest_contour = max(valid_contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(biggest_contour)
            contour_areas.append(contour_area)
            contour_areas_over_time.append(contour_area)
            avg_largest_contour_area = np.mean(contour_areas)

            if len(biggest_contour) < 5:
                print('Contour has fewer than 5 points')
                small_list.append('NaN')
                break  # Stop processing once NaN is appended

            # Fit an ellipse and check its properties
            ellipse = cv2.fitEllipse(biggest_contour)
            cv2.ellipse(masked_frame, ellipse, (0, 255, 0), 2)  # Draw the ellipse

            centroid = (int(ellipse[0][0]), int(ellipse[0][1]))
            distance_from_center = np.sqrt((centroid[0] - center[0]) ** 2 + (centroid[1] - center[1]) ** 2)

            if distance_from_center > radius:
                print('Centroid outside of the radius')
                small_list.append('NaN')
                break  # Stop processing once NaN is appended
            else:
                # Check for small condition based on contour area
                if contour_area < avg_largest_contour_area / 1.3:
                    consecutive_small_frames += 1
                    contour_areas.pop(-1)  # Remove last appended contour
                    avg_largest_contour_area = np.mean(contour_areas)
                    contour_areas.append(avg_largest_contour_area)

                    # Start small condition
                    if consecutive_small_frames > 1 and not small_condition_active:
                        small_start_frames.append(frame_count - 1)
                        small_condition_active = True
                        print(f"small_condition_active set to True at frame {frame_count}")
                else:
                    # Before setting small_condition_active to False, check if conditions are met
                    if small_condition_active and consecutive_small_frames >= 5:
                        print(f"Triggered small condition logic at frame {frame_count}")
                        small += 1
                        small_list.append(small)
                        small_centroid_x.append(int(ellipse[0][0]))
                        small_centroid_y.append(int(ellipse[0][1]))
                    else:
                        small_list.append(small)

                    # Now set small_condition_active to False and reset frames
                    if small_condition_active:
                        small_stop_frames.append(frame_count)
                        small_condition_active = False
                        print(f"small_condition_active set to False at frame {frame_count}")

                    consecutive_small_frames = 0  # Reset if condition not met

        else:
            contour_areas_over_time.append(0)
            small_list.append('NaN')
            break  # Stop processing once NaN is appended

        frame_count += 1  # Increment frame count for every processed frame

        # Display the frame with the fitted ellipse
        # cv2.imshow("Ellipse Fitting", masked_frame)

        # # Break if 'q' is pressed
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # If the video ends but the small condition is still active, append the last frame
    if small_condition_active:
        small_stop_frames.append(total_frames - 1)  # Append the last frame index
        print(f"small_condition_active ended at the last frame: {total_frames - 1}")

    # Release resources and close windows
    cap.release()
    # cv2.destroyAllWindows()

    # Print results
    print(f"Processing completed for video: {input_video_path} in {time.time() - start_time} seconds.")
    print(f"Small condition start frames: {small_start_frames}")
    print(f"Small condition stop frames: {small_stop_frames}")
    print(f"Small condition count: {small}")
    print(small_list)

    # Create the animation (assuming you have a create_animation function implemented)
    # create_animation(contour_areas_over_time, output_animation_path, fps, small_start_frames, small_stop_frames)

    return small_list, small_start_frames, small_stop_frames, small_centroid_x, small_centroid_y



def create_animation(contour_areas_over_time, output_animation_path, fps, small_start_frames, small_stop_frames):
    """Create an animation for the contour areas over time with shaded areas for the 'small' condition."""
    frame_count = len(contour_areas_over_time)
    
    fig, ax = plt.subplots()
    ax.set_xlim(0, frame_count)
    ax.set_ylim(0, max(contour_areas_over_time) if contour_areas_over_time else 10000)
    
    line, = ax.plot([], [], lw=2, label='Contour Area')
    avg_line, = ax.plot([], [], lw=2, label='Average Contour Area', color='r')

    # Shade the areas where the small condition is satisfied for at least 5 consecutive frames
    for start, stop in zip(small_start_frames, small_stop_frames):
        if stop - start >= 5:  # Only shade if at least 5 consecutive frames meet the small condition
            ax.axvspan(start, stop, facecolor='gray', alpha=0.3, label="Small Condition Active")

    def init():
        line.set_data([], [])
        avg_line.set_data([], [])
        return line, avg_line

    def update(frame):
        xdata = list(range(frame + 1))
        ydata = contour_areas_over_time[:frame + 1]
        avg_ydata = [np.mean(contour_areas_over_time[:i+1]) for i in range(frame + 1)]
        line.set_data(xdata, ydata)
        avg_line.set_data(xdata, avg_ydata)
        return line, avg_line

    ani = animation.FuncAnimation(fig, update, frames=frame_count, init_func=init, blit=True, repeat=False)
    ani.save(output_animation_path, writer=animation.FFMpegWriter(fps=fps))
    plt.close(fig)

def combine_videos(input_video_path, animation_video_path, output_combined_video_path, mask_radius, center, small_list):
    """Combine the original video with contour animation video side by side and overlay the 'small' value for each frame."""
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

        # Apply the circular mask on the first video
        masked_frame = apply_circular_mask(frame1, mask_radius, center)

        # Resize the animation frame to match the video dimensions
        frame2 = cv2.resize(frame2, (width, height))

        # Combine the two frames side by side
        combined_frame = np.hstack((masked_frame, frame2))

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
            mask_radius = int((width // 2) )
        cap.release()

        # Run the process_video function and retrieve small_list, start, and stop frames
        small_list, small_start_frames, small_stop_frames, small_x, small_y = process_video(input_video_path, output_animation_path, fps)

        # Combine videos with small_list
        # combine_videos(input_video_path, output_animation_path, output_combined_video_path, mask_radius, center, small_list)

        # Create a structured dictionary entry for this video
        small_contour_counts[video_filename] = {
            'small_count': small_list,  # List of small count over frames
            'small_start_frames': small_start_frames,  # List of start frames
            'small_stop_frames': small_stop_frames, 
             'small_x': small_x,
              'small_y': small_y # List of stop frames
        }

    return small_contour_counts





import os
import time
import re
import csv

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
                       if os.path.isdir(os.path.join(rnai_folder_path, f)) and re.match(r'seg\d+_filtered_\d+_arena_\d+_l2sec', f)]

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
                    writer.writerow(['Video Filename', 'Small Count', 'Start Frames', 'Stop Frames','small_x','small_y'])
                    
                    for video_filename, small_data in small_contour_counts.items():
                        # Assuming small_data contains small_count, start and stop frames, etc.
                        small_count = small_data.get('small_count', ['NaN'])[-1]  # Get last small_count or 'NaN'
                        small_start_frames = small_data.get('small_start_frames', [])  # Get start frames or empty list
                        small_stop_frames = small_data.get('small_stop_frames', []) 
                        small_x = small_data.get('small_x',[]) 
                        small_y =small_data.get('small_y',[])  # Get stop frames or empty list

                        # Convert lists to comma-separated values for easy readability in CSV
                        start_frames_str = ",".join(map(str, small_start_frames))  # e.g., "100,200"
                        stop_frames_str = ",".join(map(str, small_stop_frames))    # e.g., "150,250"

                        # Write small_count, and start/stop frames as a comma-separated string
                        writer.writerow([video_filename, small_count, start_frames_str, stop_frames_str,small_x,small_y])

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



