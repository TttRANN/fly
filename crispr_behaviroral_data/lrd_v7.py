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
    inverted_mask = cv2.bitwise_not(mask)
    masked_frame = cv2.bitwise_and(frame, mask)
    masked_frame += inverted_mask
    return masked_frame

def process_video(input_video_path, output_animation_path, fps):
    """Process a video to detect contours and generate an animation of contour areas over time."""
    start_time = time.time()

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {input_video_path}")
        return "NaN"

    frame_count = 0
    background_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    background_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    center = (background_width // 2, background_height // 2)
    radius = int(min(background_width, background_height) * 0.45)

    contour_areas_over_time = []
    small_list = []  
    small = 0
    consecutive_small_frames = 0  # Counter for consecutive frames meeting the small condition
    last_small_contour_frame = -30

    contour_areas = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        masked_frame = apply_circular_mask(frame, radius, center)
        gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 23, 3)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        valid_contours = [c for c in contours if 10 < cv2.contourArea(c) < 10000]

        if valid_contours:
            biggest_contour = max(valid_contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(biggest_contour)
            contour_areas.append(contour_area)
            contour_areas_over_time.append(contour_area)
            avg_largest_contour_area = np.mean(contour_areas)

            # Fit an ellipse to the biggest contour
            ellipse = cv2.fitEllipse(biggest_contour)
            centroid = (int(ellipse[0][0]), int(ellipse[0][1]))

            # Check if the centroid is inside the mask
            distance_from_center = np.sqrt((centroid[0] - center[0]) ** 2 + (centroid[1] - center[1]) ** 2)
            if distance_from_center > 0.9 * radius:
                # If the centroid is outside the mask, mark it as invalid
                small_list.append('NaN')
                small = 'NaN'
                break

            # Check if the contour area is below the threshold for small condition
            if contour_area < avg_largest_contour_area / 1.3:
                consecutive_small_frames += 1
            else:
                consecutive_small_frames = 0  # Reset counter if the condition is not met

            # If there are 10 consecutive small frames, update small
            if consecutive_small_frames >= 5 and frame_count - last_small_contour_frame >= 30:
                last_small_contour_frame = frame_count
                small += 1  # Increment small when 10 consecutive frames meet the condition
                small_list.append(small)
                consecutive_small_frames = 0  # Reset the counter after updating small
            else:
                small_list.append(small)

        else:
            contour_areas_over_time.append(0)
            small_list.append('NaN')  # Append 'NaN' if no valid contours
            small = 'NaN'
            break

        frame_count += 1

    cap.release()
    print(f"Processing completed for video: {input_video_path} in {time.time() - start_time} seconds.")

    create_animation(contour_areas_over_time, output_animation_path, fps)
    return small_list



def create_animation(contour_areas_over_time, output_animation_path, fps):
    """Create an animation for the contour areas over time."""
    frame_count = len(contour_areas_over_time)
    
    fig, ax = plt.subplots()
    ax.set_xlim(0, frame_count)
    ax.set_ylim(0, max(contour_areas_over_time) if contour_areas_over_time else 10000)
    
    line, = ax.plot([], [], lw=2, label='Contour Area')
    avg_line, = ax.plot([], [], lw=2, label='Average Contour Area', color='r')

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

    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Could not open one or both video files.")
        return

    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap1.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_combined_video_path, fourcc, fps, (width * 2, height))

    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (30, 50)  # Position to display the small value on the video
    fontScale = 1
    color = (255, 255, 0)  # Yellow color for text
    thickness = 2

    frame_idx = 0  # To keep track of the current frame

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2 or frame_idx >= len(small_list):
            break

        # Apply the circular mask on the first video
        masked_frame = apply_circular_mask(frame1, mask_radius, center)

        # Resize the animation frame to match the video dimensions
        frame2 = cv2.resize(frame2, (width, height))

        # Combine the two frames side by side
        combined_frame = np.hstack((masked_frame, frame2))

        # Overlay the 'small' value for the current frame from small_list
        small_value = small_list[frame_idx]
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
    videos_with_small_contours = 0
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
            mask_radius = int((width // 2) * 0.9)
        cap.release()

        small_contour_count = process_video(input_video_path, output_animation_path, fps)
        combine_videos(input_video_path, output_animation_path, output_combined_video_path, mask_radius, center,small_contour_count)

        small_contour_counts[video_filename] = small_contour_count

    return small_contour_counts


import time

def main(parent_directory):
    """Main function to process each RNAi folder and segment folder."""
    
    # Record the start time of the main function
    start_time = time.time()

    for rnai_folder in os.listdir(parent_directory):
        rnai_folder_path = os.path.join(parent_directory, rnai_folder)
        if not os.path.isdir(rnai_folder_path):
            continue

        seg_folders = [f for f in os.listdir(rnai_folder_path)
                       if os.path.isdir(os.path.join(rnai_folder_path, f)) and re.match(r'1seg\d+_filtered_\d+_arena_\d+', f)]

        for seg_folder in seg_folders:
            seg_number_match = re.search(r'seg(\d+)_filtered_(\d+)_arena_(\d+)', seg_folder)
            if not seg_number_match:
                print(f"Skipping folder {seg_folder}, pattern not matched.")
                continue

            csv_filename = os.path.join(rnai_folder_path, f'results_{rnai_folder}_seg{seg_number_match.group(1)}_arena{seg_number_match.group(3)}.csv')
            small_contour_counts = process_folder(parent_directory, rnai_folder, seg_folder)

            with open(csv_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                for video_filename, small_count in small_contour_counts.items():
                    writer.writerow([f'processed_{video_filename}', small_count[-1]])

            print(f'Processing complete for folder: {seg_folder}')

    # Record the end time of the main function
    end_time = time.time()

    # Calculate and print the total execution time
    total_time = end_time - start_time
    print(f"Total execution time: {total_time:.2f} seconds")



if __name__ == "__main__":
    parent_directory = '/Users/tairan/Downloads/test111'


    main(parent_directory)



