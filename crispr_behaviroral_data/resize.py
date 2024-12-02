import cv2
import sys

def reshape_video(input_file, output_file, target_width=None, target_height=None, maintain_aspect_ratio=True):
    try:
        # Open the input video
        cap = cv2.VideoCapture(input_file)

        if not cap.isOpened():
            print(f"Error: Could not open video file {input_file}")
            return

        # Get original dimensions
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if target_width is None:
            target_width = original_width
        if target_height is None:
            target_height = original_height

        if maintain_aspect_ratio:
            # Calculate aspect ratio
            aspect_ratio = original_width / original_height
            if target_width / aspect_ratio <= target_height:
                new_width = target_width
                new_height = int(target_width / aspect_ratio)
            else:
                new_height = target_height
                new_width = int(target_height * aspect_ratio)
        else:
            new_width = target_width
            new_height = target_height

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For .mp4 files
        out = cv2.VideoWriter(output_file, fourcc, fps, (new_width, new_height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize the frame
            resized_frame = cv2.resize(frame, (new_width, new_height))

            # Write the resized frame
            out.write(resized_frame)

        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print(f"Video reshaped successfully! Saved as: {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python reshape_video_opencv.py input_file output_file [target_width] [target_height] [maintain_aspect_ratio]")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        target_width = int(sys.argv[3]) if len(sys.argv) > 3 else None
        target_height = int(sys.argv[4]) if len(sys.argv) > 4 else None
        maintain_aspect_ratio = bool(int(sys.argv[5])) if len(sys.argv) > 5 else True

        reshape_video(input_file, output_file, target_width, target_height, maintain_aspect_ratio)
