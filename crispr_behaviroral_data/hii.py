import cv2

# Open a video file or connect to a camera (use 0 for default camera)
cap = cv2.VideoCapture('/Users/tairan/Downloads/rnai_gilt1-batch3/seg3/segment_7200_7440.mp4')  # Replace with 0 for webcam

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Reached the end of the video or failed to read the frame.")
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the grayscale frame
    cv2.imshow('Grayscale Video', gray_frame)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
