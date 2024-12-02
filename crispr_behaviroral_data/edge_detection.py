



import cv2
import numpy as np

# Load the static background image
background = cv2.imread('/Users/tairan/Downloads/WTR.jpg')
background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

# Open the video stream
video = cv2.VideoCapture('/Users/tairan/Downloads/cas9_dscam3_t4t5_batch4_arena_0_resized.mp4')


while True:
    ret, frame = video.read()
    if not ret:
        break
    
    # Convert the current frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize the current frame to match the background image dimensions
    frame_gray = cv2.resize(frame_gray, (background_gray.shape[1], background_gray.shape[0]))

    # Compute the absolute difference between the current frame and the background image
    background_diff = cv2.absdiff(frame_gray, background_gray)
    
    # Threshold the difference image
    _, thresholded = cv2.threshold(background_diff, 25, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
    thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
    
    # Find contours of the moving objects
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw the contours on the original frame
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # filter out small contours
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Display the results
    cv2.imshow('Frame', frame)
    cv2.imshow('Background Difference', thresholded)
    
    # Exit on 'q' key press
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release resources
video.release()
cv2.destroyAllWindows()
