import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the binary image
image_path = '/Users/tairan/Downloads/cbd.jpg'
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the binary image
# image_path = '/Users/tairan/Downloads/abc.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Find contours
contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Assume the largest contour is the fly
if contours:
    fly_contour = max(contours, key=cv2.contourArea)
    
    # Calculate moments for the centroid
    M = cv2.moments(fly_contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    # Initialize variables to store head and tail points
    tail_point = None
    max_distance = 0
    
    for point in fly_contour:
        point = point[0]
        distance = np.sqrt((point[0] - cX)**2 + (point[1] - cY)**2)
        if distance > max_distance:
            max_distance = distance
            tail_point = tuple(point)
    
    # Calculate the head point as the point 180 degrees opposite to the tail
    if tail_point:
        head_point = (2*cX - tail_point[0], 2*cY - tail_point[1])
        
        # Determine the orientation of the head
        head_orientation = np.arctan2(head_point[1] - cY, head_point[0] - cX) * 180 / np.pi
        
        # Draw head and tail points
        output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.circle(output_image, tail_point, 5, (255, 0, 0), -1)  # Tail in blue
        cv2.circle(output_image, head_point, 5, (0, 0, 255), -1)  # Head in red
        cv2.line(output_image, tail_point, head_point, (0, 255, 0), 2)  # Direction in green
        cv2.circle(output_image, (cX, cY), 5, (0, 255, 255), -1)  # Centroid in yellow
        
        # Show the positions and orientation
        print(f"Centroid: ({cX}, {cY})")
        print(f"Head Position: {head_point}")
        print(f"Tail Position: {tail_point}")
        print(f"Head Orientation: {head_orientation} degrees")
        
        # Display the output image
        plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
        plt.show()
    else:
        print("Tail point not found.")
else:
    print("No contours found.")
