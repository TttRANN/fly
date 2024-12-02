import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = '/Users/tairan/Desktop/test_ori.png'
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a binary threshold
ret, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
ret1, thresh1 = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# Find contours in the binary image
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours1, _ = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Lists to store centroids
centroids_thresh = []
centroids_thresh1 = []

# Fit an ellipse and find the centroid for each contour if the contour has more than 5 points
for contour in contours:
    if len(contour) >= 5:  # Need at least 5 points to fit an ellipse
        ellipse = cv2.fitEllipse(contour)
        cv2.ellipse(image, ellipse, (0, 255, 0), 2)  # Draw the ellipse on the original image

        # Calculate moments to find the centroid
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        centroids_thresh.append((cx, cy))

        # Draw the centroid on the image
        cv2.circle(image, (cx, cy), 5, (255, 0, 0), -1)  # Blue circle for centroid

for contour in contours1:
    if len(contour) >= 5:  # Need at least 5 points to fit an ellipse
        ellipse = cv2.fitEllipse(contour)
        cv2.ellipse(image, ellipse, (0, 255, 0), 2)  # Draw the ellipse on the original image

        # Calculate moments to find the centroid
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        centroids_thresh1.append((cx, cy))

        # Draw the centroid on the image
        cv2.circle(image, (cx, cy), 5, (200, 200, 0), -1)  # Light blue circle for centroid

# Draw vectors and arrows between corresponding centroids
for i in range(min(len(centroids_thresh), len(centroids_thresh1))):
    # Draw a red line for vector
    cv2.line(image, centroids_thresh[i], centroids_thresh1[i], (0, 0, 255), 2)
    start_point = np.array(centroids_thresh[i])
    end_point = np.array(centroids_thresh1[i])
    direction = end_point - start_point
    new_end_point = end_point + direction  # Extend by the vector's length

    cv2.arrowedLine(image, tuple(start_point), tuple(new_end_point), (200, 200, 0), 2)

    # Draw an arrow from blue to light blue
    # cv2.a rrowedLine(image, centroids_thresh[i]*2, centroids_thresh1[i]*2, (200, 200, 0), 30)  # Light blue arrow

# Display the image with fitted ellipses, centroids, vectors, and arrows
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Image with Fitted Ellipses, Centroids, Vectors, and Arrows')
plt.axis('off')
plt.show()

# Properly close all OpenCV windows (if any were opened)
cv2.destroyAllWindows()
