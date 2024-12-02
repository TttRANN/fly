import cv2
import numpy as np

# Load image, grayscale, Gaussian blur, Adaptive threshold
image = cv2.imread('/Users/tairan/Desktop/Screenshot 2024-09-03 at 16.26.49.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (9,9), 0)
thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,23,3)

# Find contours
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

# Draw contours and fit ellipses
for c in cnts:
    area = cv2.contourArea(c)
    if area > 10:
        cv2.drawContours(image, [c], -1, (36,255,12), 1)
        # Fit an ellipse to the contour
        if len(c) >= 5:  # fitEllipse requires at least 5 points
            ellipse = cv2.fitEllipse(c)
            cv2.ellipse(image, ellipse, (0,255,255), 2)  # Draw the ellipse

# Display the result
cv2.imshow('thresh', thresh)
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
