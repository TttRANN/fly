import cv2
import numpy as np

def draw_orientation(frame, centroid, orientation, length=50, color=(0, 0, 255), thickness=2):
    """
    Draws an arrow on the frame to represent the fly's orientation.

    Parameters:
    - frame: The image frame on which to draw.
    - centroid: A tuple (x, y) representing the centroid of the fly.
    - orientation: The angle of the major axis of the ellipse, in degrees.
    - length: The length of the arrow to draw (default is 50 pixels).
    - color: The color of the arrow in BGR format (default is red).
    - thickness: The thickness of the arrow line (default is 2).
    """
    # Convert orientation angle from degrees to radians
    angle_rad = np.deg2rad(orientation)
    
    # Calculate the endpoint of the arrow based on the orientation angle
    x2 = int(centroid[0] + length * np.cos(angle_rad))
    y2 = int(centroid[1] + length * np.sin(angle_rad))
    
    # Draw the arrowed line from the centroid to the calculated endpoint
    cv2.arrowedLine(frame, centroid, (x2, y2), color, thickness)

# Example usage:
# Assuming `frame` is the image where the fly is detected, and you have the centroid and orientation.

# Sample data (replace with real values):
frame = np.ones((400, 400, 3), dtype=np.uint8) * 255  # A white frame
centroid = (200, 200)
orientation = 45  # Angle in degrees

# Draw the orientation on the frame
draw_orientation(frame, centroid, orientation)

# Show the frame (for testing purposes)
cv2.imshow("Fly Orientation", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
