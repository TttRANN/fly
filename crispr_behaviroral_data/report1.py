import cv2
import numpy as np

def binarize_and_fit_ellipse(input_image_path, output_image_path, threshold=127):
    # Read the image
    image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print("Error: Could not open or find the image.")
        return

    # Binarize the image using the specified threshold
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a color image for drawing the ellipses
    output_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

    # Draw ellipses around contours
    for contour in contours:
        if len(contour) >= 5:  # At least 5 points are needed to fit an ellipse
            ellipse = cv2.fitEllipse(contour)
            cv2.ellipse(output_image, ellipse, (0, 255, 0), 20)  # Draw the ellipse in green
        else:
            print(1)

    # Save the output image with the ellipses drawn on it
    cv2.imwrite(output_image_path, output_image)
    print(f"Binarized image with ellipses saved as: {output_image_path}")




if __name__ == "__main__":
    input_image_path = '/Users/tairan/Desktop/binarized_image.PNG'  # Replace with your image path
    output_image_path = '/Users/tairan/Desktop/binarized_image1.PNG'  # Replace with desired output path
    binarize_and_fit_ellipse(input_image_path, output_image_path)
