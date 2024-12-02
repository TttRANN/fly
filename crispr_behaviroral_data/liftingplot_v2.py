import os
import cv2  # OpenCV for video reading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Define the directory where the CSV files and the video file are located
root_directory = '/Users/tairan/Downloads/29c_results_v2_c1'  # Replace with the correct path
video_path = '/Users/tairan/Downloads/output_segment_2820_3060_fly_tracking.mp4'  # Replace with the correct path to the video

# Read the video to get width and height
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error opening video file")
else:
    # Get width and height from the video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video Width: {width}, Video Height: {height}")

# Function to parse small_x and small_y columns from CSV
def parse_coordinates(column):
    coords = []
    for coord_list in column:
        if isinstance(coord_list, str) and coord_list.strip():  # Check if the value is a non-empty string
            try:
                # Convert string representation of list to actual list of floats
                coords.extend([float(coord.strip()) for coord in coord_list.strip('[]').split(',') if coord.strip()])
            except ValueError:
                print(f"Warning: Could not convert value '{coord_list}' to float. Skipping.")
        else:
            print(f"Warning: Found empty or invalid data in column: {coord_list}. Skipping.")
    return coords

# Dictionary to group coordinates by condition
grouped_coordinates = defaultdict(lambda: {'x': [], 'y': []})

# Walk through all CSV files
for subdir, _, files in os.walk(root_directory):
    for file in files:
        if file.endswith('.csv'):
            file_path = os.path.join(subdir, file)
            
            # Load the CSV file
            df = pd.read_csv(file_path)
            
            # Parse the 'small_x' and 'small_y' columns
            small_x = parse_coordinates(df['small_x'])
            small_y = parse_coordinates(df['small_y'])
            
            # Skip if no valid coordinates
            if not small_x or not small_y:
                print(f"Skipping file: {file} (no valid coordinates)")
                continue
            
            # Extract condition information from the filename
            parts = file.split('_')
            condition = '_'.join(parts[1:3])  # e.g., 'gilt1-29C'
            
            # Append the coordinates to the corresponding group
            grouped_coordinates[condition]['x'].extend(small_x)
            grouped_coordinates[condition]['y'].extend(small_y)

# Function to draw a circle and plot density with dots on it
def plot_circle_with_density_and_dots(condition, small_x, small_y, width, height):
    plt.figure(figsize=(7, 7))

    # Calculate the radius of the circle based on video width
    radius = width // 2

    # Set an aspect ratio of 1 to ensure the circle is not distorted
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='datalim')

    # Use a pastel color palette for the plot
    palette = sns.color_palette("coolwarm", as_cmap=False)

    # Plot the circle centered at (width/2, height/2)
    circle = plt.Circle((width / 2, height / 2), radius, color=palette[0], fill=False, linewidth=3, linestyle='--')
    
    # Add circle to the plot
    ax.add_patch(circle)

    # Plot the density of small_x and small_y coordinates using kdeplot
    sns.kdeplot(x=small_x, y=small_y, cmap="Blues", fill=True, thresh=0.01, levels=5, alpha=0.8)

    # Overlay the dots on top of the density plot
    plt.scatter(small_x, small_y, color=palette[4], edgecolor='k', alpha=0.7, s=100, zorder=5)

    # Set grid and limits based on video dimensions
    plt.xlim(0, width)
    plt.ylim(0, height)
    plt.grid(True, linestyle=':', color='gray', linewidth=1, alpha=0.5)  # Soft grid lines for better readability

    # Add labels and title with larger font size
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)
    plt.title(f'Condition: {condition} - Density with Dots on Circle', fontsize=16, fontweight='bold')
    
    # Add a light background to the plot for better contrast
    plt.gca().set_facecolor('whitesmoke')

    # Show the plot
    plt.show()

# Now loop through the grouped coordinates and plot the density with dots for each condition
for condition, coordinates in grouped_coordinates.items():
    if coordinates['x'] and coordinates['y']:
        plot_circle_with_density_and_dots(condition, coordinates['x'], coordinates['y'], width, height)
