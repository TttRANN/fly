import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the directory where the CSV files and the video file are located
root_directory = '/Users/tairan/Downloads/rnai_issue/jump1'

# List to store "Accumulated Angle" from each CSV file
accumulated_angles = []

# Iterate through the files in the root directory
for root, dirs, files in os.walk(root_directory):
    for file in files:
        if file.endswith('csv'):
            # Read each CSV file
            airbnb_data = pd.read_csv(os.path.join(root, file))
            plt.plot(airbnb_data["Angular Speed"],color='gray')
            accumulated_angles.append(airbnb_data["Angular Speed"])

# Combine all "Accumulated Angle" columns into a DataFrame, aligning by index
# This will handle cases where the CSV files have different lengths by filling with NaN
accumulated_df = pd.DataFrame(accumulated_angles).T

# Compute the element-wise mean, ignoring NaN values
mean_accumulated_angle = accumulated_df.mean(axis=1)

# Plot the mean curve in grey
plt.plot(mean_accumulated_angle, color='red', label='Average Accumulated Angle')

# Add labels and a title
plt.xlabel('Frame')
plt.ylabel('Accumulated Angle')
plt.title('Element-wise Average of Accumulated Angle')
plt.legend()

# Display the plot
plt.show()


                        
                        
        

        