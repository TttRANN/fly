import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Load the CSV file
file_path = '/Users/tairan/Documents/ista/crispr_behaviroral_data/CRISPR_T4T5_screen/cas9_dscam3_t4t5_batch1.csv'  # Adjust the path if needed
column_names = ['Var1', 'Var2', 'Var3', 'Var4', 'Var5', 'Var6', 'Var7', 'Var8', 'Var9', 'Var10', 'Var11', 'Var12']

data = pd.read_csv(file_path, names=column_names)

# Check if data is loaded correctly
print("Data loaded successfully. Shape:", data.shape)
print(data.head())

# Remove rows where 'Var7' is 'off' or 'Var3' is 0, 1, or 3
data = data[(data['Var7'] != 'off') & (~data['Var3'].isin([0, 1, 3]))]

# Check if data is filtered correctly
print("Data after filtering. Shape:", data.shape)
print(data.head())

# Initialize a structure to store the data
data_struct = {}
unique_trials = data['Var10'].unique()

# Function to calculate velocities
def calculate_velocities(x, y, theta, t):
    translational_vel = (np.diff(x) * np.cos(theta[:-1]) + np.diff(y) * np.sin(theta[:-1])) / np.diff(t)
    side_slip_vel = (-np.diff(x) * np.sin(theta[:-1]) + np.diff(y) * np.cos(theta[:-1])) / np.diff(t)
    angular_vel = np.diff(theta) / np.diff(t)
    return translational_vel, side_slip_vel, angular_vel

# Loop through each unique value in data.Var10
for trial_id in unique_trials:
    trial_data = data[data['Var10'] == trial_id]
    nan_count = trial_data.isna().sum().sum()
    if nan_count <= 5:
        data_struct[f'Trial{trial_id}'] = trial_data

# Initialize matrices to store velocities
velocities_matrix = []

# Loop through each trial in data_struct to calculate velocities
for trial_id in unique_trials:
    field_name = f'Trial{trial_id}'
    if field_name in data_struct:
        trial_data = data_struct[field_name]
        x = trial_data['Var4'].values
        y = trial_data['Var5'].values
        theta = trial_data['Var6'].values
        frames = np.arange(len(x))  # Create frame vector based on the index of data points
        
        # Calculate velocities
        translational_vel, side_slip_vel, angular_vel = calculate_velocities(x, y, theta, frames)
        
        # Normalize the velocities
        translational_vel /= np.max(np.abs(translational_vel))
        side_slip_vel /= np.max(np.abs(side_slip_vel))
        angular_vel /= np.max(np.abs(angular_vel))
        
        # Append trial velocities to the matrix
        velocities_matrix.append(np.column_stack((translational_vel, side_slip_vel, angular_vel)))

# Check if velocities_matrix has data
if len(velocities_matrix) == 0:
    print("No data to concatenate. Check filtering conditions and data extraction steps.")
else:
    # Concatenate all velocities into a single matrix
    velocities_matrix = np.vstack(velocities_matrix)

    # Extract the velocities for the contour plot
    translational_velocity = velocities_matrix[:, 0]
    angular_velocity = velocities_matrix[:, 2]

    # Perform kernel density estimation for the contour plot
    kde = gaussian_kde([angular_velocity, translational_velocity])

    # Create a grid for the contour plot
    x_min, x_max = angular_velocity.min(), angular_velocity.max()
    y_min, y_max = translational_velocity.min(), translational_velocity.max()
    x, y = np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x, y)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

    # Calculate the confidence interval contour level
    confidence_level = 0.68
    total = Z.sum()
    cumulative_sum = np.cumsum(Z.ravel()) / total
    cumulative_sum = cumulative_sum.reshape(Z.shape)
    confidence_contour = np.max(Z[cumulative_sum <= confidence_level])

    # Plot the contour with the 68% confidence interval
    plt.figure(figsize=(10, 8))
    plt.contour(X, Y, Z, levels=[confidence_contour], colors='r')
    plt.scatter(angular_velocity, translational_velocity, s=10, alpha=0.5)
    plt.xlabel('Normalized Angular Velocity')
    plt.ylabel('Normalized Translational Velocity')
    plt.title('2D Contour Plot with 68% Confidence Interval')
    plt.colorbar(label='Density')
    plt.show()
