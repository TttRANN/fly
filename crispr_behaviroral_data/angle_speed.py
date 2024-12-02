import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from CSV
data = pd.read_csv('/Users/tairan/Downloads/29c/rnai_BEAT-IV-29C_t4t5_batch1/rnai_BEAT-IV-29C_t4t5_batch1_filtered_2.csv')

# Assuming the sixth column contains the angle data, adjust the index if needed
angle = data.iloc[200:500, 5].values  # Adjust index if necessary


dt = 1  # Define time step, adjust if needed

# Calculate angular velocity (diff of angle / time step)
angular_velocity = np.diff(angle) / dt

# Create time array for plotting
time = np.arange(len(angle) - 1) * dt  # Time array for the angular velocity

# Plotting
plt.figure(figsize=(12, 6))

# Plot the instantaneous angle
plt.subplot(2, 1, 1)
plt.plot(angle, label='Instantaneous Angle')
plt.xlabel('Sample Index')
plt.ylabel('Angle (degrees)')
plt.title('Instantaneous Angle')
plt.legend()

# Plot the angular velocity
plt.subplot(2, 1, 2)
plt.plot(time, angular_velocity, label='Angular Velocity', color='r')
plt.xlabel('Sample Index')
plt.ylabel('Angular Velocity (degrees/second)')
plt.title('Angular Velocity')
plt.legend()

plt.tight_layout()
plt.show()
