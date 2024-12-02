import numpy as np
from dtw import *
import matplotlib.pyplot as plt

# Example 1D Trajectories with noise
trajectory1 = np.linspace(0, 10, 100)
trajectory2 = np.linspace(0, 10, 100) + np.random.normal(0, 0.5, 100)

# Compute DTW
alignment = dtw(trajectory1, trajectory2, keep_internals=True)

# Plot the alignment
alignment.plot(type="threeway")

# Show the plot
plt.show()

