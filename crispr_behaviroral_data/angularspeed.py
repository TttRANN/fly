import pandas as pd
import matplotlib.pyplot as plt

def plot_speed_angular_speed(file_path):
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Extract the necessary columns
    time = df.iloc[:, 0].values
    x_pos = df.iloc[:, 3].values
    y_pos = df.iloc[:, 4].values
    angle = df.iloc[:, 5].values

    # Calculate the speed (Euclidean distance) between consecutive positions
    speed = ((x_pos[1:] - x_pos[:-1])**2 + (y_pos[1:] - y_pos[:-1])**2)**0.5
    # Calculate the angular speed between consecutive angles
    angular_speed = abs(angle[1:] - angle[:-1])

    # Plot the speed and angular speed
    plt.figure(figsize=(14, 7))

    plt.subplot(2, 1, 1)
    plt.plot(time[1:], speed, label='Speed')
    plt.xlabel('Time')
    plt.ylabel('Speed')
    plt.title('Speed over Time')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(time[1:], angular_speed, label='Angular Speed', color='orange')
    plt.xlabel('Time')
    plt.ylabel('Angular Speed')
    plt.title('Angular Speed over Time')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

# Example usage:
plot_speed_angular_speed('/Users/tairan/Downloads/test1.csv')