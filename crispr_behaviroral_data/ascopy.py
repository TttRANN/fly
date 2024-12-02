
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import re
import os
from dtw import *
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

# Function to extract start and end time from segment name
def extract_time_from_segment(segment_name):
    match = re.search(r'segment_(\d+)_(\d+)\.mp4', segment_name)
    if match:
        start_time = int(match.group(1))
        end_time = int(match.group(2))
        return start_time, end_time
    else:
        raise ValueError(f"Invalid segment name format: {segment_name}")

# Function to save x and y positions based on peak-to-peak segments
def save_xy_trajectory(df, peaks, time, x_pos, y_pos, segment_name):
    trajectory_files = []
    for i in range(len(peaks) - 1):
        start_idx = peaks[i]
        end_idx = peaks[i + 1]

        segment_trajectory = pd.DataFrame({
            'Time': time[start_idx:end_idx],
            'X_Position': x_pos[start_idx:end_idx],
            'Y_Position': y_pos[start_idx:end_idx]
        })
        
        output_file = f'{segment_name}_trajectory_{i+1}.csv'
        segment_trajectory.to_csv(output_file, index=False)
        trajectory_files.append(output_file)
        print(f'Saved trajectory segment {i+1} to {output_file}')
    
    return trajectory_files

# Function to plot speed, angular speed, and fragment trajectory
def plot_and_fragment(df, start_time, end_time, segment_name):
    # Filter the DataFrame based on the start and end time
    filtered_df = df[(df.iloc[:, 0] >= start_time) & (df.iloc[:, 0] <= end_time)]

    # Extract the necessary columns
    time = filtered_df.iloc[:, 0].values
    x_pos = filtered_df.iloc[:, 3].values
    y_pos = filtered_df.iloc[:, 4].values
    angle = filtered_df.iloc[:, 5].values

    # Calculate the angular speed between consecutive angles
    angular_speed = abs(angle[1:] - angle[:-1])

    # Find peaks in the angular speed that are larger than 25
    peaks, _ = find_peaks(angular_speed, height=20)

    # Plot the angular speed with peaks
    plt.figure(figsize=(14, 7))
    plt.plot(time[1:], angular_speed, label='Angular Speed', color='orange')
    plt.plot(time[1:][peaks], angular_speed[peaks], "x", label='Peaks')  # Mark the peaks
    plt.xlabel('Time')
    plt.ylabel('Angular Speed')
    plt.title(f'Angular Speed over Time with Peaks > 25 for segment {start_time}_{end_time}')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Save the trajectory segments based on peak-to-peak segments
    return save_xy_trajectory(filtered_df, peaks, time[1:], x_pos[1:], y_pos[1:], segment_name)

# Function to process all segments
def process_segments(file_path, segment_names):
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Process each segment and collect trajectory files
    all_trajectory_files = []
    for segment in segment_names:
        start_time, end_time = extract_time_from_segment(segment)
        trajectory_files = plot_and_fragment(df, start_time, end_time, segment)
        all_trajectory_files.extend(trajectory_files)
    
    return all_trajectory_files
from scipy.spatial import procrustes

# Function to rotate trajectory to align with a reference trajectory
def rotate_trajectory(traj, reference_traj):
    _, traj_rotated, _ = procrustes(reference_traj, traj)
    return traj_rotated
# Function to align and rotate trajectories to minimize distance to a reference
def align_and_rotate_trajectories(trajectory_files):
    aligned_trajectories = []
    
    # Load the first trajectory as the reference
    reference_traj = pd.read_csv(trajectory_files[0])[['X_Position', 'Y_Position']].values
    
    for file in trajectory_files:
        traj = pd.read_csv(file)
        # Translate the trajectory to start at (0,0)
        x_start, y_start = traj.iloc[0]['X_Position'], traj.iloc[0]['Y_Position']
        traj['X_Position'] -= x_start
        traj['Y_Position'] -= y_start
        
        # Rotate the trajectory to minimize the distance to the reference
        traj_coords = traj[['X_Position', 'Y_Position']].values
        traj_rotated = rotate_trajectory(traj_coords, reference_traj)
        
        # Save the aligned and rotated trajectory
        traj['X_Position'] = traj_rotated[:, 0]
        traj['Y_Position'] = traj_rotated[:, 1]
        
        aligned_file = file.replace('.csv', '_aligned_rotated.csv')
        traj.to_csv(aligned_file, index=False)
        aligned_trajectories.append(aligned_file)
    
    return aligned_trajectories

# WSP-like Clustering Function (using DTW for simplicity)
def cluster_trajectories(trajectory_files):
    trajectories = []
    
    # Load all trajectory files
    for file in trajectory_files:
        traj = pd.read_csv(file)
        trajectories.append(traj[['X_Position', 'Y_Position']].values)

    # Pairwise DTW distances
    n = len(trajectories)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            try:
                dist = dtw(trajectories[i], trajectories[j]).distance
            except ValueError as e:
                print(f"Error processing trajectories {i} and {j}: {e}")
                dist = 5000  # Assign a large distance if DTW fails


            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    # Perform clustering (e.g., hierarchical clustering)
    Z = linkage(distance_matrix, 'average')
    clusters = fcluster(Z, t=5, criterion='maxclust')


    return clusters, distance_matrix, Z

# Function to visualize the clustered trajectories
def visualize_clusters(trajectory_files, clusters):
    num_clusters = len(set(clusters))
    
    for cluster_id in range(1, num_clusters + 1):
        plt.figure(figsize=(8, 6))
        for i, file in enumerate(trajectory_files):
            if clusters[i] == cluster_id:
                traj = pd.read_csv(file)
                plt.plot(traj['X_Position'], traj['Y_Position'], label=f'Trajectory {i+1}')
        
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title(f'Cluster {cluster_id} Trajectories')
        plt.grid(True)
        plt.legend()
        plt.show()
# Function to align trajectories to start at (0,0)
def align_trajectories_to_origin(trajectory_files):
    aligned_trajectories = []
    for file in trajectory_files:
        traj = pd.read_csv(file)
        x_start, y_start = traj.iloc[0]['X_Position'], traj.iloc[0]['Y_Position']
        traj['X_Position'] -= x_start
        traj['Y_Position'] -= y_start
        
        aligned_file = file.replace('.csv', '_aligned.csv')
        traj.to_csv(aligned_file, index=False)
        aligned_trajectories.append(aligned_file)
    
    return aligned_trajectories

# Function to plot dendrogram for visualizing the clustering process
def plot_dendrogram(Z, labels):
    plt.figure(figsize=(10, 7))
    dendrogram(Z, labels=labels, orientation='top', distance_sort='ascending', show_leaf_counts=True)
    plt.title('Dendrogram of Trajectory Clustering')
    plt.xlabel('Trajectory Index')
    plt.ylabel('Distance')
    plt.show()
# Example usage:
file_path = '/Users/tairan/Downloads/29c/rnai_BEAT-IV-29C_t4t5_batch1/rnai_BEAT-IV-29C_t4t5_batch1_filtered_2.csv'
segment_names = ['segment_180_420.mp4',
                'segment_660_900.mp4',
                #  'segment_35700_35940.mp4',
    'segment_4140_4380.mp4',
    'segment_14340_14580.mp4',
    'segment_15600_15840.mp4',
    'segment_21840_22080.mp4'
]

# Process the segments to extract and save trajectories
trajectory_files = process_segments(file_path, segment_names)

# Cluster the saved trajectories
clusters, distance_matrix, Z = cluster_trajectories(trajectory_files)

# Align the trajectories to the origin (0,0)
aligned_trajectory_files = align_trajectories_to_origin(trajectory_files)

# Visualize the aligned clusters
visualize_clusters(aligned_trajectory_files, clusters)

# Visualize the clustering dendrogram
plot_dendrogram(Z, labels=[f'Traj {i+1}' for i in range(len(aligned_trajectory_files))])
