import numpy as np
import os
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Helper functions
def apply_circular_mask(frame, radius, center):
    """Apply a circular mask to a frame."""
    mask = np.zeros_like(frame)
    cv2.circle(mask, center, radius, (255, 255, 255), thickness=-1)
    masked_frame = cv2.bitwise_and(frame, mask)
    return masked_frame

def calculate_curvature(centroids):
    curvatures = []
    for i in range(1, len(centroids) - 1):
        p1 = np.array(centroids[i-1])
        p2 = np.array(centroids[i])
        p3 = np.array(centroids[i+1])
        
        v1 = p1 - p2
        v2 = p3 - p2
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            curvatures.append(0)
            continue
        
        v1_norm = v1 / norm_v1
        v2_norm = v2 / norm_v2
        
        dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        angle = np.arccos(dot_product)
        distance = np.linalg.norm(p3 - p1)
        
        if distance > 0:
            curvature = angle / distance
        else:
            curvature = 0
        
        curvatures.append(curvature)
    
    return curvatures

def calculate_speed(centroids, fps):
    speeds = []
    for i in range(1, len(centroids)):
        p1 = np.array(centroids[i-1])
        p2 = np.array(centroids[i])
        distance = np.linalg.norm(p2 - p1)
        speed = distance * fps  # Speed in pixels per second
        speeds.append(speed)
    return speeds

def calculate_angular_speed(orientations, fps):
    angular_speeds = []
    for i in range(1, len(orientations)):
        delta_angle = np.abs(orientations[i] - orientations[i-1])
        if delta_angle > 180:  # Correct for crossing the 0-360 boundary
            delta_angle = 360 - delta_angle
        angular_speed = delta_angle * fps  # Angular speed in degrees per second
        angular_speeds.append(angular_speed)
    return angular_speeds

def track_fly_and_calculate_features(input_video_path, output_video_path=None):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {input_video_path}")
        return None

    background_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    background_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    center = (background_width // 2, background_height // 2)
    radius = int(min(background_width, background_height) * 0.45)

    contour_areas = []
    centroids = []
    orientations = []

    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (background_width, background_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        masked_frame = apply_circular_mask(frame, radius, center)
        gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
        
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        valid_contours = [c for c in contours if 50 < cv2.contourArea(c) < 10000]

        if valid_contours:
            biggest_contour = max(valid_contours, key=cv2.contourArea)
            ellipse = cv2.fitEllipse(biggest_contour)
            contour_areas.append(cv2.contourArea(biggest_contour))
            centroids.append((int(ellipse[0][0]), int(ellipse[0][1])))
            orientations.append(ellipse[2])

            if output_video_path:
                cv2.ellipse(frame, ellipse, (25, 25, 0), 1)
                out.write(frame)

    cap.release()
    if output_video_path:
        out.release()

    if len(centroids) > 2:
        curvatures = calculate_curvature(centroids)
        speeds = calculate_speed(centroids, fps)
        angular_speeds = calculate_angular_speed(orientations, fps)
        return curvatures, speeds, angular_speeds, contour_areas
    else:
        return None, None, None, None

def extract_features_from_video(video_path):
    curvatures, speeds, angular_speeds, contour_area = track_fly_and_calculate_features(video_path, None)
    
    if curvatures is not None and len(curvatures) > 0:
        mean_curvature = np.mean(curvatures)
        max_curvature = np.max(curvatures)
        min_curvature =np.min(curvatures)
        diff_curvatures=max_curvature-min_curvature
        # print(diff_curvatures)
        std_curvature = np.std(curvatures)
        mean_speed = np.mean(speeds)
        min_speed = np.min(speeds)
        max_speed = np.max(speeds)
        std_speed = np.std(speeds)
        mean_angular_speed = np.mean(angular_speeds)
    
        std_angular_speed = np.std(angular_speeds)
        mean_contour_area = np.mean(contour_area)
        std_contour_area = np.std(contour_area)
        max_contour_area=np.max(contour_area)
        min_contour_area=np.min(contour_area)
        median_contour_area=np.median(contour_area)
        # print(median_contour_area)

        return [mean_curvature, max_curvature, min_curvature,diff_curvatures,std_curvature, mean_speed, min_speed,max_speed,std_speed, mean_angular_speed, std_angular_speed, mean_contour_area,std_contour_area,max_contour_area,min_contour_area,median_contour_area]
    else:
        return [0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0]

def extract_features_from_folder(folder_path):
    features = []
    labels = []
    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        if os.path.isdir(label_path):
            for video_file in os.listdir(label_path):
                if video_file.endswith('.mp4'):
                    video_path = os.path.join(label_path, video_file)
                    feature_vector = extract_features_from_video(video_path)
                    features.append(feature_vector)
                    labels.append(label)
    return np.array(features), np.array(labels)

# Function to classify a new video
def classify_new_video(model, video_path):
    feature_vector = extract_features_from_video(video_path)
    prediction = model.predict([feature_vector])
    return prediction[0], feature_vector

# Function to classify all videos in a subfolder
def classify_videos_in_folder(model, folder_path):
    predictions_summary = []
    
    for video_file in os.listdir(folder_path):
        if video_file.endswith('.mp4'):
            video_path = os.path.join(folder_path, video_file)
            predicted_behavior, _ = classify_new_video(model, video_path)
            predictions_summary.append((video_file, predicted_behavior))
    
    return predictions_summary

# Function to summarize predictions
def summarize_predictions(predictions_summary):
    summary = {}
    for video_file, predicted_behavior in predictions_summary:
        if predicted_behavior not in summary:
            summary[predicted_behavior] = []
        summary[predicted_behavior].append(video_file)
    
    return summary

# Function to process all subfolders
def process_all_subfolders(main_folder_path, model):
    subfolder_summaries = {}

    for subfolder in os.listdir(main_folder_path):
        subfolder_path = os.path.join(main_folder_path, subfolder)
        if os.path.isdir(subfolder_path):
            print(f"Processing subfolder: {subfolder}")
            predictions_summary = classify_videos_in_folder(model, subfolder_path)
            summary = summarize_predictions(predictions_summary)
            subfolder_summaries[subfolder] = summary
    
    return subfolder_summaries

# Main Execution
# Define the root folder path
# main_folder_path = '/Users/tairan/Downloads/segment_14340_14580'  # Replace with your path
import numpy as np
import os
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Helper functions (apply_circular_mask, calculate_curvature, calculate_speed, 
# calculate_angular_speed, track_fly_and_calculate_features, extract_features_from_video, 
# classify_new_video, classify_videos_in_folder, summarize_predictions) remain the same

def extract_features_and_labels_from_folder(folder_path):
    features = []
    labels = []
    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        if os.path.isdir(label_path):
            for video_file in os.listdir(label_path):
                if video_file.endswith('.mp4'):
                    video_path = os.path.join(label_path, video_file)
                    feature_vector = extract_features_from_video(video_path)
                    features.append(feature_vector)
                    labels.append(label)
    return np.array(features), np.array(labels)
import csv

import csv

def process_and_classify_subfolders(main_folder_path, model, output_csv_path):
    subfolder_summaries = {}

    with open(output_csv_path, mode='w', newline='') as csv_file:
        fieldnames = ['Subfolder', 'Predicted Behavior', 'Video File']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        # Get the list of subfolders and sort them
        def extract_number(subfolder_name):
            number = ''.join(filter(str.isdigit, subfolder_name))
            return int(number) if number else float('inf')  # Use infinity for subfolders with no numbers

        subfolders = sorted(os.listdir(main_folder_path), key=extract_number)

        for subfolder in subfolders:
            subfolder_path = os.path.join(main_folder_path, subfolder)
            if os.path.isdir(subfolder_path):
                print(f"Processing subfolder: {subfolder}")
                predictions_summary = classify_videos_in_folder(model, subfolder_path)
                summary = summarize_predictions(predictions_summary)
                subfolder_summaries[subfolder] = summary

                # Write the summary to the CSV file
                for behavior, videos in summary.items():
                    for video in videos:
                        writer.writerow({
                            'Subfolder': subfolder,
                            'Predicted Behavior': behavior,
                            'Video File': video
                        })
    
    print(f"Summaries have been written to {output_csv_path}")
    return subfolder_summaries

        




# Main Execution

# Step 1: Train the model using the first main folder
training_folder_path = '/Users/tairan/Downloads/segment_14340_14580'  # Replace with your training dataset path
# scaling_factor=1.5

# Extract features and labels for training
# features, labels = extract_features_and_labels_from_folder(training_folder_path)




# Optionally drop the original feature if no longer needed
# features = features.drop(columns=['min_contour_area'])


# Split the dataset into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42,stratify=labels)

# # Train a Random Forest model
# model = RandomForestClassifier(n_estimators=300, random_state=42)
# model.fit(X_train, y_train)
# print("Unique labels in the test set:", np.unique(y_test))
# from sklearn.feature_selection import RFECV


# # Evaluate the model on the test set
# y_pred = model.predict(X_test)
# print("Training set evaluation:")
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Classification Report:\n", classification_report(y_test, y_pred))
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report

# # Assume features and labels have already been extracted
# # Split the dataset into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

# # List of different numbers of estimators to try
# n_estimators_range = [10, 50, 100, 200, 300, 400, 500, 1000]
# accuracies = []

# # Loop over different numbers of estimators
# for n_estimators in n_estimators_range:
#     model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
#     model.fit(X_train, y_train)
 
# Step 2: Use the trained model to classify videos in a different main folder
# classification_folder_path = '/Users/tairan/Downloads/processed_videos_rnai_SIDE-VIII-29C_t4t5_batch2_seg2_filtered_3_arena_3'  # Replace with your classification dataset path

# Process all subfolders in the classification folder and generate summaries
# subfolder_summaries = process_and_classify_subfolders(classification_folder_path, model)

# # Print the summary for each subfolder
# for subfolder, summary in subfolder_summaries.items():
#     print(f"Summary for {subfolder}:")
#     for behavior, videos in summary.items():
#         print(f"  Predicted behavior: {behavior}")
#         for video in videos:
#             print(f"    - {video}")

# Example usage:
# classification_folder_path = '/Users/tairan/Downloads/classification_dataset'  # Replace with your path
# output_csv_path = '/Users/tairan/Downloads/classification_summary.csv'  # Replace with your desired output path


# subfolder_summaries = process_and_classify_subfolders(classification_folder_path, model, output_csv_path)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# # Paths to the "spinning" and "drifting" folders
# spinning_folder_path = '/Users/tairan/Downloads/segment_14340_14580/spinning'
# drifting_folder_path = '/Users/tairan/Downloads/segment_14340_14580/drifting'
# print("Spinning folder contents:", os.listdir(spinning_folder_path))
# print("Drifting folder contents:", os.listdir(drifting_folder_path))

# Extract features from the "spinning" folder
# spinning_features, spinning_labels = extract_features_from_folder(spinning_folder_path)

# # Extract features from the "drifting" folder
# drifting_features, drifting_labels = extract_features_from_folder(drifting_folder_path)

# # Convert to DataFrames for easier handling
# columns = ['mean_curvature', 'max_curvature', 'min_curvature','diff_curvatures','std_curvature', 'mean_speed', 'min_speed','max_speed','std_speed', 'mean_angular_speed', 'std_angular_speed', 'mean_contour_area','std_contour_area','max_contour_area','min_contour_area','median_contour_area']
# # Extract features from the "spinning" folder
# spinning_features, spinning_labels = extract_features_from_folder(spinning_folder_path)
# print(f"Spinning features shape: {spinning_features.shape}")

# # Extract features from the "drifting" folder
# drifting_features, drifting_labels = extract_features_from_folder(drifting_folder_path)
# print(f"Drifting features shape: {drifting_features.shape}")
# spinning_df = pd.DataFrame(spinning_features, columns=columns)
# drifting_df = pd.DataFrame(drifting_features, columns=columns)
# # Combine the dataframes for plotting
# spinning_df['label'] = 'spinning'
# drifting_df['label'] = 'drifting'
# import pandas as pd
# import matplotlib.pyplot as plt

# # Define the columns
# columns = ['mean_curvature', 'max_curvature', 'min_curvature', 'diff_curvatures',
#            'std_curvature', 'mean_speed', 'min_speed', 'max_speed', 'std_speed',
#            'mean_angular_speed', 'std_angular_speed', 'mean_contour_area',
#            'std_contour_area', 'max_contour_area', 'min_contour_area', 'median_contour_area']

# # Assuming spinning_df and drifting_df are already created
# spinning_df['label'] = 'spinning'
# drifting_df['label'] = 'drifting'

# # Combine the dataframes for plotting
# combined_df = pd.concat([spinning_df, drifting_df])

# # Loop through each feature and plot the histogram in separate figures
# for column in columns:
#     plt.figure(figsize=(10, 6))
#     plt.hist(spinning_df[column], bins=20, alpha=0.7, label='Spinning')
#     plt.hist(drifting_df[column], bins=20, alpha=0.7, label='Drifting')
#     plt.title(f'Histogram of {column}')
#     plt.xlabel(column)
#     plt.ylabel('Frequency')
#     plt.legend()
    # plt.show()
# xxxxxxxxxxx
# xxxxxxxxxxx
import os

def process_videos_in_folder(folder_path, label):
    """
    Processes all videos in the specified folder and extracts features for the given label.
    
    Parameters:
    folder_path (str): Path to the folder containing the videos.
    label (str): Label for the behavior ('drifting' or 'spinning').
    
    Returns:
    list: Lists of curvatures, speeds, and angular speeds for all videos in the folder.
    """
    all_curvatures = []
    all_speeds = []
    all_angular_speeds = []
    
    for video_file in os.listdir(folder_path):
        if video_file.endswith('.mp4'):  # Assuming video files are in .mp4 format
            video_path = os.path.join(folder_path, video_file)
            curvatures, speeds, angular_speeds, _ = track_fly_and_calculate_features(video_path)
            
            if curvatures is not None and speeds is not None and angular_speeds is not None:
                all_curvatures.extend(curvatures)
                all_speeds.extend(speeds)
                all_angular_speeds.extend(angular_speeds)
    
    return all_curvatures, all_speeds, all_angular_speeds

# Example folder paths for drifting and spinning
spinning_folder_path = '/Users/tairan/Downloads/segment_segment_180_420/a'
# drifting_folder_path = '/Users/tairan/Downloads/segment_14340_14580/drifting/a'

# # Process videos in the "drifting" folder
# drifting_curvatures, drifting_speeds, drifting_angular_speeds = process_videos_in_folder(drifting_folder_path, 'drifting')

# Process videos in the "spinning" folder
spinning_curvatures, spinning_speeds, spinning_angular_speeds = process_videos_in_folder(spinning_folder_path, 'spinning')
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

def process_and_plot_videos_with_average_overlay(folder_path, label, feature='angular_speed', smooth_window=5):
    """
    Processes all videos in the specified folder, extracts the specified feature,
    plots each video in grey, and overlays the average in color on the same figure.
    
    Parameters:
    folder_path (str): Path to the folder containing the videos.
    label (str): Label for the behavior ('drifting' or 'spinning').
    feature (str): The feature to plot ('angular_speed', 'speed', or 'curvature').
    smooth_window (int): Window size for smoothing the average feature.
    """
    all_features = []

    plt.figure(figsize=(12, 6))  # Initialize the figure once

    for video_file in os.listdir(folder_path):
        if video_file.endswith('.mp4'):  # Assuming video files are in .mp4 format
            video_path = os.path.join(folder_path, video_file)
            curvatures, speeds, angular_speeds, _ = track_fly_and_calculate_features(video_path)
            
            if feature == 'angular_speed':
                feature_data = angular_speeds
                ylabel = 'Angular Speed (degrees/sec)'
            elif feature == 'speed':
                feature_data = speeds
                ylabel = 'Speed (pixels/sec)'
            elif feature == 'curvature':
                feature_data = curvatures
                ylabel = 'Curvature'
            else:
                raise ValueError(f"Unknown feature: {feature}")

            if feature_data is not None:
                all_features.append(feature_data)
                
                # Plot individual video in grey
                plt.plot(feature_data, color='grey', alpha=0.5, label=None)

    # Calculate and plot the average feature if there are any videos processed
    if all_features:
        # Find the minimum length to truncate all sequences to the same length for averaging
        min_length = min(len(f) for f in all_features)
        truncated_features = [f[:min_length] for f in all_features]

        # Calculate the average feature
        avg_feature = np.mean(truncated_features, axis=0)

        # Optionally smooth the average feature
        avg_feature_smoothed = uniform_filter1d(avg_feature, size=smooth_window)

        # Plot the smoothed average in color
        plt.plot(avg_feature_smoothed, color='blue' if label == 'drifting' else 'red', linewidth=2, label=f'Average {label}')
        plt.title(f'Instantaneous {feature.replace("_", " ").title()} for {label} (Individual in Grey, Average in Color)')
        plt.xlabel('Frame')
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()

# Example folder paths for drifting and spinning
# drifting_folder_path = '/path_to_your_data/drifting_videos'
# spinning_folder_path = '/path_to_your_data/spinning_videos'

# Plot different features by changing the 'feature' parameter
# process_and_plot_videos_with_average_overlay(drifting_folder_path, 'drifting', feature='angular_speed')
process_and_plot_videos_with_average_overlay(spinning_folder_path, 'spinning', feature='angular_speed')

# process_and_plot_videos_with_average_overlay(drifting_folder_path, 'drifting', feature='speed')
process_and_plot_videos_with_average_overlay(spinning_folder_path, 'spinning', feature='speed')

# process_and_plot_videos_with_average_overlay(drifting_folder_path, 'drifting', feature='curvature')
process_and_plot_videos_with_average_overlay(spinning_folder_path, 'spinning', feature='curvature')
