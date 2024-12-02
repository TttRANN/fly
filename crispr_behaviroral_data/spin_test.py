import numpy as np
import os
import cv2
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import export_graphviz, plot_tree
import matplotlib.pyplot as plt

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

    frame_count = 0
    background_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    background_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    center = (background_width // 2, background_height // 2)
    radius = int(min(background_width, background_height) * 0.45)
    consecutive_invalid_frames = 0

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
        
        frame_count += 1

    cap.release()
    if output_video_path:
        out.release()

    if len(centroids) > 2:
        curvatures = calculate_curvature(centroids)
        speeds = calculate_speed(centroids, fps)
        angular_speeds = calculate_angular_speed(orientations, fps)
        return curvatures, speeds, angular_speeds
    else:
        return None, None, None

def extract_features_from_video(video_path):
    curvatures, speeds, angular_speeds = track_fly_and_calculate_features(video_path, None)
    
    if curvatures is not None and len(curvatures) > 0:
        mean_curvature = np.mean(curvatures)
        # median_curvature = np.median(curvatures)
        std_curvature = np.std(curvatures)
        mean_speed = np.mean(speeds)
        std_speed = np.std(speeds)
        mean_angular_speed = np.mean(angular_speeds)
        std_angular_speed = np.std(angular_speeds)
        return [mean_curvature,  std_curvature, mean_speed, std_speed, mean_angular_speed, std_angular_speed]
    else:
        return [0,0, 0, 0, 0, 0]

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

# Define the root folder path
folder_path = '/Users/tairan/Downloads/segment_14340_14580'  # Replace with your path

# Extract features and labels
features, labels = extract_features_from_folder(folder_path)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.5, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=1000, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Visualize one of the trees in the Random Forest
# tree = model.estimators_[0]  # You can choose any tree by changing the index

# plt.figure(figsize=(20, 10))
# plot_tree(tree, filled=True, rounded=True, class_names=model.classes_, 
#           feature_names=['mean_curvature', 'std_curvature', 'mean_speed', 'std_speed', 'mean_angular_speed', 'std_angular_speed'])
# plt.show()

# # Export as DOT file
# export_graphviz(tree, out_file='tree.dot', 
#                 feature_names=['mean_curvature', 'std_curvature', 'mean_speed', 'std_speed', 'mean_angular_speed', 'std_angular_speed'], 
#                 class_names=model.classes_,
#                 filled=True, rounded=True)

# def classify_new_video(model, video_path):
#     feature_vector = extract_features_from_video(video_path)
#     prediction = model.predict([feature_vector])
#     return prediction[0]

# # Example usage:
# new_video_path = '/Users/tairan/Downloads/segment_14340_14580/42000_42240_segment_0007.mp4'
# predicted_behavior = classify_new_video(model, new_video_path)
# print(f"Predicted behavior: {predicted_behavior}")
