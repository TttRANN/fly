import cv2
import os
# print(list(range(19,0,-1)))



a = [1, 1, 1, 0, 0, 1, 1, 1,5,6,7,5,5,5,8,9]
consecutive_count = 0
consecutive_details = []

i = 0
while i < len(a) - 1:
    current_count = 1  # Start with the current element
    while i < len(a) - 1 and a[i] == a[i + 1]:
        current_count += 1
        i += 1
    if current_count > 1:
        consecutive_count += 1
        consecutive_details.append((a[i], current_count, i - current_count + 2))
    i += 1

print("Total consecutive occurrences:", consecutive_count)
for num, count, start_index in consecutive_details:
    print(f"Number {num} appears {count} times consecutively starting at index {start_index}.")

        


# # Path to the video file
# video_path = '/path/to/your/video.mp4'
# output_dir = '/path/to/output/images'
# cap = cv2.VideoCapture("/Users/tairan/Downloads/9/seg0/new.mp4")
# total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# # total_frames = 600  # Replace this with the actual number of total frames
# count=0
# for i in range(5):
#     denominator = pow(2, i)
#     frame_frag = total_frames // denominator
    
#     # Create a list of frame indices for this iteration
#     frame_indices = []
    
#     for j in range(denominator):
        
#         start_frame = j * frame_frag
#         end_frame = (j + 1) * frame_frag
#         frame_indices.append([start_frame, end_frame])
#         count=count+1

#     # Print or use the frame_indices list as needed
#     print(f"Iteration {i+1}: Frame Indices = {frame_indices}")
#     print(count)





