# Function to merge intervals continuously if gaps between stop and start are <= 10 frames
def merge_intervals(start_frames, stop_frames):
    merged_intervals = []
    current_start = start_frames[0]
    current_stop = stop_frames[0]

    for i in range(1, len(start_frames)):
        next_start = start_frames[i]
        next_stop = stop_frames[i]

        # If the gap between current stop and next start is <= 10, merge them
        if next_start - current_stop <= 10:
            current_stop = next_stop  # Extend the stop to the next stop
        else:
            merged_intervals.append((current_start, current_stop))  # Add the merged interval
            current_start = next_start
            current_stop = next_stop

    # Append the last interval
    merged_intervals.append((current_start, current_stop))
    
    return merged_intervals

# Example data (input 2)
start_frames = [26, 33, 46, 54, 100, 123, 138, 165, 194, 208, 218, 235]
stop_frames = [29, 44, 52, 75, 122, 127, 162, 167, 205, 210, 229, 239]

# Merge the intervals until they are all separable
merged_intervals = merge_intervals(start_frames, stop_frames)

# Print the result
print("Merged Intervals:", merged_intervals)
