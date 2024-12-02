import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
def correct_opencv_theta_assignment(df):
    df['direction'] = df['direction'].astype(float)
    if df['direction'].sum() > 0:
        factor = -1
    else:
        factor = 1
    cumulative_theta = df['ori'].iloc[0]
    theta_corr = [cumulative_theta * factor]
    for difference in df['ori'].diff().fillna(0).iloc[1:]:
        if difference > 90:
            difference -= 180
        elif difference < -90:
            difference += 180
        cumulative_theta += difference
        theta_corr.append(cumulative_theta * factor)
    df = df.assign(theta_corr=theta_corr)
    return df
folder_path = '/Users/tairan/Downloads/test_roshan'
# List to store data for all files
all_data = []

# Process each CSV file in the folder
for file_path in glob.glob(f"{folder_path}/*.csv"):
    file_name = os.path.basename(file_path)
    file_name = '_'.join(file_name.split("_")[0:3])
    df = pd.read_csv(file_path, names=["pos_x", "pos_y", "ori", "timestamp", "frame_number", "video_frame", "direction"])
    df = df.apply(pd.to_numeric, errors='coerce')
    df = correct_opencv_theta_assignment(df)
    
    ori_diff_minus = []  # List to store differences for -1 transitions
    ori_diff_1 = []      # List to store differences for +1 transitions
    positive = 0
    negative = 0

    for i in range(df["direction"].size - 1):
        # Check for -1 transitions
        if df["direction"][i] == 0 and df["direction"][i+1] == -1.0:
            start_point = i
        elif df["direction"][i] == -1.0 and df["direction"][i+1] == 0:
            end_point = i
            diff = np.cumsum(np.diff(df["ori"][start_point:end_point+1]))
            
            if -diff[-1] > 30 or -diff[-1] < -30:
                ori_diff_minus.append(diff[-1])
            
            if -diff[-1] > 30:
                positive += 1
            elif -diff[-1] < -30:
                negative += 1

        # Check for +1 transitions
        if df["direction"][i] == 0 and df["direction"][i+1] == 1.0:
            start_point = i
        elif df["direction"][i] == 1.0 and df["direction"][i+1] == 0:
            end_point = i
            diff = np.cumsum(np.diff(df["ori"][start_point:end_point+1]))
            
            if diff[-1] > 30 or diff[-1] < -30:           
                ori_diff_1.append(-diff[-1])
            
            if diff[-1] > 30:
                positive += 1
            elif diff[-1] < -30:
                negative += 1

    # Collect data for plotting
# Collect data for plotting
    print(positive / (negative+positive) if negative != 0 else 'No negative transitions')

    # Create DataFrames with correct lengths
    ori_diff_minus_df = pd.DataFrame({
        "Difference": ori_diff_minus,
        # -1
        "Transition Type": ["1"] * len(ori_diff_minus),
        "File": [file_name] * len(ori_diff_minus)
    })

    ori_diff_1_df = pd.DataFrame({
        "Difference": ori_diff_1,
        "Transition Type": ["1"] * len(ori_diff_1),
        "File": [file_name] * len(ori_diff_1)
    })

    file_data = pd.concat([ori_diff_minus_df, ori_diff_1_df], ignore_index=True)
    all_data.append(file_data)

# Concatenate all data
final_data = pd.concat(all_data, ignore_index=True)

# Plot the violin plot
plt.figure(figsize=(50, 6))
sns.swarmplot(x="File", y="Difference", hue="Transition Type", data=final_data, dodge=True)
plt.xticks(rotation=90)
plt.xlabel("File")
plt.ylabel("Difference in 'ori'")
plt.title("Violin Plot of 'ori' Differences by File and Transition Type")
plt.legend(title="Transition Type")
plt.tight_layout()
plt.show()
