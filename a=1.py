import os
import glob
import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output
import flymodule

# Initial mapping from substrings to group names
group_mapping_text = """
sgrna-control: Cas9Gal4SgrnaControl
dscam3: Cas9Dscam383045
cas9_r42f06-gal4_control_homozygous: Cas9Gal4HomoControl
kir2.1: Cas9Gal4Kir21
rnai_r42f06-gal4_kir2: RNAiKir21
kek1: RNAiKek1
plexa: Cas9PlexA
beat-iv: RNAiBeatIV
cg32206_67029: RNAiCG32206
fas3: RNAiFas3
rnai_r42f06-gal4_control_homozygous: RNAiHomozygousControl
"""

# Frequencies (assuming you have this list)
frequencies = [5, 10, 20]

folder_path = "/Users/tairan/Downloads/roshan" 

# Text area for group mapping input
mapping_text_area = widgets.Textarea(
    value=group_mapping_text.strip(),
    description="Group Mapping:",
    layout=widgets.Layout(width="50%", height="200px")
)

# Buttons
save_button = widgets.Button(
    description="Save Changes",
    button_style="success",
    icon="save"
)
process_button = widgets.Button(
    description="Start Processing",
    button_style="primary",
    icon="play"
)

# Output area to show saved results and processing output
output = widgets.Output()

# Function to handle save button click
def save_changes(_):
    with output:
        clear_output()
        # Update group_mapping dictionary
        mapping_lines = mapping_text_area.value.strip().splitlines()
        global group_mapping
        group_mapping = {}
        for line in mapping_lines:
            if ':' in line:
                key, value = line.split(':', 1)
                group_mapping[key.strip().lower()] = value.strip()
        print("Updated Group Mapping:")
        print(group_mapping)

# Function to handle processing
def start_processing(_):
    with output:
        clear_output()
        print("Starting processing with group mapping:")
        print(group_mapping)
        # Initialize the dictionary with updated group names
        group_names = list(set(group_mapping.values()))
        group_straightness_values = {group: {freq: [] for freq in frequencies} for group in group_names}
        
        # Main loop to apply smoothing, alignment, and responsiveness check
        for file_path in glob.glob(f"{folder_path}/*.csv"):
            file_name = os.path.basename(file_path)
            print(f"Processing {file_name}")
            if "stimulus" not in file_name.lower():
                df = pd.read_csv(file_path, names=["pos_x", "pos_y", "ori", "timestamp", "frame_number", "video_frame", "direction"])
                filename_without_extension = os.path.splitext(file_name)[0]
                file_stim_name = filename_without_extension + "_stimulus_conditions.csv"
                file_path_stimulus = f"{folder_path}/{file_stim_name}"
        
                df_stimulus = pd.read_csv(file_path_stimulus, names=["trial_number", "contrast", "spatial_frequency", "temporal_frequency", "direction"])
                df = df.apply(pd.to_numeric, errors='coerce')
        
                df['Trial_Num'] = np.nan  # Add a column for trial number
            else:
                continue
            fly = flymodule.Fly_dynamics(df)

            _, trials_list, trials_by_frequency=fly.process_direction_transitions(df_stimulus,[5,10,20])


            # Determine group name based on file name
            group_name = "Other"
            for key, value in group_mapping.items():
                if key in file_name.lower():
                    group_name = value
                    break

            print(f"Assigned to group: {group_name}")



# Bind the button clicks to their respective functions
save_button.on_click(save_changes)
process_button.on_click(start_processing)

# Display the widgets
display(widgets.VBox([mapping_text_area, widgets.HBox([save_button, process_button]), output]))

