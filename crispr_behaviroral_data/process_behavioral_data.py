#!/usr/bin/env python3.8

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from pathlib import Path
pd.options.mode.chained_assignment = None

def compute_velocity(xv, yv):
    # Calculates velocity from x and y components
    velocity = np.sqrt(xv**2 + yv**2)
    return velocity

def correct_orientation_displacement(orientation_difference):
    # Rectifies head-rear ambiguity from OpenCV
    orientations = []
    for difference in orientation_difference:
        if difference > 90:
            orientations.append(difference - 180)
        elif difference < -90:
            orientations.append(180 + difference)
        else:
            orientations.append(difference)
    return orientations

def get_cumulative_angular_displacement(start_orientation, orientation_diff):
    # Store the cumulative turn
    orientations = []
    displacement = start_orientation
    for diff in orientation_diff:
        displacement += diff
        orientations.append(displacement)
    return orientations

def analyze_optomotor_behavior(df, arena, trial):
    dff = df[(df.Arena == arena) & (df.Trial == trial)]
    # Transform pixel to mm
    dff.loc[:, ("x")] = dff.loc[:, ("x")] / 12
    dff.loc[:, ("y")] = dff.loc[:, ("y")] / 12
    # Compute x-axis velocity
    dff.loc[:, ("xv")] = savgol_filter(dff.loc[:, ("x")], 21, 4, deriv=1) / dff.loc[:, ("Timestamp")].diff().fillna(0) * 1e9

    # Compute y-axis velocity
    dff.loc[:, ("yv")] = savgol_filter(dff.loc[:, ("y")], 21, 4, deriv=1) / dff.loc[:, ("Timestamp")].diff().fillna(0) * 1e9

    # Compute angular displacement
    dff.loc[:, ("Orientation_diff")] = dff.loc[:, ("Orientation")].diff()
    dff = dff.fillna(0)
    dff.loc[:, ("Orientation_diff")] = correct_orientation_displacement(dff["Orientation_diff"])
    dff.loc[:, ("Cumulative_Orientation")] = get_cumulative_angular_displacement(df.loc[0, ("Orientation")], dff.loc[:, ("Orientation_diff")])
    if dff["Clockwise"].iloc[0] == False:
        dff.loc[:, ("Cumulative_Orientation")] = dff.loc[:, ("Cumulative_Orientation")] * (-1)
    dff["Cumulative_Orientation"] = savgol_filter(dff["Cumulative_Orientation"], 21, 4)
    dff["Angular_velocity"] = savgol_filter(dff["Cumulative_Orientation"], 21, 4, deriv=1) / dff.loc[:, ("Timestamp")].diff().fillna(0) * 1e9
    # Get translational velocity
    dff["Translational_velocity"] = compute_velocity(dff.xv, dff.yv)
    dff["Translational_velocity"] = savgol_filter(dff["Translational_velocity"], 21, 4)
    dff = dff.reset_index()
    total_displacement = dff["Cumulative_Orientation"].values[-1] - dff.loc[len(dff)-240, ("Cumulative_Orientation")]
    velocity_pre_stim = dff.loc[len(dff)-270:len(dff)-240, ("Translational_velocity")].median()
    avg_angular_velocity = dff.loc[len(dff)-240:len(dff), ("Angular_velocity")].median()
    avg_translational_velocity = dff.loc[:, ("Translational_velocity")].median()
    # latency = 0
    # print(f"{arena},{trial},{total_displacement},{velocity_pre_stim},{latency}")
    # print(dff.loc[len(dff)-360:len(dff), ["Angular_velocity"]].values)
    return [arena, trial, total_displacement, avg_angular_velocity, avg_translational_velocity, velocity_pre_stim]

    # sns.set(style="white", rc={"lines.linewidth": 2})
    # # Trial position distribution
    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, figsize=(4, 16))
    # sns.scatterplot(x="x", y="y", data=dff, hue="Frame", color="black", s=4, legend=False, ax=ax1)
    # circle = plt.Circle((0,0), 240, color="gray", fill=False)
    # ax1.add_patch(circle)
    # sns.despine(ax=ax1, left=True, bottom=True)
    # ax1.set(xlabel=None, xticklabels=[], ylabel=None, yticklabels=[])
    # ax1.set_xlim(-240, 240)
    # ax1.set_ylim(-240, 240)
    # # Trial Cumulative Orientation
    # sns.lineplot(x="Frame", y="Cumulative_Orientation", data=dff, color="black", ax=ax2)
    # ax2.axvline(len(dff)-240, color="gray", ls="--")
    # # Trial Forward Velocity
    # sns.lineplot(x="Frame", y="Translational_velocity", data=dff, color="black", ax=ax3)
    # ax3.axvline(len(dff)-240, color="gray", ls="--")
    # # Trial Angular Velocity
    # sns.lineplot(x="Frame", y="Angular_velocity", data=dff, color="black", ax=ax4)
    # ax4.axvline(len(dff)-240, color="gray", ls="--")
    # plt.tight_layout()
    # plt.show()


def process_output_file(file):
    data_dic = {}
    df = pd.read_csv(file, names=["Frame", "Timestamp", "Arena", "x", "y", "Orientation", "Stimulus", "Frequency", "Clockwise", "Trial"])
    for arena in [0, 1, 2, 3]:
        for index in range(1,99):
            data_dic[f"{arena}_{index}"] = analyze_optomotor_behavior(df, arena, index)
    dff = pd.DataFrame.from_dict(data_dic, orient="index", columns=["Arena", "Trial", "Total_Angular_Displacement", "Avg_Angular_Velocity", "Avg_Translational_Velocity", "Pre_Stim_Velocity"])
    dff["Perturbation"] = f"{file.name.split('_')[1]}_{file.name.split('_')[3]}"
    return dff

def batch_process(folder):
    df = pd.DataFrame()
    p = Path(folder)
    for file in p.iterdir():
        print(file)
        if file.name.endswith(".csv"):
            df = pd.concat([df, process_output_file(file)])
    df.to_csv("dummy.csv")

batch_process("/Users/tairan/Downloads/crispr_behaviroral_data/CRISPR_T4T5_screen/")
