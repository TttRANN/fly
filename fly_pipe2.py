import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.signal import periodogram, find_peaks, savgol_filter
import os
import matplotlib.pyplot as plt
import glob
import pingouin as pg
import seaborn as sns  # Import seaborn for plotting
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import flymodule
import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt

def swt_denoising(data, wavelet='bior2.6', level=7):
    
    if data.size == 0:
        return data
    max_level = pywt.swt_max_level(len(data))
    if max_level == 0:
        print("Data length is too short for any level of SWT decomposition.")
        return data
    level = min(level, max_level)
    # self.plot_swt_scalogram(self,data,wavelet)
    freq = pywt.scale2frequency(wavelet, 2 ** np.arange(0, level)[::-1]) / 0.016
    print(f"freq:{freq}")
    swt_coeffs = pywt.swt(data, wavelet, level=level)
    index = np.argwhere((freq > 10) & (freq < 30)).flatten()
    swt_filtered = []
    for i, coeff in enumerate(swt_coeffs):
        if i in index:
            swt_filtered.append(coeff)
        else:
            swt_filtered.append((np.zeros_like(coeff[0]), np.zeros_like(coeff[1])))
    re_data = pywt.iswt(swt_filtered, wavelet)
    return re_data

def plot_swt_scalogram(data, wavelet='bior2.6', level=7, sampling_period=0.016):
    # Check if data is a pandas DataFrame or Series and extract the values
    if isinstance(data, pd.DataFrame):
        # If DataFrame, extract the first column or adjust as needed
        data = data.iloc[:, 0].values.flatten()
    elif isinstance(data, pd.Series):
        data = data.values.flatten()
    else:
        # If not a DataFrame or Series, ensure it's a NumPy array
        data = np.asarray(data).flatten()
    
    if data.size == 0:
        print("Data is empty.")
        return
    max_level = pywt.swt_max_level(len(data))
    if max_level == 0:
        print("Data length is too short for any level of SWT decomposition.")
        return
    level = min(level, max_level)
    swt_coeffs = pywt.swt(data, wavelet, level=7)


    # Compute corresponding frequencies
    scales = 2 ** np.arange(0, level)
    frequencies = pywt.scale2frequency(wavelet, scales) / sampling_period
    # print(frequencies)
    # Combine approximation and detail coefficients for visualization
    coeff_matrix = np.vstack([coeff[0] for coeff in swt_coeffs] )
    print(coeff_matrix.shape)

    # Update frequency bands to include approximation level
    frequencies = np.concatenate([frequencies, [frequencies[-1] / 2]])

    plt.figure(figsize=(12, 6))
    plt.imshow(np.abs(coeff_matrix), extent=[0, len(data), frequencies[-1], frequencies[0]],
            cmap='jet', aspect='auto', interpolation='nearest')
    plt.colorbar(label='Magnitude')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (samples)')
    plt.title('SWT Scalogram Including Approximation Coefficients')
    plt.show()


    # After computing 'frequencies' and before 'plt.show()'
    # plt.figure(figsize=(12, 6))
    # plt.imshow(np.abs(detail_coeffs), extent=[0, len(data), frequencies[-1], frequencies[0]],
    #         cmap='jet', aspect='auto', interpolation='nearest')
    # plt.colorbar(label='Magnitude')
    # plt.ylabel('Frequency (Hz)')
    # plt.xlabel('Time (samples)')
    # plt.title('SWT Scalogram')

    # # Set y-ticks to the frequencies
    # plt.yticks(frequencies, np.round(frequencies, 2))
    # plt.show()





folder_path = "/Users/tairan/Downloads/roshan"  # Update this path accordingly

# Initialize lists to collect medians for each group
median_data = []  # This will store dictionaries of {'Group': group_name, 'Value': median_value}

# Frequencies to consider
frequencies = [5, 10, 20]

# Initialize data structures to hold trials and straightness values by frequency
trials_by_frequency = {freq: [] for freq in frequencies}

# Initialize group straightness values by frequency
group_names = [
    "Cas9Gal4SgrnaControl",
    "Cas9Dscam383045",
    "Cas9Gal4HomoControl",
    "Cas9Gal4Kir21",
    "RNAiKir21",
    "RNAiKek1",
    "Cas9PlexA",
    "RNAiBeatIV",
    "RNAiCG32206",
    "RNAiFas3",
    "RNAiHomozygousControl"
]

group_straightness_values = {group: {freq: [] for freq in frequencies} for group in group_names}

# Main loop to apply smoothing, alignment, and responsiveness check
for file_path in glob.glob(f"{folder_path}/*.csv"):
    file_name = os.path.basename(file_path)
    print(file_name)
    if "stimulus" not in file_name:
        df = pd.read_csv(file_path, names=["pos_x", "pos_y", "ori", "timestamp", "frame_number", "video_frame", "direction"])
        filename_without_extension = os.path.splitext(file_name)[0]
        file_stim_name = filename_without_extension + "_stimulus_conditions.csv"
        file_path_stimulus = f"{folder_path}/{file_stim_name}"

        df_stimulus = pd.read_csv(file_path_stimulus, names=["trial_number", "contrast", "spatial_frequency", "temporal_frequency", "direction"])
        df = df.apply(pd.to_numeric, errors='coerce')

        df['Trial_Num'] = np.nan  # Add a column for trial number
    else:
        continue
    fly=flymodule.Fly_dynamics(df)
    df=fly.correct_opencv_theta_assignment(df)
    # print(df)
    df=pd.DataFrame(df)

    angular_speed = np.diff(df["theta_corr"]) * 60
    print(angular_speed)
    print(type(angular_speed))
    plot_swt_scalogram(angular_speed)
    plot_swt_scalogram(swt_denoising(angular_speed))


    # print(dir(fly))

    # a=fly.correct_opencv_theta_assignment()
#     # print(a)
    # speed=fly.angular_speed()

#     # print(speed)
    _, trials_list, trials_by_frequency=fly.process_direction_transitions(df_stimulus,[5,10,20])
#     # print(trials_list)

    # Determine group name based on file name
    if "sgrna-control" in file_name.lower():
        group_name = "Cas9Gal4SgrnaControl"
    elif "dscam3" in file_name.lower():
        group_name = "Cas9Dscam383045"
    elif "cas9_r42f06-gal4_control_homozygous" in file_name.lower():
        group_name = "Cas9Gal4HomoControl"
    elif "kir2.1" in file_name.lower():
        group_name = "Cas9Gal4Kir21"
    elif "rnai_r42f06-gal4_kir2" in file_name.lower():
        group_name = "RNAiKir21"
    elif "kek1" in file_name.lower():
        group_name = "RNAiKek1"
    elif "plexa" in file_name.lower():
        group_name = "Cas9PlexA"
    elif "beat-iv" in file_name.lower():
        group_name = "RNAiBeatIV"
    elif "cg32206_67029" in file_name.lower():
        group_name = "RNAiCG32206"
    elif "fas3" in file_name.lower():
        group_name = "RNAiFas3"
    elif "rnai_r42f06-gal4_control_homozygous" in file_name.lower():
        group_name = "RNAiHomozygousControl"
    else:
        group_name = "Other"

    # Process responsive trials for each frequency
    speed_threshold = 45
    window_size = 50  # Set window size for rolling straightness
    step_size = 10   # Set step size for rolling straightness

    for freq in frequencies:
        # Get trials for this frequency
        trials_for_freq = [trial for trial in trials_list if trial['frequency_stim'] == freq]
        # print(trials_for_freq)

        # Filter responsive trials
        # Separate responsive trials and their speed values
        responsive_trials = []
        angular_speeds = []
        peak_index = []
        peak_syn_index = []
        peak_anti_index = []

        for trial in trials_for_freq:
            # print("-------------------")
            # print(trial)
            # print("-------------------")
            is_responsive = fly.is_fly_responsive_dual(trial, speed_threshold)

            if is_responsive:
                responsive_trials.append(trial)

        # Calculate straightness for responsive trials
        freq_straightness_values = []
        angular_speeds2=[]
        peak_index = []
        peak_index_syn = []
        peak_index_anti = []

        for trial in responsive_trials:
            df_trial = pd.DataFrame(trial)
            angular_speed = np.diff(trial["theta_corr"]) * 60
            angular_speeds2.append(np.diff(trial["theta_corr"]) * 60)
            angular_speeds.append(fly.swt_denoising(angular_speed, level=5))

            threshold_saccades = 4 * np.median(np.abs(angular_speed) / 0.6745)
            
            if len(df_trial["theta_corr"]) < 10:  # Adjust the threshold as needed
                print(f"Skipping trial {trial['trial_number']} due to insufficient data length.")
                continue
            # print("---------------------------------")
            final_peaks, _, _, final_peaks_syn, final_peaks_anti = fly.detect_saccades_cwt(df_trial,threshold_saccades)
            # plot_swt_scalogram(df_trial)
            # print("---------------------------------")
            peak_index.append(final_peaks)
            peak_index_syn.append(final_peaks_syn)
            peak_index_anti.append(final_peaks_anti)
        # swt_denoising(angular_speed, level=5)
        # plot_swt_scalogram(angular_speeds2)
        # plot_swt_scalogram(angular_speeds)



        if responsive_trials:
            # Initialize total_responsive_trials
            total_responsive_trials = len(responsive_trials)

            # Get the maximum number of frames among the responsive trials
            max_frame_length = max(len(trial['theta_corr']) for trial in responsive_trials)
            
            # Initialize saccade occurrence matrices per trial and frame
            saccade_matrix_syn = np.zeros((total_responsive_trials, max_frame_length))
            saccade_matrix_anti = np.zeros((total_responsive_trials, max_frame_length))
            
            # For each trial, mark saccade occurrences in the matrices
            for idx, (trial, peaks_syn_in_trial, peaks_anti_in_trial) in enumerate(zip(responsive_trials, peak_index_syn, peak_index_anti)):
                trial_length = len(trial['theta_corr'])
                for i in peaks_syn_in_trial:
                    # Expand peaks by ±3 frames
                    start = max(0, i - 15)
                    end = min(trial_length, i + 15 + 1)
                    saccade_matrix_syn[idx, start:end] = 1
                for i in peaks_anti_in_trial:
                    start = max(0, i - 15)
                    end = min(trial_length, i + 15 + 1)
                    saccade_matrix_anti[idx, start:end] = 1
            
            # Create boolean matrices
            saccade_matrix_syn_bool = saccade_matrix_syn == 1
            saccade_matrix_anti_bool = saccade_matrix_anti == 1
            # print(saccade_matrix_syn_bool)

            both = saccade_matrix_syn_bool & saccade_matrix_anti_bool
            syn_only = saccade_matrix_syn_bool & (~saccade_matrix_anti_bool)
            anti_only = saccade_matrix_anti_bool & (~saccade_matrix_syn_bool)
            none = (~saccade_matrix_syn_bool) & (~saccade_matrix_anti_bool)
            
            # Counts per frame
            counts_both = np.sum(both, axis=0)
            counts_syn_only = np.sum(syn_only, axis=0)
            counts_anti_only = np.sum(anti_only, axis=0)
            counts_none = np.sum(none, axis=0)
            
            # Fractions per frame
            fraction_both = counts_both / total_responsive_trials
            fraction_syn_only = counts_syn_only / total_responsive_trials
            fraction_anti_only = counts_anti_only / total_responsive_trials
            fraction_none = counts_none / total_responsive_trials
            
            # Plot the fractions per frame
            plt.figure(figsize=(10, 6), dpi=600)
            frame_numbers = np.arange(max_frame_length)

# Assuming frame_numbers, fraction_syn_only, and fraction_anti_only are numpy arrays
            # fractions = np.vstack([fraction_syn_only, 1-fraction_syn_only, 1 - fraction_syn_only - fraction_anti_only])

            # Calculate 'Smooth Turning' fraction
            fraction_smooth_turning = 1 - fraction_syn_only - fraction_anti_only

            # Prepare the fractions in the correct stacking order
            fractions = [fraction_syn_only,fraction_anti_only , 1-fraction_anti_only-fraction_syn_only]

            # Plot using stackplot
            plt.stackplot(frame_numbers, fractions, labels=['Syn Saccades', 'Anti Saccades', 'Smooth Turning'], baseline='zero',colors=['blue', 'red', 'green'],alpha=0.5)
            # plt.ylim(0, 1)  
            plt.xlabel('Frame Number')
            plt.ylabel('Fraction of Trials')
            plt.title(f'Fraction of Trials with Saccades at Each Frame for {file_name} at {freq} Hz')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"/Users/tywentking/Downloads/roshan3/{file_name}_freq_{freq}_saccade_fraction_anti_syn.png")
            plt.close()

