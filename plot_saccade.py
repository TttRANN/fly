
import numpy as np

def plot_saccade(responsive_trials,fly,file_name,freq):
    # Initialize saccade occurrence matrices per trial and frame
    # Calculate straightness for responsive trials
    freq_straightness_values = []
    fly
    angular_speeds=[]
    peak_index = []
    peak_index_syn = []
    peak_index_anti = []
    for trial in responsive_trials:
        df_trial = pd.DataFrame(trial)
        angular_speed = np.diff(trial["theta_corr"]) * 60
        # angular_speeds2.append(np.diff(trial["theta_corr"]) * 60)
        angular_speeds.append(fly.swt_denoising(angular_speed, level=5))

        threshold_saccades = 4 * np.median(np.abs(angular_speed) / 0.6745)
        
        if len(df_trial["theta_corr"]) < 10:  # Adjust the threshold as needed
            print(f"Skipping trial {trial['trial_number']} due to insufficient data length.")
            continue
        # print("---------------------------------")
        final_peaks, _, _, final_peaks_syn, final_peaks_anti = fly.detect_saccades_cwt(df_trial,threshold_saccades)

        peak_index.append(final_peaks)
        peak_index_syn.append(final_peaks_syn)
        peak_index_anti.append(final_peaks_anti)
        num_responsive_trials = len(responsive_trials)
        max_frame_length = max(len(trial['theta_corr']) for trial in responsive_trials)
        saccade_matrix_syn = np.zeros((num_responsive_trials, max_frame_length))
        saccade_matrix_anti = np.zeros((num_responsive_trials, max_frame_length))
    
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
    syn_only = saccade_matrix_syn_bool 
    anti_only = saccade_matrix_anti_bool
    
    # Counts per frame
    counts_both = np.sum(both, axis=0)
    counts_syn_only = np.sum(syn_only, axis=0)
    counts_anti_only = np.sum(anti_only, axis=0)
    
    # Fractions per frame
    fraction_syn_only = counts_syn_only / num_responsive_trials
    fraction_anti_only = counts_anti_only / num_responsive_trials
    
    # Plot the fractions per frame
    plt.figure(figsize=(10, 6), dpi=1500)
    frame_numbers = np.arange(max_frame_length)

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
