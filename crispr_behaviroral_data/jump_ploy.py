import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict

# Path to the directory containing the CSV files
root_directory = '/Users/tairan/Downloads/results_jump'  # Replace with the correct path

def process_counts_per_file(root_directory, condition_name_index=(1, 3)):
    """
    Process CSV files to count occurrences per file where:
    - Only 'Small Count' is present (non-zero, non-NaN), and 'Jump Count' is zero or NaN.
    - Only 'Jump Count' is present (non-zero, non-NaN), and 'Small Count' is zero or NaN.
    - Both 'Small Count' and 'Jump Count' are present (non-zero, non-NaN).
    Include zero counts as well.

    Args:
        root_directory (str): The root directory containing the CSV files.
        condition_name_index (tuple): The indices in the filename parts to extract the condition name.
    Returns:
        df_counts (DataFrame): DataFrame with counts per file and condition.
    """
    data = []

    # Walk through all CSV files
    for subdir, _, files in os.walk(root_directory):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(subdir, file)

                # Load the CSV file with headers
                df = pd.read_csv(file_path)

                # Check if the necessary columns exist
                if 'Small Count' not in df.columns or 'Jump Count' not in df.columns:
                    continue  # Skip this file if columns are missing

                # Extract condition information from the filename
                parts = file.split('_')
                condition = '_'.join(parts[1:3])  # Adjust indices if needed

                # Initialize counts for this file
                count_only_small = 0
                count_only_jump = 0
                count_both = 0

                # Iterate over each row in the DataFrame
                for _, row in df.iterrows():
                    small_count = row['Small Count']
                    jump_count = row['Jump Count']

                    # If 'Small Count' is NaN, ignore 'Jump Count' and skip this row
                    if pd.isna(small_count):
                        continue

                    # Check for 'Small Count' and 'Jump Count' presence
                    small_present = small_count != 0 and not pd.isna(small_count)
                    jump_present = jump_count != 0 and not pd.isna(jump_count)

                    if small_present and not jump_present:
                        count_only_small += 1
                    elif jump_present and not small_present:
                        count_only_jump += 1
                    elif small_present and jump_present:
                        count_both += 1
                    elif not small_present and not jump_present:
                        pass  # Include zero counts if needed

                # Calculate total count
                total_count = count_only_small + count_only_jump + count_both

                # Append the counts for this file to the data list
                data.append({
                    'Condition': condition,
                    'Filename': file,
                    'Only Flipping Count': count_only_small,
                    'Only Jump Count': count_only_jump,
                    'Both': count_both,
                    'Total': total_count
                })

    # Convert to DataFrame
    df_counts = pd.DataFrame(data)

    # Print the counts per file
    print("Counts per file:")
    print(df_counts)

    return df_counts

def plot_counts_all_groups(df_counts):
    """
    Plot both violin and beeswarm plots for all groups (conditions), showing counts for each 'Count Type'.
    Different colors are used for each 'Count Type' to allow differentiation.

    Args:
        df_counts (DataFrame): DataFrame with counts per file and condition.
    """
    # Melt the DataFrame to long format, include zero counts
    df_melted = df_counts.melt(
        id_vars=['Condition', 'Filename'],
        value_vars=['Only Flipping Count', 'Only Jump Count', 'Both', 'Total'],
        var_name='Count Type',
        value_name='Count'
    )

    # Custom sorting function to sort conditions as per requirements
    def custom_sort_key(group):
        group_lower = group.lower()
        if 'rnai_gal4-control' in group_lower:
            return (4, group)  # Control group always last
        elif '29c-off' in group_lower:
            return (3, group)
        elif '29c' in group_lower:
            return (2, group)
        elif '-off' in group_lower:
            return (1, group)  # -off groups second
        else:
            return (0, group)  # Non-off groups first

    # Apply custom sorting to conditions
    df_melted['SortKey'] = df_melted['Condition'].apply(custom_sort_key)
    df_melted.sort_values(by=['SortKey', 'Condition'], inplace=True)

    # Remove the SortKey as it's no longer needed
    df_melted.drop('SortKey', axis=1, inplace=True)

    # Define a colorful palette for the 'Count Type'
    count_types = df_melted['Count Type'].unique()
    palette = sns.color_palette("husl", len(count_types))
    palette_dict = dict(zip(count_types, palette))

    # Create the plot
    plt.figure(figsize=(18, 10))

    # Violin plot: show distribution of counts per 'Condition' and 'Count Type'
    sns.violinplot(
        data=df_melted,
        x='Condition',
        y='Count',
        hue='Count Type',
        palette=palette_dict,
        scale='width',
        cut=0,              # No extension beyond the data range
        inner='quartile',   # Show quartile lines inside the violin
        linewidth=1.5,        # Outline thickness
        alpha=0.6            # Transparency to allow overlay
    )

    # Beeswarm plot: overlay individual data points
    sns.swarmplot(
        data=df_melted,
        x='Condition',
        y='Count',
        hue='Count Type',
        dodge=True,                 # Separate swarms for each hue
        palette=palette_dict,       # Same palette for consistency
        size=5,                     # Size of the points
        edgecolor='gray',
        linewidth=0.5,
        alpha=0.9                   # Less transparent to stand out
    )

    # Adjust labels and title
    plt.xlabel('Condition', fontsize=16)
    plt.ylabel('Count per File', fontsize=16)
    # plt.title("Counts per File for All Groups with Violins and Beeswarm", fontsize=20)
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.yticks(fontsize=14)

    # Adjust the legend to avoid duplicate entries
    handles, labels = plt.gca().get_legend_handles_labels()
    # Since both violin and swarm plots have the same hue, we need to remove duplicates
    # We'll keep only the first occurrence of each label
    by_label = OrderedDict()
    for handle, label in zip(handles, labels):
        if label not in by_label:
            by_label[label] = handle
    plt.legend(by_label.values(), by_label.keys(), title='Count Type', fontsize=12, title_fontsize=14)

    plt.tight_layout()
    plt.show()

# Process counts per file
df_counts = process_counts_per_file(root_directory)

# Plot counts for all groups using both violin and beeswarm plots
plot_counts_all_groups(df_counts)
