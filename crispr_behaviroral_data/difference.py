import pandas as pd

# Replace these paths with the paths to your output files
output_file_method1 = '/Users/tairan/Downloads/gilt1/rnai_gilt1_t4t5_batch3/results_rnai_gilt1_t4t5_batch3_seg3_arena3.csv'  # Output from the first process_video function
output_file_method2 = '/Users/tairan/Downloads/results_rnai_v2/gilt1/results_rnai_gilt1_t4t5_t4t5_batch3_seg3_arena3.csv'  # Output from the second process_video function

# Read the CSV files into pandas DataFrames
# Read the CSV files into pandas DataFrames
df_method1 = pd.read_csv(output_file_method1)  # Adjust 'sep' if needed
df_method2 = pd.read_csv(output_file_method2)

# Print the columns to diagnose the issue
print("Columns in df_method1:", df_method1.columns.tolist())
print("Columns in df_method2:", df_method2.columns.tolist())


# Ensure 'Small Count' is treated as numeric, coerce errors to NaN
df_method1['Small Count'] = pd.to_numeric(df_method1['Small Count'], errors='coerce')
df_method2['Small Count'] = pd.to_numeric(df_method2['Small Count'], errors='coerce')

# Merge the DataFrames on 'Video Filename'
merged_df = pd.merge(
    df_method1[['Video Filename', 'Small Count']],
    df_method2[['Video Filename', 'Small Count']],
    on='Video Filename',
    suffixes=('_Method1', '_Method2')
)

# Calculate the difference in Small Counts
merged_df['Difference'] = merged_df['Small Count_Method2'] - merged_df['Small Count_Method1']

# Display the comparison
print("Comparison of Small Counts:")
print(merged_df)

# Optionally, save the comparison to a new CSV file
merged_df.to_csv('small_count_comparison.csv', index=False)
