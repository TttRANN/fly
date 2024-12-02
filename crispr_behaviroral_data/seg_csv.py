import os
import glob
import pandas as pd

# Base directory containing the subfolders
import os
import glob

# Base directory containing the subfolders
base_dir = '/Users/tairan/Downloads/testfor'

# Print the base directory to ensure it's correct
print(f"Base directory: {base_dir}")

# Find all directories with a name pattern starting with 'rnai'
sub_dirs = glob.glob(os.path.join(base_dir, 'rnai*'))

# Print the list of subdirectories
print(sub_dirs)

# List all files and folders in the base directory
all_contents = os.listdir(base_dir)
print("All contents in base_dir:")
print(all_contents)

# Values to filter on
values_to_filter = [0, 1, 2, 3]

# Loop through each subdirectory
for sub_dir in sub_dirs:
    # Find all CSV files in the subdirectory
    csv_files = glob.glob(os.path.join(sub_dir, '*.csv'))
    
    # Loop through each CSV file in the subdirectory
    for csv_file in csv_files:
        # Read the CSV file
        data = pd.read_csv(csv_file)
        
        # Loop through each value and create a separate file
        for value in values_to_filter:
            # Filter rows where the third column has the specific value
            rows_to_keep = data.iloc[:, 2] == value
            data_filtered = data[rows_to_keep]
            
            # Reindex the first column to be sequential
            data_filtered.iloc[:, 0] = range(1, len(data_filtered) + 1)
            
            # Save the result to a new CSV file within the same subdirectory
            new_file_name = f'{os.path.splitext(os.path.basename(csv_file))[0]}_filtered_{value}.csv'
            new_file_path = os.path.join(sub_dir, new_file_name)
            data_filtered.to_csv(new_file_path, index=False)
            
            print(f'Filtered data containing only {value} has been saved to {new_file_path}')

