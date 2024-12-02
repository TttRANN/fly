import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/Users/tairan/Downloads/29c/rnai_BEAT-IV-29C_t4t5_batch1/rnai_BEAT-IV-29C_t4t5_batch1_filtered_2.csv'
df = pd.read_csv(file_path, header=None)

# Extract the relevant portion of the data
selected_data = df.loc[4139:4377, [3, 4]]
print(df.dtypes)
# Convert columns to float
selected_data = selected_data.astype(float)

# Plotting the selected data
plt.figure(figsize=(10, 8))
plt.plot(selected_data[3], selected_data[4], marker='o', linestyle='-')
plt.xlabel('Column 3')
plt.ylabel('Column 4')
plt.title('Plot of Selected Data')
plt.grid(True)

# Optional: Format the y-axis with scientific notation
plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter())

plt.show()


# Use ScalarFormatter to format the y-axis
ax = plt.gca()
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.yaxis.get_major_formatter().set_scientific(False)

plt.show()
