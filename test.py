import numpy as np
import pandas as pd

file_info = [
    {"a": "abc", "b": [123], "c": [2312]},
    {"a": "abc", "b": [2, 34, 5], "c": [18]},
]

# Initialize an empty list to collect rows
rows = []



#testhahahahahhha
# test
#new 
# avcjeadnv;rwa
#ewafoerwahfrwqafwqhpf
#newbee
# Loop through each dictionary in file_info
for dic in file_info:
    # Append a row to the list
    rows.append({
        'Group': dic["a"],
        'Value': np.mean(dic["b"]),
        'Extra': np.mean(dic["c"])  # Example if you want to use 'c' data
    })

# Convert the list of rows into a DataFrame
print(type(rows))
data = pd.DataFrame(rows)

# Display the DataFrame
print(data)
