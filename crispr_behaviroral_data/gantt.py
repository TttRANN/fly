import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime

# Set global font size
plt.rcParams.update({'font.size': 50})

# Define the tasks with their start and end dates
tasks = [
    {'name': 'Setup Building and Calibration', 'start': datetime.date(2024, 11, 2), 'end': datetime.date(2024, 12, 15)},
    {'name': 'Running Fly Experiments', 'start': datetime.date(2024, 11, 2), 'end': datetime.date(2025, 3, 28)},
    {'name': 'Trajectory Analysis Pipeline', 'start': datetime.date(2024, 11, 2), 'end': datetime.date(2025, 1, 25)},
    {'name': 'Fly Dissection and Brain Imaging', 'start': datetime.date(2024, 11, 15), 'end': datetime.date(2025, 5, 30)},
    {'name': 'Project Wrap-Up', 'start': datetime.date(2025, 4, 15), 'end': datetime.date(2025, 6, 30)},
]

# Create a new figure and axis for the plot
fig, ax = plt.subplots(figsize=(20, 10))

# Loop over the tasks and add them to the plot
for i, task in enumerate(tasks):
    start_date = mdates.date2num(task['start'])
    end_date = mdates.date2num(task['end'])
    duration = end_date - start_date
    ax.barh(i, duration, left=start_date, height=0.5, align='center', color='skyblue')
    # Optional: Add task names inside the bars with larger font size
    # ax.text(start_date + duration/2, i, task['name'], va='center', ha='center', color='white', fontsize=12)

# Customize the y-axis with task names and increase font size
ax.set_yticks(range(len(tasks)))
ax.set_yticklabels([task['name'] for task in tasks], fontsize=14)
ax.invert_yaxis()  # Optional: tasks read top-to-bottom

# Format the x-axis to show dates and increase font size
ax.xaxis_date()
date_format = mdates.DateFormatter('%b %d, %Y')
ax.xaxis.set_major_formatter(date_format)
plt.xticks(rotation=45, fontsize=14)

# Set the x-axis limits
ax.set_xlim(datetime.date(2024, 11, 2), datetime.date(2025, 6, 30))

# Add labels and title with increased font sizes
ax.set_xlabel('Date', fontsize=16)
ax.set_title('Gantt Chart from November 2, 2024, to June 30, 2025', fontsize=18)

# Adjust layout to prevent clipping of tick-labels
plt.tight_layout()

# Display the plot
plt.show()
