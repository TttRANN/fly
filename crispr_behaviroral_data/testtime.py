# # Python program to illustrate fromtimestamp()
# from datetime import datetime

# timestamp = 5700829060350
# # converting timestamp to date
# dt_object = datetime.fromtimestamp(timestamp)

# print('Date and Time is:', dt_object)


# You have a millisecond-precise timestamp so first divide it by 1000 then feed it to datetime.datetime.fromtimestamp() for local timezone (or pass datetime.tzinfo of the target timezone as a second argument) or datetime.datetime.utcfromtimestamp() for UTC. Finally, use datetime.datetime.strftime() to turn it into a string of your desired format.

import datetime

# Sample timestamp in milliseconds since start of the day
timestamp = "5700846837528"

# Convert timestamp from milliseconds to seconds
timestamp_seconds = int(timestamp) / 1000

# Get today's date
today = datetime.datetime.now().date()

# Create a datetime object for the start of today
start_of_day = datetime.datetime.combine(today, datetime.time.min)

# Add the duration to the start of the day
your_dt = start_of_day + datetime.timedelta(seconds=timestamp_seconds)

# Print the datetime in a formatted string
print(your_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])  # Print milliseconds up to three digits

