import re
import matplotlib.pyplot as plt
import numpy as np

# Function to convert scientific notation to float
def sci_to_float(sci_str):
    return float(sci_str)

# Initialize dictionaries to store processing times for each line number
line_processing_times = {}
last_for_loop_times = {}

# Read the .txt file line by line
with open('Line-Wise-Time-Detection.txt', 'r') as file:
    for line in file:
        # Use regex to find lines containing processing times
        match_processing = re.search(r'Line (\d+): ([0-9.e-]+)', line)
        if match_processing:
            line_number = int(match_processing.group(1))
            processing_time = sci_to_float(match_processing.group(2))
            if line_number not in line_processing_times:
                line_processing_times[line_number] = []
            line_processing_times[line_number].append(processing_time)

        # Use regex to find lines containing Last For Loop times
        match_last_for_loop = re.search(r'Last For Loop: ([0-9.e-]+)', line)
        if match_last_for_loop:
            last_for_loop_time = sci_to_float(match_last_for_loop.group(1))
            if 'Last For Loop' not in last_for_loop_times:
                last_for_loop_times['Last For Loop'] = []
            last_for_loop_times['Last For Loop'].append(last_for_loop_time)

# Calculate the average processing time for each line number
line_average_processing_times = {}
for line_number, times in line_processing_times.items():
    line_average_processing_times[line_number] = np.mean(times)

# Calculate the average processing time for "Last For Loop" lines
last_for_loop_average_time = np.mean(last_for_loop_times['Last For Loop'])

# Extract line numbers and corresponding average times
line_numbers = list(line_average_processing_times.keys())
average_times = list(line_average_processing_times.values())

# Append "Last For Loop" as a special line number
line_numbers.append('Last For Loop')
average_times.append(last_for_loop_average_time)

# Plot time vs. line number
plt.plot(line_numbers, average_times, marker='o', linestyle='-')
plt.xlabel('Line Number')
plt.ylabel('Average Processing Time')
plt.title('Average Processing Time for Each Line')
plt.grid(True)

# Show the plot
plt.show()
