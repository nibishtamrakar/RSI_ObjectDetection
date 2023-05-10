# Nibish Tamrakar
# Plotting the data to see the trend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

person_count = []
vehicle_count = []
frame_no = []

with open("person_count.txt", "r") as f:
    for line in f:
        frame, person, vehicle = line.split()
        frame_no.append(int(frame))
        person_count.append(int(person))
        vehicle_count.append(int(vehicle))

# print(frame_no)
# print(person_count)
# print(vehicle_count)

# Plot the data as a function of frame_no
plt.plot(frame_no, person_count, label="Person count")
plt.plot(frame_no, vehicle_count, label="Vehicle count")
plt.xlabel("Frame number")
plt.ylabel("Count")
plt.legend()
plt.show()

# Fit linear regression lines to the data
person_count_coeffs = np.polyfit(frame_no, person_count, 1)
vehicle_count_coeffs = np.polyfit(frame_no, vehicle_count, 1)

# Evaluate regression lines for each data point
person_count_regression = np.polyval(person_count_coeffs, frame_no)
vehicle_count_regression = np.polyval(vehicle_count_coeffs, frame_no)

# Plot the data and the regression lines
plt.plot(frame_no, person_count, label="Person count")
plt.plot(frame_no, vehicle_count, label="Vehicle count")
plt.plot(frame_no, person_count_regression, label="Person count trend")
plt.plot(frame_no, vehicle_count_regression, label="Vehicle count trend")

# Add axis labels and a legend
plt.xlabel("Frame number")
plt.ylabel("Count")
plt.legend()

# Display the plot
plt.show()
