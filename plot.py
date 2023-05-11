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

# Average Plotting
# Create a Pandas DataFrame with the data
df = pd.DataFrame({
    'frame_no': frame_no,
    'person_count': person_count,
    'vehicle_count': vehicle_count
})
# Calculate the rolling averages for each column
window_size = 11
rolling_df = df.rolling(window=window_size).mean().dropna()
# Plot the data and the rolling averages
plt.plot(df['frame_no'], df['person_count'], label="Person count", alpha = 0.5)
plt.plot(df['frame_no'], df['vehicle_count'], label="Vehicle count", alpha = 0.5)
plt.plot(rolling_df['frame_no'], rolling_df['person_count'], label=f"Person count")
plt.plot(rolling_df['frame_no'], rolling_df['vehicle_count'], label=f"Vehicle count)")
# Add axis labels and a legend
plt.xlabel("Frame number")
plt.ylabel("Count")
plt.legend()
# Display the plot
plt.show()

# Calculate the predicted values for each data point
person_count_predicted = np.polyval(person_count_coeffs, frame_no)
vehicle_count_predicted = np.polyval(vehicle_count_coeffs, frame_no)
# Calculate the sum of squared errors (SSE) for each trend line
person_count_sse = np.sum((person_count - person_count_predicted) ** 2)
vehicle_count_sse = np.sum((vehicle_count - vehicle_count_predicted) ** 2)
# Calculate the mean squared error (MSE) for each trend line
person_count_mse = person_count_sse / len(frame_no)
vehicle_count_mse = vehicle_count_sse / len(frame_no)
# Calculate the root mean squared error (RMSE) for each trend line
person_count_rmse = np.sqrt(person_count_mse)
vehicle_count_rmse = np.sqrt(vehicle_count_mse)
# Print the RMSE for each trend line
print(f"Person count RMSE: {person_count_rmse:.2f}")
print(f"Vehicle count RMSE: {vehicle_count_rmse:.2f}")

# Display the plot
plt.show()
