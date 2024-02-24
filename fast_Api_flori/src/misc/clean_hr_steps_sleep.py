import pandas as pd
import numpy as np

# Load the data (replace 'sleep_data.csv' with the actual file path)
data = pd.read_csv('../data/copy2.csv')

# Convert '<null>' strings to actual NaN values
data.replace('<null>', np.nan, inplace=True)

# Drop rows where any of the sleep stages is NaN
data.dropna(subset=['Sleep Analysis [Core] (hr)', 'Sleep Analysis [Deep] (hr)', 'Sleep Analysis [REM] (hr)'], inplace=True)

# Calculate the total sleep time (sum of core, deep, and REM sleep)
data['Total_Sleep'] = data['Sleep Analysis [Core] (hr)'] + data['Sleep Analysis [Deep] (hr)'] + data['Sleep Analysis [REM] (hr)']

# Calculate the sleep efficiency (time in bed divided by total sleep time)
# Avoid division by zero by not including rows where Total_Sleep is zero
data = data[data['Total_Sleep'] != 0]
data['Sleep_Efficiency'] = data['Total_Sleep'] / data['Sleep Analysis [In Bed] (hr)']

# Select only the relevant columns, including the date, resting heart rate, step count, and the new sleep efficiency metric
print(data)
# List of desired columns
desired_columns = ['Date', 'Resting Heart Rate (bpm)', 'Step Count (steps)', 'Sleep_Efficiency']

# Filter the list of desired columns to only those that actually exist in the DataFrame
existing_columns = [col for col in desired_columns if col in data.columns]

# Now, select only the existing columns
final_data = data[existing_columns]
print(final_data.head())

# Export the cleaned and organized data to a new CSV file (replace 'cleaned_sleep_data.csv' with the desired output file path)
final_data.to_csv('../data/cleaned_sleep_data2.csv', index=False)

print("Data cleaned and exported successfully.")
