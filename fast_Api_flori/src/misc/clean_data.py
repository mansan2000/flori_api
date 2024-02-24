import pandas as pd
import numpy as np

# Load the data (replace 'your_file.csv' with the actual file path)
data = pd.read_csv('../data/copy.csv')

# Filter out rows where any of the sleep stages is zero or null
data = data[(data['Sleep Analysis [Core] (hr)'] != 0) & (data['Sleep Analysis [Core] (hr)'].notnull()) &
            (data['Sleep Analysis [Deep] (hr)'] != 0) & (data['Sleep Analysis [Deep] (hr)'].notnull()) &
            (data['Sleep Analysis [REM] (hr)'] != 0) & (data['Sleep Analysis [REM] (hr)'].notnull())]

# Calculate the total sleep time (sum of core, deep, and REM sleep)
data['Total_Sleep'] = data['Sleep Analysis [Core] (hr)'] + data['Sleep Analysis [Deep] (hr)'] + data['Sleep Analysis [REM] (hr)']

# Calculate the sleep efficiency (time in bed divided by total sleep time)
# Avoid division by zero by not including rows where Total_Sleep is zero
data = data[data['Total_Sleep'] != 0]
data['Sleep_Efficiency'] = data['Sleep Analysis [In Bed] (hr)'] / data['Total_Sleep']

# Select only the relevant columns, including the date and the new sleep efficiency metric
result = data[['Date', 'Sleep Analysis [In Bed] (hr)', 'Total_Sleep', 'Sleep_Efficiency']]

# Output the results
print(result)
