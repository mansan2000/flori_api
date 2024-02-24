import pandas as pd
from sklearn.model_selection import train_test_split
from tpot import TPOTRegressor


# Load the data
data = pd.read_csv('../data/cleaned_sleep_data2.csv')

# Prepare the independent and dependent variables
X = data[['Resting Heart Rate (bpm)', 'Step Count (steps)']]
y = data['Sleep_Efficiency']
# X = data[['Step Count (steps)', 'Sleep_Efficiency']]
# y = data['Resting Heart Rate (bpm)']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


tpot = TPOTRegressor(
    generations=5,  # How many iterations to run
    population_size=20,  # How many individuals in each generation
    verbosity=2,  # Show progress
    scoring='neg_mean_squared_error',  # MSE as the scoring function
    random_state=42,  # For reproducibility
    n_jobs=-1  # Use all available cores
)
tpot.fit(X_train, y_train)
print("Best pipeline test MSE:", tpot.score(X_test, y_test))
tpot.export('tpot_best_pipeline.py')
