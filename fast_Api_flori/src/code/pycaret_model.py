import pandas as pd
from pycaret.regression import *
import src.code.inference.predict as predict
# load sample dataset
# from pycaret.datasets import get_data
# data = get_data('diabetes')
# Load the data
# data = pd.read_csv('../data/cleaned_sleep_data2.csv')
data = pd.read_csv('../data/sleepdata.csv')

from sklearn.model_selection import train_test_split

resulting_metric = "Sleep Quality"
# resulting_metric = "Sleep_Efficiency"
# Assuming 'data' is your DataFrame and 'Sleep_Efficiency' is the target variable
columns_with_percentages = [resulting_metric, "Regularity"]  # Replace with the actual names of your columns

for column in columns_with_percentages:
    data[column] = data[column].str.replace('%', '').astype(float) / 100

X = data.drop(resulting_metric, axis=1) # Features
y = data[resulting_metric] # Target

# Split the data - 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)



# Setup PyCaret - note that we're now using the training data
s = setup(data=pd.concat([X_train, y_train], axis=1), target=resulting_metric, session_id=123, verbose=False)

# Compare models
best = compare_models()

# Create a model
model = create_model(best) # replace 'model_id' with the ID of your best model

# Tune the model (optional)
tuned_model = tune_model(model)

# Finalize the model - trains on the entire training dataset including the holdout
final_model = finalize_model(tuned_model)

save_model(final_model, '../models/automl_model')
# Predict on the test set
# predictions = predict_model(final_model, data=X_test)
predictions = predict.load_model_and_predict(X_test, final_model)
# Rename columns for clarity
# predictions.columns = ['Date', 'Feature1', 'Feature2', 'prediction_label']
predictions.rename(columns={'Label': 'prediction_label'}, inplace=True)


print(predictions)


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Calculate performance metrics
mae = mean_absolute_error(y_test, predictions['prediction_label'])
mse = mean_squared_error(y_test, predictions['prediction_label'])
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions['prediction_label'])

print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R²: {r2}")

# Evaluate predictions
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# # Correctly accessing the predicted values
# mae = mean_absolute_error(y_test, predictions['Label'])
# mse = mean_squared_error(y_test, predictions['Label'])
# rmse = mse ** 0.5  # Root Mean Squared Error
# r2 = r2_score(y_test, predictions['Label'])
#
# print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R²: {r2}")














# from pycaret.regression import *
# s = setup(data, target = 'Sleep_Efficiency', session_id = 123)
#
#
# best = compare_models()
#
# save_model(best, '../models/best_model')
#
#
#
#
#
# print(best)


# import pandas as pd
#
# # Example new data instance
# new_instance = {'feature1': [value1], 'feature2': [value2], ..., 'featureN': [valueN]}
#
# # Convert the instance into a DataFrame
# new_data = pd.DataFrame(new_instance)
#
# # Load the model and predict
# loaded_model = load_model('best_model')
# predictions = predict_model(loaded_model, data=new_data)
#
# # Display the prediction for the new instance
# print(predictions)
