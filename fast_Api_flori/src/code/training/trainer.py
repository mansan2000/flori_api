import pandas as pd
from pycaret.regression import *
from sklearn.model_selection import train_test_split


# load sample dataset
# from pycaret.datasets import get_data
# data = get_data('diabetes')
# Load the data
# data = pd.read_csv('../data/cleaned_sleep_data2.csv')
# data = pd.read_csv('../data/sleepdata.csv')

def train_model(data: pd.DataFrame, resulting_metric: str):
    # resulting_metric = "Sleep Quality"
    # resulting_metric = "Sleep_Efficiency"
    # Assuming 'data' is your DataFrame and 'Sleep_Efficiency' is the target variable
    columns_with_percentages = [resulting_metric, "Regularity"]  # Replace with the actual names of your columns

    print(data)

    for column in columns_with_percentages:
        data[column] = data[column].str.replace('%', '').astype(float) / 100

    X = data.drop(resulting_metric, axis=1)  # Features
    y = data[resulting_metric]  # Target

    # Split the data - 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # Setup PyCaret - note that we're now using the training data
    s = setup(data=pd.concat([X_train, y_train], axis=1), target=resulting_metric, session_id=123, verbose=False)

    # Compare models
    best = compare_models()

    # Create a model
    model = create_model(best)  # replace 'model_id' with the ID of your best model

    # Tune the model (optional)
    tuned_model = tune_model(model)

    # Finalize the model - trains on the entire training dataset including the holdout
    final_model = finalize_model(tuned_model)

    # return final_model
    import os
    print(os.getcwd())

    save_model(final_model, os.getcwd()+'/src/models/automl_model_from_api')
    # try:
    #     save_model(final_model, '../automl_model_34.pkl')
    #     print(f"Model saved successfully")
    # except Exception as e:
    #     print(f"Failed to save the model: {e}")

