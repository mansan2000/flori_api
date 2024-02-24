from pycaret.regression import predict_model, load_model, setup
import os


def load_model_and_predict(new_data, model):
    # Load the model and predict
    # print(os.getcwd())getcwd
    loaded_model = load_model(os.getcwd()+'/src/models/automl_model_from_api')
    columns_with_percentages = ["Regularity"]  # Replace with the actual names of your columns

    for column in columns_with_percentages:
        new_data[column] = new_data[column].str.replace('%', '').astype(float) / 100
    # s = setup(new_data, target="Sleep Quality")
    predictions = predict_model(loaded_model, data=new_data)
    # Rename columns for clarity
    # predictions.columns = ['Date', 'Feature1', 'Feature2', 'prediction_label']
    predictions.rename(columns={'Label': 'prediction_label'}, inplace=True)
    return predictions
