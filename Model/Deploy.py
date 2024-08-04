import joblib
import pandas as pd

# Load the trained model
model = joblib.load('model/insurance_model.pkl')

# Function to deploy the model
def deploy_model(input_data):
    """
    Function to deploy the machine learning model.
    Args:
        input_data (pd.DataFrame): Data for prediction.
    Returns:
        pd.DataFrame: Predictions made by the model.
    """
    predictions = model.predict(input_data)
    return pd.DataFrame(predictions, columns=['Predicted'])

if __name__ == "__main__":
    # Example input data
    data = {
        'age': [25, 32],
        'bmi': [22.2, 30.1],
        'children': [0, 2],
        'smoker': ['no', 'yes'],
        'region': ['southeast', 'northwest']
    }

    input_df = pd.DataFrame(data)
    result = deploy_model(input_df)
    print(result)
