import sys
import os
import pytest
import pandas as pd

# Add the project root directory to the PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Model.m1_regression_model import train_model, predict

# Sample data for testing
data = {
    'age': [19, 18, 28],
    'sex': ['female', 'male', 'female'],
    'bmi': [27.9, 33.77, 33.0],
    'children': [0, 1, 3],
    'smoker': ['yes', 'no', 'no'],
    'region': ['southwest', 'southeast', 'southeast'],
    'charges': [16884.924, 1725.5523, 4449.462]
}

# Convert the sample data to a DataFrame
df = pd.DataFrame(data)

def test_train_model():
    # Test if the model training function works without errors
    model = train_model(df)
    assert model is not None

def test_predict():
    # Test if the prediction function works correctly
    model = train_model(df)
    predictions = predict(model, df)
    assert len(predictions) == len(df)
    assert isinstance(predictions, pd.Series)

if __name__ == '__main__':
    pytest.main()