# app.py - Student Performance Prediction API

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)

# Load trained model, features list, and encoder
try:
    cls_model = joblib.load('trained_data/model_cls.pkl')
    features = joblib.load('trained_data/model_features.pkl')
    status_encoder = joblib.load('trained_data/status_encoder.pkl')
except FileNotFoundError as e:
    print(f"Error loading model files: {e}. Please run train_model.py first to generate them.")
    exit()

@app.route('/predict_performance', methods=['POST'])
def predict_performance():
    data = request.json # Get JSON data from the request body

    if not data:
        return jsonify({"error": "No JSON data provided in request body."}), 400

    # Define the exact expected input features from the JSON request
    # These must match the features used for training
    expected_numerical_inputs = [
        'Hours Studied',
        'Previous Scores',
        'Sleep Hours',
        'Sample Question Papers Practiced'
    ]
    expected_categorical_inputs = [
        'Extracurricular Activities'
    ]

    # Check if all required inputs are present
    for feature in expected_numerical_inputs + expected_categorical_inputs:
        if feature not in data:
            return jsonify({"error": f"Missing input data for required feature: '{feature}'"}), 400

    # Create a DataFrame from the incoming JSON data
    input_df = pd.DataFrame([data])

    # Convert categorical columns to one-hot encoding
    # This must match the `categorical_cols` and `drop_first` used in train_model.py
    categorical_cols_for_dummies = ['Extracurricular Activities']
    input_encoded = pd.get_dummies(input_df, columns=categorical_cols_for_dummies, drop_first=True)

    # Reindex input_encoded to match the features used during training
    # This ensures consistent column order and fills any missing one-hot encoded columns with 0
    input_encoded = input_encoded.reindex(columns=features, fill_value=0)

    # Predict student performance (Pass/Fail)
    predicted_encoded = cls_model.predict(input_encoded)[0]
    predicted_status = status_encoder.inverse_transform([predicted_encoded])[0]

    # Respond with the prediction
    response = {
        "Predicted_Status": predicted_status
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)