from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load trained model and encoders from the trained_data folder
model = joblib.load('trained_data/student_model.pkl')
encoder_participation = joblib.load('trained_data/encoder_participation.pkl')
encoder_internet = joblib.load('trained_data/encoder_internet.pkl')
encoder_result = joblib.load('trained_data/encoder_result.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract inputs from JSON request
    study_hours = float(data['study_hours'])
    attendance_rate = float(data['attendance_rate'])
    sleep_hours = float(data['sleep_hours'])
    participation = data['participation'].lower()
    has_internet = data['has_internet'].lower()

    # Encode categorical fields
    participation_encoded = encoder_participation.transform([participation])[0]
    internet_encoded = encoder_internet.transform([has_internet])[0]

    # Prepare the input features
    features = np.array([[study_hours, attendance_rate, sleep_hours,
                          participation_encoded, internet_encoded]])

    # Predict using the trained model
    result_encoded = model.predict(features)[0]
    result = encoder_result.inverse_transform([result_encoded])[0]

    # Return a simple response
    return jsonify({
        "prediction": result,
        "message": f"The model predicts the student is likely to **{result.upper()}** the course."
    })

if __name__ == '__main__':
    app.run(debug=True)
