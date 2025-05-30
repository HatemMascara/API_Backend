from flask import Flask, request, jsonify
from flask_cors import CORS  # ✅ Import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)  # ✅ Enable CORS for all domains (you can restrict if needed)

# Load trained models and configs
reg_model = joblib.load('trained_data/model_reg.pkl')
cls_model = joblib.load('trained_data/model_cls.pkl')
features = joblib.load('trained_data/model_features.pkl')
meal_plan_encoder = joblib.load('trained_data/meal_plan_encoder.pkl')

# Helper logic
def get_bmi_category(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi < 25:
        return 'Normal'
    elif 25 <= bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'

def assign_goal(bmi):
    if bmi < 18.5:
        return 'Gain'
    elif 18.5 <= bmi < 25:
        return 'Maintain'
    else:
        return 'Lose'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Compute BMI
    height_m = data['Height_cm'] / 100
    bmi = data['Weight_kg'] / (height_m ** 2)
    data['BMI'] = round(bmi, 2)
    data['BMI_Category'] = get_bmi_category(bmi)
    data['Goal'] = assign_goal(bmi)

    # Prepare input
    input_df = pd.DataFrame([data])
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=features, fill_value=0)

    # Predict
    reg_result = reg_model.predict(input_encoded)[0]
    cls_encoded = cls_model.predict(input_encoded)[0]
    meal_plan = meal_plan_encoder.inverse_transform([cls_encoded])[0]

    # Respond
    response = {
        "BMI": data['BMI'],
        "Goal": data['Goal'],
        "Recommended_Calories": round(reg_result[0], 2),
        "Recommended_Protein": round(reg_result[1], 2),
        "Recommended_Carbs": round(reg_result[2], 2),
        "Recommended_Fats": round(reg_result[3], 2),
        "Recommended_Meal_Plan": meal_plan
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
