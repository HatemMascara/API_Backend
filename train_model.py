# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.pipeline import Pipeline
import joblib
import os

# 📁 Ensure folder exists
os.makedirs("trained_data", exist_ok=True)

# 📥 Load CSV
df = pd.read_csv('csv/Diet_Recommendations.csv')

# ✅ Compute BMI
df['BMI'] = df['Weight_kg'] / ((df['Height_cm'] / 100) ** 2)

# ✅ BMI Category
def get_bmi_category(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi < 25:
        return 'Normal'
    elif 25 <= bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'

df['BMI_Category'] = df['BMI'].apply(get_bmi_category)

# ✅ Assign goal based on BMI
def assign_goal(bmi):
    if bmi < 18.5:
        return 'Gain'
    elif 18.5 <= bmi < 25:
        return 'Maintain'
    else:
        return 'Lose'

df['Goal'] = df['BMI'].apply(assign_goal)

# 🎯 Keep relevant features
selected_columns = [
    'Age', 'Gender', 'Height_cm', 'Weight_kg',
    'Smoking_Habit', 'Dietary_Habits', 'BMI', 'BMI_Category', 'Goal',
    'Recommended_Calories', 'Recommended_Protein',
    'Recommended_Carbs', 'Recommended_Fats', 'Recommended_Meal_Plan'
]
df = df[selected_columns]

# 🎯 Separate targets
y_reg = df[['Recommended_Calories', 'Recommended_Protein', 'Recommended_Carbs', 'Recommended_Fats']]

# Encode classification target
meal_plan_encoder = LabelEncoder()
y_cls = meal_plan_encoder.fit_transform(df['Recommended_Meal_Plan'])
joblib.dump(meal_plan_encoder, 'trained_data/meal_plan_encoder.pkl')

# Prepare input features
X_raw = df.drop(columns=[
    'Recommended_Calories', 'Recommended_Protein',
    'Recommended_Carbs', 'Recommended_Fats', 'Recommended_Meal_Plan'
])
X = pd.get_dummies(X_raw)
joblib.dump(X.columns.tolist(), 'trained_data/model_features.pkl')

# 📊 Train regression model
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)
reg_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', max_iter=2000, early_stopping=True, random_state=42))
])
reg_pipeline.fit(X_train_r, y_train_r)
joblib.dump(reg_pipeline, 'trained_data/model_reg.pkl')

# 🔤 Train classification model
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_cls, test_size=0.2, random_state=42)
cls_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', MLPClassifier(hidden_layer_sizes=(64,), activation='relu', max_iter=2000, early_stopping=True, random_state=42))
])
cls_pipeline.fit(X_train_c, y_train_c)
joblib.dump(cls_pipeline, 'trained_data/model_cls.pkl')

print("✅ Training complete. All models saved to 'trained_data/'")
