# train_model.py - Student Performance Classification

# 📦 Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier # Only need Classifier for Pass/Fail
from sklearn.pipeline import Pipeline
import joblib
import os

# Define the passing threshold for Performance Index
PERFORMANCE_THRESHOLD = 50 # Students with Performance Index >= 50 will be 'Pass'

# 📁 Make sure the folder for saving trained models exists
os.makedirs("trained_data", exist_ok=True)

# 📥 Load the student performance dataset
try:
    df = pd.read_csv('csv/Student_Performance.csv')
except FileNotFoundError:
    print("Error: 'csv/Student_Performance.csv' not found. Make sure the CSV is in the 'csv' folder relative to this script.")
    exit()

# ✅ Create the 'Pass'/'Fail' target column based on Performance Index
df['Pass_Fail_Status'] = df['Performance Index'].apply(
    lambda x: 'Pass' if x >= PERFORMANCE_THRESHOLD else 'Fail'
)

# 🎯 Define the target variable
TARGET_COLUMN = 'Pass_Fail_Status'

# 🔤 Encode the classification target from text labels to numbers ('Pass' -> 1, 'Fail' -> 0)
status_encoder = LabelEncoder()
y_cls = status_encoder.fit_transform(df[TARGET_COLUMN])
# Save the encoder so it can be reused during predictions
joblib.dump(status_encoder, 'trained_data/status_encoder.pkl')

# 🔢 Prepare input features
# These are the columns from the CSV that will be used to predict Pass/Fail
input_features = [
    'Hours Studied',
    'Previous Scores',
    'Extracurricular Activities',
    'Sleep Hours',
    'Sample Question Papers Practiced'
]
X_raw = df[input_features]

# Identify categorical columns for one-hot encoding
categorical_cols = X_raw.select_dtypes(include='object').columns.tolist()
# In this case, it's just 'Extracurricular Activities'
print(f"Detected categorical columns for one-hot encoding: {categorical_cols}")


# 🧠 Convert categorical variables into numerical format using one-hot encoding
X = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=True) # drop_first avoids multicollinearity
# Save the list of feature names. This is critical for the app.py to match input features
joblib.dump(X.columns.tolist(), 'trained_data/model_features.pkl')

# 📊 Split the data into training and test sets
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_cls, test_size=0.2, random_state=42)

# 🛠 Create a pipeline for classification: preprocessing + model training
cls_pipeline = Pipeline([
    ('scaler', StandardScaler()), # Standardize numerical features
    ('classifier', MLPClassifier(
        hidden_layer_sizes=(100, 50), # Example layers, adjust as needed
        activation='relu',
        max_iter=2000,
        early_stopping=True,
        random_state=42
    ))
])

# 🧠 Train the classification model
print("Training classification model...")
cls_pipeline.fit(X_train_c, y_train_c)
joblib.dump(cls_pipeline, 'trained_data/model_cls.pkl') # Save the trained classification model

# ✅ Print final message
print("✅ Training complete. Classification model saved to 'trained_data/model_cls.pkl'")
print(f"Model accuracy on test set: {cls_pipeline.score(X_test_c, y_test_c):.2f}")
