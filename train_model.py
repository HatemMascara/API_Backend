import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load your dataset
data = pd.read_csv("student_performance_dataset.csv")

# Encode categorical columns
le_participation = LabelEncoder()
le_internet = LabelEncoder()
le_result = LabelEncoder()

data['participation'] = le_participation.fit_transform(data['participation'])
data['has_internet'] = le_internet.fit_transform(data['has_internet'])
data['final_result'] = le_result.fit_transform(data['final_result'])

# Select features and label
X = data[['study_hours', 'attendance_rate', 'sleep_hours', 'participation', 'has_internet']]
y = data['final_result']

# Split dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = MLPClassifier(hidden_layer_sizes=(20,), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le_result.classes_))

# Save model and encoders
joblib.dump(model, "trained_data/student_model.pkl")
joblib.dump(le_participation, "trained_data/encoder_participation.pkl")
joblib.dump(le_internet, "trained_data/encoder_internet.pkl")
joblib.dump(le_result, "trained_data/encoder_result.pkl")
