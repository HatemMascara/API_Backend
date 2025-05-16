import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from imblearn.over_sampling import RandomOverSampler  # Import RandomOverSampler

# Load your dataset
data = pd.read_csv("student_performance_dataset.csv")

# Encode categorical columns (like participation, has_internet, and final_result)
le_participation = LabelEncoder()
le_internet = LabelEncoder()
le_result = LabelEncoder()

data['participation'] = le_participation.fit_transform(data['participation'])
data['has_internet'] = le_internet.fit_transform(data['has_internet'])
data['final_result'] = le_result.fit_transform(data['final_result'])

# Feature selection: we're keeping these columns
X = data[['study_hours', 'attendance_rate', 'sleep_hours', 'participation', 'has_internet']]
y = data['final_result']

# Optional: Re-weight the features to prioritize other columns (like attendance_rate, participation)
X['attendance_rate'] *= 2  # Give attendance more importance
X['participation'] *= 2    # Emphasize participation in class
X['has_internet'] *= 2     # Give internet access more influence

# Scale features using StandardScaler to normalize all numerical values
scaler = StandardScaler()
X[['study_hours', 'attendance_rate', 'sleep_hours']] = scaler.fit_transform(
    X[['study_hours', 'attendance_rate', 'sleep_hours']].copy()
)

# Apply Random OverSampling to balance the dataset (helps with class imbalance)
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Split dataset for training and testing (stratified split for balanced classes)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)

# Build the neural network model (MLPClassifier)
model = MLPClassifier(hidden_layer_sizes=(20,), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Evaluate model performance on the test set
y_pred = model.predict(X_test)

# Output accuracy and classification report
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le_result.classes_))

# Save the model and encoders to a directory (ensure directory exists)
os.makedirs("trained_data", exist_ok=True)
joblib.dump(model, "trained_data/student_model.pkl")
joblib.dump(le_participation, "trained_data/encoder_participation.pkl")
joblib.dump(le_internet, "trained_data/encoder_internet.pkl")
joblib.dump(le_result, "trained_data/encoder_result.pkl")
