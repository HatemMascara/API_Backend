# train_model.py

# 📦 Import libraries needed for data handling, machine learning, and saving models
import pandas as pd  # Used to load and manipulate tabular data
from sklearn.model_selection import train_test_split  # Used to divide data into training and testing sets
from sklearn.preprocessing import StandardScaler, LabelEncoder  # Used to prepare data for machine learning
from sklearn.neural_network import MLPRegressor, MLPClassifier  # Machine learning models for regression and classification
from sklearn.pipeline import Pipeline  # Allows chaining preprocessing and model steps
import joblib  # Used for saving/loading trained models and encoders
import os  # Used for file and directory operations

# 📁 Make sure the folder for saving trained models exists (create if not)
os.makedirs("trained_data", exist_ok=True)

# 📥 Load the dataset containing personal info and recommended dietary targets
df = pd.read_csv('csv/Diet_Recommendations.csv')

# ✅ Calculate BMI (Body Mass Index) from height and weight
# BMI is a derived feature and will help categorize health condition
#dataframe is different from CSV
df['BMI'] = df['Weight_kg'] / ((df['Height_cm'] / 100) ** 2)

# ✅ Define a function to classify BMI into four common health categories
def get_bmi_category(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi < 25:
        return 'Normal'
    elif 25 <= bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'

# Use the function to create a new column classifying each person's BMI
df['BMI_Category'] = df['BMI'].apply(get_bmi_category)

# ✅ Assign a general health goal (Gain, Maintain, or Lose weight) based on BMI
# This will become one of our input features for training
def assign_goal(bmi):
    if bmi < 18.5:
        return 'Gain'
    elif 18.5 <= bmi < 25:
        return 'Maintain'
    else:
        return 'Lose'

# Add the health goal as a new column in the dataset
df['Goal'] = df['BMI'].apply(assign_goal)

# 🎯 Select only the relevant columns for training and remove anything unnecessary
# Independent variables (inputs): all features that describe the person (e.g., Age, Gender, BMI, Goal)
# Dependent variables (outputs): what we are trying to predict (e.g., Recommended_Calories)
selected_columns = [
    'Age', 'Gender', 'Height_cm', 'Weight_kg',
    'Smoking_Habit', 'Dietary_Habits', 'BMI', 'BMI_Category', 'Goal',
    'Recommended_Calories', 'Recommended_Protein',
    'Recommended_Carbs', 'Recommended_Fats', 'Recommended_Meal_Plan'
]
#select all columns in df
df = df[selected_columns]

# 🎯 Define regression targets — these are continuous values that we want to predict
# These are the dependent variables for the regression model
y_reg = df[['Recommended_Calories', 'Recommended_Protein', 'Recommended_Carbs', 'Recommended_Fats']]

# 🔤 Encode the classification target (Meal Plan) from text labels to numbers
# This is the dependent variable for the classification model
meal_plan_encoder = LabelEncoder()
# Encode text like "Keto", "Balanced", etc. to 0, 1, 2...
y_cls = meal_plan_encoder.fit_transform(df['Recommended_Meal_Plan'])  
# Save encoder so it can be reused during predictions
#dump is save
joblib.dump(meal_plan_encoder, 'trained_data/meal_plan_encoder.pkl')  

# 🔢 Prepare input features by removing the outputs (targets)
# These are the independent variables that the model will learn from
# we will drop the dependent variables muna
X_raw = df.drop(columns=[
    'Recommended_Calories', 'Recommended_Protein',
    'Recommended_Carbs', 'Recommended_Fats', 'Recommended_Meal_Plan'
])

# 🧠 Convert all categorical variables into numerical format using one-hot encoding
X = pd.get_dummies(X_raw)  # For example, "Gender: Male" becomes "Gender_Male = 1"
joblib.dump(X.columns.tolist(), 'trained_data/model_features.pkl')  # Save list of features so future inputs match this format

# 📊 Split the data into training and test sets for the regression model
# This ensures the model is evaluated on unseen data
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)
#X = Independent
#y_reg = Dependent 
#20% for testing
#train_test_split automatically shuffles
#ramdom state fixed the way its shuffled
# (
#   X_train_r,  # Training inputs
#   X_test_r,   # Testing inputs
#   y_train_r,  # Training outputs (calories, protein, etc.)
#   y_test_r    # Testing outputs
# )


# 🧠 Define a regression pipeline that includes both preprocessing and model training steps
reg_pipeline = Pipeline([
    
    # 🔹 Step 1: Standardize the input features
    # Why? Because neural networks perform better when inputs are scaled — especially when features (like Age, Weight, Height) are on different units or scales.
    # StandardScaler transforms each feature so it has:
    #   - mean = 0
    #   - standard deviation = 1
    # This helps the neural network converge faster and avoids one feature dominating the learning process.
    ('scaler', StandardScaler()),

    # 🔹 Step 2: MLPRegressor — a multi-layer perceptron for regression tasks
    ('regressor', MLPRegressor(
        
        # 🧱 hidden_layer_sizes=(64, 32)
        # This creates a neural network with:
        #   - 1st hidden layer: 64 neurons
        #   - 2nd hidden layer: 32 neurons
        # These layers allow the network to learn complex patterns from the data.
        # More neurons = more learning capacity (but also more risk of overfitting if too big).
        hidden_layer_sizes=(64, 32),

        # ⚡ activation='relu'
        # Activation function for the hidden layers.
        # ReLU (Rectified Linear Unit) is defined as f(x) = max(0, x)
        # ReLU adds non-linearity to the model and avoids vanishing gradients — it's the most popular activation today.
        activation='relu',

        # 🔁 max_iter=2000
        # Maximum number of iterations (training steps) allowed.
        # If the model hasn’t converged (minimized loss) after 2000 steps, it will stop.
        # Re run the flow 2k times called epoch
        max_iter=2000,

        # ⛔ early_stopping=True
        # Stops training if validation score doesn’t improve over time (helps avoid overfitting).
        # Internally, it reserves 10% of the training data to evaluate this.
        # Internal 90/10 like 80/20
        early_stopping=True,

        # 🔒 random_state=42
        # Sets a fixed "random seed" so the neural network’s training process is reproducible.
        # Without this, the random initialization of weights would cause different results every run.
        random_state=42
    ))
])


# 🧠 Train the regression model on the training data
reg_pipeline.fit(X_train_r, y_train_r)
joblib.dump(reg_pipeline, 'trained_data/model_reg.pkl')  # Save the trained regression model

# 🔤 Split data for classification model (predicting meal plan category)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_cls, test_size=0.2, random_state=42)

# 🛠 Create a pipeline for classification: preprocessing + training
cls_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Normalize features for better performance
    ('classifier', MLPClassifier(
        hidden_layer_sizes=(64,),   # One hidden layer with 64 neurons
        activation='relu',          # Activation function for hidden layer
        max_iter=2000,              # Maximum training cycles
        early_stopping=True,        # Stop early if model is not improving
        random_state=42             # Fix seed for reproducibility
    ))
])

# 🧠 Train the classification model
cls_pipeline.fit(X_train_c, y_train_c)
joblib.dump(cls_pipeline, 'trained_data/model_cls.pkl')  # Save the trained classification model

# ✅ Print final message to indicate all models are ready
print("✅ Training complete. All models saved to 'trained_data/'")
