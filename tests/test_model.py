import pickle
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the trained model and scaler
with open("../CUSTOMER-CHURN-PREDICTION/model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

scaler = joblib.load("../CUSTOMER-CHURN-PREDICTION/scaler.pkl")

# Load the test dataset
test_data_path = "C:/Users/ajayr/OneDrive/Desktop/agenix/Customer-Churn-Prediction/Data/customer_churn_data.csv"
test_data = pd.read_csv(test_data_path)

# Print column names for debugging
print(f"Columns in test data: {test_data.columns}")

# Drop non-feature columns
X_test = test_data.drop(columns=['CustomerID', 'ChurnStatus'])
y_test = test_data['ChurnStatus']

# One-hot encoding for categorical variables
X_test = pd.get_dummies(X_test)

# Ensure test features match training features
expected_features_path = "../CUSTOMER-CHURN-PREDICTION/feature_names.pkl"

try:
    with open(expected_features_path, "rb") as f:
        expected_features = pickle.load(f)  # Load feature names used in training
except FileNotFoundError:
    raise FileNotFoundError(f"Feature names file not found: {expected_features_path}")

# Align test data columns with training features
X_test = X_test.reindex(columns=expected_features, fill_value=0)

# Scale the test data
X_test_scaled = scaler.transform(X_test)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Convert labels ('Yes', 'No') to (1, 0)
if y_test.dtype == 'object':  
    y_test = y_test.map({'No': 0, 'Yes': 1})
    y_pred = pd.Series(y_pred).map({'No': 0, 'Yes': 1})

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=1)
recall = recall_score(y_test, y_pred, zero_division=1)
f1 = f1_score(y_test, y_pred, zero_division=1)

# Display results
print("Model Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")




