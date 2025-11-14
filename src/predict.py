import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from data_preprocessing import clean_data, encode_features  # Import only relevant functions
#from model_training import train_model
from sklearn.linear_model import LogisticRegression

def save_model(model, scaler, model_path="model.pkl", scaler_path="scaler.pkl"):
    """Save trained model and scaler."""
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print("Model and scaler saved successfully!")

def load_model(model_path="model.pkl", scaler_path="scaler.pkl"):
    """Load trained model and scaler."""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def preprocess_new_data(new_data, scaler):
    """Preprocess new customer data (without requiring file path)."""
    df = pd.DataFrame([new_data]) if isinstance(new_data, dict) else pd.DataFrame(new_data)

    # Apply the same preprocessing as training data
    df = clean_data(df)
    df = encode_features(df)

    # Ensure columns match training data (fix missing columns)
    expected_columns = scaler.feature_names_in_  # Get columns from scaler
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0  # Add missing columns with default value

    df = df[expected_columns]  # Reorder columns

    # Scale features
    df_scaled = pd.DataFrame(scaler.transform(df), columns=scaler.feature_names_in_)


    return df_scaled

def make_prediction(new_data, model, scaler):
    """Make predictions on new data."""
    X_new = preprocess_new_data(new_data, scaler)
    prediction = model.predict(X_new)
    return prediction

if __name__ == "__main__":
    # Load the trained model and scaler
    model, scaler = load_model()

    # Example: Predicting churn for a new customer
    new_customer = {
        "Location": "Urban",
        "PlanType": "Premium",
        "PaymentMethod": "Credit Card",
        "Age": 35,
        "MonthlyBill": 75.0,
        "TotalUsage": 500
    }

    prediction = make_prediction(new_customer, model, scaler)
    print("Churn Prediction:", "Yes" if prediction[0] == 1 else "No")

