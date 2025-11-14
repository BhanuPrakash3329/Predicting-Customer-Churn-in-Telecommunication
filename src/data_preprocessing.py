import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

def load_data(file_path):
    """Load dataset from CSV file."""
    return pd.read_csv(file_path)

def clean_data(data):
    """Clean dataset by handling missing values and removing irrelevant columns."""
    # Drop CustomerID if present
    if 'CustomerID' in data.columns:
        data = data.drop(columns=['CustomerID'])
    
    # Fill missing numerical values with median
    for col in data.select_dtypes(include=['number']).columns:
        data[col] = data[col].fillna(data[col].median())

    
    # Fill missing categorical values with mode
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = data[col].fillna(data[col].mode()[0])

    
    return data

def encode_features(data):
    """Convert categorical variables into numerical format using one-hot encoding and label encoding."""
    label_encoder = LabelEncoder()
    
    if 'ChurnStatus' in data.columns:
        data['ChurnStatus'] = label_encoder.fit_transform(data['ChurnStatus'])  # Encode Yes/No as 1/0
    
    # One-hot encode categorical features if they exist in the dataset
    categorical_cols = ['Location', 'PlanType', 'PaymentMethod']
    available_cols = [col for col in categorical_cols if col in data.columns]
    
    if available_cols:
        data = pd.get_dummies(data, columns=available_cols, drop_first=True)
    
    return data

def scale_features(X_train, X_test):
    """Scale numerical features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def balance_data(X, y):
    """Apply SMOTE to handle class imbalance."""
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

def preprocess_pipeline(file_path):
    """Complete preprocessing pipeline for data preparation."""
    data = load_data(file_path)
    data = clean_data(data)
    data = encode_features(data)
    
    # Splitting features and target variable
    X = data.drop(columns=['ChurnStatus'])
    y = data['ChurnStatus']
    
    return X, y

if __name__ == "__main__":
    file_path = "C:/Users/ajayr/OneDrive/Desktop/agenix/Customer-Churn-Prediction/data/customer_churn_data.csv"  # Adjust as needed
    X, y = preprocess_pipeline(file_path)
    print("Data preprocessing completed successfully!")

