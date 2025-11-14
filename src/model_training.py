import pickle
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load dataset
data_path = "C:/Users/ajayr/OneDrive/Desktop/agenix/Customer-Churn-Prediction/Data/customer_churn_data.csv"
df = pd.read_csv(data_path)

# Split features and target variable
X = df.drop(columns=['CustomerID', 'ChurnStatus'])
y = df['ChurnStatus']

# One-hot encoding for categorical variables
X = pd.get_dummies(X, columns=['Location', 'PaymentMethod', 'PlanType'])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Save model properly
model_path = "C:/Users/ajayr/OneDrive/Desktop/agenix/Customer-Churn-Prediction/model.pkl"
scaler_path = "C:/Users/ajayr/OneDrive/Desktop/agenix/Customer-Churn-Prediction/scaler.pkl"
feature_names_path = "C:/Users/ajayr/OneDrive/Desktop/agenix/Customer-Churn-Prediction/feature_names.pkl"

with open(model_path, "wb") as f:
    pickle.dump(model, f)

# Save scaler properly
joblib.dump(scaler, scaler_path)

# Save feature names used during training
with open(feature_names_path, "wb") as f:
    pickle.dump(X_train.columns.tolist(), f)

print("Model, scaler, and feature names saved successfully!")


