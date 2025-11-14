import pickle
import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Load trained model, scaler, and feature names
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

scaler = joblib.load("scaler.pkl")

with open("feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Customer Churn Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON request data
        data = request.get_json()
        
        # Convert data to DataFrame
        df = pd.DataFrame([data])
        
        # One-hot encoding for categorical features
        df = pd.get_dummies(df)
        
        # Align test features with training features
        df = df.reindex(columns=feature_names, fill_value=0)
        
        # Scale features
        df_scaled = scaler.transform(df)
        
        # Make prediction
        prediction = model.predict(df_scaled)[0]
        
        # Convert prediction to human-readable format
        result = "Yes" if prediction == 1 else "No"
        
        return jsonify({"churn_prediction": result})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
