import os
import pickle
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

# Get the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

# Load trained model, scaler, and feature names
model_path = os.path.join(ROOT_DIR, "model.pkl")
scaler_path = os.path.join(ROOT_DIR, "scaler.pkl")
feature_names_path = os.path.join(ROOT_DIR, "feature_names.pkl")

with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

scaler = joblib.load(scaler_path)

with open(feature_names_path, "rb") as f:
    feature_names = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Customer Churn Prediction</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            max-width: 600px;
            width: 100%;
            padding: 40px;
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            color: #333;
            font-weight: 600;
        }
        input, select {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        input:focus, select:focus {
            outline: none;
            border-color: #667eea;
        }
        .btn {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        .btn:active {
            transform: translateY(0);
        }
        .result {
            margin-top: 30px;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            display: none;
        }
        .result.success {
            background: #d4edda;
            color: #155724;
            border: 2px solid #c3e6cb;
        }
        .result.danger {
            background: #f8d7da;
            color: #721c24;
            border: 2px solid #f5c6cb;
        }
        .result h3 {
            margin-bottom: 10px;
        }
        .loading {
            display: none;
            text-align: center;
            margin-top: 20px;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔮 Customer Churn Prediction</h1>
        <p class="subtitle">Predict customer churn using machine learning</p>
        
        <form id="predictionForm">
            <div class="form-group">
                <label for="age">Age</label>
                <input type="number" id="age" name="Age" required min="18" max="100">
            </div>
            
            <div class="form-group">
                <label for="location">Location</label>
                <select id="location" name="Location" required>
                    <option value="">Select Location</option>
                    <option value="Urban">Urban</option>
                    <option value="Rural">Rural</option>
                    <option value="Suburban">Suburban</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="planType">Plan Type</label>
                <select id="planType" name="PlanType" required>
                    <option value="">Select Plan Type</option>
                    <option value="Basic">Basic</option>
                    <option value="Standard">Standard</option>
                    <option value="Premium">Premium</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="averageCallDuration">Average Call Duration (minutes)</label>
                <input type="number" id="averageCallDuration" name="AverageCallDuration" required step="0.01" min="0">
            </div>
            
            <div class="form-group">
                <label for="dataUsage">Data Usage (GB)</label>
                <input type="number" id="dataUsage" name="DataUsage" required step="0.01" min="0">
            </div>
            
            <div class="form-group">
                <label for="numberOfCalls">Number of Calls</label>
                <input type="number" id="numberOfCalls" name="NumberOfCalls" required min="0">
            </div>
            
            <div class="form-group">
                <label for="monthlyCharges">Monthly Charges ($)</label>
                <input type="number" id="monthlyCharges" name="MonthlyCharges" required step="0.01" min="0">
            </div>
            
            <div class="form-group">
                <label for="paymentMethod">Payment Method</label>
                <select id="paymentMethod" name="PaymentMethod" required>
                    <option value="">Select Payment Method</option>
                    <option value="BankTransfer">Bank Transfer</option>
                    <option value="CreditCard">Credit Card</option>
                    <option value="DebitCard">Debit Card</option>
                    <option value="DigitalWallet">Digital Wallet</option>
                </select>
            </div>
            
            <button type="submit" class="btn">Predict Churn</button>
        </form>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Processing prediction...</p>
        </div>
        
        <div class="result" id="result">
            <h3 id="resultTitle"></h3>
            <p id="resultMessage"></p>
        </div>
    </div>
    
    <script>
        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const data = {};
            formData.forEach((value, key) => {
                data[key] = isNaN(value) ? value : parseFloat(value);
            });
            
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');
            const resultTitle = document.getElementById('resultTitle');
            const resultMessage = document.getElementById('resultMessage');
            
            loading.style.display = 'block';
            result.style.display = 'none';
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data)
                });
                
                const json = await response.json();
                
                loading.style.display = 'none';
                result.style.display = 'block';
                
                if (json.error) {
                    result.className = 'result danger';
                    resultTitle.textContent = 'Error';
                    resultMessage.textContent = json.error;
                } else {
                    const willChurn = json.churn_prediction === 'Yes';
                    result.className = willChurn ? 'result danger' : 'result success';
                    resultTitle.textContent = willChurn ? '⚠️ Customer Will Churn' : '✅ Customer Will Stay';
                    resultMessage.textContent = willChurn 
                        ? 'This customer is predicted to churn. Consider retention strategies.'
                        : 'This customer is predicted to remain with the service.';
                }
            } catch (error) {
                loading.style.display = 'none';
                result.style.display = 'block';
                result.className = 'result danger';
                resultTitle.textContent = 'Error';
                resultMessage.textContent = 'Failed to get prediction: ' + error.message;
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

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
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
