# Quick Start Guide

## 🚀 Your App is Running!

**Local URL**: http://localhost:5000

The Customer Churn Prediction app is now running with a beautiful web interface!

## 📋 Features

- **Interactive Web Interface**: Fill out a form to predict customer churn
- **REST API**: Use `/predict` endpoint for programmatic access
- **Machine Learning Model**: Trained model for accurate predictions

## 🌐 Deploy to Get a Public URL

To get a public link for your resume, deploy to one of these platforms:

### **Render.com** (Easiest - Recommended)
1. Go to https://render.com and sign up (free)
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Use these settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python app/app.py`
5. Deploy! You'll get a URL like: `https://your-app.onrender.com`

### **Railway.app** (Also Easy)
1. Go to https://railway.app and sign up (free)
2. Create new project from GitHub
3. Railway auto-detects everything
4. Deploy! You'll get a URL automatically

See `DEPLOYMENT.md` for detailed instructions.

## 🧪 Test the API

You can test the prediction API with:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 35,
    "Location": "Urban",
    "PlanType": "Premium",
    "AverageCallDuration": 20.5,
    "DataUsage": 15.2,
    "NumberOfCalls": 150,
    "MonthlyCharges": 85.50,
    "PaymentMethod": "CreditCard"
  }'
```

## 📝 For Your Resume

Once deployed, you can add:
- **Project Link**: [Your deployed URL]
- **Description**: "Customer Churn Prediction web application using machine learning (Flask, scikit-learn)"
