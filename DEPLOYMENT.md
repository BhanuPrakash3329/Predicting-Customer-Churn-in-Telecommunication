# Deployment Guide

This guide will help you deploy the Customer Churn Prediction app to a public URL for your resume.

## Option 1: Render (Recommended - Free Tier Available)

1. **Create a Render account** at https://render.com
2. **Connect your GitHub repository**:
   - Push this project to GitHub
   - In Render dashboard, click "New +" → "Web Service"
   - Connect your GitHub repository
3. **Configure the service**:
   - Name: `customer-churn-prediction`
   - Environment: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python app/app.py`
   - Port: Will be automatically set (use `PORT` environment variable)
4. **Deploy**: Click "Create Web Service"
5. **Your app will be live at**: `https://customer-churn-prediction.onrender.com` (or your custom name)

## Option 2: Railway (Free Tier Available)

1. **Create a Railway account** at https://railway.app
2. **Create a new project** and connect your GitHub repository
3. **Railway will auto-detect** Python and install dependencies
4. **Set the start command**: `python app/app.py`
5. **Deploy**: Railway will automatically deploy
6. **Your app will be live** at a Railway-provided URL

## Option 3: Heroku (Requires Credit Card, but has free tier)

1. **Install Heroku CLI** from https://devcenter.heroku.com/articles/heroku-cli
2. **Login to Heroku**: `heroku login`
3. **Create app**: `heroku create your-app-name`
4. **Deploy**: `git push heroku main`
5. **Your app will be live** at `https://your-app-name.herokuapp.com`

## Option 4: PythonAnywhere (Free Tier Available)

1. **Create account** at https://www.pythonanywhere.com
2. **Upload your files** via the Files tab
3. **Create a new Web App** in the Web tab
4. **Configure WSGI file** to point to your Flask app
5. **Reload the web app**

## Local Testing

Before deploying, test locally:
```bash
python app/app.py
```

Then visit: http://localhost:5000

## Important Notes

- Make sure all model files (`model.pkl`, `scaler.pkl`, `feature_names.pkl`) are committed to your repository
- The app uses environment variable `PORT` for deployment platforms
- For production, consider using `gunicorn` instead of Flask's development server

## Using Gunicorn (Production)

For better performance in production:

1. Add to `requirements.txt`: `gunicorn`
2. Update `Procfile`: `web: gunicorn app.app:app --bind 0.0.0.0:$PORT`
