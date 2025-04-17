🩺 Disease Prediction and Drug Recommendation System
This project is a Machine Learning-powered web application that predicts diseases based on user symptoms and recommends appropriate drugs. It combines ML classification (Random Forest) with drug data mapping and integrates the Google Gemini API to provide informative descriptions of the predicted diseases.

🔍 Features
✅ Predict disease based on user input symptoms using a trained ML model

💊 Recommend the best drug for the predicted disease using a drug review dataset

🧠 Generate disease descriptions using Gemini API (Gemini-Pro)

🌐 User-friendly interface built with Flask

📊 Two datasets used:

Disease Prediction Dataset: For training the Random Forest model

Drug Review Dataset: For recommending drugs based on disease

🚀 Deployed as a Flask web app

🛠️ Tech Stack
Machine Learning: Random Forest Classifier

Backend: Python, Flask

Frontend: HTML, Bootstrap, JavaScript

API: Gemini-Pro by Google

Deployment: Flask local server

Datasets:

Disease prediction dataset (symptoms → disease)

Drug review dataset (disease → drug info)

📦 How It Works
User enters symptoms on the web interface.

Model predicts the disease using a trained Random Forest Classifier.

A suitable drug is recommended by matching the disease to entries in the drug dataset.

Gemini API generates a short, human-readable description of the predicted disease.

All results are displayed on the result page.

🧪 Example
Input Symptoms: Fever, Cough, Fatigue

Predicted Disease: Influenza

Recommended Drug: Tamiflu

Generated Description: “Influenza is a contagious respiratory illness caused by influenza viruses. It can cause mild to severe illness, and at times can lead to hospitalization or death.”
