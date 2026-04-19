# app.py
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained Random Forest model
model = joblib.load("hotel_model_rf.pkl")

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    # 1. Get input from the form
    lead_time = float(request.form["lead_time"])
    special_requests = float(request.form["special_requests"])
    market_segment = float(request.form["market_segment"])
    
    # 2. Format features EXACTLY as trained
    features = np.array([[lead_time, special_requests, market_segment]])
    
    # 3. Predict the PROBABILITY instead of just 0 or 1
    # predict_proba returns [[prob_stay, prob_cancel]]
    prediction_prob = model.predict_proba(features)[0]
    cancel_probability = round(prediction_prob[1] * 100, 1) # Convert to percentage (e.g., 85.5)
    
    # 4. Create Advanced Risk Tiers based on the percentage
    if cancel_probability >= 70:
        result = "High Cancellation Risk"
        risk_level = "high"
    elif cancel_probability >= 40:
        result = "Medium Risk"
        risk_level = "medium"
    else:
        result = "Secure Booking"
        risk_level = "low"
        
    # Return the result, the risk tier, and the exact percentage
    return render_template("home.html", 
                           prediction_text=result, 
                           risk_level=risk_level, 
                           probability=cancel_probability)

if __name__ == "__main__":
    app.run(debug=True)