import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load saved artifacts
MODEL_PATH = "model.pkl"
FEATURES_PATH = "features.pkl"

model = None
features = []

def load_artifacts():
    global model, features
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    else:
        print(f"Warning: {MODEL_PATH} not found.")

    if os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH, "rb") as f:
            features = pickle.load(f)
    else:
        # Default fallback matching Section 6 of your script
        features = ["store", "item", "Year", "Month", "Day", "DayOfWeek"]

load_artifacts()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model file not found. Please ensure model.pkl exists."}), 500

    try:
        # Extract inputs from JSON payload
        data = request.get_json()

        store = int(data.get("store"))
        item = int(data.get("item"))
        date_str = data.get("date")

        # Process date to extract model features
        date_obj = pd.to_datetime(date_str)
        year = date_obj.year
        month = date_obj.month
        day = date_obj.day
        day_of_week = date_obj.dayofweek  # Monday = 0, Sunday = 6

        # Build feature vector matching exact feature sequence
        input_data = pd.DataFrame([{
            "store": store,
            "item": item,
            "Year": year,
            "Month": month,
            "Day": day,
            "DayOfWeek": day_of_week
        }])[features]

        # Generate prediction
        prediction = model.predict(input_data)[0]
        predicted_sales = max(0, round(float(prediction), 2))  # Ensure non-negative

        return jsonify({
            "success": True,
            "predicted_sales": predicted_sales,
            "date": date_obj.strftime("%A, %B %d, %Y"),
            "store": store,
            "item": item
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)