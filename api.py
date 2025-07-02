# app.py
from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Create Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the ML API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data
        data = request.json
        features = np.array(data["features"]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)
        prediction_proba = model.predict_proba(features).tolist()

        return jsonify({
            "prediction": int(prediction[0]),
            "prediction_probability": prediction_proba
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    print("API running")
    app.run(debug=True)
