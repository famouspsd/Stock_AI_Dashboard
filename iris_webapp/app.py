from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(name)
CORS(app)

model = joblib.load("iris_model.pkl")
scaler = joblib.load("iris_scaler.pkl")

flowers = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
feature_names = ["sepal length (cm)", "sepal width (cm)",
                 "petal length (cm)", "petal width (cm)"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = pd.DataFrame([[
        data["sepal_length"], data["sepal_width"],
        data["petal_length"], data["petal_width"]
    ]], columns=feature_names)
    scaled = scaler.transform(features)
    pred = model.predict(scaled)[0]
    proba = model.predict_proba(scaled)[0]
    return jsonify({
        "flower": flowers[pred],
        "confidence": str(round(max(proba)*100, 1)) + "%"
    })

if name == "main":
    app.run(debug=True, port=5000)
