
"""
Iris Species Predictor
-----------------------
A mini AI/ML project: Python (Flask + scikit-learn) backend
serving predictions to an HTML/CSS/JS frontend.

Run:
    pip install flask scikit-learn
    python app.py

Then open http://127.0.0.1:5000 in your browser.
"""

from flask import Flask, request, jsonify, render_template
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# ----------------------------------------------------------
# Train a small model once, at server startup.
# In a real project you'd load a pre-trained model with
# joblib.load("model.pkl") instead of training every time.
# ----------------------------------------------------------
iris = load_iris()
X, y = iris.data, iris.target
target_names = iris.target_names  # ['setosa', 'versicolor', 'virginica']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Model trained. Test accuracy: {accuracy:.2%}")


@app.route("/")
def home():
    """Serve the frontend page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON like:
    {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    """
    data = request.get_json()

    try:
        features = [
            float(data["sepal_length"]),
            float(data["sepal_width"]),
            float(data["petal_length"]),
            float(data["petal_width"]),
        ]
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "Invalid input. Provide 4 numeric measurements."}), 400

    prediction = model.predict([features])[0]
    probabilities = model.predict_proba([features])[0]

    result = {
        "species": target_names[prediction],
        "confidence": round(float(max(probabilities)) * 100, 1),
        "probabilities": {
            target_names[i]: round(float(p) * 100, 1)
            for i, p in enumerate(probabilities)
        },
    }
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
  
