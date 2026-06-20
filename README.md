# Iris Predictor

A mini AI/ML web app that predicts iris flower species from sepal/petal measurements,
using a Random Forest model trained in Python and served through a Flask API to an
HTML/CSS/JS frontend.

## Tech stack
- **Backend:** Python, Flask, scikit-learn
- **Frontend:** HTML, CSS, vanilla JavaScript (fetch API)
- **Model:** Random Forest Classifier trained on the classic Iris dataset

## How it works
1. `app.py` trains the model at startup and exposes a `/predict` endpoint
2. The frontend sends measurement values to `/predict` as JSON
3. The model returns the predicted species and a confidence breakdown for all 3 classes
4. Results render live in the browser, no page reload

## Run locally
```bash
git clone https://github.com/<your-username>/iris-predictor.git
cd iris-predictor
pip install -r requirements.txt
python app.py
```
Then open `http://127.0.0.1:5000` in your browser.

## Deploy (Render)
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app`

## Project structure
```
iris-predictor/
├── app.py
├── index.html
├── requirements.txt
└── .gitignore
```
