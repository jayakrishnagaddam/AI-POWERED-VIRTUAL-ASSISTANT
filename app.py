from flask import Flask, render_template, request
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pyttsx3
import random

app = Flask(__name__)

# Example ML model setup (Logistic Regression for NLP tasks)
vectorizer = TfidfVectorizer()
clf_nlp = LogisticRegression()

# Example data for ML model (Iris dataset for classification)
iris = load_iris()
X = iris.data
y = iris.target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
clf_iris = SVC(kernel='linear')
clf_iris.fit(X_scaled, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/voice-command', methods=['POST'])
def voice_command():
    # This is a placeholder function
    response = "Voice command received and processed."
    return response

@app.route('/nlp', methods=['POST'])
def nlp():
    text = request.form.get('text', 'default text')
    # Simple NLP prediction using the example Logistic Regression model
    text_vector = vectorizer.transform([text])
    prediction = clf_nlp.predict(text_vector)
    response = f"Predicted NLP category: {prediction[0]}"
    return response

@app.route('/tts', methods=['POST'])
def tts():
    text = request.form.get('text', 'Hello, this is a text-to-speech test.')
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    return f"Text to Speech: {text}"

@app.route('/task-automation', methods=['POST'])
def task_automation():
    tasks = ["Set a reminder", "Send an email", "Create a to-do list"]
    selected_task = random.choice(tasks)
    return f"Automated Task: {selected_task}"

@app.route('/weather', methods=['POST'])
def weather():
    # Placeholder for weather information retrieval
    weather_info = "The current weather is sunny with a temperature of 25°C."
    return f"Weather Info: {weather_info}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
