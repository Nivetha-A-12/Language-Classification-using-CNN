from flask import Flask, render_template, request, redirect, url_for
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import os

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Load the pre-trained model and label encoder
model = load_model("language_detection_model.h5")
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Define the upload folder
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the uploads folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def process_audio(file_path):
    """Process the audio file and make predictions"""
    new_audio, sr = librosa.load(file_path, sr=None)
    new_mfcc = librosa.feature.mfcc(y=new_audio, sr=sr, n_mfcc=40)
    new_feature = np.mean(new_mfcc, axis=1).reshape(1, -1)

    prediction = model.predict(new_feature)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])

    return predicted_label[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the request contains a file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    # Check if the user submitted a file
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file:
        # Save the file to the uploads folder
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Process the uploaded file
        predicted_language = process_audio(filepath)

        return render_template('result.html', predicted_language=predicted_language)

if __name__ == "__main__":
    app.run(debug=True)