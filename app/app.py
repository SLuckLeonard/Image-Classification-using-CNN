from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load your pre-trained model
model = load_model('../saved_models/best_cnn_model.keras')

# Define image preprocessing function (modify based on your model's input requirements)
def preprocess_image(image):
    image = image.resize((32, 32))  # Assuming CIFAR-10 input size
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('home'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('home'))

    if file:
        # Preprocess the image
        image = Image.open(file)
        processed_image = preprocess_image(image)

        # Make prediction
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction)

        # Return the result
        return render_template('index.html', prediction=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
