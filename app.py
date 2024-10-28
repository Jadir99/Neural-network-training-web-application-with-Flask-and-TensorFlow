from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import pandas as pd
from ml_functions import (
    train_tensor,
    train_images_tensors,
    choose_target,
)
from tensorflow.keras.models import load_model
import cv2
import numpy as np

app = Flask(__name__)

# Define the upload folder for data
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

@app.route('/train_model')
def train_model():
    """Render the training model page."""
    return render_template('train_model.html')

@app.route('/result')
def result():
    """Render the result page."""
    return render_template('result.html')

@app.route('/classification_images')
def classification_images():
    """Render the image classification page."""
    return render_template('classificationByImage.html')

@app.route('/predict')
def predict():
    """Render the prediction page."""
    return render_template('predict.html')

@app.route('/train_data', methods=["POST", "GET"])
def train_data():
    """Handle training for tabular data."""
    if request.method == "POST":
        neurons = []
        activations = []
        number_layers = int(request.form['number_layers'])

        # Collect neurons and activations from the form
        for i in range(number_layers):
            neurons.append(int(request.form.get(f'number_neurons_{i}')))
            activations.append(request.form.get(f'activation_function_{i}'))

        # Get other parameters from the form
        epoches = int(request.form['epoches'])
        loss_function = request.form['loss_function']
        optimizer_function = request.form['optimizer_function']
        Target = request.form['Target'].replace('"', '')
        task_type = request.form['task_type']

        # Handle file upload    
        file = request.files['file']
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        data = pd.read_csv(file_path)

        # Train the model
        model, history, test_loss, score = train_tensor(
            number_layers, neurons, activations, loss_function, optimizer_function, epoches, data, Target, task_type
        )

        # Render the result template with model details
        return render_template('result.html', summary=model.summary(), score=score, test_loss=test_loss,
                               activations=activations, neurons=neurons, epoches=epoches, 
                               loss_function=loss_function, optimizer_function=optimizer_function, Target=Target, task_type=task_type)

@app.route('/train_images', methods=["POST", "GET"])
def train_images():
    """Handle training for image data."""
    if request.method == "POST":
        neurons = []
        activations = []
        number_layers = int(request.form['number_layers'])

        # Collect neurons and activations from the form
        for i in range(number_layers):
            neurons.append(int(request.form.get(f'number_neurons_{i}')))
            activations.append(request.form.get(f'activation_function_{i}'))

        # Get other parameters from the form
        epoches = int(request.form['epoches'])
        loss_function = request.form['loss_function']
        optimizer_function = request.form['optimizer_function']
        Target1 = request.form['target1']
        Target2 = request.form['target2']

        # Define upload folders for each target
        app.config['UPLOAD_FOLDER_TARGET1'] = f'./static/images/uploads/{Target1}/'
        app.config['UPLOAD_FOLDER_TARGET2'] = f'./static/images/uploads/{Target2}/'

        # Create target directories if they don't exist
        os.makedirs(app.config['UPLOAD_FOLDER_TARGET1'], exist_ok=True)
        os.makedirs(app.config['UPLOAD_FOLDER_TARGET2'], exist_ok=True)

        # Handle file uploads
        images1 = request.files.getlist('images1')  # List of files for the first target
        images2 = request.files.getlist('images2')  # List of files for the second target

        for image in images1:
            if image and image.filename != '':
                filename = secure_filename(image.filename)
                upload_path = os.path.join(app.config['UPLOAD_FOLDER_TARGET1'], filename)
                image.save(upload_path)

        for image in images2:
            if image and image.filename != '':
                filename = secure_filename(image.filename)
                upload_path = os.path.join(app.config['UPLOAD_FOLDER_TARGET2'], filename)
                image.save(upload_path)

        print("Images uploaded successfully")

        # Train the model
        model, history, test_loss, score = train_images_tensors(
            number_layers, neurons, activations, loss_function, optimizer_function, epoches, Target1, Target2, alpha=0.001, img_size=(50, 50)
        )

        # Render the result template with model details
        return render_template('result.html', summary=model.summary(), score=score, test_loss=test_loss,
                               activations=activations, neurons=neurons, epoches=epoches, 
                               loss_function=loss_function, optimizer_function=optimizer_function, Target=Target1)

@app.route('/prediction_image', methods=["POST", "GET"])
def prediction_image():
    """Handle prediction based on uploaded image."""
    if request.method == "POST":
        # Handle file upload
        image = request.files['image']
        filename = secure_filename(image.filename)
        temp_path = os.path.join('./static/images/temp/', filename)
        
        # Save the uploaded image temporarily
        image.save(temp_path)

        # Load the image with OpenCV
        img = cv2.imread(temp_path)

        # Check if the image was loaded correctly
        if img is None:
            print("Error: Could not load image. Please check the file path.")
        else:
            # Preprocess the image if it loaded successfully
            img = cv2.resize(img, (50, 50))  # Resize to the model's input shape
            img = img / 255.0  # Normalize pixel values to [0, 1]
            img = np.expand_dims(img, axis=0)  # Add batch dimension
            
            # Load the model
            model = load_model('./static/models/my_model.h5')

            # Predict the class
            prediction = model.predict(img)
            predicted_class = np.argmax(prediction)
            target = choose_target(predicted_class)

        return render_template('predict.html', predicted_class=predicted_class, image=filename, proba=max(prediction[0]), target=target)

if __name__ == '__main__':
    app.run(debug=True, port=4000, use_reloader=False)
