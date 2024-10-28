from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import pandas as pd
from ml_functions import (
    train_tensor,
    train_images_tensorsr,
    choose_target,
    save_model_summary_as_image
)
from tensorflow.keras.models import load_model
import cv2
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train_model')
def train_model():
    return render_template('train_model.html')

@app.route('/result')
def result():
    return render_template('result.html')

@app.route('/classification_images')
def classification_images():
    return render_template('classificationByImage.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/train_data', methods=["POST", "GET"])
def train_data():
    if request.method == "POST":
        neurons = [int(request.form.get(f'number_neurons_{i}')) for i in range(int(request.form['number_layers']))]
        activations = [request.form.get(f'activation_function_{i}') for i in range(len(neurons))]

        epoches = int(request.form['epoches'])
        loss_function = request.form['loss_function']
        optimizer_function = request.form['optimizer_function']
        Target = request.form['Target'].replace('"', '')
        task_type = request.form['task_type']

        file = request.files['file']
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        data = pd.read_csv(file_path)

        model, history, test_loss, score = train_tensor(
            len(neurons), neurons, activations, loss_function, optimizer_function, epoches, data, Target, task_type
        )

        return render_template('result.html', summary=model.summary(), score=score, test_loss=test_loss,
                               activations=activations, neurons=neurons, epoches=epoches, 
                               loss_function=loss_function, optimizer_function=optimizer_function, Target=Target, task_type=task_type)

@app.route('/train_images', methods=["POST", "GET"])
def train_images():
    if request.method == "POST":
        neurons = [int(request.form.get(f'number_neurons_{i}')) for i in range(int(request.form['number_layers']))]
        activations = [request.form.get(f'activation_function_{i}') for i in range(len(neurons))]

        epoches = int(request.form['epoches'])
        loss_function = request.form['loss_function']
        optimizer_function = request.form['optimizer_function']
        Target1 = request.form['target1']
        Target2 = request.form['target2']

        app.config['UPLOAD_FOLDER_TARGET1'] = f'./static/images/uploads/{Target1}/'
        app.config['UPLOAD_FOLDER_TARGET2'] = f'./static/images/uploads/{Target2}/'
        os.makedirs(app.config['UPLOAD_FOLDER_TARGET1'], exist_ok=True)
        os.makedirs(app.config['UPLOAD_FOLDER_TARGET2'], exist_ok=True)

        images1 = request.files.getlist('images1')
        images2 = request.files.getlist('images2')
        for image in images1 + images2:
            if image and image.filename != '':
                folder = app.config['UPLOAD_FOLDER_TARGET1'] if image in images1 else app.config['UPLOAD_FOLDER_TARGET2']
                image.save(os.path.join(folder, secure_filename(image.filename)))

        model, history, test_loss, score = train_images_tensorsr(
            len(neurons), neurons, activations, loss_function, optimizer_function, epoches, Target1, Target2, alpha=0.001, img_size=(50, 50)
        )

        return render_template('result.html', summary=model.summary(), score=score, test_loss=test_loss,
                               activations=activations, neurons=neurons, epoches=epoches, 
                               loss_function=loss_function, optimizer_function=optimizer_function, Target=Target1)

@app.route('/prediction_image', methods=["POST", "GET"])
def prediction_image():
    if request.method == "POST":
        image = request.files['image']
        filename = secure_filename(image.filename)
        temp_path = os.path.join('./static/images/temp/', filename)
        image.save(temp_path)

        img = cv2.imread(temp_path)
        if img is not None:
            img = cv2.resize(img, (50, 50)) / 255.0
            img = np.expand_dims(img, axis=0)
            model = load_model('./static/models/my_model.h5')
            prediction = model.predict(img)
            predicted_class = np.argmax(prediction)
            target = choose_target(predicted_class)
            proba = max(prediction[0])
        return render_template('predict.html', predicted_class=predicted_class, image=str(image.filename), proba=proba, target=target)

if __name__ == '__main__':
    app.run(debug=True, port=4000, use_reloader=False)
