import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import  tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.utils import plot_model, to_categorical
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

def save_model_summary_as_image(model):
    """Save the model summary as an image."""
    try:
        plot_model(model, to_file='./static/images/plots/model_summary.png', show_shapes=True)
        print("Model summary saved as 'model_summary.png'.")
    except Exception as e:
        print(f"An error occurred: {e}")

def train_images_tensors(nbr_layers, nbr_neurals, activation, loss, optimizer, epochs, target1, target2, alpha=0.001, img_size=(50, 50)):
    """
    Train a convolutional neural network with images in specified folders for target1 and target2.
    
    Parameters:
    - nbr_layers: int - Number of layers in the model.
    - nbr_neurals: list - List of neurons in each layer.
    - activation: list - List of activation functions for each layer.
    - loss: str - Loss function.
    - optimizer: str - Optimizer to use.
    - epochs: int - Number of training epochs.
    - target1: str - Name of the first target category (e.g., 'cats').
    - target2: str - Name of the second target category (e.g., 'dogs').
    - alpha: float - Learning rate (default 0.001).
    - img_size: tuple - Size to which each image will be resized (default (50, 50)).
    """
    
    # Prepare the categories
    categories = [target1, target2]

    # Save targets for training model 
    with open("./static/models/target.txt", "w") as file:
        file.write(f'{target1},{target2}')
    
    data = []
    
    # Load images and labels
    for categ in categories:
        categ_folder = os.path.join('./static/images/uploads/', categ)
        label = categories.index(categ)
        print(f"Loading category: {categ} (label: {label})")
        
        for img_name in os.listdir(categ_folder):
            img_path = os.path.join(categ_folder, img_name)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Skipping {img_name}: unable to load image.")
                continue  # Skip if the image couldn't be loaded

            try:
                # Resize the image
                img = cv2.resize(img, img_size)
                data.append([img, label])
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
                continue
    
    # Preprocess the data
    x, y = zip(*data)
    x = np.array(x) / 255.0  # Normalize pixel values
    y = np.array(y)
    y = to_categorical(y, num_classes=len(categories))  # One-hot encode labels

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Build the CNN model
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Flatten())
    
    for i in range(nbr_layers):
        model.add(layers.Dense(units=nbr_neurals[i], activation=activation[i]))
    
    # Last dense layer has the number of targets (2 for binary classification)
    model.add(layers.Dense(len(categories), activation='softmax')) 

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
    # Train the model
    history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))

    # Plot and save model loss
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig('static/images/plots/model_loss.png')
    plt.close()  # Close the figure to free memory

    # Plot and save model accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig('static/images/plots/model_accuracy.png')
    plt.close()

    # Save the model summary
    save_model_summary_as_image(model)
    
    # Save the trained model to an .h5 file
    model_path = 'static/models/my_model.h5'
    model.save(model_path)
    
    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f'Test loss: {test_loss}, Test accuracy: {test_accuracy}')

    return model, history, test_loss, test_accuracy

def choose_target(predict_class):
    """Return the target based on the predicted class."""
    with open("./static/models/target.txt", "r") as file:
        all_targets = file.read().strip().split(',')
    return str(all_targets[predict_class])

def train_tensor(nbr_layers, nbr_neurals, activation, loss, optimizer, epoches, data, target_column, task_type, alpha=0.001):
    """
    This function handles both regression and classification tasks.
    
    Parameters:
    - nbr_layers: int - Number of layers in the model.
    - nbr_neurals: list - List of neurons in each layer.
    - activation: list - List of activation functions for each layer.
    - loss: str - Loss function for the model.
    - optimizer: str - Optimizer to use.
    - epoches: int - Number of training epochs.
    - data: DataFrame - Input data for training.
    - target_column: str - The target column for prediction.
    - task_type: str - Type of task ('regression' or 'classification').
    """
    
    # Preprocess the data
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # If it's a classification problem, perform one-hot encoding for the target
    if task_type == 'classification':
        num_classes = len(y.unique())  # Assume this is a multi-class classification problem
        y = pd.get_dummies(y)  # One-hot encoding for classification
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Build the model
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(X_train.shape[1],)))
    
    for i in range(nbr_layers):
        model.add(layers.Dense(units=nbr_neurals[i], activation=activation[i]))
    
    # Add final layer based on the task type
    if task_type == 'regression':
        model.add(layers.Dense(1))  # Single output for regression
    elif task_type == 'classification':
        model.add(layers.Dense(num_classes, activation='softmax'))  # Softmax for multi-class classification
    
    # Compile the model with appropriate loss function
    optimizer_func = getattr(tf.keras.optimizers, optimizer)(learning_rate=alpha)
    
    if task_type == 'regression':
        model.compile(loss=loss, optimizer=optimizer_func)  # E.g., 'mean_squared_error'
    elif task_type == 'classification':
        model.compile(loss='categorical_crossentropy', optimizer=optimizer_func, metrics=['accuracy'])
    
    history = model.fit(X_train, y_train, epochs=epoches, batch_size=32, validation_split=0.2)

    # Save the trained model to an .h5 file
    model_path = 'static/models/my_model.h5'
    model.save(model_path)

    # Model performance plot
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig('static/images/plots/model_loss.png')

    if task_type == 'classification' and 'accuracy' in history.history:
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['accuracy'], label='Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid()
        plt.savefig('static/images/plots/model_accuracy.png')
    
    print("Model training completed.")

# Flask app setup
app = Flask(__name__)

@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload from the user."""
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file'
    
    filename = secure_filename(file.filename)
    file.save(os.path.join('./static/images/uploads/', filename))
    return 'File uploaded successfully'

if __name__ == '__main__':
    app.run(debug=True)
