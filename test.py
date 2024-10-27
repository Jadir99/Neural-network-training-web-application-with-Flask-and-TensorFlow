from flask import Flask ,render_template,request
import os
from werkzeug.utils import secure_filename

import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from tensorflow.keras.models import load_model
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import models, layers
from keras.utils import to_categorical

app=Flask(__name__)
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


def train_images_tensorsr(nbr_layers, nbr_neurals, activation, loss, optimizer, epochs, target1, target2, alpha=0.001,img_size=(50, 50)):

    """
    Train a convolutional neural network with images in specified folders for target1 and target2.
    
    Parameters:
    - folder: str - The folder path containing subfolders for each target category.
    - target1: str - The name of the first target category (e.g., 'cats').
    - target2: str - The name of the second target category (e.g., 'dogs').
    - optimizer: str - The optimizer to use (e.g., 'adam').
    - epochs: int - The number of training epochs.
    - img_size: tuple - The size to which each image will be resized.
    """
    
    # Prepare the categories
    categories = [target1, target2]
    # save argets for training model 
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
        model.add(layers.Dense(units=nbr_neurals[i-1], activation=activation[i-1]))

    # last dense has the number of target that is 2 (classification binary)
    model.add(layers.Dense(len(categories), activation='softmax')) 

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
    # Train the model
    history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))
    # Plot and save model loss
    plt.figure(figsize=(6, 6))
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
    plt.figure(figsize=(6, 6))
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig('static/images/plots/model_accuracy.png')
    plt.close()


    
    # Save the trained model to an .h5 file
    model_path = 'static/models/my_model.h5'
    model.save(model_path)
    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f'Test loss: {test_loss}, Test accuracy: {test_accuracy}')


# dont forget to add file html to visualisze the loss function ...
    return model, history, test_loss, test_accuracy


# get the target 
def choose_target(predict_class):
    with open("./static/models/target.txt","r")as file:
        all=file.read()
    if predict_class==0:
        return str(all.split(',')[0])
    return str(all.split(',')[1])

# Prototype for training function with Pandas DataFrame
def train_tensor(nbr_layers, nbr_neurals, activation, loss, optimizer, epoches, data, target_column, task_type, alpha=0.001):
    """
    This function now handles both regression and classification tasks.
    
    task_type: 'regression' or 'classification' (string)
    """

    #preprocessing the data :
####################### don t forget ######################

    print(target_column)
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # If it's a classification problem, perform one-hot encoding for the target
    if task_type == 'classification':
        num_classes = len(y.unique())  # Assume this is a multi-class classification problem
        y = pd.get_dummies(y)  # One-hot encoding for classification
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(X_train.shape[1],)))
    
    for i in range(nbr_layers):
        model.add(layers.Dense(units=nbr_neurals[i-1], activation=activation[i-1]))
    
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

# model_performance_plot
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


    test_loss = model.evaluate(X_test, y_test)
    
    if task_type == 'regression':
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        return model, history, test_loss, r2
    elif task_type == 'classification':
        accuracy = model.evaluate(X_test, y_test)[1]  # Get test accuracy
        return model, history, test_loss, accuracy

# Specify the directory where uploaded files will be saved
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
@app.route('/train_data', methods=["POST", "GET"])
def train_data():
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
                               loss_function=loss_function, optimizer_function=optimizer_function, Target=Target)




@app.route('/train_images', methods=["POST", "GET"])
def train_images():
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

        # Define upload folders
        app.config['UPLOAD_FOLDER_TARGET1'] = f'./static/images/uploads/{Target1}/'
        app.config['UPLOAD_FOLDER_TARGET2'] = f'./static/images/uploads/{Target2}/'

        # Create target directories if they don't exist
        os.makedirs(app.config['UPLOAD_FOLDER_TARGET1'], exist_ok=True)
        os.makedirs(app.config['UPLOAD_FOLDER_TARGET2'], exist_ok=True)

        # Handle file uploads from a single input
        images1 = request.files.getlist('images1')  # List of files from the single input
        images2 = request.files.getlist('images2')  # List of files from the single input

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
        model, history, test_loss, score = train_images_tensorsr(number_layers, neurons, activations, loss_function, optimizer_function, epoches,  Target1, Target2, alpha=0.001,img_size=(50, 50))

        # Render the result template with model details
        return render_template('result.html', summary=model.summary(), score=score, test_loss=test_loss,
                               activations=activations, neurons=neurons, epoches=epoches, 
                               loss_function=loss_function, optimizer_function=optimizer_function, Target=Target1)

@app.route('/prediction_image',methods=["POST", "GET"])
def prediction_image():
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
            # upload the model
            model = load_model('./static/models/my_model.h5')
            # Predict the class
            prediction = model.predict(img)
            print(prediction[0][0])
            predicted_class = np.argmax(prediction)
            print("Predicted class:", predicted_class)
            target=choose_target(predicted_class)
        return render_template('predict.html',predicted_class=predicted_class,image=str(image.filename),proba=max(prediction[0][0],prediction[0][1]),target=target)

if __name__ == '__main__':
    app.run(debug=True,port=4000,use_reloader=False)