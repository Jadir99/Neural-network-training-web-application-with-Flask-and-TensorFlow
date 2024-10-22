from flask import Flask ,render_template,request
import os
from werkzeug.utils import secure_filename


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



import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from tensorflow.keras.models import load_model


# Prototype for training function with Pandas DataFrame
def train_tensor(nbr_layers, nbr_neurals, activation, loss, optimizer, epoches, data, target_column, task_type, alpha=0.001):
    """
    This function now handles both regression and classification tasks.
    
    task_type: 'regression' or 'classification' (string)
    """
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
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    if task_type == 'classification' and 'accuracy' in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid()

    plt.savefig('static/images/plots/model_performance_plot.png')
    
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

        for i in range(number_layers):
            neurons.append(int(request.form.get(f'number_neurons_{i}')))
            activations.append(request.form.get(f'activation_function_{i}'))

        epoches = int(request.form['epoches'])
        loss_function = request.form['loss_function']
        optimizer_function = request.form['optimizer_function']
        Target = request.form['Target'].replace('"','')
        print(str(Target))
        task_type = request.form['task_type']  # Get task type from the form

        file = request.files['file']
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        data = pd.read_csv(file_path)
        
        model, history, test_loss, score = train_tensor(
            number_layers, neurons, activations, loss_function, optimizer_function, epoches, data, Target, task_type
        )
        
        return render_template('result.html', summary=model.summary(), score=score, test_loss=test_loss, activations=activations, neurons=neurons, epoches=epoches, loss_function=loss_function, optimizer_function=optimizer_function, Target=Target)


if __name__ == '__main__':
    app.run(debug=True,port=4000,use_reloader=False)