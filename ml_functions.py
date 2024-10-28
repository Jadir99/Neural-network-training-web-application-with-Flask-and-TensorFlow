import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers
from tensorflow.keras.utils import plot_model
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def save_model_summary_as_image(model):
    try:
        plot_model(model, to_file='./static/images/plots/model_summary.png', show_shapes=True)
    except Exception as e:
        print(f"An error occurred: {e}")

def train_images_tensorsr(nbr_layers, nbr_neurals, activation, loss, optimizer, epochs, target1, target2, alpha=0.001, img_size=(50, 50)):
    categories = [target1, target2]
    data = []

    for categ in categories:
        label = categories.index(categ)
        for img_name in os.listdir(os.path.join('./static/images/uploads/', categ)):
            img_path = os.path.join('./static/images/uploads/', categ, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, img_size)
                data.append([img, label])

    x, y = zip(*data)
    x = np.array(x) / 255.0  # Normalize pixel values
    y = np.array(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Flatten())
    
    for i in range(nbr_layers):
        model.add(layers.Dense(units=nbr_neurals[i-1], activation=activation[i-1]))
    model.add(layers.Dense(len(categories), activation='softmax')) 
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
    history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))

    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.savefig('static/images/plots/model_loss.png')
    plt.close()
    model.save('static/models/my_model.h5')
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    return model, history, test_loss, test_accuracy

def choose_target(predict_class):
    with open("./static/models/target.txt","r") as file:
        all_targets = file.read()
    return all_targets.split(',')[predict_class]
