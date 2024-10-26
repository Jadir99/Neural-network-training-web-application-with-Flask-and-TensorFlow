import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import MaxPooling2D, Conv2D, Flatten, Dense
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import models, layers
# Set the folder path and categories
folder = r'IA'
categories = ['chats', 'chiens']
data = []

# Load the images and labels
for categ in categories:
    categ_folder = os.path.join(folder, categ)
    label = categories.index(categ)
    print(f"Loading category: {categ} (label: {label})")
    
    for img_name in os.listdir(categ_folder):
        img_path = os.path.join(categ_folder, img_name)
        
        # Try reading the image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipping {img_name}: unable to load image.")
            continue  # Skip if the image couldn't be loaded

        try:
            # Resize the image
            img = cv2.resize(img, (50, 50))
            
            # Append the image and its label
            data.append([img, label])
        
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            continue

# Split data into features and labels
x, y = zip(*data)
x = np.array(x)
y = np.array(y)

# Normalize pixel values
x = x / 255.0

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 3)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(20, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(len(categories), activation='softmax'))  # Update output layer to match number of categories

# Compile the model
model.compile(optimizer='adam', metrics=['accuracy'], loss='SparseCategoricalCrossentropy')
model.summary()

# Train the model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
plt.plot(model.history.history['loss'])