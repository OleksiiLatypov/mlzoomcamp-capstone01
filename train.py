import os
import math
from keras.preprocessing import image
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random as python_random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.applications import VGG16

import seaborn as sns
import math
import random
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense,Flatten,Dropout,BatchNormalization
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D,MaxPool2D


SIZE=(224,224)
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


train_path = '/workspaces/mlzoomcamp-capstone01/Covid19-dataset/train'
test_path = '/workspaces/mlzoomcamp-capstone01/Covid19-dataset/test'


name_classes = os.listdir(train_path)
print(name_classes)


covid_dir = os.path.join(train_path, 'Covid')
num_of_covid = len(os.listdir(covid_dir))

normal_dir = os.path.join(train_path, 'Normal')
num_of_normal = len(os.listdir(normal_dir))

pneumonia_dir = os.path.join(train_path, 'Viral Pneumonia')
num_of_pneumonia = len(os.listdir(pneumonia_dir))


print(f'There are {num_of_covid} images with Covid diagnosis')
print(f'There are {num_of_normal} images with Normal diagnosis')
print(f'There are {num_of_pneumonia} images with Pneumona diagnosis')

five_covid_images = random.sample(os.listdir(covid_dir), 5)
five_normal_images = random.sample(os.listdir(normal_dir), 5)
five_viral_images = random.sample(os.listdir(pneumonia_dir), 5)

print(f'5 Covid random images: {five_covid_images}')
print(f'5 Normal random images: {five_normal_images}')
print(f'5 Viral Pneumonia random images: {five_viral_images}')


def load_image_path(img_path: str, target_size=(224, 224)):
    """
    Loads an image from the given path, resizes it, and scales the pixel values to [0, 1].

    Parameters:
        img_path (str): Path to the image file.
        target_size (tuple): The target size to which the image should be resized (default is (224, 224)).

    Returns:
        np.ndarray: The processed image as a NumPy array.
    """
    # Load image with the target size
    img = image.load_img(img_path, target_size=target_size)

    # Convert the image to a NumPy array
    img_array = image.img_to_array(img)

    # Rescale the image array to the range [0, 1]
    rescale_img = img_array / 255.0

    return rescale_img


joined_images = [
    (covid_dir, five_covid_images, 'Covid'),
    (normal_dir, five_normal_images, 'Normal'),
    (pneumonia_dir, five_viral_images, 'Viral Pneumonia')
]


fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(12, 9))  # 5 rows, 3 columns
axes = axes.flatten()  # Flatten the axes array to iterate easily

# Initialize a counter for the subplot index
index = 0

# Loop through directories and images
for image_dir, image_list, label in joined_images:
    for image_name in image_list:
        image_path = os.path.join(image_dir, image_name)
        image_to_load = load_image_path(image_path)

        # Plot the image in the current subplot
        axes[index].imshow(image_to_load)
        axes[index].set_title(f'{label}\n{image_name}', fontsize=8)
        axes[index].axis("off")  # Turn off axes

        index += 1  # Move to the next subplot

# Hide any remaining unused subplots (if total images < grid size)
for i in range(index, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()


datagen_train=ImageDataGenerator(rescale=1./255., validation_split=0.2)
datagen_test=ImageDataGenerator(rescale=1./255.)


train_data=datagen_train.flow_from_directory(batch_size=32,
                                            directory=train_path,
                                            shuffle=True,
                                            classes=name_classes,
                                            target_size=SIZE,
                                            subset="training",
                                            class_mode='categorical')

validation_data=datagen_train.flow_from_directory(batch_size=32,
                                            directory=train_path,
                                            shuffle=True,
                                            classes=name_classes,
                                            subset="validation",
                                            target_size=SIZE,
                                            class_mode='categorical')

test_data=datagen_train.flow_from_directory(batch_size=1,
                                            directory=test_path,
                                            shuffle=False,
                                            classes=name_classes,
                                            target_size=SIZE,
                                            class_mode='categorical')


# Create CNN model 
def create_cnn_model(input_shape=(224, 224, 3), num_classes=3):
    """
    Function to create a CNN model with 3 convolutional layers and 2 fully connected layers.

    Parameters:
    - input_shape (tuple): Shape of the input image, default is (224, 224, 3) for RGB images.
    - num_classes (int): Number of output classes for classification, default is 3.

    Returns:
    - model: A Keras Sequential model.
    """
    model = Sequential()

    # First convolutional layer
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPool2D((2, 2)))

    # Second convolutional layer
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D((2, 2)))

    # Third convolutional layer
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D((2, 2)))

    # Flatten the 3D feature map to 1D vector for fully connected layers
    model.add(Flatten())

    # Fully connected layers
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))

    # Output layer with softmax activation for multi-class classification
    model.add(Dense(num_classes, activation='softmax'))

    return model
        
cnn_model = create_cnn_model()
print(cnn_model.summary())


cnn_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


class myCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        # Check if accuracy has reached or exceeded 95%
        if logs.get('accuracy') >= 0.99:
            print(f"\nReached More than 99% accuracy")
            self.model.stop_training = True  # Stop training early

# Initialize the custom callback instance
callback = myCallback()


hist=cnn_model.fit(train_data, epochs=30, validation_data=validation_data, callbacks=callback)

print(cnn_model.evaluate(test_data))

predictions = cnn_model.predict(test_data)
y_pred = [np.argmax(probas) for probas in predictions]
y_test = test_data.classes
class_names = test_data.class_indices.keys()

# Generate classification report
report = classification_report(y_test, y_pred, target_names=name_classes)

# Print the classification report
print(report)


#SAVE MODEL 
try:
    # Try saving the model
    cnn_model.save('cnn_model.h5')
    print("Model saved successfully!")

except Exception as e:
    # Catch any exception and print the error message
    print(f"An error occurred while saving the model: {e}")


# Load your saved model
model = load_model('cnn_model.h5')


# Convert the Keras model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(cnn_model)
tflite_model = converter.convert()

# Save the converted TensorFlow Lite model
with open('converted_model.tflite', 'wb') as file:
    file.write(tflite_model)

print("Model successfully converted to converted_model.tflite")




### IMAGE AUGMENTATION ###
# Apply rescaling and augmentation to the training data
datagen_train_aug = ImageDataGenerator(
    rescale=1./255.,             # Rescale pixel values to [0, 1]
    validation_split=0.2,        # Split the data, 20% for validation
    rotation_range=40,           # Random rotation within ±40 degrees
    width_shift_range=0.2,       # Random width shift by 20%
    height_shift_range=0.2,      # Random height shift by 20%
    shear_range=0.2,             # Random shear transformation
    zoom_range=0.2,              # Random zoom by 20%
    horizontal_flip=True,        # Random horizontal flip
    fill_mode='nearest'          # Fill empty pixels after transformation with nearest pixel value
)

# Training data generator with augmentation
train_data_aug = datagen_train_aug.flow_from_directory(
    batch_size=32,
    directory=train_path,
    shuffle=True,
    classes=name_classes,
    target_size=SIZE,
    subset="training",  # Use the 'training' subset from the split
    class_mode='categorical'
)

hist_aug = cnn_model.fit(train_data_aug, epochs=20, validation_data=validation_data, callbacks=[callback])

print(cnn_model.evaluate(test_data))

predictions=cnn_model.predict(test_data)
y_pred=[np.argmax(probas) for probas in predictions]
y_test=test_data.classes
class_names=test_data.class_indices.keys()

# Generate classification report
report = classification_report(y_test, y_pred, target_names=name_classes)

# Print the classification report
print(report)


### using pre-trained VGG16 ###

# Load the pre-trained VGG16 model, excluding the top layers (Fully Connected)
base_model = VGG16(include_top=False, input_shape=(224, 224, 3))

# Freeze the layers in the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

# Now, build the model by adding your custom layers
model_vgg = Sequential()

# Add the pre-trained VGG16 as the base (feature extractor)
model_vgg.add(base_model)

# Add a few custom layers for classification
model_vgg.add(Flatten())  # Flatten the feature map to a 1D vector
model_vgg.add(Dense(128, activation='relu'))
model_vgg.add(Dropout(0.5))  # Dropout for regularization
model_vgg.add(Dense(64, activation='relu'))
model_vgg.add(Dense(3, activation='softmax'))  # Number of classes is 3

# Compile the model
model_vgg.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# View model summary to check architecture
print(model_vgg.summary())

# Define image data generators for training and validation
datagen_train_vgg = ImageDataGenerator(
    rescale=1./255.,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


# Flow training and validation data
train_data_aug = datagen_train_vgg.flow_from_directory(
    batch_size=32,
    directory=train_path,
    shuffle=True,
    classes=name_classes,
    target_size=SIZE,  # Use the correct image size
    class_mode='categorical'
)


history_vgg = model_vgg.fit(
    train_data_aug,
    epochs=10,
    validation_data=validation_data
)

print(model_vgg.evaluate(test_data))


predictions=model_vgg.predict(test_data)
y_pred=[np.argmax(probas) for probas in predictions]
y_test=test_data.classes
class_names=test_data.class_indices.keys()

# Generate classification report
report = classification_report(y_test, y_pred, target_names=name_classes)

# Print the classification report
print(report)




model = load_model('cnn_model.h5')



# Example image path (replace with the actual image file path)
img_path = '/workspaces/mlzoomcamp-capstone01/Covid19-dataset/test/Viral Pneumonia/0102.jpeg'

# Load and preprocess the image
image_for_prediction = load_image_path(img_path)
# Add an extra batch dimension as Keras expects input in batches
image_for_prediction = np.expand_dims(image_for_prediction, axis=0)


# Make prediction using the model
prediction = model.predict(image_for_prediction)

# Get predicted class index (class with the highest probability)
predicted_class_index = np.argmax(prediction, axis=1)

# Class labels (must be the same as used during training)
class_labels = ['Covid', 'Viral Pneumonia', 'Normal']

# Get predicted class label
predicted_class_label = class_labels[predicted_class_index[0]]

# Print the raw output (probabilities for each class)
print(f"Raw prediction (probabilities): {prediction}")

print(f"Predicted class: {predicted_class_label}")