from tensorflow.keras.models import load_model
import numpy as np

# Load the saved model
model = load_model('cnn_model_with_augmentation.h5')



def load_image_path(img_path: str, target_size=(224, 224)):
    from tensorflow.keras.preprocessing import image
    # Load the image with the target size
    img = image.load_img(img_path, target_size=target_size)
    # Convert the image to a NumPy array
    img_array = image.img_to_array(img)
    # Rescale the image array to [0, 1]
    rescale_img = img_array / 255.0
    return rescale_img

# Example image path (replace with the actual image file path)
img_path = '/workspaces/ml-zoomcamp-capstone01/Covid19-dataset/test/Viral Pneumonia/0102.jpeg'

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