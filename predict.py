from flask import Flask, request, jsonify
import numpy as np
import os
import requests
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# Load the TensorFlow Lite model
TFLITE_MODEL_PATH = 'converted_model.tflite'

# Initialize the TensorFlow Lite interpreter
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class labels
class_labels = ['Covid', 'Normal', 'Viral Pneumonia']


def preprocess_image(img: Image.Image, target_size=(224, 224)):
    """
    Preprocess the image: resize, convert to array, and rescale.

    Args:
        img: PIL Image object.
        target_size: The target size for resizing the image.

    Returns:
        numpy array: The processed image ready for prediction.
    """
    # Resize the image to the target size
    img = img.resize(target_size)
    
    # Convert the image to a NumPy array
    img_array = np.array(img)
    
    # Rescale the image to [0, 1]
    rescaled_img = img_array / 255.0
    
    return rescaled_img


@app.route('/')
def home():
    return 'Flask app for image prediction with a TensorFlow Lite model!'


@app.route('/predict', methods=['POST'])
def predict():
    # Check if the request contains the 'url' key
    if 'url' not in request.json:
        return jsonify({'error': 'No URL provided'}), 400

    image_url = request.json['url']

    try:
        # Download the image from the provided URL
        response = requests.get(image_url)
        
        # Check if the response is successful
        if response.status_code != 200:
            return jsonify({'error': f'Failed to download image from URL. Status code: {response.status_code}'}), 400
        
        img = Image.open(BytesIO(response.content))

        # Preprocess the image
        image_for_prediction = preprocess_image(img, target_size=(224, 224))
        image_for_prediction = np.expand_dims(image_for_prediction, axis=0).astype(np.float32)

        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], image_for_prediction)

        # Run inference
        interpreter.invoke()

        # Get the output
        prediction = interpreter.get_tensor(output_details[0]['index'])

        # Get the predicted class index and label
        predicted_class_index = np.argmax(prediction, axis=1)
        predicted_class_label = class_labels[predicted_class_index[0]]

        return jsonify({'prediction': predicted_class_label, 'probability': prediction.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=9696)
