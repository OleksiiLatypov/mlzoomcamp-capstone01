from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import os
import requests
from io import BytesIO
from PIL import Image
from tensorflow.keras.preprocessing import image

app = Flask(__name__)



# Load the trained model once when the app starts
model = load_model('cnn_model_with_augmentation.h5')
class_labels = ['Covid', 'Normal', 'Viral Pneumonia']


def load_image_path(img: Image.Image, target_size=(224, 224)):
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
    rescale_img = img_array / 255.0
    
    return rescale_img


@app.route('/')
def home():
    return 'Flask app for image prediction with a CNN model!'

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

        image_for_prediction = load_image_path(img, target_size=(224, 224))

        image_for_prediction = np.expand_dims(image_for_prediction, axis=0)

        prediction = model.predict(image_for_prediction)

        predicted_class_index = np.argmax(prediction, axis=1)

        predicted_class_label = class_labels[predicted_class_index[0]]

        return jsonify({'prediction': predicted_class_label, 'probability': prediction.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=9696)
