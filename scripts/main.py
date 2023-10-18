from flask import Flask, request, jsonify
import numpy as np
from tensorflow import keras
from PIL import Image
from io import BytesIO

app = Flask(__name__)


model = keras.models.load_model('models/model_Balanced.keras')
# Data preprocessing
def preprocess_image(image):
  
    image = image.resize((128, 128))  
    image = image.convert('L') 
    image = np.asarray(image) / 255.0  
    image = np.expand_dims(image, axis=-1)  
    return image

# predict pased on the image that the client sends, (verify if the image exist or not)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'})

        image_data = request.files['image'].read()
        image = Image.open(BytesIO(image_data))

        processed_image = preprocess_image(image)

        predictions = model.predict(np.expand_dims(processed_image, axis=0))

        response = {
            'predictions': predictions.tolist()
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
