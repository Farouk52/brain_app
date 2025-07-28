import os
import numpy as np
from PIL import Image
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'brain_tumor_model.h5'
try:
    model = load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# Define class names in the same order as training labels
CLASS_NAMES = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']


def model_predict(img_path, model):
    """Predict the class of the uploaded image."""
    img = load_img(img_path, target_size=(299, 299))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0  # Normalize

    predictions = model.predict(x)
    predicted_class_index = np.argmax(predictions[0])
    return CLASS_NAMES[predicted_class_index]


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if file part is in the request
        if 'file' not in request.files:
            return render_template('index.html', error='No file uploaded.')

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', error='No file selected.')

        if file:
            # Secure filename and save it
            filename = secure_filename(file.filename)
            basepath = os.path.dirname(__file__)
            upload_folder = os.path.join(basepath, 'static', 'uploads')
            os.makedirs(upload_folder, exist_ok=True)
            file_path = os.path.join(upload_folder, filename)
            file.save(file_path)

            # Make prediction
            prediction = model_predict(file_path, model)

            # Show result and image
            return render_template('index.html',
                                   prediction=prediction,
                                   uploaded_image=f'uploads/{filename}')

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
