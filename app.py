# app.py
from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load pretrained model
model = load_model('cnn_model.h5')

# Load class names
TRAIN_DIR = r'C:\Users\NARMATHA R\OneDrive\Desktop\Jupter Projects\DLDS\train'
class_labels = sorted(os.listdir(TRAIN_DIR))  # Folder names as classes

# Remedy Mapping (Optional)
remedy_dict = {
    "Tomato___Early_blight": "Use fungicide sprays and remove infected leaves.",
    "Tomato___Late_blight": "Apply copper-based fungicides and improve air circulation.",
    "Tomato___Leaf_Mold": "Use resistant varieties and maintain dry foliage.",
    "Tomato___Healthy": "No remedy needed. Keep monitoring the plant.",
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    image = load_img(filepath, target_size=(128, 128))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    class_index = np.argmax(prediction[0])
    class_name = class_labels[class_index]

    if "___" in class_name:
        plant_name, disease = class_name.split("___")
    else:
        plant_name = class_name
        disease = "Unknown"

    remedy = remedy_dict.get(class_name, "No remedy info available.")

    return jsonify({
        'plant_name': plant_name,
        'disease': disease,
        'remedy': remedy
    })

if __name__ == '__main__':
    app.run(debug=True)
