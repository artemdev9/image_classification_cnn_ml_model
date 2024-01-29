from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
from keras.models import load_model

app = Flask(__name__)

model = load_model('./bdcm_convnn_v3_96.keras')

@app.route('/')
def index():
    return render_template('index.html')  

def preprocess_image(uploaded_file, target_size=(224, 224)):
    image_stream = uploaded_file.read()
    image_stream = np.frombuffer(image_stream, np.uint8)
    image = cv2.imdecode(image_stream, cv2.IMREAD_COLOR)
    image = cv2.resize(image, target_size)
    image = image / 255.0
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400

    file = request.files['file']
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type.'}), 400

    try:
        image = preprocess_image(file)
        image = np.expand_dims(image, axis=0)  
        prediction = model.predict(image)
        class_labels = ['Bicycle', 'Cars', 'Deer', 'Mountains']
        max_index = np.argmax(prediction[0])
        prediction_str = class_labels[max_index]
        return jsonify({'prediction': prediction_str})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

if __name__ == '__main__':
    app.run(debug=False)

