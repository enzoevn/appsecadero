from flask import Flask, request, render_template, send_from_directory, send_file
import os
import numpy as np
from PIL import Image
import onnxruntime as ort
from inference_onnx_secadero import run_pipeline, list_files, list_folders, calculate_mean_weight

from io import BytesIO


app = Flask(__name__)

MODELS_PATH = './models'
IMAGES_PATH = './'

@app.route('/')
def index():
    model_files = list_files(MODELS_PATH)
    image_folders = list_folders(IMAGES_PATH)

    return render_template('index.html', model_files=model_files, image_folders=image_folders)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the selected model file and image folder
    selected_model = request.form.get('model')
    selected_image_folder = request.form.get('image_folder')

    # Construct the paths
    model_path = os.path.join(MODELS_PATH, selected_model)
    image_folder_path = os.path.join(IMAGES_PATH, selected_image_folder)

    # Inference
    predictions = run_pipeline(model_path, image_folder_path)
    print(type(predictions))

    # Mean weight
    mean_weight = calculate_mean_weight(predictions)
    
    return render_template('predictions.html', predictions=predictions, mean_weight=mean_weight, image_folder=selected_image_folder)

# @app.route('/images/<folder>/<path:filename>')
# def serve_image(folder, filename):
#     return send_from_directory(os.path.join(IMAGES_PATH, folder), filename)

@app.route('/images/<folder>/<path:filename>')
def serve_image(folder, filename):
    image_path = os.path.join(IMAGES_PATH, folder, filename)
    with Image.open(image_path) as img:
        img.thumbnail((50, 50))  # Resize the image to a thumbnail
        img_io = BytesIO()
        img.save(img_io, 'JPEG', quality=85)
        img_io.seek(0)
        return send_file(img_io, mimetype='image/jpeg')

if __name__ == '__main__':
    port = 5000
    host = 'localhost'
    app.run(port=port, host=host, debug=True)
