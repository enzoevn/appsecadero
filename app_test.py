from flask import Flask, jsonify, render_template, send_from_directory
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
import time
import os
import onnxruntime as ort
from PIL import Image
import numpy as np
import uuid
from collections import OrderedDict

app = Flask(__name__)

# Load the model
model_path = "secadero.onnx"
ort_session = ort.InferenceSession(model_path)

def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0).astype(np.float32)
    return image

def run_inference(ort_session, image):
    ort_inputs = {ort_session.get_inputs()[0].name: image}
    ort_outs = ort_session.run(None, ort_inputs)
    ort_outs = ort_outs[0][0] / 4
    ort_outs = np.int32(ort_outs)
    return ort_outs

class ImageHandler(FileSystemEventHandler):
    def __init__(self):
        self.predictions = OrderedDict()

    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith('.jpg') or event.src_path.endswith('.png'):
            print(f"New image detected: {event.src_path}")
            image = preprocess_image(event.src_path)
            prediction = run_inference(ort_session, image).tolist()
            prediction_id = str(uuid.uuid4())
            self.predictions[prediction_id] = {
                "id": prediction_id,
                "name": os.path.basename(event.src_path),
                "prediction": prediction[0]
            }
            print(f"Prediction for {event.src_path}: {prediction[0]}")

    def reset_predictions(self):
        self.predictions = OrderedDict()

image_handler = ImageHandler()
observer = Observer()
observer.schedule(image_handler, path='images', recursive=False)
observer.start()

@app.route('/')
def index():
    return render_template('monitoring.html')

@app.route('/predictions', methods=['GET'])
def get_predictions():
    return jsonify(image_handler.predictions)

@app.route('/reset', methods=['POST'])
def reset_predictions():
    image_handler.reset_predictions()
    return jsonify({"status": "predictions reset"})

@app.route('/images/<filename>')
def get_image(filename):
    return send_from_directory('images', filename)

def run_flask():
    app.run(host='localhost', port=5000)

if __name__ == "__main__":
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()