from flask import Flask, jsonify, render_template, send_from_directory, request, send_file
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
from io import BytesIO
from datetime import datetime

# --- Configuration ---
MODELS_PATH = './models'  # Path to store ONNX models
IMAGES_PATH = './images'  # Path that watchdog monitors
DEFAULT_MODEL = "secadero.onnx"  # Default ONNX model

app = Flask(__name__)

# Global variable to hold the ONNX session and model path
ort_session = None
current_model_path = None

# Load the default model initially
def load_model(model_path):
    global ort_session, current_model_path
    try:
        ort_session = ort.InferenceSession(model_path)
        current_model_path = model_path
        print(f"Model loaded successfully from: {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        ort_session = None  # Ensure session is None if loading fails
        current_model_path = None

load_model(os.path.join(MODELS_PATH, DEFAULT_MODEL))  # Load initial model

# --- Image Processing and Inference Functions ---
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0).astype(np.float32)
    return image

# def run_inference(ort_session, image):
#     ort_inputs = {ort_session.get_inputs()[0].name: image}
#     ort_outs = ort_session.run(None, ort_inputs)
#     ort_outs = ort_outs[0][0] / 4
#     ort_outs = np.int32(ort_outs)
#     return ort_outs

def run_inference(ort_session, image):
    # Prepare the input data
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: image}

    if current_model_path == "./models/secadero.onnx":
        # Perform inference
        ort_outs = ort_session.run(None, ort_inputs)
        return ort_outs[0][0] / 4
    else:
        # Perform inference
        ort_outs = ort_session.run(None, ort_inputs)
        return ort_outs[0][0] * 1000

# --- Watchdog Event Handler ---
class ImageHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()  # Call the parent class's initializer
        self.predictions = dict()
        self.first_prediction = None

    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith(('.jpg', '.jpeg', '.png')): # Added jpeg extension
            print(f"New image detected: {event.src_path}")
            try:
                image = preprocess_image(event.src_path)
                if ort_session:
                    prediction = run_inference(ort_session, image).tolist()
                    prediction_id = str(uuid.uuid4())
                    now = datetime.now()
                    loss = 0
                    percentage_loss = 0
                    # Calculate the percentage of loss
                    if self.first_prediction is None:
                        self.first_prediction = prediction[0]
                    try :
                        # print(f"Self Prediction: {self.predictions}")
                        # print(f"List of keys: {list(self.predictions.keys())}")
                        loss =  self.first_prediction - prediction[0]
                        percentage_loss = (loss / self.first_prediction) * 100
                        # print(f"Percentage Loss: {percentage_loss}")
                    except Exception as e:
                        print(f"Error calculating loss: {e}")
                    self.predictions[now.strftime("%Y-%m-%d %H:%M:%S")] = {
                        "id": prediction_id,
                        "name": os.path.basename(event.src_path),
                        "prediction": prediction[0],
                        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                        "loss": loss,
                        "percentage_loss": percentage_loss
                    }
                    self.predictions = OrderedDict(sorted(self.predictions.items(), key=lambda x: x[0], reverse=True))
                    # print(f"Prediction for {event.src_path}: {prediction[0]}")
                    print(f"Predictions: {self.predictions}")
                else:
                    print("No model loaded. Skipping prediction.")
            except Exception as e:
                print(f"Error processing image {event.src_path}: {e}")
            finally:
                pass  # Always pass in the finally block to avoid unhandled exceptions.

    def reset_predictions(self):
        self.predictions = dict()

image_handler = ImageHandler()
observer = Observer()
observer.schedule(image_handler, path=IMAGES_PATH, recursive=False)
observer.start()

# --- Flask Routes ---
@app.route('/')
def index():
    model_files = list_files(MODELS_PATH)
    return render_template('monitoring.html', model_files=model_files, current_model=os.path.basename(current_model_path) if current_model_path else None) # Pass model list and current model

@app.route('/predictions', methods=['GET'])
def get_predictions():
    # print(f"Predictions in get predictions: {image_handler.predictions}")
    return jsonify(image_handler.predictions)

@app.route('/reset', methods=['POST'])
def reset_predictions():
    image_handler.reset_predictions()
    return jsonify({"status": "predictions reset"})

@app.route('/images/<filename>')
def get_image(filename):
    return send_from_directory(IMAGES_PATH, filename)

@app.route('/load_model', methods=['POST'])
def load_selected_model():
    model_name = request.form.get('model_name')
    if not model_name:
        return jsonify({"status": "error", "message": "No model name provided"}), 400

    model_path = os.path.join(MODELS_PATH, model_name)
    if not os.path.exists(model_path):
        return jsonify({"status": "error", "message": "Model file not found"}), 404

    load_model(model_path) # Use the function to load model

    return jsonify({"status": "success", "message": f"Model '{model_name}' loaded successfully"})

@app.route('/images_thumbnail/<filename>')
def serve_image(filename):
    image_path = os.path.join(IMAGES_PATH, filename)
    try:
        with Image.open(image_path) as img:
            img.thumbnail((50, 50))  # Resize the image to a thumbnail
            img_io = BytesIO()
            img.save(img_io, 'JPEG', quality=85)
            img_io.seek(0)
            return send_file(img_io, mimetype='image/jpeg')
    except FileNotFoundError:
        return "Image not found", 404
    except Exception as e:
        print(f"Error serving thumbnail: {e}")
        return "Error serving image", 500

# --- Helper Functions ---

def list_files(path):
    try:
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        return files
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"Error listing files in {path}: {e}")
        return []

def list_folders(path):  # corrected function for listing folders
    try:
        folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f)) and not f.startswith('.')]
        return folders
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"Error listing folders in {path}: {e}")
        return []

# --- Flask Execution ---
def run_flask():
    app.run(host='localhost', port=5000, debug=True, use_reloader=False)

if __name__ == "__main__":
    # Ensure models and images directories exist
    os.makedirs(MODELS_PATH, exist_ok=True)
    os.makedirs(IMAGES_PATH, exist_ok=True)

    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
