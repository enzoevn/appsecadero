"""
Flask application for monitoring and processing images using ONNX models.

This application provides functionality to:
- Load and manage ONNX models
- Monitor a directory for new images
- Process images and make predictions using loaded models
- Serve predictions and images via a web interface
"""

import os
import uuid
from collections import OrderedDict
from io import BytesIO
import time
from datetime import datetime
import threading
from flask import Flask, jsonify, render_template, send_from_directory, request, send_file
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
import onnxruntime as ort
from PIL import Image
import numpy as np

# --- Configuration ---
MODELS_PATH = './models'  # Path to store ONNX models
IMAGES_PATH = './images'  # Path that watchdog monitors
DEFAULT_MODEL = "secadero.onnx"  # Default ONNX model

OLD_MODEL_LIST = ["secadero.onnx", "secadero_2.onnx"]

app = Flask(__name__)

# Global variable to hold the ONNX session and model path
ort_session = None
current_model_path = None

def load_model(model_path):
    """
    Load an ONNX model from the specified path.
    
    Args:
        model_path (str): Path to the ONNX model file
        
    Returns:
        None
    """
    global ort_session, current_model_path
    try:
        ort_session = ort.InferenceSession(model_path)
        current_model_path = model_path
        print(f"Model loaded successfully from: {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        ort_session = None  # Ensure session is None if loading fails
        current_model_path = None

try:
    load_model(os.path.join(MODELS_PATH, DEFAULT_MODEL))  # Load initial model
except Exception as e:
    print(f"Error loading initial model: {e}")

# --- Image Processing and Inference Functions ---
def preprocess_image(image_path):
    """
    Preprocess an image for model inference.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        numpy.ndarray: Preprocessed image array
    """
    try:
        image = Image.open(image_path)
        image = image.resize((224, 224))
        image = np.array(image) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0).astype(np.float32)
        return image
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        raise

def run_inference(ort_session, image):
    """
    Run model inference on a preprocessed image.
    
    Args:
        ort_session: ONNX Runtime session
        image (numpy.ndarray): Preprocessed image array
        
    Returns:
        float: Model prediction
    """
    try:
        input_name = ort_session.get_inputs()[0].name
        ort_inputs = {input_name: image}

        model_name = os.path.basename(current_model_path)
        print(f"Model name: {model_name}")

        if model_name in OLD_MODEL_LIST:
            ort_outs = ort_session.run(None, ort_inputs)
            return ort_outs[0][0] / 4
        else:
            ort_outs = ort_session.run(None, ort_inputs)
            return ort_outs[0][0] * 1000
    except Exception as e:
        print(f"Error running inference: {e}")
        raise

# --- Watchdog Event Handler ---
class ImageHandler(FileSystemEventHandler):
    """
    Handler for monitoring image file creation events.
    
    Attributes:
        predictions (dict): Dictionary storing prediction results
        first_prediction (float): First prediction value for loss calculation
    """
    def __init__(self):
        try:
            super().__init__()
            self.predictions = dict()
            self.first_prediction = None
        except Exception as e:
            print(f"Error initializing ImageHandler: {e}")
            raise

    def on_created(self, event):
        """
        Handle file/directory creation events.
        
        Args:
            event: File system event object
        """
        try:
            if event.is_directory:
                print(f"New folder detected: {event.src_path}")
                image_path = os.path.join(event.src_path, "Input0_Camera.png")
                time.sleep(1)
                if os.path.exists(image_path):
                    self.process_image(image_path)
        except Exception as e:
            print(f"Error in on_created: {e}")

    def process_image(self, image_path):
        """
        Process an image and store prediction results.
        
        Args:
            image_path (str): Path to the image file
        """
        print(f"Processing image: {image_path}")
        try:
            image = preprocess_image(image_path)
            if ort_session:
                prediction = run_inference(ort_session, image).tolist()
                prediction_id = str(uuid.uuid4())
                now = datetime.now()
                loss = 0
                percentage_loss = 0
                
                if self.first_prediction is None:
                    self.first_prediction = prediction[0]
                try:
                    loss = self.first_prediction - prediction[0]
                    percentage_loss = (loss / self.first_prediction) * 100
                except Exception as e:
                    print(f"Error calculating loss: {e}")
                
                self.predictions[now.strftime("%Y-%m-%d %H:%M:%S")] = {
                    "id": prediction_id,
                    "name": os.path.basename(image_path),
                    "folder": os.path.basename(os.path.dirname(image_path)),
                    "prediction": prediction[0],
                    "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                    "loss": loss,
                    "percentage_loss": percentage_loss
                }
                self.predictions = OrderedDict(sorted(self.predictions.items(), key=lambda x: x[0], reverse=True))
                print(f"Predictions: {self.predictions}")
            else:
                print("No model loaded. Skipping prediction.")
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

    def reset_predictions(self):
        """Reset all stored predictions."""
        try:
            self.predictions = dict()
        except Exception as e:
            print(f"Error resetting predictions: {e}")

try:
    image_handler = ImageHandler()
    observer = Observer()
    observer.schedule(image_handler, path=IMAGES_PATH, recursive=True)
    observer.start()
except Exception as e:
    print(f"Error starting observer: {e}")
    raise

# --- Flask Routes ---
@app.route('/')
def index():
    """Render main monitoring page."""
    try:
        model_files = list_files(MODELS_PATH)
        return render_template('monitoring.html', model_files=model_files, current_model=os.path.basename(current_model_path) if current_model_path else None)
    except Exception as e:
        print(f"Error in index route: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/predictions', methods=['GET'])
def get_predictions():
    """Get all stored predictions."""
    try:
        return jsonify(image_handler.predictions)
    except Exception as e:
        print(f"Error getting predictions: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset_predictions():
    """Reset all predictions."""
    try:
        image_handler.reset_predictions()
        return jsonify({"status": "predictions reset"})
    except Exception as e:
        print(f"Error resetting predictions: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/images/<path:filepath>')
def get_image(filepath):
    """
    Serve an image file.
    
    Args:
        filepath (str): Path to the image file
    """
    try:
        print(f"Requesting image: {filepath}")
        return send_from_directory(IMAGES_PATH, filepath)
    except Exception as e:
        print(f"Error serving image: {e}")
        return "Error serving image", 500

@app.route('/load_model', methods=['POST'])
def load_selected_model():
    """Load a selected model."""
    try:
        model_name = request.form.get('model_name')
        if not model_name:
            return jsonify({"status": "error", "message": "No model name provided"}), 400

        model_path = os.path.join(MODELS_PATH, model_name)
        if not os.path.exists(model_path):
            return jsonify({"status": "error", "message": "Model file not found"}), 404

        load_model(model_path)

        return jsonify({"status": "success", "message": f"Model '{model_name}' loaded successfully"})
    except Exception as e:
        print(f"Error loading model: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/images_thumbnail/<path:filepath>')
def serve_image(filepath):
    """
    Serve a thumbnail version of an image.
    
    Args:
        filepath (str): Path to the image file
    """
    try:
        image_path = os.path.join(IMAGES_PATH, filepath)
        
        if not os.path.exists(image_path):
            print(f"Thumbnail - File not found: {image_path}")
            return "Image not found", 404
            
        with Image.open(image_path) as img:
            img.thumbnail((50, 50))
            img_io = BytesIO()
            img.save(img_io, 'JPEG', quality=85)
            img_io.seek(0)
            return send_file(img_io, mimetype='image/jpeg')
    except Exception as e:
        print(f"Error serving thumbnail: {e}")
        return "Error serving image", 500

# --- Helper Functions ---

def list_files(path):
    """
    List all files in a directory.
    
    Args:
        path (str): Directory path
        
    Returns:
        list: List of file names
    """
    try:
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        return files
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"Error listing files in {path}: {e}")
        return []

def list_folders(path):
    """
    List all folders in a directory.
    
    Args:
        path (str): Directory path
        
    Returns:
        list: List of folder names
    """
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
    """Run the Flask application."""
    try:
        app.run(host='localhost', port=5000, debug=True, use_reloader=False)
    except Exception as e:
        print(f"Error running Flask: {e}")
        raise

if __name__ == "__main__":
    try:
        # Ensure models and images directories exist
        os.makedirs(MODELS_PATH, exist_ok=True)
        os.makedirs(IMAGES_PATH, exist_ok=True)

        flask_thread = threading.Thread(target=run_flask)
        flask_thread.start()

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise
    finally:
        observer.join()
