"""
Flask application for monitoring and processing images using ONNX models.

This application provides functionality to:
- Load and manage ONNX models
- Monitor a directory for new images
- Process images and make predictions using loaded models
- Serve predictions and images via a web interface
"""

import os
import threading
import time
from io import BytesIO

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_file,
    send_from_directory,
)
from image_handler_service import ImageMonitoringService
from model_service import ModelManager
from PIL import Image
from settings import (
    IMAGES_PATH,
    MODELS_PATH,
)
from utils import list_files, preprocess_image

app = Flask(__name__)

# Global model manager instance
model_manager = ModelManager()

# Global image monitoring service
image_monitoring_service = None


# --- Flask Routes ---
@app.route("/")
def index():
    """Render main monitoring page."""
    try:
        model_files = [
            model for model in list_files(MODELS_PATH) if model.endswith(".onnx")
        ]
        print(f"Model files: {model_files}")
        return render_template(
            "monitoring.html",
            model_files=model_files,
            current_model=os.path.basename(model_manager.current_model_path)
            if model_manager.current_model_path
            else None,
        )
    except Exception as e:
        print(f"Error in index route: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/predictions", methods=["GET"])
def get_predictions():
    """Get all stored predictions."""
    try:
        if image_monitoring_service:
            return jsonify(
                image_monitoring_service.get_image_service().get_predictions()
            )
        return jsonify({})
    except Exception as e:
        print(f"Error getting predictions: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/reset", methods=["POST"])
def reset_predictions():
    """Reset all predictions."""
    try:
        if image_monitoring_service:
            image_monitoring_service.get_image_service().reset_predictions()
        return jsonify({"status": "predictions reset"})
    except Exception as e:
        print(f"Error resetting predictions: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/images/<path:filepath>")
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


@app.route("/load_model", methods=["POST"])
def load_selected_model():
    """Load a selected model."""
    try:
        model_name = request.form.get("model_name")
        if not model_name:
            return jsonify(
                {"status": "error", "message": "No model name provided"}
            ), 400
        model_path = os.path.join(MODELS_PATH, model_name)
        if not os.path.exists(model_path):
            return jsonify(
                {"status": "error", "message": f"Model file not found: {model_path}"}
            ), 404
        model_manager.load_model(model_path)
        return jsonify(
            {
                "status": "success",
                "message": f"Model '{model_name}' loaded successfully",
            }
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/images_thumbnail/<path:filepath>")
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
            img.save(img_io, "JPEG", quality=85)
            img_io.seek(0)
            return send_file(img_io, mimetype="image/jpeg")
    except Exception as e:
        print(f"Error serving thumbnail: {e}")
        return "Error serving image", 500


@app.route("/predict_image", methods=["POST"])
def predict_image():
    """
    Endpoint to predict from an uploaded image and extra parameters.
    """
    try:
        image_file = request.files.get("image")
        days = request.form.get("days", type=float)
        initial_weight = request.form.get("initial_weight", type=float)
        if not image_file:
            return jsonify({"error": "No image uploaded"}), 400
        # Save image temporarily
        temp_path = os.path.join(IMAGES_PATH, "temp_input.png")
        image_file.save(temp_path)

        # Usar el tamaño de imagen del modelo cargado
        target_size = model_manager.image_size
        print(f"[DEBUG] Using image size for prediction: {target_size}")

        image = preprocess_image(temp_path, target_size)
        kwargs = {}
        if days is not None:
            kwargs["days"] = days
        if initial_weight is not None:
            kwargs["initial_weight"] = initial_weight
        prediction = model_manager.predict(image, **kwargs)
        os.remove(temp_path)
        return jsonify({"prediction": float(prediction)})
    except Exception as e:
        print(f"Error in /predict_image: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/set_days", methods=["POST"])
def set_days():
    """Allows updating the days elapsed for a specific prediction."""
    try:
        data = request.get_json()
        prediction_id = data.get("prediction_id")
        days = data.get("days", 0)

        if image_monitoring_service:
            success = (
                image_monitoring_service.get_image_service().update_prediction_days(
                    prediction_id, days
                )
            )
            if success:
                return jsonify({"status": "success"})

        return jsonify({"status": "error", "message": "Prediction not found"}), 404
    except Exception as e:
        print(f"Error in /set_days: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/set_initial_days", methods=["POST"])
def set_initial_days():
    """Allows saving the initial days elapsed when loading a model."""
    try:
        data = request.get_json()
        days = int(data.get("days", 0))

        if image_monitoring_service:
            image_monitoring_service.get_image_service().set_initial_days(days)

        return jsonify({"status": "success"})
    except Exception as e:
        print(f"Error in /set_initial_days: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/set_initial_weight", methods=["POST"])
def set_initial_weight():
    """Allows saving the initial weight for automatic predictions."""
    try:
        data = request.get_json()
        weight = float(data.get("initial_weight", 0))

        if image_monitoring_service:
            success = image_monitoring_service.get_image_service().set_initial_weight(
                weight
            )
            if success:
                return jsonify(
                    {
                        "status": "success",
                        "message": f"Initial weight set: {weight}g",
                    }
                )
            else:
                return jsonify(
                    {
                        "status": "error",
                        "message": f"The initial weight must be between 100g and 10000g. Received value: {weight}g",
                    }
                ), 400

        return jsonify({"status": "error", "message": "Service not available"}), 500
    except Exception as e:
        print(f"Error in /set_initial_weight: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# --- Flask Execution ---
def run_flask():
    """Run the Flask application."""
    try:
        app.run(host="localhost", port=5000, debug=True, use_reloader=False)
    except Exception as e:
        print(f"Error running Flask: {e}")
        raise


if __name__ == "__main__":
    try:
        # Ensure models and images directories exist
        os.makedirs(MODELS_PATH, exist_ok=True)
        os.makedirs(IMAGES_PATH, exist_ok=True)

        # Initialize image monitoring service
        image_monitoring_service = ImageMonitoringService(model_manager)
        image_monitoring_service.start()

        # No load default model - the user must select one
        print("[INFO] Application started. Please select a model to start.")
        flask_thread = threading.Thread(target=run_flask)
        flask_thread.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        if image_monitoring_service:
            image_monitoring_service.stop()
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise
    finally:
        if image_monitoring_service:
            image_monitoring_service.stop()
