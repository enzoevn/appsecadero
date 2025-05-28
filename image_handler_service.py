"""
Service for handling and processing images using ONNX models.

This service provides functionality for:
- Monitor directories for new images
- Process images and make predictions
- Manage the storage of prediction results
- Calculate loss metrics and percentages
"""

import os
import time
import uuid
from collections import OrderedDict
from datetime import datetime

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from settings import IMAGES_PATH
from utils import extract_date_from_folder, preprocess_image


class ImageProcessingService:
    """
    Service for processing images and managing predictions.

    Attributes:
        predictions (dict): Dictionary that stores prediction results
        first_prediction (float): First prediction for loss calculation
        initial_days_reference (int): Initial days reference defined by the user
        initial_weight_reference (float): Initial weight reference for loss calculations
        model_manager: Instance of the model manager
        min_date_cache (datetime): Cached minimum date from folders to avoid recalculating
        processed_folders (set): Track processed folders to avoid reprocessing
    """

    def __init__(self, model_manager):
        """
        Initialize the image processing service.

        Args:
            model_manager: Instance of the ModelManager for making predictions
        """
        try:
            self.predictions = dict()
            self.first_prediction = None
            self.initial_days_reference = None
            self.initial_weight_reference = 2500  # Default value of 2.5kg
            self.model_manager = model_manager
            self.min_date_cache = None  # Cache para evitar recalcular la fecha mínima
            self.processed_folders = (
                set()
            )  # Track processed folders to avoid reprocessing
            self._initialize_processed_folders()
        except Exception as e:
            print(f"Error initializing ImageProcessingService: {e}")
            raise

    def _initialize_processed_folders(self):
        """Initialize the set of processed folders with existing folders at startup."""
        try:
            if os.path.exists(IMAGES_PATH):
                for folder in os.listdir(IMAGES_PATH):
                    folder_path = os.path.join(IMAGES_PATH, folder)
                    if os.path.isdir(folder_path):
                        self.processed_folders.add(folder)
                        print(
                            f"[DEBUG] Marking existing folder as processed at startup: {folder}"
                        )
        except Exception as e:
            print(f"Error initializing processed folders: {e}")

    def _reset_for_new_session(self):
        """Reset the system for a new prediction session (when days are set)."""
        try:
            self.min_date_cache = None
            self.processed_folders.clear()
            print("[DEBUG] System reset for new prediction session")
        except Exception as e:
            print(f"Error resetting for new session: {e}")

    def _get_or_update_min_date(self, current_folder_date):
        """
        Get the minimum date from cache or update it if necessary.

        Args:
            current_folder_date (datetime): Date of the current folder being processed

        Returns:
            datetime: The minimum date among all folders
        """
        try:
            # Si no tenemos cache, usar la fecha actual como nueva referencia
            if self.min_date_cache is None:
                print(
                    "[DEBUG] No min_date_cache - using current image as new reference"
                )

                if current_folder_date:
                    print(f"[DEBUG] Setting new reference date: {current_folder_date}")
                    self.min_date_cache = current_folder_date
                else:
                    print("[DEBUG] No valid current folder date, cannot set reference")
                    return None

                print(f"[DEBUG] New reference date established: {self.min_date_cache}")

            # Si la fecha actual es menor que la mínima conocida, actualizar
            elif current_folder_date and current_folder_date < self.min_date_cache:
                print(
                    f"[DEBUG] New minimum date found. Previous: {self.min_date_cache}, Current folder: {current_folder_date}"
                )
                self.min_date_cache = current_folder_date
                print(f"[DEBUG] Updated min_date_cache: {self.min_date_cache}")

            return self.min_date_cache
        except Exception as e:
            print(f"Error updating min date cache: {e}")
            # En caso de error, usar la fecha actual si está disponible
            return current_folder_date if current_folder_date else None

    def process_image(self, image_path):
        """
        Process an image and store the prediction results.

        Args:
            image_path (str): Path to the image file
        """
        print(f"[DEBUG] Entering process_image with: {image_path}")
        try:
            folder_name = os.path.basename(os.path.dirname(image_path))

            # Check if this folder was already processed to avoid reprocessing
            if folder_name in self.processed_folders:
                print(f"[DEBUG] Folder {folder_name} already processed, skipping...")
                return

            # Mark this folder as processed
            self.processed_folders.add(folder_name)
            print(f"[DEBUG] Processing new folder: {folder_name}")

            # Use the image size of the loaded model
            target_size = self.model_manager.image_size
            print(f"[DEBUG] Using image size: {target_size}")

            image = preprocess_image(image_path, target_size)

            # Only calculate days if the model actually needs them (hybrid models)
            days = 0
            kwargs = {}

            # Check if model needs days calculation BEFORE doing any date processing
            if (
                self.model_manager.infer_function == self.model_manager.infer_hybrid
                or self.model_manager.infer_function
                == self.model_manager.infer_hybrid_initialweights
            ):
                print("[DEBUG] Hybrid model detected, calculating days...")
                folder_date = extract_date_from_folder(folder_name)
                # Usar el método optimizado para obtener la fecha mínima
                min_date = self._get_or_update_min_date(folder_date)
                print(
                    f"[DEBUG] folder_name: {folder_name}, folder_date: {folder_date}, min_date: {min_date}"
                )

                if folder_date and min_date:
                    days = (folder_date - min_date).days

                # If the user defined initial days, add them
                if self.initial_days_reference is not None:
                    days += self.initial_days_reference
                print(f"[DEBUG] Days calculated (with reference): {days}")
                kwargs["days"] = days

                if (
                    self.model_manager.infer_function
                    == self.model_manager.infer_hybrid_initialweights
                ):
                    kwargs["initial_weight"] = self.initial_weight_reference
                    print(
                        f"[DEBUG] Using initial weight: {self.initial_weight_reference}g"
                    )
                    if self.initial_weight_reference < 100:
                        print(
                            f"[WARNING] Initial weight too low: {self.initial_weight_reference}g. Is it configured correctly?"
                        )
            else:
                print("[DEBUG] ResNet model detected, no days calculation needed")

            if self.model_manager.ort_session:
                print(f"[DEBUG] Calling model_manager.predict with kwargs: {kwargs}")
                prediction = self.model_manager.predict(image, **kwargs)

                # Handle both numpy arrays and scalar values
                if hasattr(prediction, "tolist"):
                    prediction_value = prediction.tolist()
                    if isinstance(prediction_value, list):
                        prediction_value = (
                            prediction_value[0] if prediction_value else 0
                        )
                else:
                    prediction_value = float(prediction)

                prediction_id = str(uuid.uuid4())
                now = datetime.now()
                loss = 0
                percentage_loss = 0

                # Calcular merma usando el peso inicial configurado
                initial_weight_for_calculation = kwargs.get(
                    "initial_weight", self.initial_weight_reference
                )
                print("[DEBUG] Values for loss calculation:")
                print(
                    f"[DEBUG] - initial_weight_reference (global): {self.initial_weight_reference}"
                )
                print(
                    f"[DEBUG] - kwargs.get('initial_weight'): {kwargs.get('initial_weight')}"
                )
                print(
                    f"[DEBUG] - initial_weight_for_calculation: {initial_weight_for_calculation}"
                )
                print(f"[DEBUG] - prediction_value: {prediction_value}")

                if (
                    initial_weight_for_calculation
                    and initial_weight_for_calculation > 0
                ):
                    try:
                        loss = initial_weight_for_calculation - prediction_value
                        percentage_loss = (loss / initial_weight_for_calculation) * 100
                        print(
                            f"[DEBUG] Loss calculated: {loss}g ({percentage_loss:.1f}%) - Initial weight: {initial_weight_for_calculation}g, Prediction: {prediction_value}g"
                        )
                    except Exception as e:
                        print(f"Error calculating loss with initial weight: {e}")
                else:
                    # Fallback to the previous method if no initial weight is configured
                    print(
                        "[DEBUG] No initial weight configured, using first prediction as reference"
                    )
                    if self.first_prediction is None:
                        self.first_prediction = prediction_value
                    try:
                        loss = self.first_prediction - prediction_value
                        percentage_loss = (loss / self.first_prediction) * 100
                        print(
                            f"[DEBUG] Loss calculated with first prediction: {loss}g ({percentage_loss:.1f}%)"
                        )
                    except Exception as e:
                        print(f"Error calculating loss with first prediction: {e}")

                self.predictions[now.strftime("%Y-%m-%d %H:%M:%S")] = {
                    "id": prediction_id,
                    "name": os.path.basename(image_path),
                    "folder": folder_name,
                    "prediction": prediction_value,
                    "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                    "loss": loss,
                    "percentage_loss": percentage_loss,
                    "days": days,
                    "initial_weight": kwargs.get("initial_weight", None),
                }

                self.predictions = OrderedDict(
                    sorted(self.predictions.items(), key=lambda x: x[0], reverse=True)
                )
                print(f"Predictions: {self.predictions}")
            else:
                print("No model loaded. Skipping prediction.")
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

    def reset_predictions(self):
        """Reset all stored predictions."""
        try:
            self.predictions = dict()
            self.first_prediction = None  # Reset first prediction reference
            self.min_date_cache = None  # Reset min date cache
            self.processed_folders.clear()  # Clear processed folders to allow reprocessing
            print("[DEBUG] Predictions, min_date_cache, and processed folders reset")
        except Exception as e:
            print(f"Error resetting predictions: {e}")

    def set_initial_days(self, days):
        """
        Set the initial days reference.

        Args:
            days (int): Number of initial days
        """
        self.initial_days_reference = days
        print(f"[DEBUG] Initial days set: {days}")
        print("[DEBUG] Cache reset - next new image will be used as reference date")
        self._reset_for_new_session()

    def set_initial_weight(self, weight):
        """
        Set the initial weight reference.

        Args:
            weight (float): Initial weight in grams

        Returns:
            bool: True if the weight is valid, False otherwise
        """
        # Validation: the initial weight must be reasonable (between 100g and 10kg)
        if weight < 100 or weight > 10000:
            print(
                f"[ERROR] The initial weight must be between 100g and 10000g. Received value: {weight}g"
            )
            return False

        self.initial_weight_reference = weight
        print(f"[DEBUG] Initial weight set: {weight}g")
        return True

    def update_prediction_days(self, prediction_id, days):
        """
        Update the days elapsed for a specific prediction.

        Args:
            prediction_id (str): ID of the prediction
            days (int): New days elapsed

        Returns:
            bool: True if updated correctly, False if not found
        """
        for pred in self.predictions.values():
            if pred["id"] == prediction_id:
                pred["days"] = days
                return True
        return False

    def get_predictions(self):
        """
        Get all stored predictions.

        Returns:
            dict: Dictionary with all predictions
        """
        return self.predictions


class ImageHandler(FileSystemEventHandler):
    """
    File system event handler for monitoring image creation.

    Attributes:
        image_service (ImageProcessingService): Image processing service
    """

    def __init__(self, image_service):
        """
        Initialize the event handler.

        Args:
            image_service (ImageProcessingService): Image processing service
        """
        try:
            super().__init__()
            self.image_service = image_service
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
            print(
                f"[DEBUG] Event created: {event.src_path}, is_directory={event.is_directory}"
            )
            if event.is_directory:
                print(f"New folder detected: {event.src_path}")
                # Asegurar que src_path es string
                folder_path = str(event.src_path)
                image_path_png = os.path.join(folder_path, "Input0_Camera0.png")
                image_path_jpg = os.path.join(folder_path, "Input0_Camera0.jpg")
                time.sleep(1)
                print(
                    f"[DEBUG] Searching for image in: {image_path_png} or {image_path_jpg}"
                )

                if os.path.exists(image_path_png):
                    print("[DEBUG] PNG image found, processing...")
                    self.image_service.process_image(image_path_png)
                elif os.path.exists(image_path_jpg):
                    print("[DEBUG] JPG image found, processing...")
                    self.image_service.process_image(image_path_jpg)
                else:
                    print("[DEBUG] Image not found.")
        except Exception as e:
            print(f"Error in on_created: {e}")


class ImageMonitoringService:
    """
    Main service for monitoring images that coordinates the observer and processing.
    """

    def __init__(self, model_manager, watch_path=IMAGES_PATH):
        """
        Initialize the monitoring service.

        Args:
            model_manager: Instance of the ModelManager
            watch_path (str): Path to monitor
        """
        try:
            self.image_service = ImageProcessingService(model_manager)
            self.image_handler = ImageHandler(self.image_service)
            self.observer = Observer()
            self.watch_path = watch_path
            self._setup_observer()
        except Exception as e:
            print(f"Error initializing ImageMonitoringService: {e}")
            raise

    def _setup_observer(self):
        """Configure the file system observer."""
        try:
            self.observer.schedule(
                self.image_handler, path=self.watch_path, recursive=True
            )
        except Exception as e:
            print(f"Error configuring observer: {e}")
            raise

    def start(self):
        """Start monitoring."""
        try:
            self.observer.start()
            print(f"[INFO] Monitoring started in: {self.watch_path}")
        except Exception as e:
            print(f"Error starting observer: {e}")
            raise

    def stop(self):
        """Stop monitoring."""
        try:
            self.observer.stop()
            self.observer.join()
            print("[INFO] Monitoring stopped")
        except Exception as e:
            print(f"Error stopping observer: {e}")

    def get_image_service(self):
        """
        Get the instance of the image processing service.

        Returns:
            ImageProcessingService: Instance of the image processing service
        """
        return self.image_service
