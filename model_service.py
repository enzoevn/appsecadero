import os
import re

import numpy as np
import onnxruntime as ort


def extract_image_size_from_model_name(model_path):
    """
    Extract the image size from the model name.
    Search for patterns like '256x256', '224x224', '640x640', etc.

    Args:
        model_path (str): Path to the model file

    Returns:
        tuple: (width, height) or None if the pattern is not found
    """
    try:
        filename = os.path.basename(model_path)
        # Buscar patrón de tamaño como 256x256, 224x224, etc.
        pattern = r"(\d+)x(\d+)"
        match = re.search(pattern, filename)
        if match:
            width = int(match.group(1))
            height = int(match.group(2))
            return (width, height)
        return None
    except Exception as e:
        print(f"Error extracting image size from model name: {e}")
        return None


class ModelManager:
    """
    Class to manage ONNX model loading and inference dispatching.
    """

    def __init__(self):
        self.ort_session = None
        self.current_model_path = None
        self.infer_function = None
        self.image_size = (224, 224)  # Tamaño por defecto

    def load_model(self, model_path):
        """
        Load an ONNX model and set the appropriate inference function based on its folder.
        Args:
            model_path (str): Path to the ONNX model file
        """
        try:
            self.ort_session = ort.InferenceSession(model_path)
            self.current_model_path = model_path

            # Extraer tamaño de imagen del nombre del modelo
            extracted_size = extract_image_size_from_model_name(model_path)
            if extracted_size:
                self.image_size = extracted_size
                print(f"[DEBUG] Image size extracted from model: {self.image_size}")
            else:
                # Usar tamaño por defecto basado en el tipo de modelo
                parent_folder = os.path.basename(os.path.dirname(model_path))
                if parent_folder in [
                    "SecaderoDaysFeature",
                    "SecaderoDaysAndWeightFeature",
                ]:
                    self.image_size = (640, 640)  # Default para modelos híbridos
                else:
                    self.image_size = (224, 224)  # Default para ResNet
                print(f"[DEBUG] Using default image size: {self.image_size}")

            # Print input names and count for debugging
            input_names = [i.name for i in self.ort_session.get_inputs()]
            print(f"[DEBUG] Model inputs ({len(input_names)}): {input_names}")
            # Detect model type by parent folder
            parent_folder = os.path.basename(os.path.dirname(model_path))
            if parent_folder == "SecaderoResnet":
                self.infer_function = self.infer_resnet
            elif parent_folder == "SecaderoDaysFeature":
                self.infer_function = self.infer_hybrid
            elif parent_folder == "SecaderoDaysAndWeightFeature":
                self.infer_function = self.infer_hybrid_initialweights
            else:
                # Default to ResNet inference if unknown
                self.infer_function = self.infer_resnet
            print(
                f"Model loaded successfully from: {model_path} (type: {self.infer_function.__name__}, size: {self.image_size})"
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            self.ort_session = None
            self.current_model_path = None
            self.infer_function = None
            self.image_size = (224, 224)

    def infer_resnet(self, image, **kwargs):
        """
        Inference for standard ResNet ONNX models.
        Args:
            image (np.ndarray): Preprocessed image
        Returns:
            float: Model prediction
        """
        if self.ort_session is None:
            raise RuntimeError("No ONNX session available for inference.")
        input_name = self.ort_session.get_inputs()[0].name
        ort_inputs = {input_name: image}
        ort_outs = self.ort_session.run(None, ort_inputs)
        print(f"[DEBUG] ResNet output: {ort_outs[0][0]}")
        return ort_outs[0][0] * 1000

    def infer_hybrid(self, image, days=None, **kwargs):
        """
        Inference for Hybrid ONNX models.
        Args:
            image (np.ndarray): Preprocessed image
        Returns:
            float: Model prediction
        """
        if self.ort_session is None:
            raise RuntimeError("No ONNX session available for inference.")
        input_names = [i.name for i in self.ort_session.get_inputs()]
        ort_inputs = {input_names[0]: image}
        if len(input_names) > 1 and days is not None:
            ort_inputs[input_names[1]] = np.array([[days]], dtype=np.float32)
        ort_outs = self.ort_session.run(None, ort_inputs)
        print(f"[DEBUG] Hybrid output: {ort_outs[0][0]}")
        return ort_outs[0][0] * 1000

    def infer_hybrid_initialweights(
        self, image, days=None, initial_weight=None, **kwargs
    ):
        """
        Inference for HybridInitialWeights ONNX models.
        Args:
            image (np.ndarray): Preprocessed image
            days (float): Days elapsed
            initial_weight (float): Initial weight in grams (will be converted to kg for model)
        Returns:
            float: Model prediction in grams
        """
        if self.ort_session is None:
            raise RuntimeError("No ONNX session available for inference.")
        input_names = [i.name for i in self.ort_session.get_inputs()]
        ort_inputs = {input_names[0]: image}
        if len(input_names) > 1 and days is not None:
            ort_inputs[input_names[1]] = np.array([[days]], dtype=np.float32)
        if len(input_names) > 2 and initial_weight is not None:
            # Convertir de gramos a kilos para el modelo
            initial_weight_kg = initial_weight / 1000.0
            ort_inputs[input_names[2]] = np.array(
                [[initial_weight_kg]], dtype=np.float32
            )
            print(
                f"[DEBUG] Initial weight sent to model: {initial_weight_kg}kg (original: {initial_weight}g)"
            )
        ort_outs = self.ort_session.run(None, ort_inputs)
        print(f"[DEBUG] HybridInitialWeights output: {ort_outs[0][0]}")
        return ort_outs[0][0] * 1000

    def predict(self, image, **kwargs):
        """
        Run inference using the selected model and function.
        Args:
            image (np.ndarray): Preprocessed image
        Returns:
            float: Model prediction
        """
        if self.ort_session is None or self.infer_function is None:
            raise RuntimeError("No model loaded or inference function not set.")
        return self.infer_function(image, **kwargs)
