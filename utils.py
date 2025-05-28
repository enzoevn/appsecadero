import os
from datetime import datetime

import numpy as np
from PIL import Image


# --- Helper Functions ---
def list_files(path):
    """
    List all ONNX model files in subdirectories, returning their relative paths.
    Args:
        path (str): Directory path
    Returns:
        list: List of relative file paths
    """
    try:
        model_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".onnx"):
                    rel_path = os.path.relpath(os.path.join(root, file), path)
                    model_files.append(rel_path)
        return model_files
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
        folders = [
            f
            for f in os.listdir(path)
            if os.path.isdir(os.path.join(path, f)) and not f.startswith(".")
        ]
        return folders
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"Error listing folders in {path}: {e}")
        return []


# --- Image Processing and Inference Functions ---
def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess an image for model inference.
    Args:
        image_path (str): Path to the image file
        target_size (tuple): Target size for resizing (width, height)
    Returns:
        numpy.ndarray: Preprocessed image array
    """
    try:
        image = Image.open(image_path)
        image = image.resize(target_size)
        image = np.array(image) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array(
            [0.229, 0.224, 0.225]
        )
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0).astype(np.float32)
        return image
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        raise


def extract_date_from_folder(folder_name):
    """
    Extracts the datetime object from a folder name with format 'YYYY-MM-DD_HH-MM-SS-XXXX.jfz'.
    """
    try:
        date_str = folder_name.split(".")[0]  # Remove extension
        date_obj = datetime.strptime(date_str, "%Y-%m-%d_%H-%M-%S-%f")
        return date_obj
    except Exception as e:
        print(f"Error parsing date from folder: {folder_name} -> {e}")
        return None


def get_min_date_from_folders(images_path):
    """
    Finds the earliest date among all image folders.
    """
    min_date = None
    for folder in os.listdir(images_path):
        folder_path = os.path.join(images_path, folder)
        if os.path.isdir(folder_path):
            date_obj = extract_date_from_folder(folder)
            if date_obj:
                if min_date is None or date_obj < min_date:
                    min_date = date_obj
    return min_date
