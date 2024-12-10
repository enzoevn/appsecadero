# Load the model onnx model  with onnxruntime
import onnxruntime as ort
from PIL import Image
import numpy as np
import os

def list_files(folder):
    return [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f.endswith('.onnx')]

def list_folders(folder):
    folders = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]
    # Filter folders with jpg files
    return [f for f in folders if len([f for f in os.listdir(os.path.join(folder, f)) if f.endswith('.jpg') or f.endswith('.png')]) > 0]

def load_model(model_path):
    """
    Load the ONNX model from the given path.

    Parameters:
    model_path (str): Path to the ONNX model file.

    Returns:
    ort.InferenceSession: The loaded ONNX model session.
    """
    return ort.InferenceSession(model_path)

def preprocess_image(image_path):
    """
    Preprocess the image for ONNX model inference.

    Parameters:
    image_path (str): Path to the image file.

    Returns:
    np.ndarray: The preprocessed image.
    """
    # Load the image
    image = Image.open(image_path)
    image = image.resize((224, 224))
    
    # Normalize the image with ImageNet mean and std
    image = np.array(image)
    image = image / 255.0
    image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0).astype(np.float32)
    
    return image

def run_inference(ort_session, image):
    """
    Run inference on the preprocessed image using the ONNX model.

    Parameters:
    ort_session (ort.InferenceSession): The loaded ONNX model session.
    image (np.ndarray): The preprocessed image.

    Returns:
    list: The output from the model inference.
    """
    ort_inputs = {ort_session.get_inputs()[0].name: image}
    ort_outs = ort_session.run(None, ort_inputs)
    ort_outs = ort_outs[0][0] / 4
    # Round the output to 2 decimal places
    ort_outs = np.int32(ort_outs)
    return ort_outs

def preprocess_folder(folder_path):
    """
    Read all the images in the folder and preprocess them for inference.

    Parameters:
    folder_path (str): Path to the folder containing images.

    Returns:
    list: List of preprocessed images.
    """
    images = []
    image_names = sorted(os.listdir(folder_path))
    image_names = [image_name for image_name in image_names if image_name.endswith('.jpg') or image_name.endswith('.png')]
    for image_path in image_names:
        # Check if the file is an image
        if not image_path.endswith('.jpg') and not image_path.endswith('.png'):
            continue
        image = preprocess_image(os.path.join(folder_path, image_path))
        images.append(image)
    return images, image_names

# Run inference on the images
def calculate_mean_weight(predictions):
    weights = [prediction for prediction in predictions.values()]

    return np.int32(np.mean(weights))


# Pipeline
model_path = "secadero.onnx"
image_path = "images"

def run_pipeline(model_path, image_folder_path):
    """
    Run the complete pipeline: load the model, preprocess images, run inference, and collect predictions.

    Parameters:
    model_path (str): Path to the ONNX model file.
    image_folder_path (str): Path to the folder containing images.

    Returns:
    dict: A dictionary with image names as keys and their corresponding predictions as values.
    """
    # Load the model
    ort_session = load_model(model_path)

    # Preprocess the images in the folder
    images, images_names = preprocess_folder(image_folder_path)

    # Run inference
    predic_items = []
    for image in images:
        prediction = run_inference(ort_session, image).tolist()
        predic_items.append(prediction)

    # Collect predictions in a dictionary
    predic_dict = {}
    for i in range(len(images_names)):
        predic_dict[images_names[i]] = predic_items[i][0]

    return predic_dict

model_path = "secadero.onnx"
image_folder_path = "images"
predictions = run_pipeline(model_path, image_folder_path)
print(type(predictions))
