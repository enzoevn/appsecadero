from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
import time
import os
from inference_onnx_secadero import run_pipeline

class ImageHandler(FileSystemEventHandler):
    def __init__(self, model_path, image_folder_path):
        self.model_path = model_path
        self.image_folder_path = image_folder_path

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(('.png', '.jpg', '.jpeg')):
            print(f"New image detected: {event.src_path}")
            predictions = run_pipeline(self.model_path, self.image_folder_path)
            print(predictions)

def start_watching(model_path, image_folder_path):
    event_handler = ImageHandler(model_path, image_folder_path)
    observer = Observer()
    observer.schedule(event_handler, path=image_folder_path, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()