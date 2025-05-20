from ultralytics import YOLO
import time

model_cache = {}

def run_yolo(model_path, frame):
    if model_path not in model_cache:
        model_cache[model_path] = YOLO(model_path)
    model = model_cache[model_path]

    start = time.time()
    results = model.track(source=frame, persist=True)
    end = time.time()

    annotated = results[0].plot()
    fps = 1 / (end - start)
    return annotated, fps
