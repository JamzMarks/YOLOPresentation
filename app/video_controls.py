import cv2
import time
import threading
from app.detection import run_yolo

class VideoController:
    def __init__(self, model_path, video_path, gui):
        self.model_path = model_path
        self.video_path = video_path
        self.gui = gui

        self.cap = None
        self.paused = False
        self.running = False

    def start(self):
        self.running = True
        self.paused = False
        threading.Thread(target=self.run, daemon=True).start()

    def toggle_pause(self):
        self.paused = not self.paused

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def run(self):
        self.cap = cv2.VideoCapture(self.video_path)
        window_name = f"YOLO Detection - {self.video_path}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while self.cap.isOpened() and self.running:
            if self.paused:
                time.sleep(0.1)
                continue

            ret, frame = self.cap.read()
            if not ret:
                break

            annotated, fps = run_yolo(self.model_path, frame)

            cv2.putText(annotated, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow(window_name, annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.stop()
