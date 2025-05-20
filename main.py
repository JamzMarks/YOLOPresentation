import cv2
import os
import time
import threading
import tkinter as tk
from tkinter import filedialog, ttk
from ultralytics import YOLO

class YOLOTrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 Video Detector")
        self.root.geometry("500x300")
        self.root.resizable(False, False)

        self.model = None
        self.model_path = None
        self.video_path = None
        self.cap = None
        self.paused = False
        self.running = False

        self.build_interface()

    def build_interface(self):
        self.clear_root()

        tk.Label(self.root, text="Selecione um modelo YOLO:", font=("Arial", 12)).pack(pady=10)

        self.model_var = tk.StringVar()
        model_files = [f for f in os.listdir("models") if f.endswith(".pt")]
        self.model_dropdown = ttk.Combobox(self.root, textvariable=self.model_var, values=model_files, state="readonly")
        self.model_dropdown.pack(pady=5)

        tk.Button(self.root, text="Selecionar Vídeo", command=self.select_video).pack(pady=5)
        self.video_label = tk.Label(self.root, text="Nenhum vídeo selecionado", wraplength=400)
        self.video_label.pack(pady=5)

        tk.Button(self.root, text="Iniciar Detecção", command=self.start_detection, bg="#4CAF50", fg="white").pack(pady=10)

    def clear_root(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def select_video(self):
        file_path = filedialog.askopenfilename(title="Selecione o vídeo", filetypes=[("Vídeos", "*.mp4 *.avi *.mov")])
        if file_path:
            self.video_path = file_path
            self.video_label.config(text=f"Vídeo: {os.path.basename(file_path)}")

    def start_detection(self):
        if not self.model_var.get():
            tk.messagebox.showerror("Erro", "Selecione um modelo YOLO")
            return
        if not self.video_path:
            tk.messagebox.showerror("Erro", "Selecione um vídeo")
            return

        self.model_path = os.path.join("models", self.model_var.get())
        self.model = YOLO(self.model_path)

        self.clear_root()
        self.build_video_controls()

        self.running = True
        self.paused = False

        threading.Thread(target=self.run_detection, daemon=True).start()

    def build_video_controls(self):
        tk.Label(self.root, text=f"Rodando: {os.path.basename(self.video_path)}", font=("Arial", 12)).pack(pady=5)

        tk.Button(self.root, text="⏸️ Pausar / Continuar", command=self.toggle_pause).pack(pady=5)
        tk.Button(self.root, text="🔁 Voltar", command=self.reset_to_menu).pack(pady=5)
        tk.Button(self.root, text="❌ Sair", command=self.root.quit, fg="white", bg="#d9534f").pack(pady=10)

    def toggle_pause(self):
        self.paused = not self.paused

    def reset_to_menu(self):
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.build_interface()

    def run_detection(self):
        self.cap = cv2.VideoCapture(self.video_path)
        window_name = f"YOLOv8 - {os.path.basename(self.video_path)}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while self.cap.isOpened() and self.running:
            if self.paused:
                time.sleep(0.1)
                continue

            ret, frame = self.cap.read()
            if not ret:
                break

            start = time.time()
            results = self.model.track(source=frame, persist=True)
            end = time.time()

            annotated = results[0].plot()
            fps = 1 / (end - start)

            cv2.putText(annotated, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow(window_name, annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        self.running = False

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOTrackerApp(root)
    root.mainloop()
