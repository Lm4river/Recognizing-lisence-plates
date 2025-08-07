import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import torch
import re
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='plate_detector.pt')
model.conf = 0.4

# Load VietOCR model
config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = 'vietocr/vgg_transformer.pth'
config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
config['predictor']['beamsearch'] = True
detector = Predictor(config)

def clean_text(text):
    return ''.join(e for e in text if e.isalnum() or e == '-')

def is_valid_plate(text):
    pattern1 = r'^[0-9]{2}[A-Z]-?[0-9]{4,5}$'
    pattern2 = r'^[0-9]{2}[A-Z]{1,2}[0-9]{1,2}-?[0-9]{4,5}$'
    return re.match(pattern1, text) or re.match(pattern2, text)

def recognize_plate(plate_img, detector):
    h, w = plate_img.shape[:2]
    pil_img = Image.fromarray(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB))

    if h / w > 0.5:
        mid = h // 2
        top_half = pil_img.crop((0, 0, w, mid))
        bottom_half = pil_img.crop((0, mid, w, h))
        top_text = detector.predict(top_half)
        bottom_text = detector.predict(bottom_half)
        full_text = clean_text(top_text + bottom_text)
    else:
        full_text = clean_text(detector.predict(pil_img))
    return full_text

class PlateRecognitionApp:
    def __init__(self, root):
        self.root = root
        root.title("Biển số xe - Nhận diện & Crop")

        self.btn_load = tk.Button(root, text="Chọn ảnh test", command=self.load_image)
        self.btn_load.pack(pady=10)

        self.canvas_orig = tk.Canvas(root, width=640, height=360)
        self.canvas_orig.pack()
        self.canvas_crop = tk.Canvas(root, width=320, height=90)
        self.canvas_crop.pack(pady=10)

        self.label_text = tk.Label(root, text="Text biển số sẽ hiện ở đây", font=("Arial", 14))
        self.label_text.pack(pady=10)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files","*.jpg *.jpeg *.png")])
        if not file_path:
            return
        self.image_path = file_path

        # Đọc ảnh
        self.image = cv2.imread(file_path)
        img_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil = img_pil.resize((640, 360))
        self.photo_orig = ImageTk.PhotoImage(img_pil)
        self.canvas_orig.create_image(0, 0, anchor='nw', image=self.photo_orig)

        self.detect_and_display()

    def detect_and_display(self):
        results = model(self.image)
        boxes = results.xyxy[0].cpu().numpy()

        if len(boxes) == 0:
            messagebox.showinfo("Kết quả", "Không phát hiện biển số nào.")
            self.label_text.config(text="Không phát hiện biển số")
            self.canvas_crop.delete("all")
            return

        # Chỉ lấy biển số đầu tiên để demo
        box = boxes[0]
        x1, y1, x2, y2, conf, cls = box.astype(int)
        cropped_plate = self.image[y1:y2, x1:x2]

        text = recognize_plate(cropped_plate, detector)

        # Hiển thị ảnh crop
        crop_rgb = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb)
        crop_pil = crop_pil.resize((320, 90))
        self.photo_crop = ImageTk.PhotoImage(crop_pil)
        self.canvas_crop.create_image(0, 0, anchor='nw', image=self.photo_crop)

        # Hiển thị text biển số
        if is_valid_plate(text):
            self.label_text.config(text=f'Biển số: {text}')
        else:
            self.label_text.config(text=f'Không hợp lệ: {text}')

if __name__ == "__main__":
    root = tk.Tk()
    app = PlateRecognitionApp(root)
    root.mainloop()
