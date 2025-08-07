import os
import cv2
import torch
import numpy as np
import re
from PIL import Image
import matplotlib.pyplot as plt
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='plate_detector.pt')
model.conf = 0.4  # confidence threshold

# Load VietOCR model
config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = 'vietocr/vgg_transformer.pth'  # sửa đúng đường dẫn model
config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
config['predictor']['beamsearch'] = True
detector = Predictor(config)

# Thư mục lưu ảnh crop biển số
save_dir = 'cropped_plates'
os.makedirs(save_dir, exist_ok=True)

# Hàm làm sạch text (giữ chữ số, chữ hoa, dấu '-')
def clean_text(text):
    return ''.join(e for e in text if e.isalnum() or e == '-')

# Kiểm tra định dạng biển số hợp lệ
def is_valid_plate(text):
    pattern1 = r'^[0-9]{2}[A-Z]-?[0-9]{4,5}$'  # biển 1 dòng
    pattern2 = r'^[0-9]{2}[A-Z]{1,2}[0-9]{1,2}-?[0-9]{4,5}$'  # biển 2 dòng sau khi nối
    return re.match(pattern1, text) or re.match(pattern2, text)

# Hàm nhận dạng biển số, hỗ trợ 1 & 2 dòng
def recognize_plate(plate_img, detector):
    h, w = plate_img.shape[:2]

    # Chuyển BGR numpy array sang PIL.Image RGB
    pil_img = Image.fromarray(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB))

    if h / w > 0.5:  # nghi ngờ biển 2 dòng
        mid = h // 2
        top_half = pil_img.crop((0, 0, w, mid))
        bottom_half = pil_img.crop((0, mid, w, h))

        top_text = detector.predict(top_half)
        bottom_text = detector.predict(bottom_half)

        full_text = clean_text(top_text + bottom_text)
    else:
        full_text = clean_text(detector.predict(pil_img))

    return full_text

if __name__ == "__main__":
    image_path = r'F:\INTERN-MOBI\PROJECT-INTERN\Final-yolov5-bsx\images\CarLongPlate39_jpg.rf.c1f8bd591eed4c2696e4ecfe365c87c2.jpg'
    image = cv2.imread(image_path)

    results = model(image)
    boxes = results.xyxy[0].cpu().numpy()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2, conf, cls = box.astype(int)
        cropped_plate = image[y1:y2, x1:x2]

        # Lưu ảnh crop biển số
        save_path = os.path.join(save_dir, f'plate_{i+1}.jpg')
        cv2.imwrite(save_path, cropped_plate)

        text = recognize_plate(cropped_plate, detector)

        if is_valid_plate(text):
            print(f'[✓] Biển số: {text}')
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (36, 255, 12), 2)
        else:
            print(f'[✗] Không hợp lệ: {text}')

    # Hiển thị kết quả bằng matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.title('Detected Plates')
    plt.axis('off')
    plt.show()
