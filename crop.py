import torch
import cv2
import os

# Cấu hình
model_path = 'plate_detector.pt'
input_folder = 'dataset/raw_images'
output_folder = 'dataset/cropped_plates'
os.makedirs(output_folder, exist_ok=True)

# Load model YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
model.conf = 0.3

# Đếm số ảnh đã crop
plate_id = 0
label_lines = []

for filename in os.listdir(input_folder):
    if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    image_path = os.path.join(input_folder, filename)
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Lỗi khi đọc ảnh: {filename}")
        continue

    # Detect
    results = model(image)
    boxes = results.xyxy[0]

    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # Crop biển số
        plate_img = image[y1:y2, x1:x2]

        # Lưu ảnh đã crop
        plate_filename = f"plate_{plate_id:04d}.jpg"
        plate_path = os.path.join(output_folder, plate_filename)
        cv2.imwrite(plate_path, plate_img)

        # Ghi nhãn trống để bạn điền sau
        label_lines.append(f"{plate_filename} UNKNOWN")

        plate_id += 1

print(f"✅ Đã crop {plate_id} biển số.")
with open("dataset/labels.txt", "w", encoding='utf-8') as f:
    f.write("\n".join(label_lines))

print("📄 Đã tạo file labels.txt (gắn nhãn bằng tay hoặc bán tự động).")
