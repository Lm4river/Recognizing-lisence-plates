# My YOLOv5 Project

This project uses YOLOv5 for object detection.

# Requirements

- Other dependencies in `requirements.txt`

# Installation

1. Clone the repo:  git clone https://github.com/Lm4river/Recognizing-lisence-plates.git 

2. Cài đặt môi trường :

python -m venv venv
venv\Scripts\activate  

Install dependencies:
pip install -r requirements.txt

3. Clone YOLOv5:  git clone https://github.com/ultralytics/yolov5.git

4. Tạo folder vietocr/(cùng cấp với folder yolov5/) :

Cách 1:
mkdir vietocr
gdown https://drive.google.com/uc?id=1bTInWz4nI-To0skD1vSzRqXyW_sARgBp -O vietocr/vgg_transformer.pth

Cách 2:
mkdir vietocr
Download qua đường link :"https://drive.google.com/uc?id=1bTInWz4nI-To0skD1vSzRqXyW_sARgBp" và move vào trong thư mục vietocr vừa tạo

5. Run file:  GUI.py