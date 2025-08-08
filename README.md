# My YOLOv5 Project

Dự án nhận diện biển số xe được xây dựng dựa trên sự kết hợp của các công nghệ và thư viện mạnh mẽ trong lĩnh vực thị giác máy tính và xử lý ảnh:

YOLOv5: Mô hình phát hiện đối tượng thời gian thực, dùng để xác định vị trí biển số xe trong ảnh.

VietOCR: Thư viện OCR tiếng Việt dựa trên Transformer, sử dụng để nhận dạng ký tự từ ảnh biển số đã được cắt.

PyTorch: Framework deep learning phục vụ huấn luyện và triển khai mô hình YOLOv5 và VietOCR.

OpenCV: Xử lý ảnh, tiền xử lý và trích xuất vùng biển số từ ảnh gốc.

Pillow (PIL): Hỗ trợ đọc, ghi và chuyển đổi định dạng ảnh.

Tkinter: Xây dựng giao diện người dùng (GUI) để tải ảnh, hiển thị kết quả phát hiện và nhận dạng.



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

6. Sử dụng ảnh trong thư mục test để kiểm nghiệm kết quả 