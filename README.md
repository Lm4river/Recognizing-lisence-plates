# My YOLOv5 Project

This project uses YOLOv5 for object detection.

# Requirements

- Other dependencies in `requirements.txt`

# Installation

1. Clone the repo:  git clone https://github.com/Lm4river/Recognizing-lisence-plates.git 

2. Create virtual environment:

python -m venv venv
venv\Scripts\activate  

Install dependencies:
pip install -r requirements.txt

3. Clone YOLOv5:  git clone https://github.com/ultralytics/yolov5.git

4. Add to head of file "yolov5\detect.py" :

import pathlib
pathlib.PosixPath = pathlib.WindowsPath

5. Run file:  GUI.py