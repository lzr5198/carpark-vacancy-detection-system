# Real-time Smart Car Park Vacancy Detection System
This repo presents our Final Year Project at HKUST, which is jointly developed by [HE Qihao](https://github.com/phyqh), [LIN Zhaorun](https://github.com/lzr5198), [YIN Zhuohao](https://github.com/Thomas-YIN), and [ZHOU Siyuan](https://github.com/szhoubd). Our system achieves high accuracy in detecting vacant parking spaces in a car park by utilzing the techniques of transfer learning of the state-of-the-art YOLOv5 model and a customized CNN model. It saves both time and energy for drivers by providing fine-grained vacancy information and reducing unnecessary cruising in the car park. Moreover, it outperforms any existing vacancy detection method in terms of cost since it can capture much more parking spaces with a single camera.

### Demo video

https://user-images.githubusercontent.com/90033411/234069586-54ca39bf-d9e7-43b0-8feb-9f1b1e390ad4.mov

### Getting started
- Download the pretrained CNN parameters tensors_res50.pt from [this link](https://drive.google.com/file/d/1LL67CrCH0qHLa8GSIlBJMYkC_Ki_FZ9w/view?usp=share_link) and copy it under `carpark-vacancy-detection-system/yolov5/CNN` directory.
- Replace the rtsp link in the `stream.txt` file.
- Run `pip install -r requirement.txt` to install the needed packages.
- Run ``python manage.py runserver`` to activate the visualization system.
- Run ``python3 ./yolov5/detect.py --source 'stream.txt'`` to activate the detection system.

### CNN Training

Clone the following directory from [this link](https://drive.google.com/drive/folders/1bqdDbgPB51MteYFeg_cvANipz-mUKFc1?usp=share_link). The training script is `FYP_CNN.ipynb`.
