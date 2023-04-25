# Smart Car Park Vacancy Detection System
This system achieves high accuracy in detecting the vacant spots in a car park utilzing the techniques of transfer learning of the state-of-the-art YOLOv5 model and a customized CNN model.

### Demo



https://user-images.githubusercontent.com/90033411/234069586-54ca39bf-d9e7-43b0-8feb-9f1b1e390ad4.mov

### Getting started
- Download the pretrained CNN parameters tensors_res50.pt from [this link](https://drive.google.com/file/d/1LL67CrCH0qHLa8GSIlBJMYkC_Ki_FZ9w/view?usp=share_link) and copy it under `carpark-vacancy-detection-system/yolov5/CNN` directory
- Run ``python manage.py runserver`` to activate the visualization system
- Run ``python3 ./yolov5/detect.py --source 'stream.txt'`` to activate the detection system.

### CNN Training
Clone the following directory, the training script is FYP_CNN.ipynb
https://drive.google.com/drive/folders/1bqdDbgPB51MteYFeg_cvANipz-mUKFc1?usp=share_link
