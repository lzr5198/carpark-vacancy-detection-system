# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     path/                           # directory
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s.xml                # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path
import requests
import urllib.parse
import torch
import numpy as np
import json
import time

cfp=os.path.abspath(os.path.dirname(__file__))

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch.nn import BatchNorm2d
from torch import sigmoid
from torch import flatten
from PIL import Image
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from matplotlib import cm
from torchvision.models import resnet50
import torch.nn.functional as F

trick_counter = 0 
trick_counter_max = 100
trick_sum = {}

boundingBoxesFileLoc = cfp + "/CNN/boundingBoxes/spot.txt"

mtime = os.path.getmtime(boundingBoxesFileLoc)


class CNN(nn.Module):
    def __init__(self, batchSize):
        super(CNN, self).__init__()
        self.backbone = resnet50(pretrained=False)
        self.backbone.load_state_dict(torch.load(cfp + '/CNN/resnet50.pth'))
        for p in self.backbone.parameters():
            p.requires_grad = False
        del self.backbone.fc
        # input: 3*128*128 greyScale so 3 channels

        # self.layer1=nn.Sequential(
        #     # Defining a 2D convolution layer
        #     Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
        #     ReLU(inplace=True),
        #     MaxPool2d(kernel_size=2, stride=2)
        # )
        # 64*32*32
        self.layer2 = nn.Sequential(
            # Defining a 2D convolution layer
            Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),

            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2)
        )
        # 128*16*16
        self.layer3 = nn.Sequential(
            # Defining a 2D convolution layer
            Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2)
        )
        # 256*8*8
        self.fc1 = nn.Linear(in_features=256 * 8 * 8, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=1)
        # -inf, inf

    def forward(self, x):
        x = self.backbone.conv1(x)
        # print(x.requires_grad)
        x = self.backbone.bn1(x)
        # print(x.requires_grad)
        x = self.backbone.relu(x)
        # print(x.requires_grad)
        x = self.backbone.maxpool(x)

        # print(x.requires_grad)
        x = self.layer2(x)
        # print(x.requires_grad)
        x = self.layer3(x)
        # print(x.requires_grad)
        x = self.fc1(flatten(x, 1))
        x = F.relu(x)
        x = self.fc2(x)
        # print(x.requires_grad)
        result = sigmoid(x)
        # print(result.requires_grad)
        return result

@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow()
        print("imgsz is: ", imgsz)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
        print("newly init shape", dataset.imgs[0].shape)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # CNN Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fypCNN = CNN(16)
    model_dict = torch.load(cfp + '/CNN/tensors_res50.pt', map_location=torch.device('cpu'))
    fypCNN.load_state_dict(model_dict)
    fypCNN = fypCNN.to(device)

    trans_resize = transforms.Resize([128, 128])
    trans_tograyscale = transforms.Grayscale(num_output_channels=1)
    trans_totensor = transforms.ToTensor()
    trans_compose = transforms.Compose([trans_resize, trans_tograyscale, trans_totensor])
    # CNN Model

    # CNN related results
    CNN_results = {}
    img_set = {}

    # Mean filtering trick 
    # For now, it seems trick_dictionary is no longer needed 
    trick_dictionary = initialize_mean_filtering()

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    start = time.time()

    for path, im, im0s, vid_cap, s in dataset:
        global trick_counter_max
        global trick_counter
        global mtime

        if os.path.getmtime(boundingBoxesFileLoc) > mtime:
            mtime = os.path.getmtime(boundingBoxesFileLoc)
            trick_dictionary = initialize_mean_filtering()
            CNN_results = {}
            time.sleep(1)

        BoundingBoxSet = initialize_bounding_box_set()
        # im0s[0] = compute_remap(im0s[0])
        with dt[0]:
            # For CNN
            ############################################# For rtsp stream #############################################
            print("img shape is: ")
            print(im0s[0].shape)
            print("-----------")

            img = im0s
            img = Image.fromarray(np.uint8(img[0])).convert('RGB')
            ############################################# For rtsp stream #############################################

            ############################################# For video #############################################
            # print("img shape is: ")
            # print(img.shape)
            # print("-----------")

            # img = Image.fromarray(np.uint8(img)).convert('RGB')
            ############################################# For video #############################################

            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

            for key in BoundingBoxSet:
                boxCoordinates = BoundingBoxSet[key]
                img_current = img.crop(((int(boxCoordinates[0])), int(boxCoordinates[1]), int(boxCoordinates[2]), int(boxCoordinates[3])))

                img_out = trans_compose(img_current).to(device)
                img_mean = img_out.mean()
                std = 0.225
                img_out = (img_out - img_mean) / std
                img_out = img_out.repeat(3,1,1)
                img_out = img_out.unsqueeze(0)

                img_set[key] = img_out

                CNN_results[key] = fypCNN(img_set[key])

                # print("CNN_result for ", key)
                # print(CNN_results[key])
                
            f = open(cfp + '/CNN/CNN_result.txt', 'w')
            for key in CNN_results:
                f.write('Bounding Box ' + str(key) + ' with coordinates: ' + str(BoundingBoxSet[key])
                        + ' produces result: ' + str(CNN_results[key].detach().numpy()[0, 0]) + '\n')
            f.close()

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        CNN_results_2 = {}
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                post_cnn = {}
                for key in CNN_results:
                    tmp = CNN_results[key].numpy()[0][0]
                    if tmp > 0.7:
                        # post_cnn[key] = 1
                        if trick_dictionary[key][trick_counter] == 0:
                            trick_sum[key] += 1
                        trick_dictionary[key][trick_counter] = 1  # This line may be unnecessary
                    else:
                        # post_cnn[key] = 0
                        if trick_dictionary[key][trick_counter] == 1:
                            trick_sum[key] -= 1
                        trick_dictionary[key][trick_counter] = 0  # This line may be unnecessary

                trick_counter = (trick_counter + 1) % trick_counter_max  # Update trick_counter

                for key in CNN_results:
                    tmp = trick_sum[key]/trick_counter_max
                    if tmp > 0.7:
                        post_cnn[key] = 1
                    else:
                        post_cnn[key] = 0
                    CNN_results_2[key] = trick_sum[key]
                f = open(cfp + '/CNN/CNN_result_2.txt', 'w')
                for key in CNN_results_2:
                    end = time.time()
                    f.write('Bounding Box ' + str(key) + ' with coordinates: ' + str(BoundingBoxSet[key])
                            + ' produces result: ' + str(CNN_results_2[key]) + ' time: ' + str(end-start) + '\n')
                f.close()
                print(post_cnn)
                cnn_result_encoded = json.dumps(post_cnn)
                
                post_data = {
                    "object_name": [],
                    "x1": [],
                    "y1": [],
                    "x2": [],
                    "y2": [],
                    "CNN_result": cnn_result_encoded,
                }
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))

                        # Can be used when integrating with CNN
                        # confidence_score = conf
                        post_data["object_name"].append(names[int(cls)])
                        post_data["x1"].append(str(int(xyxy[0].item())))
                        post_data["y1"].append(str(int(xyxy[1].item())))
                        post_data["x2"].append(str(int(xyxy[2].item())))
                        post_data["y2"].append(str(int(xyxy[3].item())))
                        # Pass params to backend
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                
                print("post_data sent from detect")
                print(post_data)
                print()

                endpoint = "http://localhost:8000/carslots/"
                requests.post(url = endpoint, data = post_data)

                # time.sleep(5)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

def initialize_mean_filtering():
    global trick_sum 
    carslot_ids = [] 
    trick_dict = {} 
    f = open(boundingBoxesFileLoc, 'r') 
    for i, line in enumerate(f):
        line = line.rstrip() 
        if i % 2 == 0:
            carslot_ids.append(line)
    f.close()
    # Initialization of global variables for the purpose of mean filtering in time 
    for i in range(len(carslot_ids)): 
        trick_dict[carslot_ids[i]] = [0] * trick_counter_max
        trick_sum[carslot_ids[i]] = 0
    return trick_dict

def initialize_bounding_box_set():
    carslot_ids = []
    box_coor = []
    BoundingBoxSet = {}

    f = open(boundingBoxesFileLoc, 'r')
    for i, line in enumerate(f):
        line = line.rstrip()
        if i % 2 == 0: carslot_ids.append(line)
        else: box_coor.append(line)
    f.close()
    for i in range(len(carslot_ids)):
        BoundingBoxSet[carslot_ids[i]] = box_coor[i].strip().split(',')[0:4]
    
    return BoundingBoxSet

def compute_remap(image):
    R = image.shape[0]//2
    W = int(2*np.pi*R)
    H = R
    mapx = np.zeros([H,W], dtype=np.float32)
    mapy = np.zeros([H,W], dtype=np.float32)

    for i in range(mapx.shape[0]):
        for j in range(mapx.shape[1]):
            angle = j/W*np.pi*2
            radius = H-i
            mapx[i,j]=R+np.sin(angle)*radius
            mapy[i,j]=R-np.cos(angle)*radius
        
    image_remap = cv2.remap(image, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) 
    return image_remap

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
