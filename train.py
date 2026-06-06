from ultralytics import YOLO
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from datetime import datetime

def train():
    # model = YOLO("MYyolo11.yaml")
    # model = YOLO("MyConv/MYyolo11-C3k2_GhostModule.yaml")
    # model = YOLO("MYyolo11-Shape-IoU.yaml")
    # model = YOLO("MyAttention/MYyolo11-iEMA.yaml")
    # model = YOLO("MyFusion/MYyolo11-C3k2_GhostModule-Shape-IoU.yaml")
    # model = YOLO("MyFusion/MYyolo11-C3k2_GhostModule-iEMA.yaml")
    # model = YOLO("./MyAttention/MYyolo11-iEMA-Shape-IoU.yaml")
    model = YOLO("MyFusion/MYyolo11-C3k2_GhostModule-iEMA-ShapeIoU.yaml")


    # amp=False,
    model.train(data="SARDet-100K.yaml",
                imgsz=640,
                epochs=400,
                patience=30,
                # amp=False,
                batch=16,
                workers=0,
                device='',
                optimizer='SGD',
                close_mosaic=10,
                resume=False,
                project='runs/train',
                name='exp',
                single_cls=False,
                cache=False,)
    result = model.val()

if __name__ == '__main__':
    train()

