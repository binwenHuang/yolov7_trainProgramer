# *coding:utf-8 *

import numpy as np

import cv2
import torch
from numpy import random
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, \
    set_logging
from utils.torch_utils import select_device
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    '''
    @param
        img: need predict img
        new_shape: resize shape size
        color: bgr
    @return
        img: resize img
        ratio: height and  width's ratio of origin and resize img
        (dw, dh): divide padding into 2 sides
    '''
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    r = min(r, 1.0)
    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


# some configs
# 更换模型路径
weights = os.path.join()
opt_device = '0'  # device = 'cpu' or '0' or '0,1,2,3'
imgsz = 640
opt_conf_thres = 0.01  # conf score thresh
opt_iou_thres = 0.5  # iou thresh
# Initialize
set_logging()
device = select_device(opt_device)  # cpu or gpu
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model
imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
if half:
    model.half()  # to FP16

# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]


# here is predict func, you can predict(your imgs)
def predict(im0s):
    '''
    @param
        im0s: import img
    @return
        ret: a list that [label, score, (x, y, w, h)]
    '''
    # Run inference
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init src_img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    # Set Dataloader & Run inference
    img = letterbox(im0s, new_shape=imgsz)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    # pred = model(src_img, augment=opt.augment)[0]
    pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt_conf_thres, opt_iou_thres)

    # Process detections
    ret = []
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
            # Write results
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]}'
                prob = round(float(conf) * 100, 2)  # round 2
                xywh = xyxy
                x = (int(xywh[0]) + int(xywh[2])) / 2
                y = (int(xywh[1]) + int(xywh[3])) / 2
                w = (int(xywh[2]) - int(xywh[0]))
                h = (int(xywh[3]) - int(xywh[1]))
                position = (x, y, w, h)
                ret_i = [label, prob, position]
                ret.append(ret_i)
                print(ret)
    return ret