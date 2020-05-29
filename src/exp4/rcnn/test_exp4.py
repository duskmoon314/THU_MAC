import json
import os
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision import transforms as T
import torch.distributed as dist
from PIL import Image, ImageDraw
from collections import defaultdict, deque
import datetime
import pickle
import time
import numpy as np
from engine import train_one_epoch, evaluate
import utils

img = Image.open("./image_exp/Detection/test/10056.jpg").convert("RGB")
img_tensor = T.ToTensor()(img)

model = torch.load("./model_RCNN.pth")
model.eval()

pred = model([img_tensor.to("cuda:0")])
out = ImageDraw.Draw(img)
for b in pred[0]['boxes']:
    out.rectangle(b.detach().cpu().numpy().tolist(), outline="green", width=2)
img.save("./out.png")
print(pred)

result = {"imgs": {}, "types": [
    "pl30",
    "pl5",
    "w57",
    "io",
    "p5",
    "ip",
    "i2",
    "i5",
    "i4",
    "pl80",
    "p11",
    "pl60",
    "p26",
    "pl40",
    "p23",
    "pl50",
    "pn",
    "pne",
    "po"
]}


test_dir = "./image_exp/Detection/test"
img_list = os.listdir(test_dir)
for img_ in img_list:
    img_dir = os.path.join(test_dir, img_)
    img = Image.open(img_dir).convert("RGB")
    img = T.ToTensor()(img).to("cuda:0")
    pred = model([img])
    result['imgs'][os.path.splitext(img_)[0]] = {"objects":
                                                 [{"bbox":
                                                   {"xmin": pred[0]['boxes'][x][0].item(),
                                                    "ymin": pred[0]['boxes'][x][1].item(),
                                                    "xmax": pred[0]['boxes'][x][2].item(),
                                                    "ymax": pred[0]['boxes'][x][3].item()},
                                                   "category": result['types'][pred[0]['labels'][x].item()],
                                                   "score": pred[0]['scores'][x].item()
                                                   } for x in range(len(pred[0]['boxes']))]}
with open("./exp4_test_mn.json", "w", newline="", encoding="utf8") as target:
    json.dump(result, target)
