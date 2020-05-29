import json
import os
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
import torch.distributed as dist
from PIL import Image
from engine import train_one_epoch


class exp4Dataset(object):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        with open(os.path.join(root, "train_annotations.json"), "r", encoding="utf8") as jsonFile:
            annotation = json.load(jsonFile)
            self.classnames = annotation['types']
            self.img_objects = annotation['imgs']
        self.imgs = list(sorted(os.listdir(os.path.join(root, "train"))))

    def __getitem__(self, idx):
        img_dir = os.path.join(self.root, "train", self.imgs[idx])
        img = Image.open(img_dir).convert("RGB")
        img_id = os.path.splitext(self.imgs[idx])[0]
        objs = self.img_objects[img_id]['objects']
        bboxs = []
        categories = []
        for o in objs:
            bboxs.append([o["bbox"]["xmin"], o["bbox"]["ymin"],
                          o["bbox"]["xmax"], o["bbox"]["ymax"]])
            categories.append(self.classnames.index(o["category"]))

        boxes = torch.as_tensor(bboxs, dtype=torch.float32)
        labels = torch.as_tensor(categories, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * \
            (boxes[:, 2] - boxes[:, 0])
        target["iscrowd"] = torch.zeros((len(objs),), dtype=torch.int64)
        if self.transforms is not None:
            img = self.transforms(img)
        return img, target

    def __len__(self):
        return len(self.imgs)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load("exp2_ft_model.pth", map_location=device)
backbone = nn.Sequential(model.Conv2d_1a_3x3,
                         model.Conv2d_2a_3x3,
                         model.Conv2d_2b_3x3,
                         nn.MaxPool2d(kernel_size=3, stride=2),
                         model.Conv2d_3b_1x1,
                         model.Conv2d_4a_3x3,
                         nn.MaxPool2d(kernel_size=3, stride=2),
                         model.Mixed_5b,
                         model.Mixed_5c,
                         model.Mixed_5d,
                         model.Mixed_6a,
                         model.Mixed_6b,
                         model.Mixed_6c,
                         model.Mixed_6d,
                         model.Mixed_6e,
                         model.Mixed_7a,
                         model.Mixed_7b,
                         model.Mixed_7c,
                         nn.AdaptiveAvgPool2d((1, 1)))
backbone.out_channels = 2048

# model = torchvision.models.mobilenet_v2(pretrained=True)
# backbone = model.features
# backbone.out_channels = 1280

anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))

roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0, 1, 2, 3, 4, 5],
                                                output_size=7,
                                                sampling_ratio=2)

model_RCNN = FasterRCNN(backbone,
                        num_classes=19,
                        rpn_anchor_generator=anchor_generator,
                        box_roi_pool=roi_pooler)
# model_RCNN = torch.load("model_RCNN_mobilenet.pth")
model_RCNN = model_RCNN.to(device)


def collate_fn(batch):
    return tuple(zip(*batch))


dataset = exp4Dataset("./image_exp/Detection",
                      T.Compose([T.RandomHorizontalFlip(0.5), T.ToTensor()]))

data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)

params = [p for p in model_RCNN.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

# let's train it for 10 epochs
num_epochs = 10

for epoch in range(num_epochs):
    train_one_epoch(model_RCNN, optimizer, data_loader,
                    device, epoch, print_freq=1000)
    # update the learning rate
    lr_scheduler.step()

    torch.save(model_RCNN, "model_RCNN.pth")
