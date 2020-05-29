from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchvision import datasets, models, transforms
import time
import os
import copy
import json
from PIL import Image

import argparse

parser = argparse.ArgumentParser(
    description="THU_MAC exp3 transfer learning py")
parser.add_argument('--train_data', type=str,
                    default='./image_exp/Classification/DataFewShot/Train', help="train data dir default='./image_exp/Classification/DataFewShot/Train'")
parser.add_argument('--val_data', type=str,
                    default='./Test-dev', help="val data dir default='./Test-dev'")
parser.add_argument('--test_data', type=str,
                    default='./image_exp/Classification/DataFewShot/Test', help="test data dir default='./image_exp/Classification/DataFewShot/Test'")
parser.add_argument('--save_model', type=str,
                    default="./exp3_knn_model.pth", help="model save dir default='./exp3_knn_model.pth'")
parser.add_argument('--test_pred', type=str,
                    default="./exp3_knn_pred.json", help="test pred json default='./exp3_knn_pred.json'")
args = parser.parse_args()

data_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.44087456, 0.39025736, 0.43862119], [
        0.18242574, 0.19140723, 0.18536106])
])

data_dir = args.train_data
image_dataset = datasets.ImageFolder(data_dir, data_transform)

train_loader = torch.utils.data.DataLoader(image_dataset,
                                           batch_size=11,
                                           shuffle=True,
                                           num_workers=4)
class_names = image_dataset.classes
dataset_sizes = len(image_dataset)

# put validation datasets in the directory
val_dir = args.val_data
val_datasets = datasets.ImageFolder(val_dir, data_transform)
val_loader = torch.utils.data.DataLoader(val_datasets,
                                         batch_size=11,
                                         shuffle=True,
                                         num_workers=4)
val_names = val_datasets.classes
val_sizes = len(val_datasets)

test_transforms = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor()
])

test_normalize = transforms.Normalize([0.44087456, 0.39025736, 0.43862119], [
    0.18242574, 0.19140723, 0.18536106])

exp3_test_dir = args.test_data
exp3_test_data = os.listdir(exp3_test_dir)
exp3_test_result = {}
exp3_label = ['p1', 'p12', 'p14', 'p17', 'p19',
              'p22', 'p25', 'p27', 'p3', 'p6', 'p9']


class KNN(object):
    def __init__(self):
        pass

    def fit(self, train_imgs, labels):
        self.train_imgs = train_imgs.numpy()
        self.labels = labels.numpy()

    def predict(self, test_dataset):
        dis = self.compute_distances(test_dataset)
        return self.predict_labels(dis)

    def compute_distances(self, test_dataset):
        num_test = test_dataset.shape[0]
        num_train = self.train_imgs.shape[0]
        test_dataset = test_dataset.numpy()
        test_dataset = test_dataset.reshape(num_test, -1)
        dist = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                dist[i][j] = np.sum((self.train_imgs[j] - test_dataset[i])**2)
        return dist

    def predict_labels(self, dist, k=1):
        num_test = dist.shape[0]
        y_pred = np.zeros(num_test, dtype=int)
        for i in range(num_test):
            pred_x = np.argsort(dist[i])[:k]
            pred_y = [self.labels[val] for val in pred_x]
            y_pred[i] = pred_y[0]
        return y_pred


knn = KNN()
for images, labels in train_loader:
    images = images.view(11, -1)
    knn.fit(images, labels)

correct = 0
for val_img, val_labels in val_loader:
    y_pred = knn.predict(val_img)
    for i in range(11):
        if (y_pred[i] == val_labels.numpy()[i]):
            correct = correct + 1
print('Val Acc:', correct / val_sizes)

for data_dir in exp3_test_data:
    img = Image.open(os.path.join(exp3_test_dir, data_dir))
    inputs = test_transforms(img)
    inputs = test_normalize(inputs[0:3]).unsqueeze(0)
    outputs = knn.predict(inputs)
    exp3_test_result[data_dir] = exp3_label[outputs[0]]
with open(args.test_pred, "w", newline="", encoding="utf8") as target:
    json.dump(exp3_test_result, target)

torch.save(knn, args.save_model)
