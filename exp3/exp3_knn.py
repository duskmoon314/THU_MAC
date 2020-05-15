from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import json
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = '../image_exp/Classification/DataFewShot/Train'
image_dataset = datasets.ImageFolder(data_dir, data_transform)

train_loader = torch.utils.data.DataLoader(image_dataset,
                                           batch_size=11,
                                           shuffle=True,
                                           num_workers=8)
class_names = image_dataset.classes
dataset_sizes = len(image_dataset)

# put validation datasets in the directory
val_dir = '../image_exp/Classification/DataFewShot/Val'
val_datasets = datasets.ImageFolder(val_dir, data_transform)
val_loader = torch.utils.data.DataLoader(val_datasets,
                                         batch_size=11,
                                         shuffle=True,
                                         num_workers=8)
val_names = val_datasets.classes
val_sizes = len(val_datasets)


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
    print('Acc:', correct / val_sizes)
