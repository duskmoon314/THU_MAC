from argparse import ArgumentParser
import cv2
from sklearn import svm
from skimage.feature import hog
import os
import random
import numpy as np
import json
from tqdm import tqdm
from sklearn.externals import joblib

parser = ArgumentParser()
parser.add_argument('-p', '--path', type=str, default='image_exp/Classification/Data/Test')
parser.add_argument('-w', '--weight', type=str, default='./exp1.model')
args = parser.parse_args()
DATA_TEST = args.path

classifier = joblib.load(args.weight)
test_data = []
test_imgs = os.listdir(DATA_TEST)
print('loading test images')
for i in tqdm(range(len(test_imgs))):
    path = os.path.join(DATA_TEST, test_imgs[i])
    img = cv2.imread(path)
    img = cv2.resize(img, (60, 60)) # average size
    fd = hog(img, orientations=9, pixels_per_cell=(6, 6),
                cells_per_block=(2, 2), multichannel=True)
    test_data.append(fd)
    
print('testing...')
test_predict = classifier.predict(test_data)
test_result = {}
for i in tqdm(range(len(test_imgs))):
    test_result[test_imgs[i]] = test_predict[i]
    
json.dump(test_result, open('exp1_test.json','w'))
print('test result is saved in exp1_test.json')