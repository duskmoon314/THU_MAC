from argparse import ArgumentParser
import cv2
from sklearn import svm
from skimage.feature import hog
import os
import random
import numpy as np
import json
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument('-p', '--path', type=str, default='image_exp/Classification/Data/Train')
DATA_TRAIN = parser.parse_args().path

train_data = []
categroies = os.listdir(DATA_TRAIN)

# load training data

for category in categroies:
    path = os.path.join(DATA_TRAIN, category)
    print('loading category :' + category)
    for file in os.listdir(path):
        img = cv2.imread(os.path.join(path, file))
        img = cv2.resize(img, (60, 60)) # average size
        fd = hog(img, orientations=9, pixels_per_cell=(6, 6),
                    cells_per_block=(2, 2), multichannel=True)
        train_data.append((fd, category))

random.shuffle(train_data)
print('success!')

# divide into train and validation

n = int(0.7 * len(train_data))
train_set = train_data[:n]
val_set = train_data[n:]

print(len(train_set))
print(len(val_set))

# unzip dataset
X_train, Y_train = map(list, zip(*train_set))
X_train = np.array(X_train)
Y_train = np.array(Y_train)

X_test, Y_test = map(list, zip(*val_set))
X_test = np.array(X_test)
Y_test = np.array(Y_test)

classifier = svm.SVC()
classifier.fit(X_train, Y_train)
predicted = classifier.predict(X_test)

correct = 0
for i in range(len(X_test)):
    if predicted[i] == Y_test[i]:
        correct += 1

print(correct/len(X_test))
# save model
from sklearn.externals import joblib
joblib.dump(classifier,'exp1.model')