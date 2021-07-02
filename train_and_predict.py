import numpy as np
import os
import math
import random

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.model_selection import train_test_split

grDataPath = '/bigdata/ethan/gr-dataset'
vioDataPath = '/bigdata/ethan/vio-dataset'

# hyper parameters
batchSize = 2048

testYDic = {}

# Divide into training set and testing set
trainList = ['181', '183', '184', '186', '187', '189', '1810', '192', '193', '197', '199', '1910']
testList = ['182', '185', '188', '191', '196', '198']

# Load all training samples to an array
trainX = np.array([])
testX = np.array([])
trainY = np.array([])
testY = np.array([])

# forming training set
for name in trainList:
    xName = os.path.join(grDataPath, f'entire_data_ispd1{name[1]}_test{name[2:]}.npy')
    yName = os.path.join(vioDataPath, f'{name[1]}t{name[2:]}.npy')
    x = np.load(xName)
    y = np.load(yName)
    trainX = np.append(trainX, x[:,:,0:23].flatten())
    trainY= np.concatenate((trainY, y.sum(axis=2)!=0), axis=None)
trainX = np.reshape(trainX, (-1, 23))

# forming testing set
for name in testList:
    xName = os.path.join(grDataPath, f'entire_data_ispd1{name[1]}_test{name[2:]}.npy')
    yName = os.path.join(vioDataPath, f'{name[1]}t{name[2:]}.npy')
    x = np.load(xName)
    y = np.load(yName)
    testX = np.append(testX, x[:,:,0:23].flatten())
    testY= np.concatenate((testY, y.sum(axis=2)!=0), axis=None)
testX = np.reshape(testX, (-1, 23))

numTrSample = trainX.shape[0]
def getBatch(size = 256):
    for ndx in range(0, numTrSample, size):
        yield trainX[ndx:min(ndx + size, numTrSample)], trainY[ndx:min(ndx + size, numTrSample)]

# scale
scaler = StandardScaler()
scaler.fit(trainX)  # Don't cheat - fit only on training data
trainX = scaler.transform(trainX)
testX = scaler.transform(testX)

clf = SGDClassifier(alpha=.0001, loss='log', penalty='l2', n_jobs=-1, shuffle=True, max_iter=100, verbose=0, tol=0.001)

EPOCHS = 10000
for epoch in range(EPOCHS):
    batcherator = getBatch(batchSize)
    for index, (chunkX, chunky) in enumerate(batcherator):
        clf.partial_fit(chunkX, chunky, classes=[0, 1])

        predictedY = clf.predict(testX)
        acc = accuracy_score(testY, predictedY)
        TN, FP, FN, TP = confusion_matrix(testY, predictedY).ravel()
        print(f'epoch {epoch}, index {index}: accuracy {acc}, tnr {TN/(TN+FP)}, fpr {FP/(FP+TN)}, fnr {FN/(TP+FN)}, tpr {TP/(TP+FN)}, ppv {TP/(TP+FP)}, npv {TN/(TN+FN)}, FDR {FP/(TP+FP)}')
        