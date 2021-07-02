import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import os
import math
import random

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
    y = (y.sum(axis=2)!=0).reshape((-1,1))
    y = np.concatenate((1-y, y), axis=1)
    trainX = np.append(trainX, x[:,:,0:23].flatten())
    trainY= np.concatenate((trainY, y), axis=None)
trainX = np.reshape(trainX, (-1, 23))
trainY = np.reshape(trainY, (-1, 2))

# forming testing set
for name in testList:
    xName = os.path.join(grDataPath, f'entire_data_ispd1{name[1]}_test{name[2:]}.npy')
    yName = os.path.join(vioDataPath, f'{name[1]}t{name[2:]}.npy')
    x = np.load(xName)
    y = np.load(yName)
    y = (y.sum(axis=2)!=0).reshape((-1,1))
    y = np.concatenate((1-y, y), axis=1)
    testX = np.append(testX, x[:,:,0:23].flatten())
    testY= np.concatenate((testY, y), axis=None)
testX = torch.from_numpy(np.reshape(testX, (-1, 23))).to(device='cuda').float()
testY = torch.from_numpy(np.reshape(testY, (-1, 2))).to(device='cuda').long()

numPositive = (trainY[:,1] == 1).sum() + (testY[:,1] == 1).sum()
numNegtive = (trainY[:,1] == 0).sum() + (testY[:,1] == 0).sum()
weights = torch.tensor([numNegtive, numPositive], dtype=torch.float32, device='cuda')
weights = weights / weights.sum()
weights = 1.0 / weights
print(weights)

def getBatch(samples, targets, size = 256):
    numSample = samples.shape[0]
    lables = list(range(numSample))
    random.shuffle(lables)
    batch = np.zeros((size, samples.shape[1]))
    target = np.zeros((size, 2))
    for ndx in range(size):
        batch[ndx,:] = samples[lables[ndx], :]
        target[ndx, :] = targets[lables[ndx], :]
    return torch.from_numpy(batch), torch.from_numpy(target)

def confMatrix(output, target):
    confusionVector = (output[:,1]>0.5).float() / target[:,1].float()
    tp = torch.sum(confusionVector == 1).item()
    fp = torch.sum(confusionVector == float('inf')).item()
    tn = torch.sum(torch.isnan(confusionVector)).item()
    fn = torch.sum(confusionVector == 0).item()
    print(f'tp {tp}, fp {fp}, tn {tn}, fn {fn}')
    correct = ((output[:,1]>0.5).float() == target[:,1].float()).float().sum()
    acc = 100 * correct / output.shape[0]
    tnr = 100 * tn/(tn+fp) if (tn+fp) != 0 else 0
    fpr = 100 * fp/(fp+tn) if (fp+tn) != 0 else 0
    fnr = 100 * fn/(tp+fn) if (tp+fn) != 0 else 0
    tpr = 100 * tp/(tp+fn) if (tp+fn) != 0 else 0
    return acc, tnr, fpr, fnr, tpr

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(23, 25)
        self.fc2 = nn.Linear(25, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x))
        return x

net = Net().cuda()
print(net)

# start training
for epoch in range(10000):
    inputX, targetY = getBatch(trainX, trainY, 2048)
    inputX = inputX.to(device='cuda').float()
    targetY= targetY.to(device='cuda').long()
 
    optimizer = optim.SGD(net.parameters(), lr=0.1)
    optimizer.zero_grad()

    outputY = net(inputX)
    criterion = nn.CrossEntropyLoss(weight=weights)
    loss = criterion(outputY, torch.max(targetY, 1)[1])

    loss.backward()
    optimizer.step()

    acc, tnr, fpr, fnr, tpr = confMatrix(outputY, targetY)
    print(f'Train: epoch {epoch}, loss {loss:.5f}: accuracy {acc:.5f}, tpr {tpr:.5f}, tnr {tnr:.5f}, fpr {fpr:.5f}, fnr {fnr:.5f}')

    if epoch%10 == 0:
        testOutputY = net(testX)
        loss = criterion(testOutputY, torch.max(testY, 1)[1])

        acc, tnr, fpr, fnr, tpr = confMatrix(testOutputY, testY)
        print(f'Test: epoch {epoch}, loss {loss:.5f}: accuracy {acc:.5f}, tpr {tpr:.5f}, tnr {tnr:.5f}, fpr {fpr:.5f}, fnr {fnr:.5f}')
        with open('dnn11_log.log', 'a') as outfile:
            outfile.write(f'Test: epoch {epoch}, loss {loss:.5f}: accuracy {acc:.5f}, tpr {tpr:.5f}, tnr {tnr:.5f}, fpr {fpr:.5f}, fnr {fnr:.5f}\n')
