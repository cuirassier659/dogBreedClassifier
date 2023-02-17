########################
#  Important Imports
#######################
from __future__ import print_function, division
import matplotlib
import torch
import torch.utils.data
import torch.nn as nn
from sklearn.model_selection import train_test_split
import time
import os
import copy
# from sklearn.model_selection import StratifiedGroupKFold, KFold
from torch.optim import lr_scheduler
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import pandas as pd
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from torchvision.models import ResNet101_Weights

matplotlib.use("TkAgg")
np.random.seed(0)


# torch.utils.data.Dataset is an abstract class representing a dataset. Your custom dataset should inherit Dataset
# and override the following methods:
# __len__ so that len(dataset) returns the size of the dataset.
# __getitem__ to supp
# Sort the indexing such that dataset[i] can be used to get iith sample.

class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, dat_set, transform=None):
        self.dat_set = dat_set
        self.transform = transform

    def __len__(self):
        return len(self.dat_set)

    def __getitem__(self, index):
        image, label = self.dat_set[index]
        if self.transform:
            image = self.transform(image)
            return image, label


#############################
#  Preparation For Possible
#      Visualization
#      And Optimized
#        GPU Use
############################
cudnn.benchmark = True
plt.ion()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######################
#  Data Normalization
#       For
#   Pre Processing
# and pre processing
# for normalized
# training subset
######################
image_normalization = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
)

pre_processing = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(200),
    transforms.RandomGrayscale(),
])

###########################
#     Images Folder
#          For
#      The DataSet
#          And
#        Labeling
###########################
imgDir = r'D:/Images'
imgDSet = datasets.ImageFolder(imgDir, transform=image_normalization)
trainLabels=imgDSet.class_to_idx;


#####################
# Doing a
# Train,Validation
#      &
#    Test
# Splitting
# Instead of
#  K-fold
#####################
test_l = int(len(imgDSet) * 0.2)
trn_n_val = len(imgDSet) - test_l
val_l = int(trn_n_val * 0.2)
train_l = trn_n_val - val_l
testSet, valSet, trainSet = torch.utils.data.random_split(imgDSet, [test_l, val_l, train_l])

trainFSet = CustomDataset(trainSet,transform=pre_processing)
loadedTrain = DataLoader(trainFSet,shuffle=True)
loadedTest = DataLoader(testSet, shuffle=True)
loadedVal = DataLoader(valSet, shuffle=True)




########################
#
# Loading a Pre-Trained
#       Network
#   Using ResNet101
#
########################

mod = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
mod.fc = nn.Sequential(nn.Linear(mod.fc.in_features, 120),
                       nn.LogSoftmax(dim=1))
mod
mod.to(device)

###############
#
#   Training
#
###############

funcLoss = nn.NLLLoss()
optimizer = optim.SGD(mod.parameters(), lr=0.05)
n_epochs = 10

for epoch in range(n_epochs):
    running_loss = 0.0
    for i, data in enumerate(loadedTrain, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()

        outputs = mod(inputs)
        loss = funcLoss(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:    
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
    outputs = outputs.to('cpu')

print('Finished Training')


correct = 0
total = 0

# Set the model to evaluation mode
mod.eval()

# Disable gradient calculation to save memory and processing time
with torch.no_grad():
    for data in loadedTest:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = mod(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print('Accuracy of the network on the test images: %d %%' % accuracy)
