########################
#  Important Imports
#######################
from __future__ import print_function, division
import matplotlib
import torch
import torch.utils.data
import torch.nn as nn
import sklearn
import time
import os
import copy
from sklearn.model_selection import StratifiedGroupKFold, KFold
from torch.optim import lr_scheduler
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import pandas as pd
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt


np.random.seed(0)


# torch.utils.data.Dataset is an abstract class representing a dataset. Your custom dataset should inherit Dataset
# and override the following methods:
# __len__ so that len(dataset) returns the size of the dataset.
# __getitem__ to support the indexing such that dataset[i] can be used to get iith sample.

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
######################
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
)

pre_processing = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(200),
    transforms.RandomGrayscale()
])

###########################
#     Images Folder
#          For
#      The DataSet
###########################
imgDir = r'D:/Images'
imgDSet = datasets.ImageFolder(imgDir)

test_l = int(len(imgDSet) * 0.2)
tr_n_v = len(imgDSet) - test_l
val_l = int(tr_n_v * 0.2)
train_l = tr_n_v - val_l
