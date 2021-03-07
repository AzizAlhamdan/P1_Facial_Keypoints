## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 256, 3)
        self.conv4 = nn.Conv2d(256, 512, 3)

        # Linear Layers
        self.fc1 = nn.Linear(512*10*10, 5000)
        self.fc2 = nn.Linear(5000, 1000)
        self.out = nn.Linear(1000, 136)

        # Pooling Layer
        self.pool = nn.AvgPool2d(2,2)

        # Dropout Layer
        self.dropout_1 = nn.Dropout(.15)
        self.dropout_2 = nn.Dropout(.3)
        self.dropout_3 = nn.Dropout(.45)

        # Batchnorm Layer
        self.c1_batchnorm = nn.BatchNorm2d(32)
        self.c3_batchnorm = nn.BatchNorm2d(256)


        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = F.relu(self.pool(self.c1_batchnorm(self.conv1(x)))) # 1x200x200 --> 32x98x98
        x = F.relu(self.pool(self.conv2(x))) # 32x98x98 --> 64x47x47
        x = self.dropout_3(x)
        x = F.relu(self.pool(self.c3_batchnorm(self.conv3(x)))) # 64x47x47 --> 256x22x22
        x = F.relu(self.pool(self.conv4(x))) # 256x22x22 --> 512x10x10
        x = self.dropout_2(x)

        x = x.view(x.size(0),-1) # Flattening process
        x = F.relu(self.fc1(x)) # (512x10x10) --> 5000
        x = self.dropout_1(x)
        x = F.relu(self.fc2(x)) # 5000 --> 1000
        x = self.dropout_1(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return self.out(x) # 1000 --> 136
