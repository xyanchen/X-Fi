import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
# from model import locNN
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
from torchvision.transforms import Resize


# 3x3 Convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1,  bias=False)

# Residual Block
class ResidualBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# ResNet Module
class ResNet(nn.Module):
    def __init__(self, block, layers):
        super(ResNet, self).__init__()

        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
        )
        self.encoder_conv1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.encoder_conv1.weight.data = torch.unsqueeze(torch.mean(resnet_raw_model1.conv1.weight.data, dim=1), dim=1)
        self.encoder_bn1 = nn.BatchNorm2d(64)
        self.encoder_relu = nn.ReLU(inplace=True)

        self.layer1 = self.make_layer(block, self.in_channels, layers[0])
        self.layer2 = self.make_layer(block, 64, layers[1], 2)
        self.layer3 = self.make_layer(block, 128, layers[2], 2)
        self.layer4 = self.make_layer(block, 128, layers[3], 2)

        self.decode = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=0, bias=False),
        )

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels * block.expansion):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels * block.expansion, stride=stride),
                nn.BatchNorm2d(out_channels * block.expansion))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.transpose(x, 2, 3)  # 16,2,114,3,32
        x = torch.flatten(x, 3, 4)  # 16,2,114,96
        torch_resize = Resize([136, 136])
        x = torch_resize(x) #16,2,136,136
        x = self.encoder_conv1(x)  ##16,2,136,136
        x = self.encoder_bn1(x)  ##size16,64,136,136
        x = self.encoder_relu(x)  ##size(16,64,136,136)
        x = self.layer1(x) # 16,64,136,136
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #
        x = self.decode(x)
        return x
