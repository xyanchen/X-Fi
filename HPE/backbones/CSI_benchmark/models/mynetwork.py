import torchvision
import torch.nn as nn
import torch
import cv2
from torchvision.transforms import Resize
# from position_embedding import PositionEmbeddingSine
# from detr import Transformer
# from models.CTrans import ChannelTransformer
import numpy as np

class posenet(nn.Module):
    """Our RoadSeg takes rgb and another (depth or normal) as input,
    and outputs freespace predictions.
    """
    def __init__(self):
        super(posenet, self).__init__()
        resnet_raw_model1 = torchvision.models.resnet34(pretrained=True)
        filters = [64, 64, 128, 256, 512]
        self.encoder_conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        #self.encoder_conv1.weight.data = torch.unsqueeze(torch.mean(resnet_raw_model1.conv1.weight.data, dim=1), dim=1)
        self.encoder_bn1 = resnet_raw_model1.bn1
        self.encoder_relu = resnet_raw_model1.relu
        self.encoder_maxpool = resnet_raw_model1.maxpool
        self.encoder_layer1 = resnet_raw_model1.layer1
        self.encoder_layer2 = resnet_raw_model1.layer2
        self.encoder_layer3 = resnet_raw_model1.layer3
        self.encoder_layer4 = resnet_raw_model1.layer4

        self.decode = nn.Sequential(
            nn.Conv2d(512, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0, bias=False),

           # nn.Conv2d(64, 2, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),

        )
        self.m = torch.nn.AvgPool2d((1, 4))
        self.bn1 = nn.BatchNorm2d(3)
        self.bn2 = nn.BatchNorm2d(512)
        self.rl = nn.ReLU(inplace=True)



    def forward(self,x): #16,2,3,114,32
        x = torch.transpose(x, 2, 3) #16,2,114,3,32
        x = torch.flatten(x, 3, 4)# 16,2,114,96
        torch_resize = Resize([136,32])
        x = torch_resize(x)
        # print(x.shape)
        x = self.encoder_conv1(x)  ##16,2,136,136
        # print(x.shape)
        x = self.encoder_bn1(x)  ##size16,64,136,136
        x = self.encoder_relu(x)  ##size(1,64,192,624)

        #x = self.encoder_maxpool(x)

        x = self.encoder_layer1(x)

        x = self.encoder_layer2(x)

        x = self.encoder_layer3(x)

        x = self.encoder_layer4(x)
        # print(x.shape)

        x = self.decode(x)
        x = self.m(x).squeeze() # b,2,17
        if x.shape[0] == 3:
            x = x.unsqueeze(0)

        x = torch.transpose(x, 1, 2)

        return x



def weights_init(m):
    # classname = m.__class__.__name__
    # if classname.find('Conv') != -1:
    #     nn.init.xavier_normal_(m.weight.data)
    #     nn.init.xavier_normal_(m.bias.data)
    # elif classname.find('BatchNorm2d') != -1:
    #     nn.init.normal_(m.weight.data, 1.0)
    #     nn.init.constant_(m.bias.data, 0.0)
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        #nn.init.xavier_normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
