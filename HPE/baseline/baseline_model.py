import numpy as np
import glob
import scipy.io as sio
import torch
from torch import nn
import csv
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import yaml
from evaluate import error
import time
import re
import random
import sys

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

from torchvision.transforms import Resize
from syn_DI_dataset import make_dataset, make_dataloader
parentdir = "C:/Users/Chen_Xinyan/Desktop/Modality_Invariant/HPE"
sys.path.insert(0, parentdir)
from RGB_benchmark.rgb_ResNet18.RGB_ResNet import *
from depth_benchmark.depth_ResNet18 import *
from mmwave_benchmark.mmwave_point_transformer import *
from lidar_benchmark.lidar_point_transformer import *
from lidar_benchmark.pointnet_util import *

class rgb_feature_extractor(nn.Module):
    def __init__(self, rgb_model):
        super(rgb_feature_extractor, self).__init__()
        self.part = nn.Sequential(*list(rgb_model.children())[:-2])
    def forward(self, x):
        x = self.part(x).view(x.size(0), 512, -1)
        x = x.permute(0, 2, 1)
        return x

class depth_feature_extractor(nn.Module):
    def __init__(self, depth_model):
        super(depth_feature_extractor, self).__init__()
        self.part = nn.Sequential(*list(depth_model.children())[:-2])
    def forward(self, x):
        x = self.part(x).view(x.size(0), 512, -1)
        x = x.permute(0, 2, 1)
        return x

class mmwave_feature_extractor(nn.Module):
    def __init__(self, mmwave_model):
        super(mmwave_feature_extractor, self).__init__()
        self.part = nn.Sequential(*list(mmwave_model.children())[:-1])
    def forward(self, x):
        x, _ = self.part(x)
        return x

class lidar_feature_extractor(nn.Module):
    def __init__(self, lidar_model):
        super(lidar_feature_extractor, self).__init__()
        # self.model = lidar_model
        npoints, nblocks, nneighbor, n_c, d_points = 1024, 5, 16, 51, 3
        self.fc1 = lidar_model.backbone.fc1
        self.transformer1 = lidar_model.backbone.transformer1
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(nblocks - 4):
            channel = 32 * 2 ** (i + 1)
            self.transition_downs.append(lidar_model.backbone.transition_downs[i])
            self.transformers.append(lidar_model.backbone.transformers[i])
        self.nblocks = nblocks
    
    def forward(self, x):
        xyz = x[..., :3]
        points = self.transformer1(xyz, self.fc1(x))[0]

        xyz_and_feats = [(xyz, points)]
        for i in range(self.nblocks - 4):
            xyz, points = self.transition_downs[i](xyz, points)
            points = self.transformers[i](xyz, points)[0]
            xyz_and_feats.append((xyz, points))
        points = points.view(points.size(0), -1, 512)
        return points

class csi_feature_extractor(nn.Module):
    def __init__(self, model):
        super(csi_feature_extractor, self).__init__()
        self.part = nn.Sequential(
            model.encoder_conv1,
            model.encoder_bn1,
            model.encoder_relu,
            model.encoder_layer1,
            model.encoder_layer2,
            model.encoder_layer3,
            model.encoder_layer4, 
            # torch.nn.AvgPool2d((1, 4))
        )
    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.transpose(x, 2, 3) #16,2,114,3,32
        x = torch.flatten(x, 3, 4)# 16,2,114,96
        torch_resize = Resize([136,32])
        x = torch_resize(x)
        x = self.part(x).view(x.size(0), 512, -1)
        x = x.permute(0, 2, 1)
        return x

"################################# for single modality #################################"

class single_feature_extrator(nn.Module):
    def __init__(self, modality_list):
        "modality_list: list of string, choose from ['RGB', 'Depth', 'mmWave', 'Lidar', 'Wifi']"
        super(single_feature_extrator, self).__init__()
        if 'rgb' in modality_list:
            rgb_model = RGB_ResNet18()
            rgb_model.load_state_dict(torch.load(parentdir + '/RGB_benchmark/rgb_ResNet18/RGB_Resnet18_copy.pt'))
            rgb_extractor = rgb_feature_extractor(rgb_model)
            rgb_extractor.eval()
            self.rgb_extractor = rgb_extractor
        if 'depth' in modality_list:
            depth_model = Depth_ResNet18()
            depth_model.load_state_dict(torch.load(parentdir + '/depth_benchmark/depth_Resnet18.pt'))
            depth_extractor = depth_feature_extractor(depth_model)
            depth_extractor.eval()
            self.depth_extractor = depth_extractor
        if 'mmwave' in modality_list:
            mmwave_model = mmwave_PointTransformerReg()
            mmwave_model.load_state_dict(torch.load(parentdir + '/mmwave_benchmark/mmwave_all_random.pt'))
            mmwave_extractor = mmwave_feature_extractor(mmwave_model)
            mmwave_extractor.eval()
            self.mmwave_extractor = mmwave_extractor
        if 'lidar' in modality_list:
            lidar_model = lidar_PointTransformerReg()
            lidar_model.load_state_dict(torch.load(parentdir +'/lidar_benchmark/lidar_all_random.pt'))
            lidar_extractor = lidar_feature_extractor(lidar_model)
            lidar_extractor.eval()
            self.lidar_extractor = lidar_extractor
        if 'wifi-csi' in modality_list:
            sys.path.insert(0, parentdir + '/CSI_benchmark')
            csi_model = torch.load(parentdir + '/CSI_benchmark/protocol3_random_1.pkl')
            csi_extractor = csi_feature_extractor(csi_model)
            csi_extractor.eval()
            self.csi_extractor = csi_extractor
        
        
        
    def forward(self, input, exist_list):
        if exist_list[0] == True:
            feature = self.rgb_extractor(input)
            "shape b x 49 x 512"
        elif exist_list[1] == True:
            feature = self.depth_extractor(input)
            "shape b x 49 x 512"
        elif exist_list[2] == True:
            feature = self.mmwave_extractor(input)
            "shape b x n_p x 512"
        elif exist_list[3] == True:
            feature = self.lidar_extractor(input)
            "shape b x 32 x 512"
        elif exist_list[4] == True:
            feature = self.csi_extractor(input)
            "shape b x 17 x 512"
        else:
            raise ValueError("No modality is selected!")
        return feature

class regression_Head(nn.Sequential):
    def __init__(self, emb_size=512, num_classes=17*3):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size), 
            nn.Linear(emb_size, num_classes))
    
    def forward(self, x):
        return super().forward(x).view(-1, 17, 3)

class single_model(nn.Module):
    def __init__(self, modality_list):
        super(single_model, self).__init__()
        self.feature_extractor = single_feature_extrator(modality_list)
        self.regression_head = regression_Head()
    def forward(self, input, exist_list):
        feature = self.feature_extractor(input, exist_list)
        out = self.regression_head(feature)
        return out   
    
"#######################################################################################"


"################################# for dual modality #################################"

class Dual_feature_extrator(nn.Module):
    def __init__(self, modality_list):
        "modality_list: list of string, choose from ['RGB', 'Depth', 'mmWave', 'Lidar', 'Wifi']"
        super(Dual_feature_extrator, self).__init__()
        if 'rgb' in modality_list:
            rgb_model = RGB_ResNet18()
            rgb_model.load_state_dict(torch.load(parentdir + '/RGB_benchmark/rgb_ResNet18/RGB_Resnet18_copy.pt'))
            rgb_extractor = rgb_feature_extractor(rgb_model)
            rgb_extractor.eval()
            self.rgb_extractor = rgb_extractor
        if 'depth' in modality_list:
            depth_model = Depth_ResNet18()
            depth_model.load_state_dict(torch.load(parentdir + '/depth_benchmark/depth_Resnet18.pt'))
            depth_extractor = depth_feature_extractor(depth_model)
            depth_extractor.eval()
            self.depth_extractor = depth_extractor
        if 'mmwave' in modality_list:
            mmwave_model = mmwave_PointTransformerReg()
            mmwave_model.load_state_dict(torch.load(parentdir + '/mmwave_benchmark/mmwave_all_random.pt'))
            mmwave_extractor = mmwave_feature_extractor(mmwave_model)
            mmwave_extractor.eval()
            self.mmwave_extractor = mmwave_extractor
        if 'lidar' in modality_list:
            lidar_model = lidar_PointTransformerReg()
            lidar_model.load_state_dict(torch.load(parentdir +'/lidar_benchmark/lidar_all_random.pt'))
            lidar_extractor = lidar_feature_extractor(lidar_model)
            lidar_extractor.eval()
            self.lidar_extractor = lidar_extractor
        if 'wifi-csi' in modality_list:
            sys.path.insert(0, parentdir + '/CSI_benchmark')
            csi_model = torch.load(parentdir + '/CSI_benchmark/protocol3_random_1.pkl')
            csi_extractor = csi_feature_extractor(csi_model)
            csi_extractor.eval()
            self.csi_extractor = csi_extractor
        
        
        
    def forward(self, input_1, input_2, exist_list):
        "input_1, input_2 must follow the sequence of modality_list: ['RGB', 'Depth', 'mmWave', 'Lidar', 'Wifi']"
        flag = False
        if exist_list[0] == True:
            if flag == False:
                feature_1 = self.rgb_extractor(input_1)
                flag = True
            else:
                feature_2 = self.rgb_extractor(input_2)
            "shape b x 49 x 512"
        if exist_list[1] == True:
            if flag == False:
                feature_1 = self.depth_extractor(input_1)
                flag = True
            else:
                feature_2 = self.depth_extractor(input_2)
            "shape b x 49 x 512"
        if exist_list[2] == True:
            if flag == False:
                feature_1 = self.mmwave_extractor(input_1)
                flag = True
            else:
                feature_2 = self.mmwave_extractor(input_2)
            "shape b x n_p x 512"
        if exist_list[3] == True:
            if flag == False:
                feature_1 = self.lidar_extractor(input_1)
                flag = True
            else:
                feature_2 = self.lidar_extractor(input_2)
            "shape b x 32 x 512"
        if exist_list[4] == True:
            if flag == False:
                feature_1 = self.csi_extractor(input_1)
                flag = True
            else:
                feature_2 = self.csi_extractor(input_2)
            "shape b x 17*4 x 512"
        elif flag == False:
            raise ValueError("No modality is selected!")
        # print("feature_1", feature_1.shape)
        # print("feature_2", feature_2.shape)
        feature = torch.cat((feature_1, feature_2), dim=1)

        return feature


class dual_model(nn.Module):
    def __init__(self, modality_list):
        super(dual_model, self).__init__()
        self.feature_extractor = Dual_feature_extrator(modality_list)
        self.regression_head = regression_Head()
    def forward(self, input_1, input_2, exist_list):
        feature = self.feature_extractor(input_1, input_2, exist_list)
        out = self.regression_head(feature)
        return out   
    
"#######################################################################################"

"################################# for Triple modality #################################"

class Triple_feature_extrator(nn.Module):
    def __init__(self, modality_list):
        "modality_list: list of string, choose from ['RGB', 'Depth', 'mmWave', 'Lidar', 'Wifi']"
        super(Triple_feature_extrator, self).__init__()
        if 'rgb' in modality_list:
            rgb_model = RGB_ResNet18()
            rgb_model.load_state_dict(torch.load(parentdir + '/RGB_benchmark/rgb_ResNet18/RGB_Resnet18_copy.pt'))
            rgb_extractor = rgb_feature_extractor(rgb_model)
            rgb_extractor.eval()
            self.rgb_extractor = rgb_extractor
        if 'depth' in modality_list:
            depth_model = Depth_ResNet18()
            depth_model.load_state_dict(torch.load(parentdir + '/depth_benchmark/depth_Resnet18.pt'))
            depth_extractor = depth_feature_extractor(depth_model)
            depth_extractor.eval()
            self.depth_extractor = depth_extractor
        if 'mmwave' in modality_list:
            mmwave_model = mmwave_PointTransformerReg()
            mmwave_model.load_state_dict(torch.load(parentdir + '/mmwave_benchmark/mmwave_all_random.pt'))
            mmwave_extractor = mmwave_feature_extractor(mmwave_model)
            mmwave_extractor.eval()
            self.mmwave_extractor = mmwave_extractor
        if 'lidar' in modality_list:
            lidar_model = lidar_PointTransformerReg()
            lidar_model.load_state_dict(torch.load(parentdir +'/lidar_benchmark/lidar_all_random.pt'))
            lidar_extractor = lidar_feature_extractor(lidar_model)
            lidar_extractor.eval()
            self.lidar_extractor = lidar_extractor
        if 'wifi-csi' in modality_list:
            sys.path.insert(0, parentdir + '/CSI_benchmark')
            csi_model = torch.load(parentdir + '/CSI_benchmark/protocol3_random_1.pkl')
            csi_extractor = csi_feature_extractor(csi_model)
            csi_extractor.eval()
            self.csi_extractor = csi_extractor
        
        
        
    def forward(self, input_1, input_2, input_3, exist_list):
        "input_1, input_2 must follow the sequence of modality_list: ['RGB', 'Depth', 'mmWave', 'Lidar', 'Wifi']"
        flag_1 = False
        flag_2 = False
        if exist_list[0] == True:
            if flag_1 == False and flag_2 == False:
                feature_1 = self.rgb_extractor(input_1)
                flag_1 = True
            elif flag_1 == True and flag_2 == False:
                feature_2 = self.rgb_extractor(input_2)
                flag_2 = True
            else:
                feature_3 = self.rgb_extractor(input_3)
            "shape b x 49 x 512"
        if exist_list[1] == True:
            if flag_1 == False and flag_2 == False:
                feature_1 = self.depth_extractor(input_1)
                flag_1 = True
            elif flag_1 == True and flag_2 == False:
                feature_2 = self.depth_extractor(input_2)
                flag_2 = True
            else:
                feature_3 = self.depth_extractor(input_3)
            "shape b x 49 x 512"
        if exist_list[2] == True:
            if flag_1 == False and flag_2 == False:
                feature_1 = self.mmwave_extractor(input_1)
                flag_1 = True
            elif flag_1 == True and flag_2 == False:
                feature_2 = self.mmwave_extractor(input_2)
                flag_2 = True
            else:
                feature_3 = self.mmwave_extractor(input_3)
            "shape b x n_p x 512"
        if exist_list[3] == True:
            if flag_1 == False and flag_2 == False:
                feature_1 = self.lidar_extractor(input_1)
                flag_1 = True
            elif flag_1 == True and flag_2 == False:
                feature_2 = self.lidar_extractor(input_2)
                flag_2 = True
            else:
                feature_3 = self.lidar_extractor(input_3)
            "shape b x 32 x 512"
        if exist_list[4] == True:
            if flag_1 == False and flag_2 == False:
                feature_1 = self.csi_extractor(input_1)
                flag_1 = True
            elif flag_1 == True and flag_2 == False:
                feature_2 = self.csi_extractor(input_2)
                flag_2 = True
            else:
                feature_3 = self.csi_extractor(input_3)
            "shape b x 17*4 x 512"
        # elif flag == False:
        #     raise ValueError("No modality is selected!")
        # print("feature_1", feature_1.shape)
        # print("feature_2", feature_2.shape)
        feature = torch.cat((feature_1, feature_2,feature_3), dim=1)

        return feature

class triple_model(nn.Module):
    def __init__(self, modality_list):
        super(triple_model, self).__init__()
        self.feature_extractor = Triple_feature_extrator(modality_list)
        self.regression_head = regression_Head()
    def forward(self, input_1, input_2, input_3, exist_list):
        feature = self.feature_extractor(input_1, input_2, input_3, exist_list)
        out = self.regression_head(feature)
        return out   
    
"#######################################################################################"

"################################# for Quadra modality #################################"

class Quadra_feature_extrator(nn.Module):
    def __init__(self, modality_list):
        "modality_list: list of string, choose from ['RGB', 'Depth', 'mmWave', 'Lidar', 'Wifi']"
        super(Quadra_feature_extrator, self).__init__()
        if 'rgb' in modality_list:
            rgb_model = RGB_ResNet18()
            rgb_model.load_state_dict(torch.load(parentdir + '/RGB_benchmark/rgb_ResNet18/RGB_Resnet18_copy.pt'))
            rgb_extractor = rgb_feature_extractor(rgb_model)
            rgb_extractor.eval()
            self.rgb_extractor = rgb_extractor
        if 'depth' in modality_list:
            depth_model = Depth_ResNet18()
            depth_model.load_state_dict(torch.load(parentdir + '/depth_benchmark/depth_Resnet18.pt'))
            depth_extractor = depth_feature_extractor(depth_model)
            depth_extractor.eval()
            self.depth_extractor = depth_extractor
        if 'mmwave' in modality_list:
            mmwave_model = mmwave_PointTransformerReg()
            mmwave_model.load_state_dict(torch.load(parentdir + '/mmwave_benchmark/mmwave_all_random.pt'))
            mmwave_extractor = mmwave_feature_extractor(mmwave_model)
            mmwave_extractor.eval()
            self.mmwave_extractor = mmwave_extractor
        if 'lidar' in modality_list:
            lidar_model = lidar_PointTransformerReg()
            lidar_model.load_state_dict(torch.load(parentdir +'/lidar_benchmark/lidar_all_random.pt'))
            lidar_extractor = lidar_feature_extractor(lidar_model)
            lidar_extractor.eval()
            self.lidar_extractor = lidar_extractor
        if 'wifi-csi' in modality_list:
            sys.path.insert(0, parentdir + '/CSI_benchmark')
            csi_model = torch.load(parentdir + '/CSI_benchmark/protocol3_random_1.pkl')
            csi_extractor = csi_feature_extractor(csi_model)
            csi_extractor.eval()
            self.csi_extractor = csi_extractor
        
        
        
    def forward(self, input_1, input_2, input_3, input_4, exist_list):
        "input_1, input_2 must follow the sequence of modality_list: ['RGB', 'Depth', 'mmWave', 'Lidar', 'Wifi']"
        flag_1 = False
        flag_2 = False
        flag_3 = False
        if exist_list[0] == True:
            if (flag_1 == False and flag_2 == False) and (flag_3 == False):
                feature_1 = self.rgb_extractor(input_1)
                flag_1 = True
            elif (flag_1 == True and flag_2 == False) and (flag_3 == False):
                feature_2 = self.rgb_extractor(input_2)
                flag_2 = True
            elif (flag_1 == True and flag_2 == True) and (flag_3 == False):
                feature_3 = self.rgb_extractor(input_3)
                flag_3 = True
            else:
                feature_4 = self.rgb_extractor(input_4)
            "shape b x 49 x 512"
        if exist_list[1] == True:
            if (flag_1 == False and flag_2 == False) and (flag_3 == False):
                feature_1 = self.depth_extractor(input_1)
                flag_1 = True
            elif (flag_1 == True and flag_2 == False) and (flag_3 == False):
                feature_2 = self.depth_extractor(input_2)
                flag_2 = True
            elif (flag_1 == True and flag_2 == True) and (flag_3 == False):
                feature_3 = self.depth_extractor(input_3)
                flag_3 = True
            else:
                feature_4 = self.depth_extractor(input_4)
            "shape b x 49 x 512"
        if exist_list[2] == True:
            if (flag_1 == False and flag_2 == False) and (flag_3 == False):
                feature_1 = self.mmwave_extractor(input_1)
                flag_1 = True
            elif (flag_1 == True and flag_2 == False) and (flag_3 == False):
                feature_2 = self.mmwave_extractor(input_2)
                flag_2 = True
            elif (flag_1 == True and flag_2 == True) and (flag_3 == False):
                feature_3 = self.mmwave_extractor(input_3)
                flag_3 = True
            else:
                feature_4 = self.mmwave_extractor(input_4)
            "shape b x n_p x 512"
        if exist_list[3] == True:
            if (flag_1 == False and flag_2 == False) and (flag_3 == False):
                feature_1 = self.lidar_extractor(input_1)
                flag_1 = True
            elif (flag_1 == True and flag_2 == False) and (flag_3 == False):
                feature_2 = self.lidar_extractor(input_2)
                flag_2 = True
            elif (flag_1 == True and flag_2 == True) and (flag_3 == False):
                feature_3 = self.lidar_extractor(input_3)
                flag_3 = True
            else:
                feature_4 = self.lidar_extractor(input_4)
            "shape b x 32 x 512"
        if exist_list[4] == True:
            if (flag_1 == False and flag_2 == False) and (flag_3 == False):
                feature_1 = self.csi_extractor(input_1)
                flag_1 = True
            elif (flag_1 == True and flag_2 == False) and (flag_3 == False):
                feature_2 = self.csi_extractor(input_2)
                flag_2 = True
            elif (flag_1 == True and flag_2 == True) and (flag_3 == False):
                feature_3 = self.csi_extractor(input_3)
                flag_3 = True
            else:
                feature_4 = self.csi_extractor(input_4)
            "shape b x 17*4 x 512"
        # elif flag == False:
        #     raise ValueError("No modality is selected!")
        # print("feature_1", feature_1.shape)
        # print("feature_2", feature_2.shape)
        feature = torch.cat((feature_1, feature_2,feature_3,feature_4), dim=1)

        return feature

class quadra_model(nn.Module):
    def __init__(self, modality_list):
        super(quadra_model, self).__init__()
        self.feature_extractor = Quadra_feature_extrator(modality_list)
        self.regression_head = regression_Head()
    def forward(self, input_1, input_2, input_3, input_4, exist_list):
        feature = self.feature_extractor(input_1, input_2, input_3, input_4, exist_list)
        out = self.regression_head(feature)
        return out   
    
"#######################################################################################"

class Five_feature_extrator(nn.Module):
    def __init__(self, modality_list):
        "modality_list: list of string, choose from ['RGB', 'Depth', 'mmWave', 'Lidar', 'Wifi']"
        super(Five_feature_extrator, self).__init__()
        if 'rgb' in modality_list:
            rgb_model = RGB_ResNet18()
            rgb_model.load_state_dict(torch.load(parentdir + '/RGB_benchmark/rgb_ResNet18/RGB_Resnet18_copy.pt'))
            rgb_extractor = rgb_feature_extractor(rgb_model)
            rgb_extractor.eval()
            self.rgb_extractor = rgb_extractor
        if 'depth' in modality_list:
            depth_model = Depth_ResNet18()
            depth_model.load_state_dict(torch.load(parentdir + '/depth_benchmark/depth_Resnet18.pt'))
            depth_extractor = depth_feature_extractor(depth_model)
            depth_extractor.eval()
            self.depth_extractor = depth_extractor
        if 'mmwave' in modality_list:
            mmwave_model = mmwave_PointTransformerReg()
            mmwave_model.load_state_dict(torch.load(parentdir + '/mmwave_benchmark/mmwave_all_random.pt'))
            mmwave_extractor = mmwave_feature_extractor(mmwave_model)
            mmwave_extractor.eval()
            self.mmwave_extractor = mmwave_extractor
        if 'lidar' in modality_list:
            lidar_model = lidar_PointTransformerReg()
            lidar_model.load_state_dict(torch.load(parentdir +'/lidar_benchmark/lidar_all_random.pt'))
            lidar_extractor = lidar_feature_extractor(lidar_model)
            lidar_extractor.eval()
            self.lidar_extractor = lidar_extractor
        if 'wifi-csi' in modality_list:
            sys.path.insert(0, parentdir + '/CSI_benchmark')
            csi_model = torch.load(parentdir + '/CSI_benchmark/protocol3_random_1.pkl')
            csi_extractor = csi_feature_extractor(csi_model)
            csi_extractor.eval()
            self.csi_extractor = csi_extractor
        
        
        
    def forward(self, input_1, input_2, input_3, input_4, input_5, exist_list):
        "input_1, input_2 must follow the sequence of modality_list: ['RGB', 'Depth', 'mmWave', 'Lidar', 'Wifi']"
        feature_1 = self.rgb_extractor(input_1)
        feature_2 = self.depth_extractor(input_2)
        feature_3 = self.mmwave_extractor(input_3)
        feature_4 = self.lidar_extractor(input_4)
        feature_5 = self.csi_extractor(input_5)
        "shape b x 17*4 x 512"
        # elif flag == False:
        #     raise ValueError("No modality is selected!")
        # print("feature_1", feature_1.shape)
        # print("feature_2", feature_2.shape)
        feature = torch.cat((feature_1, feature_2,feature_3,feature_4, feature_5), dim=1)

        return feature
    
class regression_Head(nn.Sequential):
    def __init__(self, emb_size=512, num_classes=17*3):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size), 
            nn.Linear(emb_size, num_classes))
    
    def forward(self, x):
        return super().forward(x).view(-1, 17, 3)
    
class Five_model(nn.Module):
    def __init__(self, modality_list):
        super(Five_model, self).__init__()
        self.feature_extractor = Five_feature_extrator(modality_list)
        self.regression_head = regression_Head()
    def forward(self, input_1, input_2, input_3, input_4, input_5, exist_list):
        feature = self.feature_extractor(input_1, input_2, input_3, input_4, input_5, exist_list)
        out = self.regression_head(feature)
        return out   
    
"#######################################################################################"
# def selective_pos_enc(xyz, npoint):
#     """
#     Input:
#         xyz: input points position data, [B, N, 3]
#     Return:
#         new_xyz: sampled points position data, [B, S, 3]
#         out: new features of sampled points, [B, S, C]
#     """
#     fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint]
#     torch.cuda.empty_cache()
#     new_xyz = index_points(xyz, fps_idx)
#     torch.cuda.empty_cache()
#     return new_xyz



# class linear_projector(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(linear_projector, self).__init__()
#         '''linear layer for each modality'''
#         # self.rgb_linear_projection = nn.Linear(input_dim, output_dim)
#         # self.depth_linear_projection = nn.Linear(input_dim, output_dim)
#         # self.mmwave_linear_projection = nn.Linear(input_dim, output_dim)
#         # self.lidar_linear_projection = nn.Linear(input_dim, output_dim)
#         # self.csi_linear_projection = nn.Linear(input_dim, output_dim)
#         '''Conv 1d layer for each modality'''
#         self.rgb_linear_projection = nn.Sequential(
#             nn.Conv1d(input_dim, output_dim, 1),
#             nn.BatchNorm1d(output_dim),
#             nn.ReLU(),
#             nn.Linear(49, 32),
#             nn.ReLU()
#         )
#         self.depth_linear_projection = nn.Sequential(
#             nn.Conv1d(input_dim, output_dim, 1),
#             nn.BatchNorm1d(output_dim),
#             nn.ReLU(),
#             nn.Linear(49, 32),
#             nn.ReLU()
#         )
#         self.mmwave_linear_projection = nn.Sequential(
#             nn.Conv1d(input_dim, output_dim, 1),
#             nn.BatchNorm1d(output_dim),
#             nn.ReLU()
#         )
#         self.lidar_linear_projection = nn.Sequential(
#             nn.Conv1d(input_dim, output_dim, 1),
#             nn.BatchNorm1d(output_dim),
#             nn.ReLU(),
#             nn.Linear(32, 32),
#             nn.ReLU()
#         )
#         self.csi_linear_projection = nn.Sequential(
#             nn.Conv1d(input_dim, output_dim, 1),
#             nn.BatchNorm1d(output_dim),
#             nn.ReLU(),
#             nn.Linear(17*4, 32),
#             nn.ReLU()
#         )
#         self.pos_enc_layer = nn.Sequential(
#             nn.Conv1d(3, 128, 1),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Conv1d(128, output_dim, 1),
#             nn.BatchNorm1d(output_dim),
#             nn.ReLU(),
#         )
#     # def forward(self, rgb_feature, depth_feature, mmwave_feature, lidar_feature, csi_feature, modality_list):
#     #     feature_list = [rgb_feature, depth_feature, mmwave_feature, lidar_feature, csi_feature]
#     #     if sum (modality_list) == 0:
#     #         print('WARNING: modality_list is empty!')
#     #         feature = torch.zeros_like(lidar_feature, device=torch.device('cuda'))
#     #         feature = self.linear_projection(feature)
#     #     else:
#     #         real_feature_list = []
#     #         for i in range(len(modality_list)):
#     #             if modality_list[i] == True:
#     #                 real_feature_list.append(feature_list[i])
#     #             else:
#     #                 continue
#     #         feature = torch.cat(real_feature_list, dim=1)
#     #         feature = self.linear_projection(feature)
#     #     return feature
#     def forward(self, rgb_feature, depth_feature, mmwave_feature, lidar_feature, csi_feature, lidar_points, modality_list):
#         # feature_list = [rgb_feature, depth_feature, mmwave_feature, lidar_feature, csi_feature]
#         if sum (modality_list) == 0:
#             print('WARNING: modality_list is empty!')
#             feature = torch.zeros((lidar_feature.shape[0], 32, 512), device=torch.device('cuda'))
#         else:
#             real_feature_list = []
#             if modality_list[0] == True:
#                 real_feature_list.append(self.rgb_linear_projection(rgb_feature.permute(0, 2, 1)))
#             if modality_list[1] == True:
#                 real_feature_list.append(self.depth_linear_projection(depth_feature.permute(0, 2, 1)))
#             if modality_list[2] == True:
#                 real_feature_list.append(self.mmwave_linear_projection(mmwave_feature.permute(0, 2, 1)))
#             if modality_list[3] == True:
#                 real_feature_list.append(self.lidar_linear_projection(lidar_feature.permute(0, 2, 1)))
#             if modality_list[4] == True:
#                 real_feature_list.append(self.csi_linear_projection(csi_feature.permute(0, 2, 1)))
#             # feature = torch.cat(real_feature_list, dim=1)
#             # for i in range(len(real_feature_list)):
#             #     print("real_feature_list", real_feature_list[i].shape)
#             feature = torch.cat(real_feature_list, dim=2).permute(0, 2, 1)
#             # feature = self.linear_projection(feature)
#             if modality_list[3] == True:
#                 feature_shape = feature.shape
#                 new_xyz = selective_pos_enc(lidar_points, feature_shape[1])
#                 # print("new_xyz shape", new_xyz.shape)
#                 'new_xyz shape: B, eature_shape[1], 3'
#                 pos_enc = self.pos_enc_layer(new_xyz.permute(0, 2, 1)).permute(0, 2, 1)
#                 # print("pos_enc shape", pos_enc.shape)
#                 feature = feature + pos_enc
#             else:
#                 pass
#         return feature

# class MultiHeadAttention(nn.Module):
#     def __init__(self, emb_size = 512, num_heads = 8, dropout = 0.0):
#         super().__init__()
#         self.emb_size = emb_size
#         self.num_heads = num_heads
#         self.qkv = nn.Linear(emb_size, emb_size*3)
#         self.att_drop = nn.Dropout(dropout)
#         self.projection = nn.Linear(emb_size, emb_size)
    
#     def forward(self, x, mask = None):
#         qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
#         queries, keys, values = qkv[0], qkv[1], qkv[2]
#         energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
#         if mask is not None:
#             fill_value = torch.finfo(torch.float32).min
#             energy.mask_fill(~mask, fill_value)
        
#         scaling = self.emb_size ** (1/2)
#         att = F.softmax(energy, dim=-1) / scaling
#         att = self.att_drop(att)
#         # sum up over the third axis
#         out = torch.einsum('bhal, bhlv -> bhav ', att, values)
#         out = rearrange(out, "b h n d -> b n (h d)")
#         out = self.projection(out)
#         return out

# class ResidualAdd(nn.Module):
#     def __init__(self, fn):
#         super().__init__()
#         self.fn = fn
        
#     def forward(self, x, **kwargs):
#         res = x
#         x = self.fn(x, **kwargs)
#         x += res
#         return x

# class FeedForwardBlock(nn.Sequential):
#     def __init__(self, emb_size, expansion = 4, drop_p = 0.):
#         super().__init__(
#             nn.Linear(emb_size, expansion * emb_size),
#             nn.GELU(),
#             nn.Dropout(drop_p),
#             nn.Linear(expansion * emb_size, emb_size),
#         )
        
# class TransformerEncoderBlock(nn.Sequential):
#     def __init__(self,
#                  emb_size = 512,
#                  drop_p = 0.5,
#                  forward_expansion = 4,
#                  forward_drop_p = 0.,
#                  ** kwargs):
#         super().__init__(
#             ResidualAdd(nn.Sequential(
#                 nn.LayerNorm(emb_size),
#                 MultiHeadAttention(emb_size, **kwargs),
#                 nn.Dropout(drop_p)
#             )),
#             ResidualAdd(nn.Sequential(
#                 nn.LayerNorm(emb_size),
#                 FeedForwardBlock(
#                     emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
#                 nn.Dropout(drop_p)
#             )
#             ))
        
# class TransformerEncoder(nn.Sequential):
#     def __init__(self, depth = 1, **kwargs):
#         super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])
        
# class regression_Head(nn.Sequential):
#     def __init__(self, emb_size, num_classes):
#         super().__init__(
#             Reduce('b n e -> b e', reduction='mean'),
#             nn.LayerNorm(emb_size), 
#             nn.Linear(emb_size, num_classes))
        
# class ViT(nn.Sequential):
#     def __init__(self,
#                 emb_size = 512,
#                 depth = 2,
#                 *,
#                 num_classes = 17*3,
#                 **kwargs):
#         super().__init__(
#             TransformerEncoder(depth, emb_size=emb_size, **kwargs),
#             regression_Head(emb_size, num_classes)
#         )

# class modality_invariant_model(nn.Module):
#     def __init__(self):
#         super(modality_invariant_model, self).__init__()
#         self.feature_extractor = feature_extrator()
#         self.linear_projector = linear_projector(512, 512)
#         self.vit = ViT()
#     def forward(self, rgb_data, depth_data, mmwave_data, lidar_data, csi_data, modality_list):
#         rgb_feature, depth_feature, mmwave_feature, lidar_feature, csi_feature = self.feature_extractor(rgb_data, depth_data, mmwave_data, lidar_data, csi_data)
#         # print("rgb_hidden_feature", rgb_feature)
#         # print("depth_hidden_feature", depth_feature)
#         # print("mmwave_hidden_feature", mmwave_feature)
#         # print("lidar_hidden_feature", lidar_feature)
#         # print("csi_hidden_feature", csi_feature)
#         feature = self.linear_projector(rgb_feature, depth_feature, mmwave_feature, lidar_feature, csi_feature, lidar_data, modality_list)
#         # print("feature after linear mapping", feature)
#         # print("feature shape", feature.shape)
#         out = self.vit(feature)
#         # print("output", out)
#         out = out.view(-1, 17, 3)
#         return out

''' 打断点方法： # %% '''


