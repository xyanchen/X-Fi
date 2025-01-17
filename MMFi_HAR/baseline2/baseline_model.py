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
path = os.getcwd()
parentdir = os.path.abspath(os.path.join(path, os.pardir))
sys.path.insert(0, parentdir)
from backbones.RGB_benchmark.RGB_ResNet import *
from backbones.depth_benchmark.depth_ResNet18 import *
from backbones.mmwave_benchmark.mmwave_point_transformer_TD import *
from backbones.lidar_benchmark.lidar_point_transformer import *
from backbones.lidar_benchmark.pointnet_util import *

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

class classification_Head(nn.Sequential):
    def __init__(self, emb_size=512, num_classes=27):
        super(classification_Head,self).__init__()
        self.norm = nn.LayerNorm(emb_size)
        self.fc = nn.Linear(emb_size, num_classes)
    
    def forward(self, x):
        x = torch.mean(x, dim=1)
        x = self.norm(x)
        x = self.fc(x)
        return x

"################################# for single modality #################################"

class single_feature_extrator(nn.Module):
    def __init__(self, modality_list):
        "modality_list: list of string, choose from ['RGB', 'Depth', 'mmWave', 'Lidar']"
        super(single_feature_extrator, self).__init__()
        if 'rgb' in modality_list:
            rgb_model = RGB_ResNet18()
            rgb_model.load_state_dict(torch.load(os.path.join(parentdir,'backbones/RGB_benchmark/RGB_Resnet18.pt')))
            rgb_extractor = rgb_feature_extractor(rgb_model)
            rgb_extractor.eval()
            self.rgb_extractor = rgb_extractor
        if 'depth' in modality_list:
            depth_model = Depth_ResNet18()
            depth_model.load_state_dict(torch.load(os.path.join(parentdir,'backbones/depth_benchmark/depth_Resnet18.pt')))
            depth_extractor = depth_feature_extractor(depth_model)
            depth_extractor.eval()
            self.depth_extractor = depth_extractor
        if 'mmwave' in modality_list:
            mmwave_model = mmwave_PointTransformerReg()
            mmwave_model.load_state_dict(torch.load(os.path.join(parentdir,'backbones/mmwave_benchmark/mmwave_all_random_TD.pt')))
            mmwave_extractor = mmwave_feature_extractor(mmwave_model)
            mmwave_extractor.eval()
            self.mmwave_extractor = mmwave_extractor
        if 'lidar' in modality_list:
            lidar_model = lidar_PointTransformer_cls()
            lidar_model.load_state_dict(torch.load(os.path.join(parentdir,'backbones/lidar_benchmark/lidar_all_random.pt')))
            lidar_extractor = lidar_feature_extractor(lidar_model)
            lidar_extractor.eval()
            self.lidar_extractor = lidar_extractor
        
        
        
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
        else:
            raise ValueError("No modality is selected!")
        return feature

class single_model(nn.Module):
    def __init__(self, modality_list):
        super(single_model, self).__init__()
        self.feature_extractor = single_feature_extrator(modality_list)
        self.classification_head = classification_Head()
    def forward(self, input, exist_list):
        feature = self.feature_extractor(input, exist_list)
        out = self.classification_head(feature)
        return out  
    
"#######################################################################################"


"################################# for dual modality #################################"

class Dual_feature_extrator(nn.Module):
    def __init__(self, modality_list):
        "modality_list: list of string, choose from ['RGB', 'Depth', 'mmWave', 'Lidar']"
        super(Dual_feature_extrator, self).__init__()
        if 'rgb' in modality_list:
            rgb_model = RGB_ResNet18()
            rgb_model.load_state_dict(torch.load(os.path.join(parentdir,'backbones/RGB_benchmark/RGB_Resnet18.pt')))
            rgb_extractor = rgb_feature_extractor(rgb_model)
            rgb_extractor.eval()
            self.rgb_extractor = rgb_extractor
        if 'depth' in modality_list:
            depth_model = Depth_ResNet18()
            depth_model.load_state_dict(torch.load(os.path.join(parentdir,'backbones/depth_benchmark/depth_Resnet18.pt')))
            depth_extractor = depth_feature_extractor(depth_model)
            depth_extractor.eval()
            self.depth_extractor = depth_extractor
        if 'mmwave' in modality_list:
            mmwave_model = mmwave_PointTransformerReg()
            mmwave_model.load_state_dict(torch.load(os.path.join(parentdir,'backbones/mmwave_benchmark/mmwave_all_random_TD.pt')))
            mmwave_extractor = mmwave_feature_extractor(mmwave_model)
            mmwave_extractor.eval()
            self.mmwave_extractor = mmwave_extractor
        if 'lidar' in modality_list:
            lidar_model = lidar_PointTransformer_cls()
            lidar_model.load_state_dict(torch.load(os.path.join(parentdir,'backbones/lidar_benchmark/lidar_all_random.pt')))
            lidar_extractor = lidar_feature_extractor(lidar_model)
            lidar_extractor.eval()
            self.lidar_extractor = lidar_extractor
        
        
        
    def forward(self, input_1, input_2, exist_list):
        "input_1, input_2 must follow the sequence of modality_list: ['RGB', 'Depth', 'mmWave', 'Lidar']"
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
        feature = torch.cat((feature_1, feature_2), dim=1)

        return feature


class dual_model(nn.Module):
    def __init__(self, modality_list):
        super(dual_model, self).__init__()
        self.feature_extractor = Dual_feature_extrator(modality_list)
        self.classification_head = classification_Head()
    def forward(self, input_1, input_2, exist_list):
        feature = self.feature_extractor(input_1, input_2, exist_list)
        out = self.classification_head(feature)
        return out     
    
"#######################################################################################"

"################################# for Triple modality #################################"

class Triple_feature_extrator(nn.Module):
    def __init__(self, modality_list):
        "modality_list: list of string, choose from ['RGB', 'Depth', 'mmWave', 'Lidar']"
        super(Triple_feature_extrator, self).__init__()
        if 'rgb' in modality_list:
            rgb_model = RGB_ResNet18()
            rgb_model.load_state_dict(torch.load(os.path.join(parentdir,'backbones/RGB_benchmark/RGB_Resnet18.pt')))
            rgb_extractor = rgb_feature_extractor(rgb_model)
            rgb_extractor.eval()
            self.rgb_extractor = rgb_extractor
        if 'depth' in modality_list:
            depth_model = Depth_ResNet18()
            depth_model.load_state_dict(torch.load(os.path.join(parentdir,'backbones/depth_benchmark/depth_Resnet18.pt')))
            depth_extractor = depth_feature_extractor(depth_model)
            depth_extractor.eval()
            self.depth_extractor = depth_extractor
        if 'mmwave' in modality_list:
            mmwave_model = mmwave_PointTransformerReg()
            mmwave_model.load_state_dict(torch.load(os.path.join(parentdir,'backbones/mmwave_benchmark/mmwave_all_random_TD.pt')))
            mmwave_extractor = mmwave_feature_extractor(mmwave_model)
            mmwave_extractor.eval()
            self.mmwave_extractor = mmwave_extractor
        if 'lidar' in modality_list:
            lidar_model = lidar_PointTransformer_cls()
            lidar_model.load_state_dict(torch.load(os.path.join(parentdir,'backbones/lidar_benchmark/lidar_all_random.pt')))
            lidar_extractor = lidar_feature_extractor(lidar_model)
            lidar_extractor.eval()
            self.lidar_extractor = lidar_extractor
        
        
        
    def forward(self, input_1, input_2, input_3, exist_list):
        "input_1, input_2 must follow the sequence of modality_list: ['RGB', 'Depth', 'mmWave', 'Lidar']"
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
        feature = torch.cat((feature_1, feature_2,feature_3), dim=1)

        return feature
class triple_model(nn.Module):
    def __init__(self, modality_list):
        super(triple_model, self).__init__()
        self.feature_extractor = Triple_feature_extrator(modality_list)
        self.classification_head = classification_Head()
    def forward(self, input_1, input_2, input_3, exist_list):
        feature = self.feature_extractor(input_1, input_2, input_3, exist_list)
        out = self.classification_head(feature)
        return out     
    
"#######################################################################################"

"################################# for Quadra modality #################################"

class Quadra_feature_extrator(nn.Module):
    def __init__(self, modality_list):
        "modality_list: list of string, choose from ['RGB', 'Depth', 'mmWave', 'Lidar']"
        super(Quadra_feature_extrator, self).__init__()
        if 'rgb' in modality_list:
            rgb_model = RGB_ResNet18()
            rgb_model.load_state_dict(torch.load(os.path.join(parentdir,'backbones/RGB_benchmark/RGB_Resnet18.pt')))
            rgb_extractor = rgb_feature_extractor(rgb_model)
            rgb_extractor.eval()
            self.rgb_extractor = rgb_extractor
        if 'depth' in modality_list:
            depth_model = Depth_ResNet18()
            depth_model.load_state_dict(torch.load(os.path.join(parentdir,'backbones/depth_benchmark/depth_Resnet18.pt')))
            depth_extractor = depth_feature_extractor(depth_model)
            depth_extractor.eval()
            self.depth_extractor = depth_extractor
        if 'mmwave' in modality_list:
            mmwave_model = mmwave_PointTransformerReg()
            mmwave_model.load_state_dict(torch.load(os.path.join(parentdir,'backbones/mmwave_benchmark/mmwave_all_random_TD.pt')))
            mmwave_extractor = mmwave_feature_extractor(mmwave_model)
            mmwave_extractor.eval()
            self.mmwave_extractor = mmwave_extractor
        if 'lidar' in modality_list:
            lidar_model = lidar_PointTransformer_cls()
            lidar_model.load_state_dict(torch.load(os.path.join(parentdir,'backbones/lidar_benchmark/lidar_all_random.pt')))
            lidar_extractor = lidar_feature_extractor(lidar_model)
            lidar_extractor.eval()
            self.lidar_extractor = lidar_extractor
        
        
        
    def forward(self, input_1, input_2, input_3, input_4, exist_list):
        "input_1, input_2 must follow the sequence of modality_list: ['RGB', 'Depth', 'mmWave', 'Lidar']"
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
        feature = torch.cat((feature_1, feature_2,feature_3,feature_4), dim=1)

        return feature

class quadra_model(nn.Module):
    def __init__(self, modality_list):
        super(quadra_model, self).__init__()
        self.feature_extractor = Quadra_feature_extrator(modality_list)
        self.classification_head = classification_Head()
    def forward(self, input_1, input_2, input_3, input_4, exist_list):
        feature = self.feature_extractor(input_1, input_2, input_3, input_4, exist_list)
        out = self.classification_head(feature)
        return out  
    
"#######################################################################################"
