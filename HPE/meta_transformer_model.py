import torch 
import torch.nn as nn

import numpy as np
import glob
import scipy.io as sio
import csv
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize
import yaml
from evaluate import error
import time
import re
import random
import sys

from RGB_benchmark.rgb_ResNet18.RGB_ResNet import *
from depth_benchmark.depth_ResNet18 import *
from mmwave_benchmark.mmwave_point_transformer_TD import *
from lidar_benchmark.lidar_point_transformer import *
from lidar_benchmark.pointnet_util import *

"map each modality num of features to 32 "
" Use modality fusion transformer"

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

class feature_extrator(nn.Module):
    def __init__(self):
        super(feature_extrator, self).__init__()
        
        rgb_model = RGB_ResNet18()
        rgb_model.load_state_dict(torch.load('./RGB_benchmark/rgb_ResNet18/RGB_Resnet18_copy.pt'))
        rgb_extractor = rgb_feature_extractor(rgb_model)
        rgb_extractor.eval()

        depth_model = Depth_ResNet18()
        depth_model.load_state_dict(torch.load('depth_benchmark/depth_Resnet18.pt'))
        depth_extractor = depth_feature_extractor(depth_model)
        depth_extractor.eval()
        
        mmwave_model = mmwave_PointTransformerReg()
        mmwave_model.load_state_dict(torch.load('mmwave_benchmark/mmwave_all_random_TD.pt'))
        mmwave_extractor = mmwave_feature_extractor(mmwave_model)
        mmwave_extractor.eval()

        lidar_model = lidar_PointTransformerReg()
        lidar_model.load_state_dict(torch.load('lidar_benchmark/lidar_all_random.pt'))
        lidar_extractor = lidar_feature_extractor(lidar_model)
        lidar_extractor.eval()

        sys.path.insert(0, './CSI_benchmark')
        csi_model = torch.load('CSI_benchmark/protocol3_random_1.pkl')
        csi_extractor = csi_feature_extractor(csi_model)
        csi_extractor.eval()

        self.rgb_extractor = rgb_extractor
        self.depth_extractor = depth_extractor
        self.mmwave_extractor = mmwave_extractor
        self.lidar_extractor = lidar_extractor
        self.csi_extractor = csi_extractor

    def forward(self, rgb_data, depth_data, mmwave_data, lidar_data, csi_data, modality_list):
        if len(modality_list) == 0:
            raise ValueError("At least one modality should be selected")
        else:
            real_feature_dict = {}
            if 'rgb' in modality_list:
                rgb_feature = self.rgb_extractor(rgb_data)
                "shape b x 49 x 512"
                real_feature_dict['rgb'] = rgb_feature
            if 'depth' in modality_list:
                depth_feature = self.depth_extractor(depth_data)
                "shape b x 49 x 512"
                real_feature_dict['depth'] = depth_feature
            if 'mmwave' in modality_list:
                mmwave_feature = self.mmwave_extractor(mmwave_data)
                "shape b x 32 x 512"
                real_feature_dict['mmwave'] = mmwave_feature
            if 'lidar' in modality_list:
                lidar_feature = self.lidar_extractor(lidar_data)
                "shape b x 32 x 512"
                real_feature_dict['lidar'] = lidar_feature
            if 'wifi' in modality_list:
                csi_feature = self.csi_extractor(csi_data)
                "shape b x (17*4) x 512"
                real_feature_dict['wifi'] = csi_feature
        # print(real_feature_dict.keys())
        return real_feature_dict

class token_mapper(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super(token_mapper, self).__init__()
        self.conv = nn.Conv1d(in_channels, embed_dim, kernel_size=1, stride=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        # x.shape = (batch_size, in_channels, num_tokens)
        x = self.conv(x)
        x = self.relu(x)
        return x
    
class multimodal_encoder(nn.Module):
    def __init__(self, in_channels, embed_dim):
        # modality_list = ['rgb', 'depth', 'mmwave', 'lidar', 'wifi']
        super(multimodal_encoder, self).__init__()
        self.rgb_token_mapper = token_mapper(in_channels, embed_dim)
        self.depth_token_mapper = token_mapper(in_channels, embed_dim)
        self.mmwave_token_mapper = token_mapper(in_channels, embed_dim)
        self.lidar_token_mapper = token_mapper(in_channels, embed_dim)
        self.wifi_token_mapper = token_mapper(in_channels, embed_dim)
        
        ckpt = torch.load("meta_transformer/Meta-Transformer_base_patch16_encoder.pth")
        from timm.models.vision_transformer import Block
        self.encoder = nn.Sequential(*[
            Block(
                dim=embed_dim,
                num_heads=12,
                mlp_ratio=4.,
                qkv_bias=True,
                norm_layer=nn.LayerNorm,
                act_layer=nn.GELU
            )
            for i in range(12)])
        # print(self.encoder.state_dict()['0.attn.qkv.weight'])
        self.encoder.load_state_dict(ckpt)
        # print(self.encoder.state_dict()['0.attn.qkv.weight'])
    def forward(self, input_dict, modality_list):
        token_list = []
        if 'rgb' in modality_list:
            rgb_data = input_dict['rgb'].permute(0,2,1)
            rgb_data = self.rgb_token_mapper(rgb_data)
            token_list.append(rgb_data)
        if 'depth' in modality_list:
            depth_data = input_dict['depth'].permute(0,2,1)
            depth_data = self.depth_token_mapper(depth_data)
            token_list.append(depth_data)
        if 'mmwave' in modality_list:
            mmwave_data = input_dict['mmwave'].permute(0,2,1)
            mmwave_data = self.mmwave_token_mapper(mmwave_data)
            token_list.append(mmwave_data)
        if 'lidar' in modality_list:
            lidar_data = input_dict['lidar'].permute(0,2,1)
            lidar_data = self.lidar_token_mapper(lidar_data)
            token_list.append(lidar_data)
        if 'wifi' in modality_list:
            wifi_data = input_dict['wifi'].permute(0,2,1)
            wifi_data = self.wifi_token_mapper(wifi_data)
            token_list.append(wifi_data)
        
        features = torch.cat(token_list, dim=2)
        features = features.permute(0, 2, 1)
        encoded_features = self.encoder(features)
        
        return encoded_features
        
class regression_head(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(regression_head, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)
    def forward(self, x):
        x = self.fc(x)
        return x

class meta_transformer(nn.Module):
    def __init__(self, in_channels = 512, embed_dim  = 768, out_channels=17*3):
        super(meta_transformer, self).__init__()
        self.feature_extractor = feature_extrator()
        self.encoder = multimodal_encoder(in_channels, embed_dim)
        # self.regression_head = regression_head(230, out_channels) # for all modalities
        self.regression_head = regression_head(130, out_channels) # for rgb depth lidar modalities
    def forward(self, rgb_data, depth_data, mmwave_data, lidar_data, csi_data, modality_list):
        input_dict = self.feature_extractor(rgb_data, depth_data, mmwave_data, lidar_data, csi_data, modality_list)
        x = self.encoder(input_dict, modality_list)
        x = x.mean(dim=2)
        x = self.regression_head(x)
        x = x.view(-1, 17, 3)
        return x
        
            