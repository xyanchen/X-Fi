import numpy as np
import glob
import scipy.io as sio
import torch
from torch import nn
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

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

from RGB_benchmark.rgb_ResNet18.RGB_ResNet import *
from depth_benchmark.depth_ResNet18 import *
from mmwave_benchmark.mmwave_point_transformer_TD import *
from lidar_benchmark.lidar_point_transformer import *
from lidar_benchmark.pointnet_util import *

from immfusion.src.modeling.bert.transformer import FFN, PredictorLG, Transformer, TokenFusionTransformer

# for IDR, each global modality feature has 128 features
# for IDRLW, each global modality feature has 48 features

class rgb_feature_extractor(nn.Module):
    def __init__(self, rgb_model):
        super(rgb_feature_extractor, self).__init__()
        self.part = nn.Sequential(*list(rgb_model.children())[:-2])
        self.final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=48,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.BatchNorm2d(48, momentum=0.1),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        x = self.part(x)
        yy = self.final_layer(x)
        yy = F.avg_pool2d(yy, kernel_size=yy.size()
                                 [2:]).view(yy.size(0), -1)
        y = x.view(x.size(0), 512, -1)
        y = y.permute(0, 2, 1)
        return y, yy

class depth_feature_extractor(nn.Module):
    def __init__(self, depth_model):
        super(depth_feature_extractor, self).__init__()
        self.part = nn.Sequential(*list(depth_model.children())[:-2])
        self.final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=48,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.BatchNorm2d(48, momentum=0.1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.part(x)
        yy = self.final_layer(x)
        yy = F.avg_pool2d(yy, kernel_size=yy.size()
                                 [2:]).view(yy.size(0), -1)
        y = x.view(x.size(0), 512, -1)
        y = y.permute(0, 2, 1)
        return y, yy

class mmwave_feature_extractor(nn.Module):
    def __init__(self, mmwave_model):
        super(mmwave_feature_extractor, self).__init__()
        self.part = nn.Sequential(*list(mmwave_model.children())[:-1])
        self.final_layer = nn.Sequential(
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256,48),
            nn.BatchNorm1d(48),
            nn.ReLU()
        )
    def forward(self, x):
        y, _ = self.part(x)
        yy = y.permute(0,2,1)
        yy = F.avg_pool1d(yy, kernel_size = yy.size()[-1]).view(yy.size(0), -1)
        yy = self.final_layer(yy)
        return y, yy

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

        self.final_layer = nn.Sequential(
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256,48),
            nn.BatchNorm1d(48),
            nn.ReLU()
        )
    
    def forward(self, x):
        xyz = x[..., :3]
        points = self.transformer1(xyz, self.fc1(x))[0]

        xyz_and_feats = [(xyz, points)]
        for i in range(self.nblocks - 4):
            xyz, points = self.transition_downs[i](xyz, points)
            points = self.transformers[i](xyz, points)[0]
            xyz_and_feats.append((xyz, points))
        points = points.view(points.size(0), -1, 512)

        yy = points.permute(0,2,1)
        yy = F.avg_pool1d(yy, kernel_size = yy.size()[-1]).view(yy.size(0), -1)
        yy = self.final_layer(yy)
        return points, yy

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
        self.final_layer = nn.Sequential(
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256,48),
            nn.BatchNorm1d(48),
            nn.ReLU()
        )
    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.transpose(x, 2, 3) #16,2,114,3,32
        x = torch.flatten(x, 3, 4)# 16,2,114,96
        torch_resize = Resize([136,32])
        x = torch_resize(x)
        x = self.part(x).view(x.size(0), 512, -1)
        y = x.permute(0, 2, 1)
        yy = F.avg_pool1d(x, kernel_size = x.size()[-1]).view(x.size(0), -1)
        yy = self.final_layer(yy)
        return y, yy

# class feature_extrator(nn.Module):
#     def __init__(self):
#         super(feature_extrator, self).__init__()
        
#         rgb_model = RGB_ResNet18()
#         rgb_model.load_state_dict(torch.load('./RGB_benchmark/rgb_ResNet18/RGB_Resnet18_copy.pt'))
#         rgb_extractor = rgb_feature_extractor(rgb_model)
#         rgb_extractor.eval()

#         depth_model = Depth_ResNet18()
#         depth_model.load_state_dict(torch.load('depth_benchmark/depth_Resnet18.pt'))
#         depth_extractor = depth_feature_extractor(depth_model)
#         depth_extractor.eval()
        
#         mmwave_model = mmwave_PointTransformerReg()
#         mmwave_model.load_state_dict(torch.load('mmwave_benchmark/mmwave_all_random_TD.pt'))
#         mmwave_extractor = mmwave_feature_extractor(mmwave_model)
#         mmwave_extractor.eval()

#         # lidar_model = lidar_PointTransformerReg()
#         # lidar_model.load_state_dict(torch.load('lidar_benchmark/lidar_all_random.pt'))
#         # lidar_extractor = lidar_feature_extractor(lidar_model)
#         # lidar_extractor.eval()

#         # sys.path.insert(0, './CSI_benchmark')
#         # csi_model = torch.load('CSI_benchmark/protocol3_random_1.pkl')
#         # csi_extractor = csi_feature_extractor(csi_model)
#         # csi_extractor.eval()

#         self.rgb_extractor = rgb_extractor
#         self.depth_extractor = depth_extractor
#         self.mmwave_extractor = mmwave_extractor
#         # self.lidar_extractor = lidar_extractor
#         # self.csi_extractor = csi_extractor

#     def forward(self, rgb_data, depth_data, mmwave_data, lidar_data, csi_data, modality_list):
#         if sum(modality_list) == 0:
#             raise ValueError("At least one modality should be selected")
#         else:
#             real_feature_list = []
#             if modality_list[0] == True:
#                 rgb_feature = self.rgb_extractor(rgb_data)
#                 "shape b x 49 x 512"
#                 real_feature_list.append(rgb_feature)
#             if modality_list[1] == True:
#                 depth_feature = self.depth_extractor(depth_data)
#                 "shape b x 49 x 512"
#                 real_feature_list.append(depth_feature)
#             if modality_list[2] == True:
#                 mmwave_feature = self.mmwave_extractor(mmwave_data)
#                 "shape b x 32 x 512"
#                 real_feature_list.append(mmwave_feature)
#             if modality_list[3] == True:
#                 lidar_feature = self.lidar_extractor(lidar_data)
#                 "shape b x 32 x 512"
#                 real_feature_list.append(lidar_feature)
#             if modality_list[4] == True:
#                 csi_feature = self.csi_extractor(csi_data)
#                 "shape b x (17*4) x 512"
#                 real_feature_list.append(csi_feature)
        
#         return real_feature_list


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size = 512, num_heads = 8, dropout = 0.0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(emb_size, emb_size*3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
    
    def forward(self, x, mask = None):
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion = 4, drop_p = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size = 512,
                 drop_p = 0.0,
                 forward_expansion = 4,
                 forward_drop_p = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))

class FusionTransformer(torch.nn.Module):
        def __init__(self,
                num_emb = 48*5,
                emb_size = 512,
                drop_p = 0.,
                forward_expansion = 4,
                forward_drop_p = 0.,
                depth = 3,
                ** kwargs):
            super(FusionTransformer,self).__init__()
            self.depth = depth
            self.TD1 = nn.Sequential(
                nn.Conv1d(num_emb, 256, 1),
                nn.BatchNorm1d(256),
                nn.ReLU())
            self.TD2 = nn.Sequential(
                nn.Conv1d(256, 128, 1),
                nn.BatchNorm1d(128),
                nn.ReLU())

            self.transformer_layers = nn.ModuleList([])
            for _ in range(depth):
                self.transformer_layers.append(TransformerEncoderBlock())
            
        def forward(self, feature_embedding):
            layer_idx = 0
            for layer in self.transformer_layers:
                feature_embedding = layer(feature_embedding)
                if layer_idx == 0:
                    feature_embedding = self.TD1(feature_embedding)
                    layer_idx += 1
                elif layer_idx == 1:
                    feature_embedding = self.TD2(feature_embedding)
                    layer_idx += 1
                else:
                    continue
            return feature_embedding
            


class AdaptiveFusion(torch.nn.Module):
    def __init__(self):
        super(AdaptiveFusion, self).__init__()

        rgb_model = RGB_ResNet18()
        rgb_model.load_state_dict(torch.load('./RGB_benchmark/rgb_ResNet18/RGB_Resnet18_copy.pt'))
        rgb_extractor = rgb_feature_extractor(rgb_model)

        depth_model = Depth_ResNet18()
        depth_model.load_state_dict(torch.load('depth_benchmark/depth_Resnet18.pt'))
        depth_extractor = depth_feature_extractor(depth_model)
        
        mmwave_model = mmwave_PointTransformerReg()
        mmwave_model.load_state_dict(torch.load('mmwave_benchmark/mmwave_all_random_TD.pt'))
        mmwave_extractor = mmwave_feature_extractor(mmwave_model)

        lidar_model = lidar_PointTransformerReg()
        lidar_model.load_state_dict(torch.load('lidar_benchmark/lidar_all_random.pt'))
        lidar_extractor = lidar_feature_extractor(lidar_model)

        sys.path.insert(0, './CSI_benchmark')
        csi_model = torch.load('CSI_benchmark/protocol3_random_1.pkl')
        csi_extractor = csi_feature_extractor(csi_model)

        self.rgb_extractor = rgb_extractor
        self.depth_extractor = depth_extractor
        self.mmwave_extractor = mmwave_extractor
        self.lidar_extractor = lidar_extractor
        self.csi_extractor = csi_extractor

        self.GIM = Transformer(dim=48*5, depth=3, heads=8, dim_head=128, mlp_dim=4096)

        self.trans_encoder = FusionTransformer(num_emb = 470)

        self.regression_head = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 51)
        )

    def forward(self, rgb_inputs, depth_inputs, mmwave_inputs, lidar_inputs, wifi_inputs):
                        
        global_feats = []
        local_feats = []

        rgb_local_feature, rgb_glob_feature = self.rgb_extractor(rgb_inputs)
        global_feats.append(rgb_glob_feature)
        local_feats.append(rgb_local_feature)

        depth_local_feature, depth_glob_feature = self.depth_extractor(depth_inputs)
        global_feats.append(depth_glob_feature)
        local_feats.append(depth_local_feature)

        mmwave_local_feature, mmwave_glob_feature = self.mmwave_extractor(mmwave_inputs)
        global_feats.append(mmwave_glob_feature)
        local_feats.append(mmwave_local_feature)

        lidar_local_feature, lidar_glob_feature = self.lidar_extractor(lidar_inputs)
        global_feats.append(lidar_glob_feature)
        local_feats.append(lidar_local_feature)

        wifi_local_feature, wifi_glob_feature = self.csi_extractor(wifi_inputs)
        global_feats.append(wifi_glob_feature)
        local_feats.append(wifi_local_feature)
        # concatinate global and local features
        global_feats = torch.cat(global_feats, dim=1)
        local_feats = torch.cat(local_feats, dim=1)

        # integrate global features
        # print(global_feats.shape)
        # torch.Size([16, 128*3])
        # torch.Size([16, 48*5])
        # print(local_feats.shape)
        # torch.Size([16, 130, 512])
        global_feats = self.GIM(global_feats)
        # print(global_feats.shape)
        # torch.Size([16, 16, 128*3])

        fusion_feat = torch.sum(global_feats, dim=1, keepdim=True).expand(-1,512, -1).permute(0,2,1)

        features = torch.cat([fusion_feat, local_feats], dim=1)

        # print(features.shape)
        # torch.Size([16, 514, 512])
        # torch.Size([16, 230+48*5, 512])

        features = self.trans_encoder(features)
        # print(features.shape)
        # torch.Size([16, 128, 512])
        
        features = torch.mean(features, dim = -1)

        x = self.regression_head(features)

        x = x.view(x.size(0), 17, 3)

        return x
