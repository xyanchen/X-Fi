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


from mixture_of_experts import MoE
from syn_DI_dataset import make_dataset, make_dataloader
from RGB_benchmark.rgb_ResNet18.RGB_ResNet import *
from depth_benchmark.depth_ResNet18 import *
from mmwave_benchmark.mmwave_point_transformer_TD import *
from lidar_benchmark.lidar_point_transformer import *
from lidar_benchmark.pointnet_util import *

"map each modality num of features to 32 "
" Use modality fusion transforemr"

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
        
    def forward(self, rgb_data, depth_data, mmwave_data, lidar_data, csi_data):
        rgb_feature = self.rgb_extractor(rgb_data)
        "shape b x 49 x 512"
        depth_feature = self.depth_extractor(depth_data)
        "shape b x 49 x 512"
        mmwave_feature = self.mmwave_extractor(mmwave_data)
        "shape b x 32 x 512"
        lidar_feature = self.lidar_extractor(lidar_data)
        "shape b x 32 x 512"
        csi_feature = self.csi_extractor(csi_data)
        "shape b x (17*4) x 512"
        return rgb_feature, depth_feature, mmwave_feature, lidar_feature, csi_feature

def selective_pos_enc(xyz, npoint):
    """
    Input:
        xyz: input points position data, [B, N, 3]
    Return:
        new_xyz: sampled points position data, [B, S, 3]
        out: new features of sampled points, [B, S, C]
    """
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint]
    torch.cuda.empty_cache()
    new_xyz = index_points(xyz, fps_idx)
    torch.cuda.empty_cache()
    return new_xyz



class linear_projector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(linear_projector, self).__init__()
        '''linear layer for each modality'''
        # self.rgb_linear_projection = nn.Linear(input_dim, output_dim)
        # self.depth_linear_projection = nn.Linear(input_dim, output_dim)
        # self.mmwave_linear_projection = nn.Linear(input_dim, output_dim)
        # self.lidar_linear_projection = nn.Linear(input_dim, output_dim)
        # self.csi_linear_projection = nn.Linear(input_dim, output_dim)
        '''Conv 1d layer for each modality'''
        self.rgb_linear_projection = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, 1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Linear(49, 32),
            nn.ReLU()
        )
        self.depth_linear_projection = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, 1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Linear(49, 32),
            nn.ReLU()
        )
        self.mmwave_linear_projection = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, 1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        self.lidar_linear_projection = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, 1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        self.csi_linear_projection = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, 1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Linear(17*4, 32),
            nn.ReLU()
        )
        self.pos_enc_layer = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, output_dim, 1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )
    # def forward(self, rgb_feature, depth_feature, mmwave_feature, lidar_feature, csi_feature, modality_list):
    #     feature_list = [rgb_feature, depth_feature, mmwave_feature, lidar_feature, csi_feature]
    #     if sum (modality_list) == 0:
    #         print('WARNING: modality_list is empty!')
    #         feature = torch.zeros_like(lidar_feature, device=torch.device('cuda'))
    #         feature = self.linear_projection(feature)
    #     else:
    #         real_feature_list = []
    #         for i in range(len(modality_list)):
    #             if modality_list[i] == True:
    #                 real_feature_list.append(feature_list[i])
    #             else:
    #                 continue
    #         feature = torch.cat(real_feature_list, dim=1)
    #         feature = self.linear_projection(feature)
    #     return feature
    def forward(self, rgb_feature, depth_feature, mmwave_feature, lidar_feature, csi_feature, lidar_points, modality_list):
        # feature_list = [rgb_feature, depth_feature, mmwave_feature, lidar_feature, csi_feature]
        if sum (modality_list) == 0:
            print('WARNING: modality_list is empty!')
            feature = torch.zeros((lidar_feature.shape[0], 32, 512), device=torch.device('cuda'))
        else:
            real_feature_list = []
            if modality_list[0] == True:
                real_feature_list.append(self.rgb_linear_projection(rgb_feature.permute(0, 2, 1)))
            if modality_list[1] == True:
                real_feature_list.append(self.depth_linear_projection(depth_feature.permute(0, 2, 1)))
            if modality_list[2] == True:
                real_feature_list.append(self.mmwave_linear_projection(mmwave_feature.permute(0, 2, 1)))
            if modality_list[3] == True:
                real_feature_list.append(self.lidar_linear_projection(lidar_feature.permute(0, 2, 1)))
            if modality_list[4] == True:
                real_feature_list.append(self.csi_linear_projection(csi_feature.permute(0, 2, 1)))
            # feature = torch.cat(real_feature_list, dim=1)
            # for i in range(len(real_feature_list)):
            #     print("real_feature_list", real_feature_list[i].shape)
            feature = torch.cat(real_feature_list, dim=2).permute(0, 2, 1)
            # feature = self.linear_projection(feature)
            if modality_list[3] == True:
                feature_shape = feature.shape
                new_xyz = selective_pos_enc(lidar_points, feature_shape[1])
                # print("new_xyz shape", new_xyz.shape)
                'new_xyz shape: B, eature_shape[1], 3'
                pos_enc = self.pos_enc_layer(new_xyz.permute(0, 2, 1)).permute(0, 2, 1)
                # print("pos_enc shape", pos_enc.shape)
                feature = feature + pos_enc
            else:
                pass
        return feature

class MoE_Projection(nn.Module):
    def __init__(self, dim, hidden_dim, num_experts, qkv_dim):
        super().__init__()
        self.moe = MoE(
            dim = dim,
            num_experts = num_experts,               # increase the experts (# parameters) of your model without increasing computation
            hidden_dim = hidden_dim,           # size of hidden dimension in each expert, defaults to 4 * dimension
            activation = nn.LeakyReLU,      # use your preferred activation, will default to GELU
            second_policy_train = 'random', # in top_2 gating, policy for whether to use a second-place expert
            second_policy_eval = 'random',  # all (always) | none (never) | threshold (if gate value > the given threshold) | random (if gate value > threshold * random_uniform(0, 1))
            second_threshold_train = 0.2,
            second_threshold_eval = 0.2,
            capacity_factor_train = 1.25,   # experts have fixed capacity per batch. we need some extra capacity in case gating is not perfectly balanced.
            capacity_factor_eval = 2.,      # capacity_factor_* should be set to a value >=1
            loss_coef = 1e-2                # multiplier on the auxiliary expert balancing auxiliary loss
        )
        # self.norm = nn.LayerNorm(dim)
        # self.to_qkv = nn.Linear(dim, qkv_dim * 3, bias = False)

    def forward(self, x):
        x, aux_loss = self.moe(x)
        # x = self.norm(x)
        # qkv = list(self.to_qkv(x).chunk(3, dim = -1))
        
        # return x, qkv, aux_loss
        return x, aux_loss

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        # self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.LayerNorm(dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, qkv):
    # def forward(self, x):
        # x = self.norm(x)
        # qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

def qurey_fusion(qkv_list, idx):
    
    # qkv_list: [(q_1, k_1, v_1), (q_2, k_2, v_2), ...]
    for i in range(len(qkv_list)):
        # print(qkv_list[i][0].shape)
        qkv_list[i][0] = qkv_list[idx][0]
    
    return qkv_list

class qvk_projection(nn.Module):
    def __init__(self, dim, expension = 2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.MLP = nn.Sequential(
            nn.Linear(dim, dim * expension),
            nn.LayerNorm(dim * expension),
            nn.ReLU(),
            nn.Linear(dim * expension, dim)
        )
        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
    
    def forward(self, x):
        x = self.MLP(x)
        qkv = list(self.to_qkv(self.norm(x)).chunk(3, dim = -1))
        return x, qkv

def fuision_idx(num_modalities, model_depth):
    
    " generate qurey_fusion_idx "
    qurey_fusion_idx = []
    repeat_list = list(range(num_modalities))
    for _ in range(model_depth):
        idx = random.randint(0,num_modalities-1)
        while idx not in repeat_list:
            idx = random.randint(0,num_modalities-1)
        repeat_list.remove(idx)
        qurey_fusion_idx.append(idx)
        if len(repeat_list) == 0:
            repeat_list = list(range(num_modalities))
    
    return qurey_fusion_idx

class fusion_transformer(nn.Module):
    def __init__(self, dim, hidden_dim, num_experts, qkv_dim, num_heads, dim_heads, num_modalities):
        super().__init__()
        self.layers = nn.ModuleList([])
        # self.moe = MoE_Projection(dim, hidden_dim, num_experts, qkv_dim)
        # self.norm = nn.LayerNorm(dim)
        # self.dropout = nn.Dropout(dropout)
        for _ in range(num_modalities):
            self.layers.append(qvk_projection(dim))
        self.mutihead_attention = Attention(dim, num_heads, dim_heads)
        self.feed_forward = FeedForward(dim, hidden_dim)
    def forward(self, x, modality_list, qurey_fusion_idx):
    # def forward(self, x, modality_list):
        if sum(modality_list) == 0:
            num_modalities = 1
        else:
            num_modalities = sum(modality_list)
        
        features = list(x.chunk(num_modalities, dim = -2))
        qkv_list = []
        # aux_loss_list = []
        # for i in range(num_modalities):
        #     # features[i], qkv, aux_loss = self.moe(features[i])
        #     features[i], aux_loss = self.moe(features[i])
        #     # qkv_list.append(qkv)
        #     aux_loss_list.append(aux_loss)
        featrue_idx = 0
        layer_idx = 0
        for layer in self.layers:
            if modality_list[layer_idx] == True:
                features[featrue_idx], qkv = layer(features[featrue_idx])
                qkv_list.append(qkv)
                featrue_idx += 1
                layer_idx += 1
            else:
                if layer_idx == len(modality_list) - 1 and featrue_idx == 0:
                    features[featrue_idx], qkv = layer(features[featrue_idx])
                    qkv_list.append(qkv)
                    layer_idx += 1
                else:
                    layer_idx += 1
                continue
        # x = torch.cat(features, dim = -2)
        
        fused_qkv_list = qurey_fusion(qkv_list, qurey_fusion_idx)
        for j in range(num_modalities):
            features[j] = self.mutihead_attention(fused_qkv_list[j]) + features[j]
            features[j] = self.feed_forward(features[j]) + features[j]
        x = torch.cat(features, dim = -2)
        # x = self.mutihead_attention(x) + x
        # x = self.feed_forward(x) + x
        return x

class regression_Head(nn.Sequential):
    def __init__(self, emb_size=512, num_classes=17*3):
        super().__init__()
        self.norm = nn.LayerNorm(emb_size)
        self.fc = nn.Linear(emb_size, num_classes)
    
    def forward(self, x):
        # print(x.shape)
        x = torch.mean(x, dim=1)
        # print(x.shape)
        x = self.norm(x)
        x = self.fc(x)
        x = x.view(x.size(0), 17, 3)
        return x

class MI_Transformer(nn.Module):
    def __init__(self, dim, hidden_dim, num_experts, qkv_dim, num_heads, dim_heads, depth, num_modalities):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(fusion_transformer(dim, hidden_dim, num_experts, qkv_dim, num_heads, dim_heads, num_modalities))
        self.regression_head = regression_Head(dim, 17*3)
        self.depth = depth

    def forward(self, x, modality_list):
        # print(x.shape)
        if sum(modality_list) == 0:
            num_modalities = 1
        else:
            num_modalities = sum(modality_list)
        qurey_fusion_idx_ls = fuision_idx(num_modalities, self.depth)
        i = 0
        for layer in self.layers:
            qurey_fusion_idx = qurey_fusion_idx_ls[i]
            # print(qurey_fusion_idx)
            x = layer(x, modality_list, qurey_fusion_idx)
            # x = layer(x, modality_list)
            # print(x.shape)
            i += 1
        x = self.regression_head(x)
        return x
        

class modality_invariant_model(nn.Module):
    def __init__(self):
        super(modality_invariant_model, self).__init__()
        self.feature_extractor = feature_extrator()
        self.linear_projector = linear_projector(512, 512)
        self.MIT = MI_Transformer(
            dim = 512,
            hidden_dim = 512,
            num_experts = 8,
            qkv_dim = 512,
            num_heads = 8,
            dim_heads = 64,
            depth = 4,
            num_modalities = 5
        )
    def forward(self, rgb_data, depth_data, mmwave_data, lidar_data, csi_data, modality_list):
        rgb_feature, depth_feature, mmwave_feature, lidar_feature, csi_feature = self.feature_extractor(rgb_data, depth_data, mmwave_data, lidar_data, csi_data)
        # print("rgb_hidden_feature", rgb_feature)
        # print("depth_hidden_feature", depth_feature)
        # print("mmwave_hidden_feature", mmwave_feature)
        # print("lidar_hidden_feature", lidar_feature)
        # print("csi_hidden_feature", csi_feature)
        feature = self.linear_projector(rgb_feature, depth_feature, mmwave_feature, lidar_feature, csi_feature, lidar_data, modality_list)
        # print("feature after linear mapping", feature)
        # print("feature shape", feature.shape)
        out = self.MIT(feature, modality_list)
        # print("output", out)
        # out = out.view(-1, 17, 3)
        return out

''' 打断点方法： # %% '''


