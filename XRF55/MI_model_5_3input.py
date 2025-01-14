import numpy as np
import glob
import scipy.io as sio
import torch
from torch import nn
import torch.nn.functional as F
import csv
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize
import yaml
import time
import re
import random
import sys

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

from backbone_models.mmWave.ResNet import *

"map each modality num of features to 32 "
" Use modality fusion transforemr"

class mmwave_feature_extractor(nn.Module):
    def __init__(self, mmwave_model):
        super(mmwave_feature_extractor, self).__init__()
        self.part = nn.Sequential(*list(mmwave_model.children())[:-2])
    def forward(self, x):
        x = self.part(x).view(x.size(0), 512, -1)
        x = x.permute(0, 2, 1)
        return x
    # shape: B, 512, 32

class wifi_feature_extractor(nn.Module):
    def __init__(self, wifi_model):
        super(wifi_feature_extractor, self).__init__()
        self.part = nn.Sequential(*list(wifi_model.children())[:-2])
    def forward(self, x):
        x = self.part(x).view(x.size(0), 512, -1)
        x = x.permute(0, 2, 1)
        return x
    # shape: B, 512, 4

class rfid_feature_extractor(nn.Module):
    def __init__(self, rfid_model):
        super(rfid_feature_extractor, self).__init__()
        self.part = nn.Sequential(*list(rfid_model.children())[:-3])
    def forward(self, x):
        x = self.part(x).view(x.size(0), 512, -1)
        x = x.permute(0, 2, 1)
        return x 
    # shape: B, 512, 5




class feature_extrator(nn.Module):
    def __init__(self):
        super(feature_extrator, self).__init__()
        
        mmwave_model = torch.load('./backbone_models/mmWave/ResNet18.pt')
        mmwave_extractor = mmwave_feature_extractor(mmwave_model)
        mmwave_extractor.eval()

        wifi_model = torch.load('./backbone_models/WIFI/ResNet18.pt')
        wifi_extractor = wifi_feature_extractor(wifi_model)
        wifi_extractor.eval()

        rfid_model = torch.load('./backbone_models/RFID/ResNet18.pt')
        rfid_extractor = rfid_feature_extractor(rfid_model)
        rfid_extractor.eval()

        self.mmwave_extractor = mmwave_extractor
        self.wifi_extractor = wifi_extractor
        self.rfid_extractor = rfid_extractor

    def forward(self, mmwave_data, wifi_data, rfid_data, modality_list):
        if sum(modality_list) == 0:
            raise ValueError("At least one modality should be selected")
        else:
            real_feature_list = []
            if modality_list[0] == True:
                mmwave_feature = self.mmwave_extractor(mmwave_data)
                # print("mmwave pre-trained model loaded")
                "shape b x 32 x 512"
                real_feature_list.append(mmwave_feature)
            if modality_list[1] == True:
                wifi_feature = self.wifi_extractor(wifi_data)
                # print("wifi pre-trained model loaded")
                "shape b x 3 x 512"
                real_feature_list.append(wifi_feature)
            if modality_list[2] == True:
                rfid_feature = self.rfid_extractor(rfid_data)
                # print("rfid pre-trained model loaded")
                "shape b x 5 x 512"
                real_feature_list.append(rfid_feature)
        
        return real_feature_list


class linear_projector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(linear_projector, self).__init__()
        '''Conv 1d layer for each modality'''
        self.mmwave_linear_projection = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, 1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        self.wifi_linear_projection = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, 1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Linear(4, 32),
            nn.ReLU()
        )
        self.rfid_linear_projection = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, 1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Linear(5, 32),
            nn.ReLU()
        )
    def forward(self, feature_list, modality_list):
        # example:
        # feature_list = [wifi_feature, rfid_feature]
        # modality_list = [False, True, True]
        feature_flag = 0
        for i in range(len(modality_list)):
            if modality_list[i] == True:
                if i == 0:
                    mmwave_feature = feature_list[feature_flag]
                    # print("mmwave_feature shape", mmwave_feature.shape)
                elif i == 1:
                    wifi_feature = feature_list[feature_flag]
                    # print("wifi_feature shape", wifi_feature.shape)
                elif i == 2:
                    rfid_feature = feature_list[feature_flag]
                    # print("rfid_feature shape", rfid_feature.shape)
                feature_flag += 1
            else:
                continue
        if sum (modality_list) == 0:
            raise ValueError("At least one modality should be selected")
        else:
            projected_feature_list = []
            if modality_list[0] == True:
                projected_feature_list.append(self.mmwave_linear_projection(mmwave_feature.permute(0, 2, 1)))
            if modality_list[1] == True:
                projected_feature_list.append(self.wifi_linear_projection(wifi_feature.permute(0, 2, 1)))
            if modality_list[2] == True:
                projected_feature_list.append(self.rfid_linear_projection(rfid_feature.permute(0, 2, 1)))
            projected_feature = torch.cat(projected_feature_list, dim=2).permute(0, 2, 1)
            "projected_feature shape: B, 32*n, 512"
            # if modality_list[3] == True:
            #     feature_shape = projected_feature.shape
            #     new_xyz = selective_pos_enc(lidar_points, feature_shape[1])
            #     # print("new_xyz shape", new_xyz.shape)
            #     'new_xyz shape: B, 32, 3'
            #     pos_enc = self.pos_enc_layer(new_xyz.permute(0, 2, 1)).permute(0, 2, 1)
            #     # print("pos_enc shape", pos_enc.shape)
            #     'pos_enc shape: B, 32, 512'
            #     # pos_enc_repeat = pos_enc.repeat(1, sum(modality_list), 1)
            #     projected_feature += pos_enc
            # else:
            #     pass
        return projected_feature

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size = 512, num_heads = 8, dropout = 0.0):
        super(MultiHeadAttention,self).__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(emb_size, emb_size*3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        self.pool = nn.AdaptiveAvgPool2d((32,None))
    
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
        out = self.pool(out)
        return out

class qkv_Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super(qkv_Attention,self).__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)


        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.LayerNorm(dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, qkv):
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super(FeedForward,self).__init__()
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

class kv_projection(nn.Module):
    def __init__(self, dim, expension = 2):
        super(kv_projection,self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.MLP = nn.Sequential(
            nn.Linear(dim, dim * expension),
            nn.LayerNorm(dim * expension),
            nn.ReLU(),
            nn.Linear(dim * expension, dim)
        )
        # self.to_q = nn.Linear(dim, dim, bias = False)
        self.to_k = nn.Linear(dim, dim, bias = False)
        self.to_v = nn.Linear(dim, dim, bias = False)
    
    def forward(self, x):
        x = self.MLP(x)
        # q = self.to_q(self.norm(x))
        k = self.to_k(self.norm(x))
        v = self.to_v(self.norm(x))
        kv = [k, v]
        return kv

class cross_modal_transformer(nn.Module):
    def __init__(self, num_feature=32, max_num_modality=4, dim_expansion=2, emb_size = 512, num_heads = 8, dropout=0.3):
        super(cross_modal_transformer,self).__init__()
        self.attention = MultiHeadAttention(emb_size, num_heads, dropout)
        self.ffw = FeedForward(emb_size, emb_size*dim_expansion, dropout)
        self.pool = nn.AdaptiveAvgPool2d((32,None))
    
    def forward(self, feature_embedding, modality_list):
        feature_embedding_ = self.attention(feature_embedding) + self.pool(feature_embedding)
        out_feature_embedding = self.ffw(feature_embedding_) + feature_embedding_
        return out_feature_embedding

class fusion_transformer(nn.Module):
    def __init__(self, dim, hidden_dim, num_heads, dim_heads, dropout):
        super(fusion_transformer,self).__init__()
        self.mutihead_attention = qkv_Attention(dim, num_heads, dim_heads, dropout)
        self.feed_forward = FeedForward(dim, hidden_dim, dropout)
    
    def forward(self, feature_embedding, kv):
    # def forward(self, x, modality_list):
        qkv = (feature_embedding, kv[0], kv[1])
        x = self.mutihead_attention(qkv) + feature_embedding
        new_feature_embedding = self.feed_forward(x) + x
        # x = self.mutihead_attention(x) + x
        # x = self.feed_forward(x) + x
        return new_feature_embedding


class cross_attention_transformer_block(nn.Module):
    def __init__(self, dim, hidden_dim, num_heads, dim_heads, num_modality, dropout):
        super(cross_attention_transformer_block,self).__init__()
        self.transformer_layers = nn.ModuleList([])
        for _ in range(num_modality):
            self.transformer_layers.append(fusion_transformer(dim, hidden_dim, num_heads, dim_heads, dropout))

    def forward(self, feature_embedding, kv_list, modality_list):
        "each layer fusion"
        transformer_layer_idx = 0
        feature_idx_ = 0
        features_list = []
        for layer in self.transformer_layers:
            # print("transformer_layer_idx", transformer_layer_idx)
            # print("feature_idx_", feature_idx_)
            if modality_list[transformer_layer_idx] == True:
                # print(qkv_list[feature_idx_][1:])
                new_feature_embedding = layer(feature_embedding, kv_list[feature_idx_])
                features_list.append(new_feature_embedding)
                transformer_layer_idx += 1
                feature_idx_ += 1
            else:
                transformer_layer_idx += 1
        feature_embedding = torch.cat(features_list, dim=1)
        return feature_embedding

class classification_Head(nn.Sequential):
    def __init__(self, emb_size=512, num_classes=27):
        super(classification_Head,self).__init__()
        self.norm = nn.LayerNorm(emb_size)
        self.fc = nn.Linear(emb_size, num_classes)
    
    def forward(self, x):
        # print(x.shape)
        x = torch.mean(x, dim=1)
        # print(x.shape)
        x = self.norm(x)
        x = self.fc(x)
        # x = x.view(x.size(0), 17, 3)
        return x
    
class MI_Transformer(nn.Module):
    def __init__(self, num_modalities, dim, qkv_hidden_expansion, hidden_dim, num_feature, num_heads, dim_heads, model_depth, dropout, num_classes):
        super(MI_Transformer,self).__init__()
        self.kv_layers = nn.ModuleList([])
        self.cross_attention_transformer = nn.ModuleList([])
        for _ in range(num_modalities):
            self.kv_layers.append(kv_projection(dim, qkv_hidden_expansion))
        # for _ in range(model_depth):
        #     self.transformer_layers.append(fusion_transformer_block(dim, hidden_dim, num_heads, dim_heads, num_modalities))
        self.cross_attention_transformer = cross_attention_transformer_block(dim, hidden_dim, num_heads, dim_heads, num_modalities, dropout)
        self.cross_modal_transformer = cross_modal_transformer(num_feature, num_modalities, qkv_hidden_expansion, dim, num_heads, dropout)
        self.depth = model_depth
        # self.qkv_layers: [qkv_layer_1, qkv_layer_2, ...]
        # self.transformer_layers: [
        #     [mod1_transformer_layer_1, mod2_transformer_layer_1, ...], 
        #     [mod1_transformer_layer_2, mod2_transformer_layer_2, ...],
        #     ...
        # ]
        # self.FFD = FeedForward(dim, hidden_dim)
        self.classification_head = classification_Head(dim, num_classes)

    def forward(self, feature, modality_list):
        num_modalities = sum(modality_list)
        feature_list = list(feature.chunk(num_modalities, dim = 1))

        kv_list = []
        kv_layer_idx = 0
        feature_idx = 0
        for kv_layer in self.kv_layers:
            if modality_list[kv_layer_idx] == True:
                kv = kv_layer(feature_list[feature_idx])
                kv_list.append(kv)
                feature_idx += 1
                kv_layer_idx += 1
            else:
                kv_layer_idx += 1
                continue
        "output kv_list: [[k_1, v_1], [k_2, v_2], ...]"
        feature_embedding = self.cross_modal_transformer(feature, modality_list)
        "feature_embedding shape: B, 32, 512 (averaged sum of all features)"

        for i in range(self.depth):
            # print("dominant_q", dominant_q.shape)
            # print("qkv_list", qkv_list)
            # print("modality_list", modality_list)
            "feature_embedding shape: B, 32, 512"
            feature_embedding = self.cross_attention_transformer(feature_embedding, kv_list, modality_list)
            "feature_embedding shape: B, 32*n, 512"
            feature_embedding = self.cross_modal_transformer(feature_embedding, modality_list)
            "feature_embedding shape: B, 32, 512"
        # for transformer_layer in self.transformer_layers:
        #     dominant_q = transformer_layer(dominant_q, qkv_list, modality_list)

        x = self.classification_head(feature_embedding)
        return x
        


class modality_invariant_model(nn.Module):
    def __init__(self, model_depth, num_classes):
        super(modality_invariant_model, self).__init__()
        self.feature_extractor = feature_extrator()
        self.linear_projector = linear_projector(512, 512)
        self.MIT = MI_Transformer(
            num_modalities = 3,
            dim = 512,
            qkv_hidden_expansion = 2,
            hidden_dim = 512,
            num_feature = 32,
            num_heads = 8,
            dim_heads = 64,
            model_depth = model_depth,
            dropout = 0.,
            num_classes = num_classes
        )
    def forward(self,  mmwave_data, wifi_data, rfid_data, modality_list):
        # features = self.feature_extractor(rgb_data, depth_data, mmwave_data, lidar_data, csi_data, modality_list)
        feature_list = self.feature_extractor(mmwave_data, wifi_data, rfid_data, modality_list)
        # print("rgb_hidden_feature", rgb_feature)
        # print("depth_hidden_feature", depth_feature)
        # print("mmwave_hidden_feature", mmwave_feature)
        # print("lidar_hidden_feature", lidar_feature)
        # print("csi_hidden_feature", csi_feature)
        # projected_features = self.linear_projector(features, lidar_data, modality_list)
        projected_features = self.linear_projector(feature_list, modality_list)
        # print("feature after linear mapping", feature)
        # print("feature shape", feature.shape)
        out = self.MIT(projected_features, modality_list)
        # print("output", out)
        # out = out.view(-1, 17, 3)
        return out

''' 打断点方法： # %% '''


