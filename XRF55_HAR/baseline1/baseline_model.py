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

path = os.getcwd()
parentdir = os.path.abspath(os.path.join(path, os.pardir))
sys.path.insert(0, parentdir)

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

class classification_Head(nn.Sequential):
    def __init__(self, emb_size=512, num_classes=55):
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

"################################# for single modality #################################"
class single_feature_extrator(nn.Module):
    def __init__(self, modality_list):
        "modality_list: list of string, choose from ['mmwave', 'wifi', 'rfid']"
        super(single_feature_extrator, self).__init__()
        if 'mmwave' in modality_list:
            mmwave_model = torch.load(os.path.join(parentdir, 'backbone_models/mmWave/mmwave_ResNet18.pt'))
            mmwave_extractor = mmwave_feature_extractor(mmwave_model)
            mmwave_extractor.eval()
            self.mmwave_extractor = mmwave_extractor
        if 'wifi' in modality_list:
            wifi_model = torch.load(os.path.join(parentdir, 'backbone_models/WIFI/wifi_ResNet18.pt'))
            wifi_extractor = wifi_feature_extractor(wifi_model)
            wifi_extractor.eval()
            self.wifi_extractor = wifi_extractor
        if 'rfid' in modality_list:
            rfid_model = torch.load(os.path.join(parentdir, 'backbone_models/RFID/rfid_ResNet18.pt'))
            rfid_extractor = rfid_feature_extractor(rfid_model)
            rfid_extractor.eval()
            self.rfid_extractor = rfid_extractor
        
        
        
    def forward(self, input, exist_list):
        if exist_list[0] == True:
            feature = self.mmwave_extractor(input)
            "shape b x 49 x 512"
        elif exist_list[1] == True:
            feature = self.wifi_extractor(input)
            "shape b x 49 x 512"
        elif exist_list[2] == True:
            feature = self.rfid_extractor(input)
            "shape b x n_p x 512"
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
        "modality_list: list of string, choose from ['mmwave', 'wifi', 'rfid']"
        super(Dual_feature_extrator, self).__init__()
        if 'mmwave' in modality_list:
            mmwave_model = torch.load(os.path.join(parentdir, 'backbone_models/mmWave/mmwave_ResNet18.pt'))
            mmwave_extractor = mmwave_feature_extractor(mmwave_model)
            mmwave_extractor.eval()
            self.mmwave_extractor = mmwave_extractor
        if 'wifi' in modality_list:
            wifi_model = torch.load(os.path.join(parentdir, 'backbone_models/WIFI/wifi_ResNet18.pt'))
            wifi_extractor = wifi_feature_extractor(wifi_model)
            wifi_extractor.eval()
            self.wifi_extractor = wifi_extractor
        if 'rfid' in modality_list:
            rfid_model = torch.load(os.path.join(parentdir, 'backbone_models/RFID/rfid_ResNet18.pt'))
            rfid_extractor = rfid_feature_extractor(rfid_model)
            rfid_extractor.eval()
            self.rfid_extractor = rfid_extractor
        
        
        
    def forward(self, input_1, input_2, exist_list):
        flag = False
        if exist_list[0] == True:
            if flag == False:
                feature_1 = self.mmwave_extractor(input_1)
                flag = True
            else:
                feature_2 = self.mmwave_extractor(input_2)
            "shape b x 49 x 512"
        if exist_list[1] == True:
            if flag == False:
                feature_1 = self.wifi_extractor(input_1)
                flag = True
            else:
                feature_2 = self.wifi_extractor(input_2)
            "shape b x 49 x 512"
        if exist_list[2] == True:
            if flag == False:
                feature_1 = self.rfid_extractor(input_1)
                flag = True
            else:
                feature_2 = self.rfid_extractor(input_2)
            "shape b x n_p x 512"
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
        "modality_list: list of string, choose from ['mmwave', 'wifi', 'rfid']"
        super(Triple_feature_extrator, self).__init__()
        if 'mmwave' in modality_list:
            mmwave_model = torch.load(os.path.join(parentdir, 'backbone_models/mmWave/mmwave_ResNet18.pt'))
            mmwave_extractor = mmwave_feature_extractor(mmwave_model)
            mmwave_extractor.eval()
            self.mmwave_extractor = mmwave_extractor
        if 'wifi' in modality_list:
            wifi_model = torch.load(os.path.join(parentdir, 'backbone_models/WIFI/wifi_ResNet18.pt'))
            wifi_extractor = wifi_feature_extractor(wifi_model)
            wifi_extractor.eval()
            self.wifi_extractor = wifi_extractor
        if 'rfid' in modality_list:
            rfid_model = torch.load(os.path.join(parentdir, 'backbone_models/RFID/rfid_ResNet18.pt'))
            rfid_extractor = rfid_feature_extractor(rfid_model)
            rfid_extractor.eval()
            self.rfid_extractor = rfid_extractor
        
        
        
    def forward(self, input_1, input_2, input_3, exist_list):
        feature_1 = self.mmwave_extractor(input_1)
        feature_2 = self.wifi_extractor(input_2)
        feature_3 = self.rfid_extractor(input_3)
        "shape b x 32 x 512"
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



