import numpy as np
import glob
import scipy.io as sio
import torch
from torch import nn
import random
import csv
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import yaml
from evaluate import error
import time
import re
from syn_DI_dataset import make_dataset, make_dataloader
from baseline_model import single_model

def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.

    dict_keys(['modality', 'scene', 'subject', 'action', 'idx', 'output', 
    'input_rgb', 'input_depth', 'input_lidar', 'input_mmwave'])
    '''
    kpts = []
    [kpts.append(np.array(t['output'])) for t in batch]
    kpts = torch.FloatTensor(np.array(kpts))
    # rgb
    rgb_data = np.array([(t['input_rgb']) for t in batch ])
    rgb_data = torch.FloatTensor(rgb_data).permute(0,3,1,2)
    # depth
    depth_data = np.array([(t['input_depth']) for t in batch ])
    depth_data = torch.FloatTensor(depth_data).permute(0,3,1,2)
    # mmwave
    ## padd
    mmwave_data = [torch.Tensor(t['input_mmwave']) for t in batch ]
    mmwave_data = torch.nn.utils.rnn.pad_sequence(mmwave_data)
    ## compute mask
    mmwave_data = mmwave_data.permute(1,0,2)
    # lidar
    ## padd
    lidar_data = [torch.Tensor(t['input_lidar']) for t in batch ]
    lidar_data = torch.nn.utils.rnn.pad_sequence(lidar_data)
    ## compute mask
    lidar_data = lidar_data.permute(1,0,2)
    # wifi-csi
    wifi_data = np.array([(t['input_wifi-csi']) for t in batch ])
    wifi_data = torch.FloatTensor(wifi_data)

    exist_list = [True, False, False, False, False]

    return rgb_data, depth_data, lidar_data, mmwave_data, wifi_data, kpts, exist_list


def get_result(rgb_model,depth_model,mmwave_model,lidar_model,wifi_model, tensor_loader, device):
    rgb_model.eval()
    depth_model.eval()
    mmwave_model.eval()
    lidar_model.eval()
    wifi_model.eval()
    for i, data in tqdm(enumerate(tensor_loader)):
        rgb_data, depth_data, lidar_data, mmwave_data, wifi_data, kpts, exist_list = data
        rgb_data = rgb_data.to(device)
        depth_data = depth_data.to(device)
        lidar_data = lidar_data.to(device)
        mmwave_data = mmwave_data.to(device)
        wifi_data = wifi_data.to(device)
        kpts.to(device)
        labels = kpts.type(torch.FloatTensor)
        rgb_model.to('cuda')
        rgb_outputs = rgb_model(rgb_data, [True, False, False, False, False])
        rgb_model.to('cpu')
        del rgb_data
        depth_model.to('cuda')
        depth_outputs = depth_model(depth_data, [False, True, False, False, False])
        depth_model.to('cpu')
        del depth_data
        mmwave_model.to('cuda')
        mmwave_outputs = mmwave_model(mmwave_data, [False, False, True, False, False])
        mmwave_model.to('cpu')
        del mmwave_data
        lidar_model.to('cuda')
        lidar_outputs = lidar_model(lidar_data, [False, False, False, True, False])
        lidar_model.to('cpu')
        del lidar_data
        wifi_model.to('cuda')
        wifi_outputs = wifi_model(wifi_data, [False, False, False, False, True])
        wifi_model.to('cpu')
        del wifi_data
        
        rgb_outputs = rgb_outputs.detach().cpu().numpy()
        depth_outputs = depth_outputs.detach().cpu().numpy()
        mmwave_outputs = mmwave_outputs.detach().cpu().numpy()
        lidar_outputs = lidar_outputs.detach().cpu().numpy()
        wifi_outputs = wifi_outputs.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        if i == 0:
            rgb_result = rgb_outputs
            depth_result = depth_outputs
            mmwave_result = mmwave_outputs
            lidar_result = lidar_outputs
            wifi_result = wifi_outputs
            all_label = labels
        else:
            rgb_result = np.vstack((rgb_result, rgb_outputs))
            depth_result = np.vstack((depth_result, depth_outputs))
            mmwave_result = np.vstack((mmwave_result, mmwave_outputs))
            lidar_result = np.vstack((lidar_result, lidar_outputs))
            wifi_result = np.vstack((wifi_result, wifi_outputs))
            all_label = np.vstack((all_label, labels))
    np.save('./baseline_results/rgb_result.npy', rgb_result)
    np.save('./baseline_results/depth_result.npy', depth_result)
    np.save('./baseline_results/mmwave_result.npy', mmwave_result)
    np.save('./baseline_results/lidar_result.npy', lidar_result)
    np.save('./baseline_results/wifi_result.npy', wifi_result)
    np.save('./baseline_results/all_label.npy', all_label)
    return



dataset_root = 'd:\Data\My_MMFi_Data\MMFi_Dataset'
with open('config.yaml', 'r') as fd:
    config = yaml.load(fd, Loader=yaml.FullLoader)

train_dataset, val_dataset = make_dataset(dataset_root, config)

rng_generator = torch.manual_seed(config['init_rand_seed'])
train_loader = make_dataloader(train_dataset, is_training=True, generator=rng_generator, **config['loader'], collate_fn = collate_fn_padd)
val_loader = make_dataloader(val_dataset, is_training=False, generator=rng_generator, **config['loader'], collate_fn = collate_fn_padd)

rgb_model = single_model(['rgb'])
rgb_model.load_state_dict(torch.load('./baseline_weights/RGB.pt'))
rgb_model.cuda()

depth_model = single_model(['depth'])
depth_model.load_state_dict(torch.load('./baseline_weights/Depth.pt'))
depth_model.cuda()

mmwave_model = single_model(['mmwave'])
mmwave_model.load_state_dict(torch.load('./baseline_weights/mmWave.pt'))
mmwave_model.cuda()

lidar_model = single_model(['lidar'])
lidar_model.load_state_dict(torch.load('./baseline_weights/Lidar.pt'))
lidar_model.cuda()

wifi_model = single_model(['wifi-csi'])
wifi_model.load_state_dict(torch.load('./baseline_weights/Wifi.pt'))
wifi_model.cuda()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
get_result(rgb_model,depth_model,mmwave_model,lidar_model,wifi_model, val_loader, device)