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
import time
import re
from XRF55_Dataset import XRF55_Datase 
from baseline_model import single_model


def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    dict_keys(['modality', 'scene', 'subject', 'action', 'idx', 'output', 
    'input_rgb', 'input_depth', 'input_lidar', 'input_mmwave'])
    '''
    labels = []
    [labels.append(float(t[3])) for t in batch]
    labels = torch.FloatTensor(labels)

    # mmwave
    mmwave_data = np.array([(t[2]) for t in batch ])
    mmwave_data = torch.FloatTensor(mmwave_data)

    # wifi-csi
    wifi_data = np.array([(t[0]) for t in batch ])
    wifi_data = torch.FloatTensor(wifi_data)

    # rfid
    rfid_data = np.array([(t[1]) for t in batch ])
    rfid_data = torch.FloatTensor(rfid_data)
    
    modality_list = [True, True, True]

    return mmwave_data, wifi_data, rfid_data, labels, modality_list

def get_result(mmwave_model,wifi_model,rfid_model, tensor_loader, device):
    mmwave_model.eval()
    wifi_model.eval()
    rfid_model.eval()
    # test_mpjpe = 0
    # test_pampjpe = 0
    # test_mse = 0
    # random.seed(config['modality_existances']['val_random_seed'])
    for i, data in tqdm(enumerate(tensor_loader)):
        # start_time = time.time()
        mmwave_data, wifi_data, rfid_data, label, exist_list = data
        mmwave_data = mmwave_data.to(device)
        wifi_data = wifi_data.to(device)
        rfid_data = rfid_data.to(device)
        label.to(device)
        labels = label.type(torch.FloatTensor)
        mmwave_model.to(device)
        mmwave_outputs = mmwave_model(mmwave_data, [True, False, False])
        mmwave_model.to('cpu')
        del mmwave_data
        wifi_model.to(device)
        wifi_outputs = wifi_model(wifi_data, [False, True, False])
        wifi_model.to('cpu')
        del wifi_data
        rfid_model.to(device)
        rfid_outputs = rfid_model(rfid_data, [False, False, True])
        rfid_model.to('cpu')
        del rfid_data
        
        # rgb_outputs = rgb_outputs.to(device)
        mmwave_outputs = mmwave_outputs.detach().cpu().numpy()
        wifi_outputs = wifi_outputs.detach().cpu().numpy()
        rfid_outputs = rfid_outputs.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        
        if i == 0:
            mmwave_result = mmwave_outputs
            wifi_result = wifi_outputs
            rfid_result = rfid_outputs
            all_label = labels
        else:
            mmwave_result = np.vstack((mmwave_result, mmwave_outputs))
            wifi_result = np.vstack((wifi_result, wifi_outputs))
            rfid_result = np.vstack((rfid_result, rfid_outputs))
            all_label = np.hstack((all_label, labels))

    np.save('./baseline_results/mmwave_result.npy', mmwave_result)
    np.save('./baseline_results/wifi_result.npy', wifi_result)
    np.save('./baseline_results/rfid_result.npy', rfid_result)
    np.save('./baseline_results/all_label.npy', all_label)

    return

train_dataset = XRF55_Datase(root_dir="D:\Data\XRF55\XRF_dataset", scene='all', is_train=True)
test_dataset = XRF55_Datase(root_dir="D:\Data\XRF55\XRF_dataset", scene='all', is_train=False)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn_padd)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn_padd)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mmwave_model = single_model(['mmwave'])
mmwave_model.load_state_dict(torch.load('./baseline_weights/mmwave.pt'))
mmwave_model.to(device)

wifi_model = single_model(['wifi'])
wifi_model.load_state_dict(torch.load('./baseline_weights/wifi.pt'))
wifi_model.to(device)

rfid_model = single_model(['rfid'])
rfid_model.load_state_dict(torch.load('./baseline_weights/rfid.pt'))
rfid_model.to(device)

get_result(mmwave_model,wifi_model,rfid_model, test_dataloader, device)