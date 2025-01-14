import os
import shutil

import torch
import torch.utils.data as Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
import csv
import time


def split_train_test_new(root_ns="./dataset/Raw_dataset/", dst_wr="./dataset/XRF_dataset/", zoo=14):

    dst_train_rfid = os.path.join(dst_wr, "train_data\RFID")
    dst_train_wifi = os.path.join(dst_wr, "train_data\WiFi")
    dst_train_mmwave = os.path.join(dst_wr, "train_data\mmWave")
    dst_test_rfid = os.path.join(dst_wr, "test_data\RFID")
    dst_test_wifi = os.path.join(dst_wr, "test_data\WiFi")
    dst_test_mmwave = os.path.join(dst_wr, "test_data\mmWave")

    if not os.path.exists(dst_train_rfid):
        os.makedirs(dst_train_rfid)
    if not os.path.exists(dst_train_wifi):
        os.makedirs(dst_train_wifi)
    if not os.path.exists(dst_train_mmwave):
        os.makedirs(dst_train_mmwave)
    if not os.path.exists(dst_test_rfid):
        os.makedirs(dst_test_rfid)
    if not os.path.exists(dst_test_wifi):
        os.makedirs(dst_test_wifi)
    if not os.path.exists(dst_test_mmwave):
        os.makedirs(dst_test_mmwave)

    for scene in os.listdir(root_ns):
        for scene_sub in os.listdir(os.path.join(root_ns, scene)):
            temp_rfid_train = os.path.join(dst_train_rfid, scene, scene_sub)
            if not os.path.exists(temp_rfid_train):
                os.makedirs(temp_rfid_train)
            temp_wifi_train = os.path.join(dst_train_wifi, scene, scene_sub)
            if not os.path.exists(temp_wifi_train):
                os.makedirs(temp_wifi_train)
            temp_mmwave_train = os.path.join(dst_train_mmwave, scene, scene_sub)
            if not os.path.exists(temp_mmwave_train):
                os.makedirs(temp_mmwave_train)
            temp_rfid_test = os.path.join(dst_test_rfid, scene, scene_sub)
            if not os.path.exists(temp_rfid_test):
                os.makedirs(temp_rfid_test)
            temp_wifi_test = os.path.join(dst_test_wifi, scene, scene_sub)
            if not os.path.exists(temp_wifi_test):
                os.makedirs(temp_wifi_test)
            temp_mmwave_test = os.path.join(dst_test_mmwave, scene, scene_sub)
            if not os.path.exists(temp_mmwave_test):
                os.makedirs(temp_mmwave_test)
            
            for file in tqdm(os.listdir(os.path.join(root_ns, scene, scene_sub, 'RFID'))):
                filename = file.split(".")[0]  # act name
                fileidx = filename.split("_")[0]  # actor idx
                actidx = int(filename.split("_")[2])
                rfid_root = os.path.join(root_ns, scene, scene_sub, 'RFID')
                wifi_root = os.path.join(root_ns, scene, scene_sub, 'WiFi')
                mmwave_root = os.path.join(root_ns, scene, scene_sub, 'mmWave')
                if actidx <= zoo: # out of 20 samples of each action for each person, the first "zoo" are selected as the training set and the rest as the test set.
                    shutil.copy(os.path.join(rfid_root,file), os.path.join(temp_rfid_train,file))
                    shutil.copy(os.path.join(wifi_root,file),os.path.join(temp_wifi_train,file))
                    shutil.copy(os.path.join(mmwave_root,file),os.path.join(temp_mmwave_train,file))
                else:
                    shutil.copy(os.path.join(rfid_root,file), os.path.join(temp_rfid_test,file))
                    shutil.copy(os.path.join(wifi_root,file),os.path.join(temp_wifi_test,file))
                    shutil.copy(os.path.join(mmwave_root,file),os.path.join(temp_mmwave_test,file))

if __name__ == '__main__':

    split_train_test_new(root_ns="D:\Data\XRF55\Raw_dataset", dst_wr="D:\Data\XRF55\XRF_dataset", zoo=14)