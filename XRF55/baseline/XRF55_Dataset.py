import logging
import torch
import csv
import os
import glob
from torch.utils.data.dataset import Dataset, IterableDataset
import pandas as pd
import numpy as np
import h5py

class XRF55_Datase(Dataset):
    def __init__(self, root_dir, scene, is_train):
        """
        Args:
            root_dir (string): Directory with all the data.
            scene (string): scene name, opts: "all", list of strings.
            is_train (bool): train or test.
        """
        super(XRF55_Datase, self).__init__()
        if scene == "all":
            scene = ["Scene1", "Scene2", "Scene3", "Scene4"]
        else:
            scene = scene
        self.root_dir = root_dir
        # self.scene = scene
        # self.is_train = is_train
        if is_train:
            self.path = os.path.join(self.root_dir, 'train_data')
        else:
            self.path = os.path.join(self.root_dir, 'test_data')
        self.RFID_name_list = []
        for scene_sub in scene:
            sub_list = glob.glob(os.path.join(self.path, 'RFID', scene_sub, scene_sub, '*.npy'))
            sub_list.sort()
            self.RFID_name_list += sub_list

    def __len__(self):
        return len(self.RFID_name_list)

    def __getitem__(self, idx):
        RFID_file_name = self.RFID_name_list[idx]
        WIFI_file_name = RFID_file_name.replace('RFID', 'WiFi')
        mmWave_file_name = RFID_file_name.replace('RFID', 'mmWave')
        wifi_data = self.load_wifi(WIFI_file_name)
        rfid_data = self.load_rfid(RFID_file_name)
        mmwave_data = self.load_mmwave(mmWave_file_name).reshape(17, 256, 128)
        label = int(os.path.basename(os.path.normpath(RFID_file_name)).split('_')[1]) - 1
        return wifi_data, rfid_data, mmwave_data, label

    def load_rfid(self, filename):
        rfid_data = np.load(filename)
        # return torch.from_numpy(rfid_data).float()
        return rfid_data


    def load_wifi(self,filename):
        wifi_data = np.load(filename)
        # return torch.from_numpy(wifi_data).float()
        return wifi_data


    def load_mmwave(self, filename):
        mmWave_data = np.load(filename)
        # return torch.from_numpy(mmWave_data).float()
        return mmWave_data