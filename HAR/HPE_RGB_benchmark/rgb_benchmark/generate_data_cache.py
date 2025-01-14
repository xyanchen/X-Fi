import os
import os.path
import sys
import argparse
import time
import yaml
import numpy as np
import torch

from mmfi_lib.mmfi import make_dataset, make_dataloader


def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2

    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X / w * 2 - torch.from_numpy(np.array([1, h / w]))


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Code Implementation for VideoPose3D on MMFi-RGB Dataset")
    # parser.add_argument("protocol_setting", type=str, help="Protocol and setting for this trial")
    # args = parser.parse_args()

    setting_list = ["p1s1", "p1s2", "p1s3", "p2s1", "p2s2", "p2s3", "p3s1", "p3s2", "p3s3"]
    # setting_list = ["p3s1", "p3s2", "p3s3"]

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    dataset_root = "/mnt/mmfi_dataset"

    checkpoint_frequency = 10

    for protocol_setting in setting_list:
        # protocol_setting = "p1s3"
        # protocol_setting = args.protocol_setting

        with open("configs/" + protocol_setting + ".yaml", 'r') as fd:
            config = yaml.load(fd, Loader=yaml.FullLoader)

        train_dataset, val_dataset = make_dataset(dataset_root, config)

        rng_generator = torch.manual_seed(config['init_rand_seed'])
        train_loader = make_dataloader(train_dataset, is_training=True, generator=rng_generator, **config['train_loader'])
        val_loader = make_dataloader(val_dataset, is_training=False, generator=rng_generator, **config['validation_loader'])

        poses_3d_in = []
        poses_2d_in = []

        if os.path.isfile(os.path.join("./data_cache", protocol_setting, "train", "poses_2d_in.pt")) and \
                os.path.isfile(os.path.join("./data_cache", protocol_setting, "train", "poses_3d_in.pt")):
            poses_2d_in = torch.load(os.path.join("./data_cache", protocol_setting, "train", "poses_2d_in.pt"))
            poses_3d_in = torch.load(os.path.join("./data_cache", protocol_setting, "train", "poses_3d_in.pt"))
            print("Train data has been loaded.")
        else:
            for idx, data in enumerate(train_loader):
                poses_2d_in = normalize_screen_coordinates(data['input_rgb'], w=640, h=480)
                poses_3d_in = data['output']
            os.makedirs(os.path.join("./data_cache", protocol_setting, "train"), exist_ok=True)
            torch.save(poses_2d_in, os.path.join("./data_cache", protocol_setting, "train", "poses_2d_in.pt"))
            torch.save(poses_3d_in, os.path.join("./data_cache", protocol_setting, "train", "poses_3d_in.pt"))
            print("Saved the data cache into " + os.path.join("./data_cache", protocol_setting, "train"))

        if os.path.exists(os.path.join("./data_cache", protocol_setting, "test", "poses_2d_in.pt")) and \
                os.path.exists(os.path.join("./data_cache", protocol_setting, "test", "poses_3d_in.pt")):
            poses_2d_in = torch.load(os.path.join("./data_cache", protocol_setting, "test", "poses_2d_in.pt"))
            poses_3d_in = torch.load(os.path.join("./data_cache", protocol_setting, "test", "poses_3d_in.pt"))
            print("Test data has been loaded.")
        else:
            for idx, data in enumerate(val_loader):
                poses_2d_in = normalize_screen_coordinates(data['input_rgb'], w=640, h=480)
                poses_3d_in = data['output']
            os.makedirs(os.path.join("./data_cache", protocol_setting, "test"), exist_ok=True)
            torch.save(poses_2d_in, os.path.join("./data_cache", protocol_setting, "test", "poses_2d_in.pt"))
            torch.save(poses_3d_in, os.path.join("./data_cache", protocol_setting, "test", "poses_3d_in.pt"))
            print("Saved the data cache into " + os.path.join("./data_cache", protocol_setting, "test"))




