import torch
import yaml
import glob
import scipy.io as scio
import random
import numpy as np
import os
from mmfi import make_dataset, make_dataloader, Csi_dataset
from torch.utils.data import Dataset, DataLoader
import csv
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import os.path
from models.mynetwork import posenet, weights_init
import torch.nn as nn
from torch.autograd import Variable
from evaluation import compute_pck_pckh
from models.wisppn_resnet import ResNet, ResidualBlock, Bottleneck
from mmfi_evaluation import calulate_error

batchsize = 32

def process_csi(csi):
    csi_data = csi.numpy()
    csi_data[np.isinf(csi_data)] = np.nan
    for i in range(10):  # 32
        temp_col = csi_data[:, :, :, i]
        nan_num = np.count_nonzero(temp_col != temp_col)
        if nan_num != 0:
            temp_not_nan_col = temp_col[temp_col == temp_col]
            temp_col[np.isnan(temp_col)] = temp_not_nan_col.mean()
    csi_data = torch.tensor((csi_data - np.min(csi_data)) / (np.max(csi_data) - np.min(csi_data)))
    # isnan = torch.isinf(csi_data).any()
    # if isnan == True:
    #     print(isnan)
    return csi_data

if __name__ == '__main__':
    dataset_root = '/data3/MMFi_Dataset'
    with open('config_vali.yaml', 'r') as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)

    train_dataset, val_dataset = make_dataset(dataset_root, config)

    rng_generator = torch.manual_seed(config['init_rand_seed'])
    # train_loader = make_dataloader(train_dataset, is_training=True, generator=rng_generator, **config['train_loader'])
    val_loader = make_dataloader(val_dataset, is_training=False, generator=rng_generator, **config['validation_loader'])
    net_dir = '/data3/weights_train/wisppn_15_lr0.0007_tanh_protocol3_cross_scene_split369_experiment3.pkl'
    wisppn = torch.load(net_dir)
    wisppn = wisppn.cuda().eval()
    criterion_L2 = nn.MSELoss().cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, file = os.path.split(net_dir)
    save_root = '/data3/MMFi_Evaluation'


    # TODO: Code for training or validation
    # sample = train_dataset[0]

    wisppn.eval()
    valid_loss_iter = []
    mpjpe_iter = []
    pampjpe_iter = []
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            # csi = data['input_wifi-csi'].flatten(0, 1)
            # output = data['output'].flatten(0, 1)
            # csi_list = list(csi.chunk(csi.shape[0], dim=0))
            # output_list = list(output.chunk(output.shape[0], dim=0))
            # small_dataset = Csi_dataset(csi_list, output_list)
            # dataloader_train = DataLoader(small_dataset, batch_size=1, shuffle=False)
            # for idx_small, data_frame in enumerate(dataloader_train):
            #     csi_data = data_frame['csi_data']
            #
            #     # xy_keypoint = data['keypoint'].unsqueeze(0)
            #     csi_data = csi_data.cuda()
            #     csi_data = csi_data.type(torch.cuda.FloatTensor)
            #     # confidence = label[:, 2:3, :, :]
            #     xy_keypoint = data_frame['keypoint'][:, :, 0:2].cuda().squeeze()
            #     confidence = data_frame['keypoint'][:, :, 0:2].cuda()
            csi = data['input_wifi-csi'].flatten(0, 1)
            output = data['output'].flatten(0, 1)
            csi_list = list(csi.chunk(csi.shape[0], dim=0))
            output_list = list(output.chunk(output.shape[0], dim=0))
            pred_xy_keypoint_list = []
            for item in range(297):
                csi_data = csi_list[item]
                csi_data = process_csi(csi_data)
                csi_data = csi_data.unsqueeze(0).cuda()
                csi_data = csi_data.type(torch.cuda.FloatTensor)
                xy_keypoint = output_list[item].cuda()
                pred_xy = wisppn(csi_data)  # 4,2,17,17
                # for m in range(1):
                #     for n in range(2):
                #         a = pred_xy[m,n,:,:]
                #         pred_keypoint[m,n,:] = torch.diag(a)
                # pred_keypoint = torch.diag(pred_xy)
                # m = torch.nn.AvgPool2d((1, 4))
                # pred_xy_keypoint = m(pred_xy).squeeze(3)
                # pred_xy_keypoint = torch.transpose(pred_xy_keypoint, 1, 2)

                loss = criterion_L2(pred_xy, xy_keypoint)

                # loss = criterion_L2(torch.mul(confidence, pred_xy), torch.mul(confidence, xy))
                valid_loss_iter.append(loss.cpu().detach().numpy())
                # keypoint = torch.transpose(keypoint,1,2)
                # keypoint = keypoint[:,0:2,:]

                # pred_xy_keypoint = torch.transpose(pred_xy_keypoint, 0, 1)
                # xy_keypoint = torch.transpose(xy_keypoint, 0, 1)
                pred_xy = pred_xy.cpu().detach().numpy()
                xy_keypoint = xy_keypoint.cpu().detach().numpy()
                # pck = compute_pck_pckh(pred_xy_keypoint, xy_keypoint, 0.5)
                #
                # pck_50_iter.append(compute_pck_pckh(pred_xy_keypoint, xy_keypoint, 0.5))
                # pck_20_iter.append(compute_pck_pckh(pred_xy_keypoint, xy_keypoint, 0.2))
                mpjpe, pampjpe = calulate_error(pred_xy, xy_keypoint)
                mpjpe_iter.append(mpjpe)
                pampjpe_iter.append(pampjpe)
                pred_xy_keypoint = pred_xy.squeeze()
                pred_xy_keypoint_list.append(pred_xy_keypoint)

            # np.savetxt(os.path.join(save_root, temp[7].split('.')[0] + '.csv'), xy_keypoint)

            npz = np.array(pred_xy_keypoint_list)
            save_dir = os.path.join(save_root, config['protocol'], config['split_to_use'], config['modality'], 'round3')
            os.makedirs(save_dir, exist_ok=True)
            sample_name = data['subject'][0] + '_' + data['action'][0] + '.npz'
            np.savez(os.path.join(save_dir, sample_name), joints=npz)
            print('result of {} has been saved'.format(sample_name))
        valid_mean_loss = np.mean(valid_loss_iter)
        mpjpe_overall = np.mean(mpjpe_iter, 0)
        pampjpe_overall = np.mean(pampjpe_iter, 0)
        # mpjpe_overall = mpjpe[17]
        # pampjpe_overall = pampjpe[17]
        print('validation result with loss: %.6f, mpjpe: %.6f, pampjpe: %.6f' % (
        valid_mean_loss, mpjpe_overall, pampjpe_overall))






