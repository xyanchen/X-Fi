import torch
import yaml
import glob
import scipy.io as scio
import random
import numpy as np
import os
from mmfi import make_dataset, make_dataloader#, Csi_dataset
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


class EarlyStopping:

    def __init__(self, patience=7, verbose=False, delta=0, path='weight7-stop.pth', trace_func=print):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), self.path)
        torch.save(model, self.path)
        self.val_loss_min = val_loss

if __name__ == '__main__':
    dataset_root = '/data3/MMFi_Dataset'
    with open('config.yaml', 'r') as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)

    train_dataset, val_dataset = make_dataset(dataset_root, config)

    rng_generator = torch.manual_seed(config['init_rand_seed'])
    train_loader = make_dataloader(train_dataset, is_training=True, generator=rng_generator, **config['train_loader'])
    val_loader = make_dataloader(val_dataset, is_training=False, generator=rng_generator, **config['validation_loader'])
    wisppn = posenet()
    # wisppn.apply(weights_init)
    wisppn = wisppn.cuda()
    criterion_L2 = nn.MSELoss().cuda()
    optimizer = torch.optim.SGD(wisppn.parameters(), lr=0.05, momentum=0.9)


    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=0.5)
    num_epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pck_50_overall_max = 0
    earlystopping = EarlyStopping(patience=5, delta=(-0.005), path='weights/wisppn_' + config['protocol'] + '_' + config['split_to_use'] + '.pkl')
    mpjpe_overall_min = 100
    # TODO: Code for training or validation
    # sample = train_dataset[0]
    for epoch_index in range(num_epochs):
        loss = 0
        train_loss_iter = []
        wisppn.train()
        for idx, data in enumerate(train_loader):
            csi_data = data['input_wifi-csi'].cuda().unsqueeze(1)
            csi_data = csi_data.type(torch.cuda.FloatTensor)
            xy_keypoint = data['output'].cuda()

            pred_xy = wisppn(csi_data)  # b,2,17,4


            loss = criterion_L2(pred_xy_keypoint, xy_keypoint) / batchsize
            train_loss_iter.append(loss.cpu().detach().numpy())

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            lr = optimizer.state_dict()['param_groups'][0]['lr']
            message = '(epoch: %d, iters: %d, lr: %.5f, loss: %.3f) ' % (epoch_index, idx, lr, loss)
            print(message)
        scheduler.step()
        train_mean_loss = np.mean(train_loss_iter)
        print('end of the epoch: %d, with loss: %.3f' % (epoch_index, train_mean_loss,))
        # earlystopping(train_mean_loss, wisppn)
        # if earlystopping.early_stop:
        #     print('early stop')
        #     break
        current_num_to_test =+ 1
        if (epoch_index % 5 == 0) & (epoch_index > 19):

            wisppn.eval()
            valid_loss_iter = []
            mpjpe_iter = []
            pampjpe_iter = []
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    csi_data = data['input_wifi-csi'].cuda().unsqueeze(1)
                    csi_data = csi_data.type(torch.cuda.FloatTensor)
                    xy_keypoint = data['output'].cuda()#.squeeze()


                    # pred_keypoint = torch.tensor(np.zeros([1,2,17]))
                    pred_xy = wisppn(csi_data)  # 4,2,17,17
                    # for m in range(1):
                    #     for n in range(2):
                    #         a = pred_xy[m,n,:,:]
                    #         pred_keypoint[m,n,:] = torch.diag(a)
                    # pred_keypoint = torch.diag(pred_xy)
                    m = torch.nn.AvgPool2d((1, 4))
                    pred_xy_keypoint = m(pred_xy).squeeze(3)
                    pred_xy_keypoint = torch.transpose(pred_xy_keypoint, 1, 2)

                    loss = criterion_L2(pred_xy_keypoint, xy_keypoint)

                    # loss = criterion_L2(torch.mul(confidence, pred_xy), torch.mul(confidence, xy))
                    valid_loss_iter.append(loss.cpu().detach().numpy())
                    # keypoint = torch.transpose(keypoint,1,2)
                    # keypoint = keypoint[:,0:2,:]

                    # pred_xy_keypoint = torch.transpose(pred_xy_keypoint, 0, 1)
                    # xy_keypoint = torch.transpose(xy_keypoint, 0, 1)
                    pred_xy_keypoint = pred_xy_keypoint.cpu().detach().numpy()
                    xy_keypoint = xy_keypoint.cpu().detach().numpy()
                    # pck = compute_pck_pckh(pred_xy_keypoint, xy_keypoint, 0.5)
                    #
                    # pck_50_iter.append(compute_pck_pckh(pred_xy_keypoint, xy_keypoint, 0.5))
                    # pck_20_iter.append(compute_pck_pckh(pred_xy_keypoint, xy_keypoint, 0.2))
                    mpjpe, pampjpe = calulate_error(pred_xy_keypoint, xy_keypoint)
                    mpjpe_iter.append(mpjpe)
                    pampjpe_iter.append(pampjpe)

                valid_mean_loss = np.mean(valid_loss_iter)
                mpjpe_overall = np.mean(mpjpe_iter, 0)
                pampjpe_overall = np.mean(pampjpe_iter, 0)
                # mpjpe_overall = mpjpe[17]
                # pampjpe_overall = pampjpe[17]
                print('validation result with loss: %.3f, mpjpe: %.3f, pampjpe: %.3f' % (
                valid_mean_loss, mpjpe_overall, pampjpe_overall))

                if mpjpe_overall < mpjpe_overall_min:
                    print('saving the model at the end of epoch %d with mpjpe: %.3f' % (epoch_index, mpjpe_overall))
                    torch.save(wisppn, 'weights/wisppn_100_lr25_' + config['protocol'] + '_' + config['split_to_use'] + str(int(mpjpe_overall*1000)) + '_experiment1.pkl')
                    mpjpe_overall_min = mpjpe_overall
                    pampjpe_overall_final = pampjpe_overall
                    best_idx = epoch_index


    print('save the best model at epoch: %d, with the min mpjpe is : %.6f, and corresponding pampjpe is : %.6f' % (best_idx, mpjpe_overall_min, pampjpe_overall_final))





