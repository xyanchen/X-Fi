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
from baseline_model import single_model, dual_model, triple_model, quadra_model

def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.

    dict_keys(['modality', 'scene', 'subject', 'action', 'idx', 'output', 
    'input_rgb', 'input_depth', 'input_lidar', 'input_mmwave'])
    '''
    ## get sequence lengths
    for t in batch:
        dict_keys = t.keys()
    #     print(a)
    # #     # print(t[0].shape,t[1].shape)
    all_actions = {'A01': 0., 'A02': 1., 'A03': 2., 'A04': 3., 'A05': 4., 
                   'A06': 5., 'A07': 6., 'A08': 7., 'A09': 8., 'A10': 9.,
                   'A11': 10., 'A12': 11., 'A13': 12., 'A14': 13., 'A15': 14., 
                   'A16': 15., 'A17': 16., 'A18': 17., 'A19': 18., 'A20': 19., 
                   'A21': 20., 'A22': 21., 'A23': 22., 'A24': 23., 'A25': 24., 
                   'A26': 25., 'A27': 26.}
    labels = []
    [labels.append(all_actions[t['action']]) for t in batch]
    labels = torch.FloatTensor(labels)

    # lengths = torch.tensor([t['input_mmwave'].shape[0] for t in batch ])

    return_data = []
    exist_list = []
    # rgb
    if 'input_rgb' in dict_keys:
        rgb_data = np.array([(t['input_rgb']) for t in batch ])
        rgb_data = torch.FloatTensor(rgb_data).permute(0,3,1,2)
        return_data.append(rgb_data)
        exist_list.append(True)
    else:
        exist_list.append(False)

    # depth
    if 'input_depth' in dict_keys:
        depth_data = np.array([(t['input_depth']) for t in batch ])
        depth_data = torch.FloatTensor(depth_data).permute(0,3,1,2)
        return_data.append(depth_data)
        exist_list.append(True)
    else:
        exist_list.append(False)

    # mmwave
    ## padd
    if 'input_mmwave' in dict_keys:
        mmwave_data = [torch.Tensor(t['input_mmwave']) for t in batch ]
        mmwave_data = torch.nn.utils.rnn.pad_sequence(mmwave_data)
        ## compute mask
        mmwave_data = mmwave_data.permute(1,0,2)
        return_data.append(mmwave_data)
        exist_list.append(True)
    else:
        exist_list.append(False)

    # lidar
    ## padd
    if 'input_lidar' in dict_keys:
        lidar_data = [torch.Tensor(t['input_lidar']) for t in batch ]
        lidar_data = torch.nn.utils.rnn.pad_sequence(lidar_data)
        ## compute mask
        lidar_data = lidar_data.permute(1,0,2)
        return_data.append(lidar_data)
        exist_list.append(True)
    else:
        exist_list.append(False)

    # # wifi-csi
    # if 'input_wifi-csi' in dict_keys:
    #     wifi_data = np.array([(t['input_wifi-csi']) for t in batch ])
    #     wifi_data = torch.FloatTensor(wifi_data)
    #     return_data.append(wifi_data)
    #     exist_list.append(True)
    # else:
    #     exist_list.append(False)
    "要改"
    # exist_list = [False, True, False, False, True]
    input_1 = return_data[0]
    input_2 = return_data[1]
    input_3 = return_data[2]
    input_4 = return_data[3]
    # input_5 = return_data[4]

    # return rgb_data, depth_data, lidar_data, mmwave_data, wifi_data, kpts, modality_list
    return input_1, input_2, input_3, input_4, labels, exist_list
    # return input_1, labels, exist_list

def test(model, tensor_loader, criterion, device):
    model.eval()
    test_acc = 0
    test_loss = 0
    # random.seed(config['modality_existances']['val_random_seed'])
    for data in tqdm(tensor_loader):
        # start_time = time.time()
        input_1, input_2, input_3, input_4, label, exist_list = data
        # input_1, label, exist_list = data
        input_1 = input_1.to(device)
        input_2 = input_2.to(device)
        input_3 = input_3.to(device)
        input_4 = input_4.to(device)
        # input_5 = input_5.to(device)
        label.to(device)
        labels = label.type(torch.LongTensor)
        # outputs = model(input_1, input_2,input_3, exist_list)
        outputs = model(input_1, input_2, input_3, input_4, exist_list)
        outputs = outputs.type(torch.FloatTensor)
        outputs.to(device)
        # t2 = time.time()
        # forward_time = t2 - t1
        loss = criterion(outputs,labels)
        predict_y = torch.argmax(outputs,dim=1).to(device)
        accuracy = (predict_y == labels.to(device)).sum().item()/labels.size(0)
        test_acc += accuracy
        test_loss += loss.item() * labels.size(0)

        outputs = outputs.detach().cpu().numpy()
        labels = labels.cpu().numpy()
        # t3 = time.time()
        # record_time = t3 - t2
        # print('load_time: ', load_time)
        # print('forward_time: ', forward_time)
        # print('record_time: ', record_time)
    test_acc = test_acc/len(tensor_loader)
    test_loss = test_loss/len(tensor_loader.dataset)
    print("validation accuracy:{:.4f}, loss:{:.5f}".format(float(test_acc),float(test_loss)))
    return test_acc

def train(model, train_loader, test_loader, num_epochs, learning_rate, criterion, device, modality):
    optimizer = torch.optim.AdamW(model.parameters(),
        lr = learning_rate
    )
    # optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[20,40],gamma=0.1)
    "要改"
    name = ''
    for mod in modality:
        name = name + mod + '_'
    name = name + '.pt'
    parameter_dir = './HAR_baseline_weights/Quadra/' + name
    best_test_acc = 0.
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        # random.seed(epoch)
        num_iter = 400
        for i, data in enumerate(tqdm(train_loader)):
            if i < num_iter:
                input_1, input_2, input_3, input_4, label, exist_list = data
                # input_1, label, exist_list = data
                input_1 = input_1.to(device)
                input_2 = input_2.to(device)
                input_3 = input_3.to(device)
                input_4 = input_4.to(device)
                # input_5 = input_5.to(device)
                labels = label.to(device)
                labels = labels.type(torch.LongTensor)
                
                optimizer.zero_grad()
                outputs = model(input_1, input_2, input_3, input_4, exist_list)
                # outputs = model(input_1, exist_list)
                # print(outputs)
                outputs = outputs.to(device)
                outputs = outputs.type(torch.FloatTensor)
                loss = criterion(outputs,labels)
                if loss == float('nan'):
                    print('nan')
                    print(outputs)
                    print(labels)
                    
                loss.backward()
                # print(length)
                # print("loss is ", loss.item())
                optimizer.step()
                
                epoch_loss += loss.item()
                predict_y = torch.argmax(outputs,dim=1).to(device)
                epoch_accuracy += (predict_y == labels.to(device)).sum().item() / labels.size(0)
            else:
                break
            # print("epoch loss is ", epoch_loss)
        # epoch_loss = epoch_loss/len(train_loader.dataset)
        "要改"
        epoch_loss = epoch_loss/num_iter
        epoch_accuracy = epoch_accuracy/num_iter
        print('Epoch:{}, Accuracy:{:.4f},Loss:{:.9f}'.format(epoch+1, float(epoch_accuracy),float(epoch_loss)))
        if (epoch+1) % 5 == 0:
            test_acc = test(
                model=model,
                tensor_loader=test_loader,
                criterion = criterion,
                device= device
            )
            # if test_mpjpe <= best_test_mpjpe:
            #     print(f"best test mpjpe is:{test_mpjpe}")
            #     best_test_mpjpe = test_mpjpe
            #     torch.save(model.state_dict(), parameter_dir)
        # scheduler.step()
    torch.save(model.state_dict(), parameter_dir)
    return

dataset_root = 'd:\Data\My_MMFi_Data\MMFi_Dataset'
with open('config_all.yaml', 'r') as fd:
    config = yaml.load(fd, Loader=yaml.FullLoader)

for i in range(len(config['modality_list'])):
    config['modality'] = config['modality_list'][i]
    train_dataset, val_dataset = make_dataset(dataset_root, config)



    rng_generator = torch.manual_seed(config['init_rand_seed'])
    train_loader = make_dataloader(train_dataset, is_training=True, generator=rng_generator, **config['loader'], collate_fn = collate_fn_padd)
    val_loader = make_dataloader(val_dataset, is_training=False, generator=rng_generator, **config['loader'], collate_fn = collate_fn_padd)

    avg_time = 0
    for epoch in range(3):
        # random.seed(config['modality_existances']['train_random_seed'])
        i= 0
        for data in val_loader:
            # start_time = time.time()
            input_1, input_2, input_3, input_4, label, exist_list = data
            # input_1, label, exist_list = data
            # epoch_time = time.time() - start_time
            # print(rgb_data[0].shape, depth_data[0].shape, lidar_data[0].shape, mmwave_data[0].shape, wifi_data[0].shape,kpts.shape, lengths.shape)
            print(input_1.shape, input_2.shape, input_3.shape, input_4.shape, label.shape)
            # print(input_1.shape, label.shape)
            # print('epoch_time: ', epoch_time)
            # avg_time += epoch_time
            # print(rgb_data, depth_data, lidar_data, mmwave_data, wifi_data, kpts, modality_list)
            print(exist_list)
            i += 1
            if i > 1:
                print('............................................................................................')
                break



    # model = model(['RGB'])
    if len(config['modality']) == 1:
        model = single_model(config['modality'])
    elif len(config['modality']) == 2:
        model = dual_model(config['modality'])
    elif len(config['modality']) == 3:
        model = triple_model(config['modality'])
    elif len(config['modality']) == 4:
        model = quadra_model(config['modality'])
    # elif len(config['modality']) == 5:
    #     model = Five_model(config['modality'])
    model.cuda()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device) 

    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # name = ''
    # for mod in config['modality']:
    #     name = name + mod + '_'
    # name = name + '.pt'
    # parameter_dir = './baseline_weights/Triple/' + name
    # model.load_state_dict(torch.load(parameter_dir))
    # model.to(device)
    # test(
    #     model=model,
    #     tensor_loader= val_loader,
    #     criterion1 = train_criterion,
    #     criterion2 = test_criterion,
    #     device= device
    # )
    train(
        model=model,
        train_loader= train_loader,
        test_loader= val_loader,    
        num_epochs= 15,
        learning_rate=1e-4,
        criterion=criterion,
        device=device,
        modality = config['modality']
            )