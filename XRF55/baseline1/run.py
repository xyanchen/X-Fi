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
from baseline_model import single_model, dual_model, triple_model

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

    return mmwave_data, wifi_data, rfid_data, labels

def test(model, tensor_loader, criterion, device, modality_list):
    model.eval()
    test_acc = 0
    test_loss = 0
    random.seed(3407)
    for data in tqdm(tensor_loader):
        mmwave_data, wifi_data, rfid_data, labels = data
        if len(modality_list) == 1:
            if modality_list[0] == 'mmwave':
                input_1 = mmwave_data.to(device)
                exist_list = [True, False, False]
            elif modality_list[0] == 'wifi':
                input_1 = wifi_data.to(device)
                exist_list = [False, True, False]
            elif modality_list[0] == 'rfid':
                input_1 = rfid_data.to(device)
                exist_list = [False, False, True]
            
            outputs = model(input_1, exist_list)
        elif len(modality_list) == 2:
            if modality_list[0] == 'mmwave' and modality_list[1] == 'wifi':
                input_1 = mmwave_data.to(device)
                input_2 = wifi_data.to(device)
                exist_list = [True, True, False]
            elif modality_list[0] == 'mmwave' and modality_list[1] == 'rfid':
                input_1 = mmwave_data.to(device)
                input_2 = rfid_data.to(device)
                exist_list = [True, False, True]
            elif modality_list[0] == 'wifi' and modality_list[1] == 'rfid':
                input_1 = wifi_data.to(device)
                input_2 = rfid_data.to(device)
                exist_list = [False, True, True]
            outputs = model(input_1, input_2, exist_list)
        elif len(modality_list) == 3:
            input_1 = mmwave_data.to(device)
            input_2 = wifi_data.to(device)
            input_3 = rfid_data.to(device)
            exist_list = [True, True, True]
            outputs = model(input_1, input_2, input_3, exist_list)
        

        labels.to(device)
        labels = labels.type(torch.LongTensor)

        
        outputs = outputs.type(torch.FloatTensor)
        outputs.to(device)
        loss = criterion(outputs,labels)
        predict_y = torch.argmax(outputs,dim=1).to(device)
        accuracy = (predict_y == labels.to(device)).sum().item() / labels.size(0)
        test_acc += accuracy
        test_loss += loss.item() * labels.size(0)
    test_acc = test_acc/len(tensor_loader)
    test_loss = test_loss/len(tensor_loader.dataset)
    print("validation accuracy:{:.4f}, loss:{:.5f}".format(float(test_acc),float(test_loss)))
    return test_acc

def train(model, train_loader, test_loader, modality_list, num_epochs, learning_rate, criterion, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    if len(modality_list) == 1:
        parameter_dir = os.path.join('./baseline/baseline_weights/Single/', modality_list[0] + '.pt')
    elif len(modality_list) == 2:
        parameter_dir = os.path.join('./baseline/baseline_weights/Dual/', modality_list[0] + '_' + modality_list[1] + '.pt')
    elif len(modality_list) == 3:
        parameter_dir = os.path.join('./baseline/baseline_weights/Triple/', modality_list[0] + '_' + modality_list[1] + '_' + modality_list[2] + '.pt')
    print('parameter_dir: ', parameter_dir)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        random.seed(epoch)
        for i, data in enumerate(tqdm(train_loader)):
            mmwave_data, wifi_data, rfid_data, labels = data
            if len(modality_list) == 1:
                if modality_list[0] == 'mmwave':
                    input_1 = mmwave_data.to(device)
                    exist_list = [True, False, False]
                elif modality_list[0] == 'wifi':
                    input_1 = wifi_data.to(device)
                    exist_list = [False, True, False]
                elif modality_list[0] == 'rfid':
                    input_1 = rfid_data.to(device)
                    exist_list = [False, False, True]
                
                outputs = model(input_1, exist_list)
            elif len(modality_list) == 2:
                if modality_list[0] == 'mmwave' and modality_list[1] == 'wifi':
                    input_1 = mmwave_data.to(device)
                    input_2 = wifi_data.to(device)
                    exist_list = [True, True, False]
                elif modality_list[0] == 'mmwave' and modality_list[1] == 'rfid':
                    input_1 = mmwave_data.to(device)
                    input_2 = rfid_data.to(device)
                    exist_list = [True, False, True]
                elif modality_list[0] == 'wifi' and modality_list[1] == 'rfid':
                    input_1 = wifi_data.to(device)
                    input_2 = rfid_data.to(device)
                    exist_list = [False, True, True]
                outputs = model(input_1, input_2, exist_list)
            elif len(modality_list) == 3:
                input_1 = mmwave_data.to(device)
                input_2 = wifi_data.to(device)
                input_3 = rfid_data.to(device)
                exist_list = [True, True, True]
                outputs = model(input_1, input_2, input_3, exist_list)
            
            labels.to(device)
            labels = labels.type(torch.LongTensor)
            
            optimizer.zero_grad()

            outputs = outputs.type(torch.FloatTensor)
            outputs.to(device)
            loss = criterion(outputs,labels)
            if loss == float('nan'):
                print('nan')
                print(outputs)
                print(labels)
                
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            predict_y = torch.argmax(outputs,dim=1).to(device)
            epoch_accuracy += (predict_y == labels.to(device)).sum().item() / labels.size(0)
        epoch_loss = epoch_loss/len(train_loader)
        epoch_accuracy = epoch_accuracy/len(train_loader)
        print('Epoch:{}, Accuracy:{:.4f},Loss:{:.9f}'.format(epoch+1, float(epoch_accuracy),float(epoch_loss)))
        if (epoch+1) % 5 == 0:
            test_acc = test(
                model=model,
                tensor_loader=test_loader,
                criterion = criterion,
                device= device,
                modality_list = modality_list
            )
    torch.save(model.state_dict(), parameter_dir)
    return

def main():
    train_dataset = XRF55_Datase(root_dir="D:\Data\XRF55\XRF_dataset", scene='all', is_train=True)
    test_dataset = XRF55_Datase(root_dir="D:\Data\XRF55\XRF_dataset", scene='all', is_train=False)

    with open('./config_all.yaml', 'r') as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)


    for i in range(len(config['modality_list'])):
        modality_list = config['modality_list'][i]
        print('modality_list: ', modality_list)
        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn_padd)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn_padd)
        # model = model(['RGB'])
        if len(modality_list) == 1:
            model = single_model(modality_list)
        elif len(modality_list) == 2:
            model = dual_model(modality_list)
        elif len(modality_list) == 3:
            model = triple_model(modality_list)
        else:
            print('error')
        model.cuda()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)

        criterion = nn.CrossEntropyLoss()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train(
            model=model,
            train_loader= train_dataloader,
            test_loader= test_dataloader,
            modality_list = modality_list,  
            num_epochs= 40,
            learning_rate=1e-4,
            criterion=criterion,
            device=device
                )
    return

if __name__ == '__main__':
    main()