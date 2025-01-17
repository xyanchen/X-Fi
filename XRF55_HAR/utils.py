import numpy as np
import torch
import random
from tqdm import tqdm
from datetime import datetime

def generate_none_empth_modality_list():
    modality_list = random.choices(
        [True, False],
        k= 1,
        weights=[50, 50]
    )
    wifi_ = random.choices(
        [True, False],
        k= 1,
        weights=[90, 10]
    )
    modality_list.append(wifi_[0])
    rfid_ = random.choices(
        [True, False],
        k= 1,
        weights=[60, 40]
    )
    modality_list.append(rfid_[0])
    # print(modality_list)
    if sum(modality_list) == 0:
        modality_list = generate_none_empth_modality_list()
        return modality_list
    else:
        return modality_list

def collate_fn_padd(batch):
    '''
    Padds batch of variable length
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
    
    modality_list = generate_none_empth_modality_list()

    return mmwave_data, wifi_data, rfid_data, labels, modality_list

def har_test(model, tensor_loader, criterion, device,val_random_seed):
    model.eval()
    test_acc = 0
    test_loss = 0
    random.seed(val_random_seed)
    for data in tqdm(tensor_loader):
        mmwave_data, wifi_data, rfid_data, labels, modality_list = data
        mmwave_data = mmwave_data.to(device)
        wifi_data = wifi_data.to(device)
        rfid_data = rfid_data.to(device)
        labels.to(device)
        labels = labels.type(torch.LongTensor)
        outputs = model(mmwave_data, wifi_data,  rfid_data, modality_list)
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

def har_train(model, train_loader, test_loader, num_epochs, learning_rate, criterion, device, save_dir, val_random_seed):
    optimizer = torch.optim.AdamW(
        [
                {'params': model.linear_projector.parameters()},
                {'params': model.X_Fusion_block.parameters()}
            ],
        lr = learning_rate
    )
    now_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    parameter_dir = save_dir + '/checkpoint_' + now_time + '.pth'
    best_test_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        random.seed(epoch)
        for i, data in enumerate(tqdm(train_loader)):
            mmwave_data, wifi_data, rfid_data, labels, modality_list = data
            mmwave_data = mmwave_data.to(device)
            wifi_data = wifi_data.to(device)
            rfid_data = rfid_data.to(device)
            labels.to(device)
            labels = labels.type(torch.LongTensor)
            
            optimizer.zero_grad()

            outputs = model(mmwave_data, wifi_data,  rfid_data, modality_list)
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
            test_acc = har_test(
                model=model,
                tensor_loader=test_loader,
                criterion = criterion,
                device= device,
                val_random_seed=val_random_seed
            )
            if test_acc >= best_test_acc:
                print(f"best test accuracy is:{test_acc}")
                best_test_acc = test_acc
    torch.save(model.state_dict(), parameter_dir)
    return

def multi_test(model, tensor_loader, criterion, device):
    model.eval()
    mmwave_test_loss = 0
    mmwave_test_accuracy = 0

    wifi_test_loss = 0
    wifi_test_accuracy = 0
    
    rfid_test_loss = 0
    rfid_test_accuracy = 0

    mmwave_wifi_test_loss = 0
    mmwave_wifi_test_accuracy = 0

    mmwave_rfid_test_loss = 0
    mmwave_rfid_test_accuracy = 0

    wifi_rfid_test_loss = 0
    wifi_rfid_test_accuracy = 0

    mmwave_wifi_rfid_test_loss = 0
    mmwave_wifi_rfid_test_accuracy = 0

    for data in tqdm(tensor_loader):
        mmwave_data, wifi_data, rfid_data, labels, _ = data
        mmwave_data = mmwave_data.to(device)
        wifi_data = wifi_data.to(device)
        rfid_data = rfid_data.to(device)
        labels.to(device)
        labels = labels.type(torch.LongTensor)


        ' SINGLE MODALITY '
        ### mmwave
        mmwave_modality_list = [True, False, False]
        mmwave_outputs = model(mmwave_data, wifi_data,  rfid_data, mmwave_modality_list)
        mmwave_outputs = mmwave_outputs.type(torch.FloatTensor)
        mmwave_outputs.to(device)
        mmwave_test_loss += criterion(mmwave_outputs,labels).item() * mmwave_data.size(0)
        mmwave_predict_y = torch.argmax(mmwave_outputs,dim=1).to(device)
        mmwave_test_accuracy += (mmwave_predict_y == labels.to(device)).sum().item() / labels.size(0)
        mmwave_outputs = mmwave_outputs.detach().cpu()
        mmwave_predict_y = mmwave_predict_y.detach().cpu()
        ### wifi-cis
        wifi_modality_list = [False, True, False]
        wifi_outputs = model(mmwave_data, wifi_data,  rfid_data, wifi_modality_list)
        wifi_outputs = wifi_outputs.type(torch.FloatTensor)
        wifi_outputs.to(device)
        wifi_test_loss += criterion(wifi_outputs,labels).item() * wifi_data.size(0)
        wifi_predict_y = torch.argmax(wifi_outputs,dim=1).to(device)
        wifi_test_accuracy += (wifi_predict_y == labels.to(device)).sum().item() / labels.size(0)
        wifi_outputs = wifi_outputs.detach().cpu()
        wifi_predict_y = wifi_predict_y.detach().cpu()
        ### rfid
        rfid_modality_list = [False, False,True]
        rfid_outputs = model(mmwave_data, wifi_data,  rfid_data, rfid_modality_list)
        rfid_outputs = rfid_outputs.type(torch.FloatTensor)
        rfid_outputs.to(device)
        rfid_test_loss += criterion(rfid_outputs,labels).item() * rfid_data.size(0)
        rfid_predict_y = torch.argmax(rfid_outputs,dim=1).to(device)
        rfid_test_accuracy += (rfid_predict_y == labels.to(device)).sum().item() / labels.size(0)
        rfid_outputs = rfid_outputs.detach().cpu()
        rfid_predict_y = rfid_predict_y.detach().cpu()
        
        'Dual modality'
        ### mmwave + wifi
        mmwave_wifi_modality_list = [True, True, False]
        mmwave_wifi_outputs = model(mmwave_data, wifi_data,  rfid_data, mmwave_wifi_modality_list)
        mmwave_wifi_outputs = mmwave_wifi_outputs.type(torch.FloatTensor)
        mmwave_wifi_outputs.to(device)
        mmwave_wifi_test_loss += criterion(mmwave_wifi_outputs,labels).item() * mmwave_data.size(0)
        mmwave_wifi_predict_y = torch.argmax(mmwave_wifi_outputs,dim=1).to(device)
        mmwave_wifi_test_accuracy += (mmwave_wifi_predict_y == labels.to(device)).sum().item() / labels.size(0)
        mmwave_wifi_outputs = mmwave_wifi_outputs.detach().cpu()
        mmwave_wifi_predict_y = mmwave_wifi_predict_y.detach().cpu()

        ### mmwave + rfid
        mmwave_rfid_modality_list = [True, False, True]
        mmwave_rfid_outputs = model(mmwave_data, wifi_data,  rfid_data, mmwave_rfid_modality_list)
        mmwave_rfid_outputs = mmwave_rfid_outputs.type(torch.FloatTensor)
        mmwave_rfid_outputs.to(device)
        mmwave_rfid_test_loss += criterion(mmwave_rfid_outputs,labels).item() * mmwave_data.size(0)
        mmwave_rfid_predict_y = torch.argmax(mmwave_rfid_outputs,dim=1).to(device)
        mmwave_rfid_test_accuracy += (mmwave_rfid_predict_y == labels.to(device)).sum().item() / labels.size(0)
        mmwave_rfid_outputs = mmwave_rfid_outputs.detach().cpu()
        mmwave_rfid_predict_y = mmwave_rfid_predict_y.detach().cpu()

        ### wifi + rfid
        wifi_rfid_modality_list = [False, True, True]
        wifi_rfid_outputs = model(mmwave_data, wifi_data,  rfid_data, wifi_rfid_modality_list)
        wifi_rfid_outputs = wifi_rfid_outputs.type(torch.FloatTensor)
        wifi_rfid_outputs.to(device)
        wifi_rfid_test_loss += criterion(wifi_rfid_outputs,labels).item() * wifi_data.size(0)
        wifi_rfid_predict_y = torch.argmax(wifi_rfid_outputs,dim=1).to(device)
        wifi_rfid_test_accuracy += (wifi_rfid_predict_y == labels.to(device)).sum().item() / labels.size(0)
        wifi_rfid_outputs = wifi_rfid_outputs.detach().cpu()
        wifi_rfid_predict_y = wifi_rfid_predict_y.detach().cpu()

        'Three modality'
        ### mmwave + wifi + rfid
        mmwave_wifi_rfid_modality_list = [True, True, True]
        mmwave_wifi_rfid_outputs = model(mmwave_data, wifi_data,  rfid_data, mmwave_wifi_rfid_modality_list)
        mmwave_wifi_rfid_outputs = mmwave_wifi_rfid_outputs.type(torch.FloatTensor)
        mmwave_wifi_rfid_outputs.to(device)
        mmwave_wifi_rfid_test_loss += criterion(mmwave_wifi_rfid_outputs,labels).item() * mmwave_data.size(0)
        mmwave_wifi_rfid_predict_y = torch.argmax(mmwave_wifi_rfid_outputs,dim=1).to(device)
        mmwave_wifi_rfid_test_accuracy += (mmwave_wifi_rfid_predict_y == labels.to(device)).sum().item() / labels.size(0)
        mmwave_wifi_rfid_outputs = mmwave_wifi_rfid_outputs.detach().cpu()
        mmwave_wifi_rfid_predict_y = mmwave_wifi_rfid_predict_y.detach().cpu()


    'single modality'
    ### mmwave
    mmwave_test_loss = mmwave_test_loss/len(tensor_loader.dataset)
    mmwave_test_accuracy = mmwave_test_accuracy/len(tensor_loader)
    print("modality: {}, Cross Entropy Loss: {:.8f}, Accuracy: {:.8f}".format('mmWave',float(mmwave_test_loss), float(mmwave_test_accuracy)))
    ### wifi
    wifi_test_loss = wifi_test_loss/len(tensor_loader.dataset)
    wifi_test_accuracy = wifi_test_accuracy/len(tensor_loader)
    print("modality: {}, Cross Entropy Loss: {:.8f}, Accuracy: {:.8f}\n".format('WiFi-CSI',float(wifi_test_loss), float(wifi_test_accuracy)))
    ### rfid
    rfid_test_loss = rfid_test_loss/len(tensor_loader.dataset)
    rfid_test_accuracy = rfid_test_accuracy/len(tensor_loader)
    print("modality: {}, Cross Entropy Loss: {:.8f}, Accuracy: {:.8f}\n".format('RFID',float(rfid_test_loss), float(rfid_test_accuracy)))

    'dual modality'
    ### mmwave + wifi
    mmwave_wifi_test_loss = mmwave_wifi_test_loss/len(tensor_loader.dataset)
    mmwave_wifi_test_accuracy = mmwave_wifi_test_accuracy/len(tensor_loader)
    print("modality: {}, Cross Entropy Loss: {:.8f}, Accuracy: {:.8f}".format('mmWave+WiFi-CSI',float(mmwave_wifi_test_loss), float(mmwave_wifi_test_accuracy)))
    
    ### mmwave + rfid
    mmwave_rfid_test_loss = mmwave_rfid_test_loss/len(tensor_loader.dataset)
    mmwave_rfid_test_accuracy = mmwave_rfid_test_accuracy/len(tensor_loader)
    print("modality: {}, Cross Entropy Loss: {:.8f}, Accuracy: {:.8f}".format('mmWave+RFID',float(mmwave_rfid_test_loss), float(mmwave_rfid_test_accuracy)))

    ### wifi + rfid
    wifi_rfid_test_loss = wifi_rfid_test_loss/len(tensor_loader.dataset)
    wifi_rfid_test_accuracy = wifi_rfid_test_accuracy/len(tensor_loader)
    print("modality: {}, Cross Entropy Loss: {:.8f}, Accuracy: {:.8f}".format('WiFi-CSI+RFID',float(wifi_rfid_test_loss), float(wifi_rfid_test_accuracy)))

    'three modality'
    ### mmwave + wifi + rfid
    mmwave_wifi_rfid_test_loss = mmwave_wifi_rfid_test_loss/len(tensor_loader.dataset)
    mmwave_wifi_rfid_test_accuracy = mmwave_wifi_rfid_test_accuracy/len(tensor_loader)
    print("modality: {}, Cross Entropy Loss: {:.8f}, Accuracy: {:.8f}".format('mmWave+WiFi-CSI+RFID',float(mmwave_wifi_rfid_test_loss), float(mmwave_wifi_rfid_test_accuracy)))
    return