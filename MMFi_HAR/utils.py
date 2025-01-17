import numpy as np
import torch
import random
from tqdm import tqdm
from datetime import datetime

def generate_none_empth_modality_list():
    rgb_ = random.choices(
        [True, False],
        k= 1,
        weights=[50, 50]
    )
    depth_ = random.choices(
        [True, False],
        k= 1,
        weights=[60, 40]
    )
    mmwave_ = random.choices(
        [True, False],
        k= 1,
        weights=[50, 50]
    )
    lidar_ = random.choices(
        [True, False],
        k= 1,
        weights=[90, 10]
    )
    modality_list = rgb_ + depth_ + mmwave_ + lidar_
    if sum(modality_list) == 0:
        modality_list = generate_none_empth_modality_list()
        return modality_list
    else:
        return modality_list

def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    dict_keys(['modality', 'scene', 'subject', 'action', 'idx', 'output', 
    'input_rgb', 'input_depth', 'input_lidar', 'input_mmwave'])
    '''
    all_actions = {'A01': 0., 'A02': 1., 'A03': 2., 'A04': 3., 'A05': 4., 
                   'A06': 5., 'A07': 6., 'A08': 7., 'A09': 8., 'A10': 9.,
                   'A11': 10., 'A12': 11., 'A13': 12., 'A14': 13., 'A15': 14., 
                   'A16': 15., 'A17': 16., 'A18': 17., 'A19': 18., 'A20': 19., 
                   'A21': 20., 'A22': 21., 'A23': 22., 'A24': 23., 'A25': 24., 
                   'A26': 25., 'A27': 26.}
    labels = []
    [labels.append(all_actions[t['action']]) for t in batch]
    labels = torch.FloatTensor(labels)

    lengths = torch.tensor([t['input_mmwave'].shape[0] for t in batch ])

    # rgb
    rgb_data = np.array([(t['input_rgb']) for t in batch ])
    rgb_data = torch.FloatTensor(rgb_data).permute(0,3,1,2)

    # depth
    depth_data = np.array([(t['input_depth']) for t in batch ])
    depth_data = torch.FloatTensor(depth_data).permute(0,3,1,2)

    # mmwave
    mmwave_data = [torch.Tensor(t['input_mmwave']) for t in batch ]
    mmwave_data = torch.nn.utils.rnn.pad_sequence(mmwave_data)
    mmwave_data = mmwave_data.permute(1,0,2)

    # lidar
    lidar_data = [torch.Tensor(t['input_lidar'].copy()) for t in batch ]
    lidar_data = torch.nn.utils.rnn.pad_sequence(lidar_data)
    lidar_data = lidar_data.permute(1,0,2)
    
    modality_list = generate_none_empth_modality_list()

    return rgb_data, depth_data, mmwave_data, lidar_data, labels, modality_list

def har_test(model, tensor_loader, criterion, device,val_random_seed):
    model.eval()
    test_acc = 0
    test_loss = 0
    random.seed(val_random_seed)
    for i, data in enumerate(tqdm(tensor_loader)):
        rgb_data, depth_data, mmwave_data, lidar_data, label, modality_list = data
        rgb_data = rgb_data.to(device)
        depth_data = depth_data.to(device)
        lidar_data = lidar_data.to(device)
        mmwave_data = mmwave_data.to(device)
        label.to(device)
        labels = label.type(torch.LongTensor)
        outputs = model(rgb_data, depth_data,  mmwave_data, lidar_data, modality_list)
        outputs = outputs.type(torch.FloatTensor)
        outputs.to(device)
        loss = criterion(outputs,labels)
        predict_y = torch.argmax(outputs,dim=1).to(device)
        accuracy = (predict_y == labels.to(device)).sum().item()/labels.size(0)
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
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[15],gamma=0.1)
    now_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    parameter_dir = save_dir + '/checkpoint_' + now_time + '.pth'
    best_test_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        random.seed(epoch)
        num_iter = 1000
        for i, data in enumerate(tqdm(train_loader)):
            if i < num_iter:
                rgb_data, depth_data, mmwave_data, lidar_data, label, modality_list = data
                rgb_data = rgb_data.to(device)
                depth_data = depth_data.to(device)
                lidar_data = lidar_data.to(device)
                mmwave_data = mmwave_data.to(device)
                labels = label.to(device)
                labels = labels.type(torch.LongTensor)
                optimizer.zero_grad()
                outputs = model(rgb_data, depth_data, mmwave_data, lidar_data, modality_list)
                outputs = outputs.to(device)
                outputs = outputs.type(torch.FloatTensor)
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
            else:
                break
        epoch_loss = epoch_loss/num_iter
        epoch_accuracy = epoch_accuracy/num_iter
        print('Epoch:{}, Accuracy:{:.4f},Loss:{:.9f}'.format(epoch+1, float(epoch_accuracy),float(epoch_loss)))
        if (epoch+1) % 10 == 0:
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
        scheduler.step()
    torch.save(model.state_dict(), parameter_dir)
    return

def multi_test(model, tensor_loader, criterion, device, val_random_seed):
    model.eval()
    rgb_test_loss = 0
    rgb_test_accuracy = 0

    depth_test_loss = 0
    depth_test_accuracy = 0
    

    lidar_test_loss = 0
    lidar_test_accuracy = 0

    mmwave_test_loss = 0
    mmwave_test_accuracy = 0

    rgb_depth_test_loss = 0
    rgb_depth_test_accuracy = 0

    rgb_lidar_test_loss = 0
    rgb_lidar_test_accuracy = 0

    rgb_mmwave_test_loss = 0
    rgb_mmwave_test_accuracy = 0

    depth_lidar_test_loss = 0
    depth_lidar_test_accuracy = 0

    depth_mmwave_test_loss = 0
    depth_mmwave_test_accuracy = 0

    lidar_mmwave_test_loss = 0
    lidar_mmwave_test_accuracy = 0

    rgb_depth_lidar_test_loss = 0
    rgb_depth_lidar_test_accuracy = 0

    rgb_depth_mmwave_test_loss = 0
    rgb_depth_mmwave_test_accuracy = 0

    rgb_lidar_mmwave_test_loss = 0
    rgb_lidar_mmwave_test_accuracy = 0

    depth_lidar_mmwave_test_loss = 0
    depth_lidar_mmwave_test_accuracy = 0

    rgb_depth_lidar_mmwave_test_loss = 0
    rgb_depth_lidar_mmwave_test_accuracy = 0

    random.seed(val_random_seed)
    for data in tqdm(tensor_loader):
        rgb_data, depth_data, mmwave_data, lidar_data, label, _ = data
        rgb_data = rgb_data.to(device)
        depth_data = depth_data.to(device)
        lidar_data = lidar_data.to(device)
        mmwave_data = mmwave_data.to(device)
        label.to(device)
        labels = label.type(torch.LongTensor)


        ' SINGLE MODALITY '
        ### rgb
        rgb_modality_list = [True, False, False, False]
        rgb_outputs = model(rgb_data, depth_data,  mmwave_data, lidar_data, rgb_modality_list)
        rgb_outputs = rgb_outputs.type(torch.FloatTensor)
        rgb_outputs.to(device)
        rgb_test_loss += criterion(rgb_outputs,labels).item() * rgb_data.size(0)
        rgb_predict_y = torch.argmax(rgb_outputs,dim=1).to(device)
        rgb_test_accuracy += (rgb_predict_y == labels.to(device)).sum().item() / labels.size(0)
        rgb_outputs = rgb_outputs.detach().cpu()
        rgb_predict_y = rgb_predict_y.detach().cpu()
        ### depth
        depth_modality_list = [False, True, False, False]
        depth_outputs = model(rgb_data, depth_data,  mmwave_data, lidar_data, depth_modality_list)
        depth_outputs = depth_outputs.type(torch.FloatTensor)
        depth_outputs.to(device)
        depth_test_loss += criterion(depth_outputs,labels).item() * rgb_data.size(0)
        depth_predict_y = torch.argmax(depth_outputs,dim=1).to(device)
        depth_test_accuracy += (depth_predict_y == labels.to(device)).sum().item() / labels.size(0)
        depth_outputs = depth_outputs.detach().cpu()
        depth_predict_y = depth_predict_y.detach().cpu()
        ### lidar
        lidar_modality_list = [False, False, False, True]
        lidar_outputs = model(rgb_data, depth_data,  mmwave_data, lidar_data, lidar_modality_list)
        lidar_outputs = lidar_outputs.type(torch.FloatTensor)
        lidar_outputs.to(device)
        lidar_test_loss += criterion(lidar_outputs,labels).item() * rgb_data.size(0)
        lidar_predict_y = torch.argmax(lidar_outputs,dim=1).to(device)
        lidar_test_accuracy += (lidar_predict_y == labels.to(device)).sum().item() / labels.size(0)
        lidar_outputs = lidar_outputs.detach().cpu()
        lidar_predict_y = lidar_predict_y.detach().cpu()
        ### mmwave
        mmwave_modality_list = [False, False, True, False]
        mmwave_outputs = model(rgb_data, depth_data,  mmwave_data, lidar_data, mmwave_modality_list)
        mmwave_outputs = mmwave_outputs.type(torch.FloatTensor)
        mmwave_outputs.to(device)
        mmwave_test_loss += criterion(mmwave_outputs,labels).item() * rgb_data.size(0)
        mmwave_predict_y = torch.argmax(mmwave_outputs,dim=1).to(device)
        mmwave_test_accuracy += (mmwave_predict_y == labels.to(device)).sum().item() / labels.size(0)
        mmwave_outputs = mmwave_outputs.detach().cpu()
        mmwave_predict_y = mmwave_predict_y.detach().cpu()
        
        'Dual modality'
        ### rgb + depth
        rgb_depth_modality_list = [True, True, False, False]
        rgb_depth_outputs = model(rgb_data, depth_data,  mmwave_data, lidar_data, rgb_depth_modality_list)
        rgb_depth_outputs = rgb_depth_outputs.type(torch.FloatTensor)
        rgb_depth_outputs.to(device)
        rgb_depth_test_loss += criterion(rgb_depth_outputs,labels).item() * rgb_data.size(0)
        rgb_depth_predict_y = torch.argmax(rgb_depth_outputs,dim=1).to(device)
        rgb_depth_test_accuracy += (rgb_depth_predict_y == labels.to(device)).sum().item() / labels.size(0)
        rgb_depth_outputs = rgb_depth_outputs.detach().cpu()
        rgb_depth_predict_y = rgb_depth_predict_y.detach().cpu()
        ### rgb + lidar
        rgb_lidar_modality_list = [True, False, False, True]
        rgb_lidar_outputs = model(rgb_data, depth_data,  mmwave_data, lidar_data, rgb_lidar_modality_list)
        rgb_lidar_outputs = rgb_lidar_outputs.type(torch.FloatTensor)
        rgb_lidar_outputs.to(device)
        rgb_lidar_test_loss += criterion(rgb_lidar_outputs,labels).item() * rgb_data.size(0)
        rgb_lidar_predict_y = torch.argmax(rgb_lidar_outputs,dim=1).to(device)
        rgb_lidar_test_accuracy += (rgb_lidar_predict_y == labels.to(device)).sum().item() / labels.size(0)
        rgb_lidar_outputs = rgb_lidar_outputs.detach().cpu()
        rgb_lidar_predict_y = rgb_lidar_predict_y.detach().cpu()
        ### rgb + mmwave
        rgb_mmwave_modality_list = [True, False, True, False]
        rgb_mmwave_outputs = model(rgb_data, depth_data,  mmwave_data, lidar_data, rgb_mmwave_modality_list)
        rgb_mmwave_outputs = rgb_mmwave_outputs.type(torch.FloatTensor)
        rgb_mmwave_outputs.to(device)
        rgb_mmwave_test_loss += criterion(rgb_mmwave_outputs,labels).item() * rgb_data.size(0)
        rgb_mmwave_predict_y = torch.argmax(rgb_mmwave_outputs,dim=1).to(device)
        rgb_mmwave_test_accuracy += (rgb_mmwave_predict_y == labels.to(device)).sum().item() / labels.size(0)
        rgb_mmwave_outputs = rgb_mmwave_outputs.detach().cpu()
        rgb_mmwave_predict_y = rgb_mmwave_predict_y.detach().cpu()
        ### depth + lidar
        depth_lidar_modality_list = [False, True, False, True]
        depth_lidar_outputs = model(rgb_data, depth_data,  mmwave_data, lidar_data, depth_lidar_modality_list)
        depth_lidar_outputs = depth_lidar_outputs.type(torch.FloatTensor)
        depth_lidar_outputs.to(device)
        depth_lidar_test_loss += criterion(depth_lidar_outputs,labels).item() * rgb_data.size(0)
        depth_lidar_predict_y = torch.argmax(depth_lidar_outputs,dim=1).to(device)
        depth_lidar_test_accuracy += (depth_lidar_predict_y == labels.to(device)).sum().item() / labels.size(0)
        depth_lidar_outputs = depth_lidar_outputs.detach().cpu()
        depth_lidar_predict_y = depth_lidar_predict_y.detach().cpu()
        ### depth + mmwave
        depth_mmwave_modality_list = [False, True, True, False]
        depth_mmwave_outputs = model(rgb_data, depth_data,  mmwave_data, lidar_data, depth_mmwave_modality_list)
        depth_mmwave_outputs = depth_mmwave_outputs.type(torch.FloatTensor)
        depth_mmwave_outputs.to(device)
        depth_mmwave_test_loss += criterion(depth_mmwave_outputs,labels).item() * rgb_data.size(0)
        depth_mmwave_predict_y = torch.argmax(depth_mmwave_outputs,dim=1).to(device)
        depth_mmwave_test_accuracy += (depth_mmwave_predict_y == labels.to(device)).sum().item() / labels.size(0)
        depth_mmwave_outputs = depth_mmwave_outputs.detach().cpu()
        depth_mmwave_predict_y = depth_mmwave_predict_y.detach().cpu()
        ### lidar + mmwave
        lidar_mmwave_modality_list = [False, False, True, True]
        lidar_mmwave_outputs = model(rgb_data, depth_data,  mmwave_data, lidar_data, lidar_mmwave_modality_list)
        lidar_mmwave_outputs = lidar_mmwave_outputs.type(torch.FloatTensor)
        lidar_mmwave_outputs.to(device)
        lidar_mmwave_test_loss += criterion(lidar_mmwave_outputs,labels).item() * rgb_data.size(0)
        lidar_mmwave_predict_y = torch.argmax(lidar_mmwave_outputs,dim=1).to(device)
        lidar_mmwave_test_accuracy += (lidar_mmwave_predict_y == labels.to(device)).sum().item() / labels.size(0)
        lidar_mmwave_outputs = lidar_mmwave_outputs.detach().cpu()
        lidar_mmwave_predict_y = lidar_mmwave_predict_y.detach().cpu()

        'Three modality'
        ### rgb + depth + lidar
        rgb_depth_lidar_modality_list = [True, True, False, True]
        rgb_depth_lidar_outputs = model(rgb_data, depth_data,  mmwave_data, lidar_data, rgb_depth_lidar_modality_list)
        rgb_depth_lidar_outputs = rgb_depth_lidar_outputs.type(torch.FloatTensor)
        rgb_depth_lidar_outputs.to(device)
        rgb_depth_lidar_test_loss += criterion(rgb_depth_lidar_outputs,labels).item() * rgb_data.size(0)
        rgb_depth_lidar_predict_y = torch.argmax(rgb_depth_lidar_outputs,dim=1).to(device)
        rgb_depth_lidar_test_accuracy += (rgb_depth_lidar_predict_y == labels.to(device)).sum().item() / labels.size(0)
        rgb_depth_lidar_outputs = rgb_depth_lidar_outputs.detach().cpu()
        rgb_depth_lidar_predict_y = rgb_depth_lidar_predict_y.detach().cpu()
        ### rgb + depth + mmwave
        rgb_depth_mmwave_modality_list = [True, True, True, False]
        rgb_depth_mmwave_outputs = model(rgb_data, depth_data,  mmwave_data, lidar_data, rgb_depth_mmwave_modality_list)
        rgb_depth_mmwave_outputs = rgb_depth_mmwave_outputs.type(torch.FloatTensor)
        rgb_depth_mmwave_outputs.to(device)
        rgb_depth_mmwave_test_loss += criterion(rgb_depth_mmwave_outputs,labels).item() * rgb_data.size(0)
        rgb_depth_mmwave_predict_y = torch.argmax(rgb_depth_mmwave_outputs,dim=1).to(device)
        rgb_depth_mmwave_test_accuracy += (rgb_depth_mmwave_predict_y == labels.to(device)).sum().item() / labels.size(0)
        rgb_depth_mmwave_outputs = rgb_depth_mmwave_outputs.detach().cpu()
        rgb_depth_mmwave_predict_y = rgb_depth_mmwave_predict_y.detach().cpu()
        ### rgb + lidar + mmwave
        rgb_lidar_mmwave_modality_list = [True, False, True, True]
        rgb_lidar_mmwave_outputs = model(rgb_data, depth_data,  mmwave_data, lidar_data, rgb_lidar_mmwave_modality_list)
        rgb_lidar_mmwave_outputs = rgb_lidar_mmwave_outputs.type(torch.FloatTensor)
        rgb_lidar_mmwave_outputs.to(device)
        rgb_lidar_mmwave_test_loss += criterion(rgb_lidar_mmwave_outputs,labels).item() * rgb_data.size(0)
        rgb_lidar_mmwave_predict_y = torch.argmax(rgb_lidar_mmwave_outputs,dim=1).to(device)
        rgb_lidar_mmwave_test_accuracy += (rgb_lidar_mmwave_predict_y == labels.to(device)).sum().item() / labels.size(0)
        rgb_lidar_mmwave_outputs = rgb_lidar_mmwave_outputs.detach().cpu()
        rgb_lidar_mmwave_predict_y = rgb_lidar_mmwave_predict_y.detach().cpu()
        ### depth + lidar + mmwave
        depth_lidar_mmwave_modality_list = [False, True, True, True]
        depth_lidar_mmwave_outputs = model(rgb_data, depth_data,  mmwave_data, lidar_data, depth_lidar_mmwave_modality_list)
        depth_lidar_mmwave_outputs = depth_lidar_mmwave_outputs.type(torch.FloatTensor)
        depth_lidar_mmwave_outputs.to(device)
        depth_lidar_mmwave_test_loss += criterion(depth_lidar_mmwave_outputs,labels).item() * rgb_data.size(0)
        depth_lidar_mmwave_predict_y = torch.argmax(depth_lidar_mmwave_outputs,dim=1).to(device)
        depth_lidar_mmwave_test_accuracy += (depth_lidar_mmwave_predict_y == labels.to(device)).sum().item() / labels.size(0)
        depth_lidar_mmwave_outputs = depth_lidar_mmwave_outputs.detach().cpu()
        depth_lidar_mmwave_predict_y = depth_lidar_mmwave_predict_y.detach().cpu()

        'Four modality'
        ### rgb + depth + lidar + mmwave
        rgb_depth_lidar_mmwave_modality_list = [True, True, True, True]
        rgb_depth_lidar_mmwave_outputs = model(rgb_data, depth_data,  mmwave_data, lidar_data, rgb_depth_lidar_mmwave_modality_list)
        rgb_depth_lidar_mmwave_outputs = rgb_depth_lidar_mmwave_outputs.type(torch.FloatTensor)
        rgb_depth_lidar_mmwave_outputs.to(device)
        rgb_depth_lidar_mmwave_test_loss += criterion(rgb_depth_lidar_mmwave_outputs,labels).item() * rgb_data.size(0)
        rgb_depth_lidar_mmwave_predict_y = torch.argmax(rgb_depth_lidar_mmwave_outputs,dim=1).to(device)
        rgb_depth_lidar_mmwave_test_accuracy += (rgb_depth_lidar_mmwave_predict_y == labels.to(device)).sum().item() / labels.size(0)
        rgb_depth_lidar_mmwave_outputs = rgb_depth_lidar_mmwave_outputs.detach().cpu()
        rgb_depth_lidar_mmwave_predict_y = rgb_depth_lidar_mmwave_predict_y.detach().cpu()


    'single modality'
    ### rgb
    rgb_test_loss = rgb_test_loss/len(tensor_loader.dataset)
    rgb_test_accuracy = rgb_test_accuracy/len(tensor_loader)
    print("modality: {}, Cross Entropy Loss: {:.8f}, Accuracy: {:.8f}".format('RGB',float(rgb_test_loss), float(rgb_test_accuracy)))
    ### depth
    depth_test_loss = depth_test_loss/len(tensor_loader.dataset)
    depth_test_accuracy = depth_test_accuracy/len(tensor_loader)
    print("modality: {}, Cross Entropy Loss: {:.8f}, Accuracy: {:.8f}".format('Depth',float(depth_test_loss), float(depth_test_accuracy)))
    ### lidar
    lidar_test_loss = lidar_test_loss/len(tensor_loader.dataset)
    lidar_test_accuracy = lidar_test_accuracy/len(tensor_loader)
    print("modality: {}, Cross Entropy Loss: {:.8f}, Accuracy: {:.8f}".format('Lidar',float(lidar_test_loss), float(lidar_test_accuracy)))
    ### mmwave
    mmwave_test_loss = mmwave_test_loss/len(tensor_loader.dataset)
    mmwave_test_accuracy = mmwave_test_accuracy/len(tensor_loader)
    print("modality: {}, Cross Entropy Loss: {:.8f}, Accuracy: {:.8f}".format('mmWave',float(mmwave_test_loss), float(mmwave_test_accuracy)))
    
    'dual modality'
    ### rgb + depth
    rgb_depth_test_loss = rgb_depth_test_loss/len(tensor_loader.dataset)
    rgb_depth_test_accuracy = rgb_depth_test_accuracy/len(tensor_loader)
    print("modality: {}, Cross Entropy Loss: {:.8f}, Accuracy: {:.8f}".format('RGB+Depth',float(rgb_depth_test_loss), float(rgb_depth_test_accuracy)))
    ### rgb + lidar
    rgb_lidar_test_loss = rgb_lidar_test_loss/len(tensor_loader.dataset)
    rgb_lidar_test_accuracy = rgb_lidar_test_accuracy/len(tensor_loader)
    print("modality: {}, Cross Entropy Loss: {:.8f}, Accuracy: {:.8f}".format('RGB+Lidar',float(rgb_lidar_test_loss), float(rgb_lidar_test_accuracy)))
    ### rgb + mmwave
    rgb_mmwave_test_loss = rgb_mmwave_test_loss/len(tensor_loader.dataset)
    rgb_mmwave_test_accuracy = rgb_mmwave_test_accuracy/len(tensor_loader)
    print("modality: {}, Cross Entropy Loss: {:.8f}, Accuracy: {:.8f}".format('RGB+mmWave',float(rgb_mmwave_test_loss), float(rgb_mmwave_test_accuracy)))
    ### depth + lidar
    depth_lidar_test_loss = depth_lidar_test_loss/len(tensor_loader.dataset)
    depth_lidar_test_accuracy = depth_lidar_test_accuracy/len(tensor_loader)
    print("modality: {}, Cross Entropy Loss: {:.8f}, Accuracy: {:.8f}".format('Depth+Lidar',float(depth_lidar_test_loss), float(depth_lidar_test_accuracy)))
    ### depth + mmwave
    depth_mmwave_test_loss = depth_mmwave_test_loss/len(tensor_loader.dataset)
    depth_mmwave_test_accuracy = depth_mmwave_test_accuracy/len(tensor_loader)
    print("modality: {}, Cross Entropy Loss: {:.8f}, Accuracy: {:.8f}".format('Depth+mmWave',float(depth_mmwave_test_loss), float(depth_mmwave_test_accuracy)))
    ### lidar + mmwave
    lidar_mmwave_test_loss = lidar_mmwave_test_loss/len(tensor_loader.dataset)
    lidar_mmwave_test_accuracy = lidar_mmwave_test_accuracy/len(tensor_loader)
    print("modality: {}, Cross Entropy Loss: {:.8f}, Accuracy: {:.8f}".format('Lidar+mmWave',float(lidar_mmwave_test_loss), float(lidar_mmwave_test_accuracy)))
    
    'three modality'
    ### rgb + depth + lidar
    rgb_depth_lidar_test_loss = rgb_depth_lidar_test_loss/len(tensor_loader.dataset)
    rgb_depth_lidar_test_accuracy = rgb_depth_lidar_test_accuracy/len(tensor_loader)
    print("modality: {}, Cross Entropy Loss: {:.8f}, Accuracy: {:.8f}".format('RGB+Depth+Lidar',float(rgb_depth_lidar_test_loss), float(rgb_depth_lidar_test_accuracy)))
    ### rgb + depth + mmwave
    rgb_depth_mmwave_test_loss = rgb_depth_mmwave_test_loss/len(tensor_loader.dataset)
    rgb_depth_mmwave_test_accuracy = rgb_depth_mmwave_test_accuracy/len(tensor_loader)
    print("modality: {}, Cross Entropy Loss: {:.8f}, Accuracy: {:.8f}".format('RGB+Depth+mmWave',float(rgb_depth_mmwave_test_loss), float(rgb_depth_mmwave_test_accuracy)))
    ### rgb + lidar + mmwave
    rgb_lidar_mmwave_test_loss = rgb_lidar_mmwave_test_loss/len(tensor_loader.dataset)
    rgb_lidar_mmwave_test_accuracy = rgb_lidar_mmwave_test_accuracy/len(tensor_loader)
    print("modality: {}, Cross Entropy Loss: {:.8f}, Accuracy: {:.8f}".format('RGB+Lidar+mmWave',float(rgb_lidar_mmwave_test_loss), float(rgb_lidar_mmwave_test_accuracy)))
    ### depth + lidar + mmwave
    depth_lidar_mmwave_test_loss = depth_lidar_mmwave_test_loss/len(tensor_loader.dataset)
    depth_lidar_mmwave_test_accuracy = depth_lidar_mmwave_test_accuracy/len(tensor_loader)
    print("modality: {}, Cross Entropy Loss: {:.8f}, Accuracy: {:.8f}".format('Depth+Lidar+mmWave',float(depth_lidar_mmwave_test_loss), float(depth_lidar_mmwave_test_accuracy)))
    
    'four modality'
    ### rgb + depth + lidar + mmwave
    rgb_depth_lidar_mmwave_test_loss = rgb_depth_lidar_mmwave_test_loss/len(tensor_loader.dataset)
    rgb_depth_lidar_mmwave_test_accuracy = rgb_depth_lidar_mmwave_test_accuracy/len(tensor_loader)
    print("modality: {}, Cross Entropy Loss: {:.8f}, Accuracy: {:.8f}".format('RGB+Depth+Lidar+mmWave',float(rgb_depth_lidar_mmwave_test_loss), float(rgb_depth_lidar_mmwave_test_accuracy)))
    
    return