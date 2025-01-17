import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import yaml
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
    all_actions = {'A01': 0., 'A02': 1., 'A03': 2., 'A04': 3., 'A05': 4., 
                   'A06': 5., 'A07': 6., 'A08': 7., 'A09': 8., 'A10': 9.,
                   'A11': 10., 'A12': 11., 'A13': 12., 'A14': 13., 'A15': 14., 
                   'A16': 15., 'A17': 16., 'A18': 17., 'A19': 18., 'A20': 19., 
                   'A21': 20., 'A22': 21., 'A23': 22., 'A24': 23., 'A25': 24., 
                   'A26': 25., 'A27': 26.}
    labels = []
    [labels.append(all_actions[t['action']]) for t in batch]
    labels = torch.FloatTensor(labels)

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
    if 'input_mmwave' in dict_keys:
        mmwave_data = [torch.Tensor(t['input_mmwave']) for t in batch ]
        mmwave_data = torch.nn.utils.rnn.pad_sequence(mmwave_data)
        mmwave_data = mmwave_data.permute(1,0,2)
        return_data.append(mmwave_data)
        exist_list.append(True)
    else:
        exist_list.append(False)

    # lidar
    if 'input_lidar' in dict_keys:
        lidar_data = [torch.Tensor(t['input_lidar']) for t in batch ]
        lidar_data = torch.nn.utils.rnn.pad_sequence(lidar_data)
        lidar_data = lidar_data.permute(1,0,2)
        return_data.append(lidar_data)
        exist_list.append(True)
    else:
        exist_list.append(False)

    return return_data, labels, exist_list

def test(model, tensor_loader, criterion, device):
    model.eval()
    test_acc = 0
    test_loss = 0
    for data in tqdm(tensor_loader):
        input_data, label, exist_list = data
        for i, modal_data in enumerate(input_data):
            modal_data = modal_data.to(device)
            globals()[f'input_{str(i+1)}'] = modal_data
        label.to(device)
        labels = label.type(torch.LongTensor)
        if len(input_data) == 1:
            outputs = model(input_1, exist_list)
        elif len(input_data) == 2:
            outputs = model(input_1, input_2, exist_list)
        elif len(input_data) == 3:
            outputs = model(input_1, input_2, input_3, exist_list)
        elif len(input_data) == 4:
            outputs = model(input_1, input_2, input_3, input_4, exist_list)
        else:
            print('error in input_data')
        outputs = outputs.type(torch.FloatTensor)
        outputs.to(device)

        loss = criterion(outputs,labels)
        predict_y = torch.argmax(outputs,dim=1).to(device)
        accuracy = (predict_y == labels.to(device)).sum().item()/labels.size(0)
        test_acc += accuracy
        test_loss += loss.item() * labels.size(0)

        outputs = outputs.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

    test_acc = test_acc/len(tensor_loader)
    test_loss = test_loss/len(tensor_loader.dataset)
    print("validation accuracy:{:.4f}, loss:{:.5f}".format(float(test_acc),float(test_loss)))
    return test_acc

def train(model, train_loader, test_loader, num_epochs, learning_rate, criterion, device, modality):
    optimizer = torch.optim.AdamW(model.parameters(),
        lr = learning_rate
    )
    name = ''
    for mod in modality:
        name = name + mod + '_'
    name = name + '.pt'
    parameter_dir = './baseline_weights/' + name
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        num_iter = 400
        for i, data in enumerate(tqdm(train_loader)):
            if i < num_iter:
                input_data, label, exist_list = data
                for i, modal_data in enumerate(input_data):
                    modal_data = modal_data.to(device)
                    globals()[f'input_{str(i+1)}'] = modal_data
                labels = label.to(device)
                labels = labels.type(torch.LongTensor)
                optimizer.zero_grad()
                if len(input_data) == 1:
                    outputs = model(input_1, exist_list)
                elif len(input_data) == 2:
                    outputs = model(input_1, input_2, exist_list)
                elif len(input_data) == 3:
                    outputs = model(input_1, input_2, input_3, exist_list)
                elif len(input_data) == 4:
                    outputs = model(input_1, input_2, input_3, input_4, exist_list)
                else:
                    print('error in exist_list')
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
        if (epoch+1) % 5 == 0:
            test_acc = test(
                model = model,
                tensor_loader = test_loader,
                criterion = criterion,
                device = device
            )
            print(f"test accuracy is:{test_acc}")
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

    # avg_time = 0
    # for _ in range(3):
    #     i= 0
    #     for data in val_loader:
    #         input_data, label, exist_list = data
    #         for modal_data in input_data:
    #             print(modal_data.shape)
    #         print(label.shape)
    #         print(exist_list)
    #         i += 1
    #         if i > 1:
    #             print('............................................................................................')
    #             break

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device) 

    if len(config['modality']) == 1:
        model = single_model(config['modality'])
    elif len(config['modality']) == 2:
        model = dual_model(config['modality'])
    elif len(config['modality']) == 3:
        model = triple_model(config['modality'])
    elif len(config['modality']) == 4:
        model = quadra_model(config['modality'])
    model.cuda()
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    test(
        model = model,
        tensor_loader = val_loader,
        criterion = criterion,
        device = device
    )
    train(
        model = model,
        train_loader = train_loader,
        test_loader = val_loader,
        num_epochs= 35,
        learning_rate=1e-4,
        criterion = criterion,
        device = device,
        modality = config['modality']
            )