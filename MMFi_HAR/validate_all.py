import numpy as np
import torch
from torch import nn
import yaml
from syn_DI_dataset import make_dataset, make_dataloader
from utils import collate_fn_padd, multi_test
from X_Fi import X_Fi
import argparse

def main():
    parser = argparse.ArgumentParser('X-Fi model for MMFI HAR')
    parser.add_argument('--dataset', type=str, required=True, help='path to dataset, e.g. d:/Data/My_MMFi_Data/MMFi_Dataset')
    parser.add_argument('--pt_weights', type=str, required=True, help='path to pretrained model weights, e.g. ./pre-trained_weights/mmfi_har_checkpoint.pt')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Available validation resources:{device}')

    with open('config.yaml', 'r') as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)

    train_dataset, val_dataset = make_dataset(args.dataset, config)
    rng_generator = torch.manual_seed(config['init_rand_seed'])
    val_loader = make_dataloader(val_dataset, is_training=False, generator=rng_generator, **config['val_loader'], collate_fn = collate_fn_padd)
    
    torch.manual_seed(3407)
    model = X_Fi(model_depth=2)
    model.to(device)
    model.load_state_dict(torch.load(args.pt_weights))

    criterion = nn.CrossEntropyLoss()
    val_random_seed = config['modality_existances']['val_random_seed']
    multi_test(model, val_loader, criterion, device, val_random_seed)
    
    return

if __name__ == '__main__':
    main()