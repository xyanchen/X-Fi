import torch
from torch import nn
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import yaml
from evaluate import error
import time

from syn_DI_dataset import make_dataset, make_dataloader
from utils import collate_fn_padd, hpe_train
from X_Fi import X_Fi

def main():

    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device) 

    # load config
    dataset_root = 'd:\Data\My_MMFi_Data\MMFi_Dataset'
    with open('config.yaml', 'r') as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
        
    # load dataset and dataloader
    train_dataset, val_dataset = make_dataset(dataset_root, config)
    rng_generator = torch.manual_seed(config['init_rand_seed'])
    train_loader = make_dataloader(train_dataset, is_training=True, generator=rng_generator, **config['loader'], collate_fn = collate_fn_padd)
    val_loader = make_dataloader(val_dataset, is_training=False, generator=rng_generator, **config['loader'], collate_fn = collate_fn_padd)

    # load model
    torch.manual_seed(3407)
    model = X_Fi()
    model.to(device)

    # Train the model
    
    train_criterion = nn.MSELoss()
    test_criterion = error
    hpe_train(
        model=model,
        train_loader = train_loader,
        test_loader = val_loader,
        num_epochs = config['training_epochs'],
        learning_rate = config['learning_rate'],
        train_criterion = train_criterion,
        test_criterion = test_criterion,
        device = device,
        save_dir = './pre-trained_weights',
        val_random_seed = config['modality_existances']['val_random_seed']
    )
    return
    
if __name__ == '__main__':
    main()