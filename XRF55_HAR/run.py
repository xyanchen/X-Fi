import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from XRF55_Dataset import XRF55_Datase 
from utils import collate_fn_padd, har_train
from X_Fi import X_Fi
import argparse

def main():
    parser = argparse.ArgumentParser('X-Fi model for XRF55 HAR')
    parser.add_argument('--dataset', type=str, required=True, help='path to dataset, e.g. "D:/Data/XRF55/XRF_dataset')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Available training resources:{device}')

    train_dataset = XRF55_Datase(root_dir=args.dataset, scene='all', is_train=True)
    test_dataset = XRF55_Datase(root_dir=args.dataset, scene='all', is_train=False)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn_padd)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn_padd)
    
    torch.manual_seed(3407)
    model = X_Fi(model_depth=5, num_classes=55)
    model.to(device)

    training_epoch = 100
    learning_rate = 1e-4
    criterion = nn.CrossEntropyLoss()
    har_train(
        model = model,
        train_loader = train_dataloader,
        test_loader = test_dataloader,
        num_epochs = training_epoch,
        learning_rate = learning_rate,
        criterion=criterion,
        device=device,
        save_dir = './pre-trained_weights',
        val_random_seed = 3407
            )
    
    return

if __name__ == '__main__':
    main()