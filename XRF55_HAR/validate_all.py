import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from XRF55_Dataset import XRF55_Datase 
from utils import collate_fn_padd, multi_test
from X_Fi import X_Fi
import argparse

def main():
    parser = argparse.ArgumentParser('X-Fi model for XRF55 HAR')
    parser.add_argument('--dataset', type=str, required=True, help='path to dataset, e.g. D:/Data/XRF55/XRF_dataset')
    parser.add_argument('--pt_weights', type=str, required=True, help='path to pretrained model weights, e.g. ./pre-trained_weights/XRF55_har_checkpoint.pt')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Available validation resources:{device}')

    test_dataset = XRF55_Datase(root_dir=args.dataset, scene='all', is_train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn_padd)
    
    torch.manual_seed(3407)
    model = X_Fi(model_depth=5, num_classes=55)
    model.to(device)
    model.load_state_dict(torch.load(args.pt_weights))

    criterion = nn.CrossEntropyLoss()
    multi_test(model, test_dataloader, criterion, device)
    
    return

if __name__ == '__main__':
    main()