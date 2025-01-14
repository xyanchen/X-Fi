import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

class rfid_BiLSTM(nn.Module):
    def __init__(self,num_classes):
        super(rfid_BiLSTM,self).__init__()
        self.lstm = nn.LSTM(23,64,num_layers=1,bidirectional=True)
        self.fc = nn.Linear(64,num_classes)
    def forward(self,x):
        x = x.view(-1,23,148)
        x = x.permute(2,0,1)
        _, (ht,ct) = self.lstm(x)
        outputs = self.fc(ht[-1])
        return outputs