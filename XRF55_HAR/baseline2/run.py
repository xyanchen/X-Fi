import numpy as np
import glob
import scipy.io as sio
import torch
from torch import nn
import random
import csv
import os

mmwave_result=np.load('./baseline_results/mmwave_result.npy')
wifi_result=np.load('./baseline_results/wifi_result.npy')
rfid_result=np.load('./baseline_results/rfid_result.npy')
all_label=np.load('./baseline_results/all_label.npy')

predict_y = np.argmax(mmwave_result,axis=1)
epoch_accuracy = (predict_y == all_label).sum() / all_label.size
print('mmwave accuracy:', epoch_accuracy)

predict_y = np.argmax(wifi_result,axis=1)
epoch_accuracy = (predict_y == all_label).sum() / all_label.size
print('wifi accuracy:', epoch_accuracy)

predict_y = np.argmax(rfid_result,axis=1)
epoch_accuracy = (predict_y == all_label).sum() / all_label.size
print('rfid accuracy:', epoch_accuracy)

RW_result = (mmwave_result+ wifi_result)/2
predict_y = np.argmax(RW_result,axis=1)
epoch_accuracy = (predict_y == all_label).sum() / all_label.size
print('RW accuracy:', epoch_accuracy)

RRF_result = (mmwave_result+ rfid_result)/2
predict_y = np.argmax(RRF_result,axis=1)
epoch_accuracy = (predict_y == all_label).sum() / all_label.size
print('RRF accuracy:', epoch_accuracy)

WRF_result = (wifi_result+ rfid_result)/2
predict_y = np.argmax(WRF_result,axis=1)
epoch_accuracy = (predict_y == all_label).sum() / all_label.size
print('WRF accuracy:', epoch_accuracy)

RWR_result = (mmwave_result+ wifi_result+rfid_result)/3
predict_y = np.argmax(RWR_result,axis=1)
epoch_accuracy = (predict_y == all_label).sum() / all_label.size
print('RWR accuracy:', epoch_accuracy)