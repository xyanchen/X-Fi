import numpy as np
import glob
import scipy.io as sio
import torch
from torch import nn
import random
import csv
import os

rgb_result=np.load('./baseline_results/rgb_result.npy')
depth_result=np.load('./baseline_results/depth_result.npy')
mmwave_result=np.load('./baseline_results/mmwave_result.npy')
lidar_result=np.load('./baseline_results/lidar_result.npy')
all_label=np.load('./baseline_results/all_label.npy')

predict_y = np.argmax(rgb_result,axis=1)
epoch_accuracy = (predict_y == all_label).sum() / all_label.size
print('rgb accuracy:', epoch_accuracy)

predict_y = np.argmax(depth_result,axis=1)
epoch_accuracy = (predict_y == all_label).sum() / all_label.size
print('depth accuracy:', epoch_accuracy)

predict_y = np.argmax(mmwave_result,axis=1)
epoch_accuracy = (predict_y == all_label).sum() / all_label.size
print('mmwave accuracy:', epoch_accuracy)

predict_y = np.argmax(lidar_result,axis=1)
epoch_accuracy = (predict_y == all_label).sum() / all_label.size
print('lidar accuracy:', epoch_accuracy)

ID_result = (rgb_result+depth_result)/2
predict_y = np.argmax(ID_result,axis=1)
epoch_accuracy = (predict_y == all_label).sum() / all_label.size
print('ID accuracy:', epoch_accuracy)

IL_result = (rgb_result+lidar_result)/2
predict_y = np.argmax(IL_result,axis=1)
epoch_accuracy = (predict_y == all_label).sum() / all_label.size
print('IL accuracy:', epoch_accuracy)

IR_result = (rgb_result+mmwave_result)/2
predict_y = np.argmax(IR_result,axis=1)
epoch_accuracy = (predict_y == all_label).sum() / all_label.size
print('IR accuracy:', epoch_accuracy)

DL_result = (depth_result+lidar_result)/2
predict_y = np.argmax(DL_result,axis=1)
epoch_accuracy = (predict_y == all_label).sum() / all_label.size
print('DL accuracy:', epoch_accuracy)

DR_result = (depth_result+mmwave_result)/2
predict_y = np.argmax(DR_result,axis=1)
epoch_accuracy = (predict_y == all_label).sum() / all_label.size
print('DR accuracy:', epoch_accuracy)

LR_result = (lidar_result+mmwave_result)/2
predict_y = np.argmax(LR_result,axis=1)
epoch_accuracy = (predict_y == all_label).sum() / all_label.size
print('LR accuracy:', epoch_accuracy)

IDL_result = (rgb_result+depth_result+lidar_result)/3
predict_y = np.argmax(IDL_result,axis=1)
epoch_accuracy = (predict_y == all_label).sum() / all_label.size
print('IDL accuracy:', epoch_accuracy)

IDR_result = (rgb_result+depth_result+mmwave_result)/3
predict_y = np.argmax(IDR_result,axis=1)
epoch_accuracy = (predict_y == all_label).sum() / all_label.size
print('IDR accuracy:', epoch_accuracy)

ILR_result = (rgb_result+lidar_result+mmwave_result)/3
predict_y = np.argmax(ILR_result,axis=1)
epoch_accuracy = (predict_y == all_label).sum() / all_label.size
print('ILR accuracy:', epoch_accuracy)

DLR_result = (depth_result+lidar_result+mmwave_result)/3
predict_y = np.argmax(DLR_result,axis=1)
epoch_accuracy = (predict_y == all_label).sum() / all_label.size
print('DLR accuracy:', epoch_accuracy)

IDLR_result = (rgb_result+depth_result+lidar_result+mmwave_result)/4
predict_y = np.argmax(IDLR_result,axis=1)
epoch_accuracy = (predict_y == all_label).sum() / all_label.size
print('IDLR accuracy:', epoch_accuracy)