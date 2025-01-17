import numpy as np
import glob
import scipy.io as sio
import torch
from torch import nn
import random
import csv
import os

from evaluate import error

rgb_result=np.load('./baseline_results/rgb_result.npy')
depth_result=np.load('./baseline_results/depth_result.npy')
mmwave_result=np.load('./baseline_results/mmwave_result.npy')
lidar_result=np.load('./baseline_results/lidar_result.npy')
wifi_result=np.load('./baseline_results/wifi_result.npy')
all_label=np.load('./baseline_results/all_label.npy')

mpjpe, pampjpe = error(rgb_result,all_label)
print('RGB mpjpe:',mpjpe)
print('RGB pampjpe:',pampjpe)

mpjpe, pampjpe = error(depth_result,all_label)
print('Depth mpjpe:',mpjpe)
print('Depth pampjpe:',pampjpe)

mpjpe, pampjpe = error(lidar_result,all_label)
print('Lidar mpjpe:',mpjpe)
print('Lidar pampjpe:',pampjpe)

mpjpe, pampjpe = error(mmwave_result,all_label)
print('mmWave mpjpe:',mpjpe)
print('mmWave pampjpe:',pampjpe)

mpjpe, pampjpe = error(wifi_result,all_label)
print('Wifi mpjpe:',mpjpe)
print('Wifi pampjpe:',pampjpe)

ID_result = (rgb_result + depth_result)/2
mpjpe, pampjpe = error(ID_result,all_label)
print('ID mpjpe:',mpjpe)
print('ID pampjpe:',pampjpe)

IL_result = (rgb_result + lidar_result)/2
mpjpe, pampjpe = error(IL_result,all_label)
print('IL mpjpe:',mpjpe)
print('IL pampjpe:',pampjpe)

IR_result = (rgb_result + mmwave_result)/2
mpjpe, pampjpe = error(IR_result,all_label)
print('IR mpjpe:',mpjpe)
print('IR pampjpe:',pampjpe)

IW_result = (rgb_result + wifi_result)/2
mpjpe, pampjpe = error(IW_result,all_label)
print('IW mpjpe:',mpjpe)
print('IW pampjpe:',pampjpe)

DL_result = (depth_result + lidar_result)/2
mpjpe, pampjpe = error(DL_result,all_label)
print('DL mpjpe:',mpjpe)
print('DL pampjpe:',pampjpe)

DR_result = (depth_result + mmwave_result)/2
mpjpe, pampjpe = error(DR_result,all_label)
print('DR mpjpe:',mpjpe)
print('DR pampjpe:',pampjpe)

RL_result = (mmwave_result + lidar_result)/2
mpjpe, pampjpe = error(RL_result,all_label)
print('RL mpjpe:',mpjpe)
print('RL pampjpe:',pampjpe)

RW_result = (mmwave_result + wifi_result)/2
mpjpe, pampjpe = error(RW_result,all_label)
print('RW mpjpe:',mpjpe)
print('RW pampjpe:',pampjpe)

LW_result = (lidar_result + wifi_result)/2
mpjpe, pampjpe = error(LW_result,all_label)
print('LW mpjpe:',mpjpe)
print('LW pampjpe:',pampjpe)

IDL_result = (rgb_result + depth_result + lidar_result)/3
mpjpe, pampjpe = error(IDL_result,all_label)
print('IDL mpjpe:',mpjpe)
print('IDL pampjpe:',pampjpe)

IDR_result = (rgb_result + depth_result + mmwave_result)/3
mpjpe, pampjpe = error(IDR_result,all_label)
print('IDR mpjpe:',mpjpe)
print('IDR pampjpe:',pampjpe)

IDW_result = (rgb_result + depth_result + wifi_result)/3
mpjpe, pampjpe = error(IDW_result,all_label)
print('IDW mpjpe:',mpjpe)
print('IDW pampjpe:',pampjpe)

ILR_result = (rgb_result + lidar_result + mmwave_result)/3
mpjpe, pampjpe = error(ILR_result,all_label)
print('ILR mpjpe:',mpjpe)
print('ILR pampjpe:',pampjpe)

ILW_result = (rgb_result + lidar_result + wifi_result)/3
mpjpe, pampjpe = error(ILW_result,all_label)
print('ILW mpjpe:',mpjpe)
print('ILW pampjpe:',pampjpe)

IRW_result = (rgb_result + mmwave_result + wifi_result)/3
mpjpe, pampjpe = error(IRW_result,all_label)
print('IRW mpjpe:',mpjpe)
print('IRW pampjpe:',pampjpe)

DLR_result = (depth_result + lidar_result + mmwave_result)/3
mpjpe, pampjpe = error(DLR_result,all_label)
print('DLR mpjpe:',mpjpe)
print('DLR pampjpe:',pampjpe)

DLW_result = (depth_result + lidar_result + wifi_result)/3
mpjpe, pampjpe = error(DLW_result,all_label)
print('DLW mpjpe:',mpjpe)
print('DLW pampjpe:',pampjpe)

DRW_result = (depth_result + mmwave_result + wifi_result)/3
mpjpe, pampjpe = error(DRW_result,all_label)
print('DRW mpjpe:',mpjpe)
print('DRW pampjpe:',pampjpe)

LRW_result = (lidar_result + mmwave_result + wifi_result)/3
mpjpe, pampjpe = error(LRW_result,all_label)
print('LRW mpjpe:',mpjpe)
print('LRW pampjpe:',pampjpe)

IDLR_result = (rgb_result + depth_result + lidar_result + mmwave_result)/4
mpjpe, pampjpe = error(IDLR_result,all_label)
print('IDLR mpjpe:',mpjpe)
print('IDLR pampjpe:',pampjpe)

IDLW_result = (rgb_result + depth_result + lidar_result + wifi_result)/4
mpjpe, pampjpe = error(IDLW_result,all_label)
print('IDLW mpjpe:',mpjpe)
print('IDLW pampjpe:',pampjpe)

IDRW_result = (rgb_result + depth_result + mmwave_result + wifi_result)/4
mpjpe, pampjpe = error(IDRW_result,all_label)
print('IDRW mpjpe:',mpjpe)
print('IDRW pampjpe:',pampjpe)

ILRW_result = (rgb_result + lidar_result + mmwave_result + wifi_result)/4
mpjpe, pampjpe = error(ILRW_result,all_label)
print('ILRW mpjpe:',mpjpe)
print('ILRW pampjpe:',pampjpe)

DLRW_result = (depth_result + lidar_result + mmwave_result + wifi_result)/4
mpjpe, pampjpe = error(DLRW_result,all_label)
print('DLRW mpjpe:',mpjpe)
print('DLRW pampjpe:',pampjpe)

IDLRW_result = (rgb_result + depth_result + lidar_result + mmwave_result + wifi_result)/5
mpjpe, pampjpe = error(IDLRW_result,all_label)
print('IDLRW mpjpe:',mpjpe)
print('IDLRW pampjpe:',pampjpe)