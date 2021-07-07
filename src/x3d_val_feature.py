# 对验证集提取x3d视频特征
import os
import json 
import random
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader 
import numpy as np 

from transformers import AdamW, AutoModel 
from tqdm import tqdm 

from PIL import Image
import torchvision.transforms as T

from config.cfg import loading_config
from src.models.x3d.video_model_builder import X3D
from utils.gap_score import get_tag_id_dict, parse_input_json, parse_gt_json, calculate_gap 

device_id = 'cuda:1'

# 创建labels--ids转换字典
label_id_dic = {}
with open("./utils/label_id.txt") as f:
    while True: 
        line = f.readline()
        if line == '':
            break
        pair = line.strip().split('\t')
        label_id_dic[pair[0]] = int(pair[1])
# 创建ids--labels转换字典，输出需要转换回来
id_label_dic = {value: key for key, value in label_id_dic.items()}

# read path file: 'train.txt'
datafile_dir = './utils/datafile/transform/'
train_datafile = datafile_dir + 'train.txt'
with open(train_datafile, 'r') as f:
    train_path_file = f.read().splitlines() # path_file是一个list
# read path file: 'val.txt'
val_datafile = datafile_dir + 'val.txt'
with open(val_datafile, 'r') as f:
    val_path_file = f.read().splitlines() # path_file是一个list
    
# Dataset, is_train用来在测试时返回.mp4文件名 
class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, path_file, label_id_dic, is_train):
        self.video_path = path_file[0::3]
        self.labels = path_file[1::3]
        self.label_id_dic = label_id_dic
        self.is_train = is_train

    def __getitem__(self, idx):
        data = {}
        ## ---------video-------------------
        video = torch.load(self.video_path[idx])
        if self.is_train == True:
            h_start = random.randint(0, 32)
            h_end = h_start + 224
            w_start = random.randint(0, 32)
            w_end = w_start + 224
        else:
            h_start = 16
            h_end = h_start + 224
            w_start = 16
            w_end = w_start + 224
        data['video'] = video[:, :, h_start:h_end, w_start:w_end]     

        ## ----------label to id-------------- 
        data['labels'] = torch.tensor(eval(self.labels[idx]))
        
        ## ----------video .mp4 name------------
        if self.is_train == False:
            data['video_path'] = os.path.basename(self.video_path[idx]).split('.')[0] + '.mp4'
            # data['video_path'] = os.path.basename(self.video_path[idx]).split('#')[0] + '.mp4'
            
        return data

    def __len__(self):
        return len(self.video_path)
    
    
# dataset
train_dataset = VideoDataset(path_file=train_path_file, label_id_dic=label_id_dic, is_train=False)
val_dataset = VideoDataset(path_file=val_path_file, label_id_dic=label_id_dic, is_train=False)
# dataloader
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)


# Training on cuda
device = torch.device(device_id) if torch.cuda.is_available() else torch.device('cpu')
print(device)

cfg = loading_config('./config/X3D_M.yaml')
# checkpoint='./checkpoint/original/x3d_m.pth'
checkpoint = './checkpoint/x3d.pth'
model = X3D(cfg, is_classify=False)
model.load_state_dict(torch.load(checkpoint))

model.to(device)
model.eval()

features = []

dataloader = val_loader

with torch.no_grad():
    for batch in tqdm(dataloader, ncols=40):
        video = batch['video'].to(device)
        feature = model(video)
        features.append(feature.cpu())

    
features = torch.cat(features)

print(features.shape)
save_dir = '/home/tione/notebook/dataset/x3d/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)  
np.save('/home/tione/notebook/dataset/x3d/val_features.npy', features)
print('done')