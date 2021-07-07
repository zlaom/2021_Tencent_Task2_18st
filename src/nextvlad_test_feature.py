import os
import json 
import random
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader 
import numpy as np 

from transformers import BertTokenizer, AdamW, AutoModel 
from tqdm import tqdm 

from PIL import Image
import torchvision.transforms as T

from src.models.nextvlad.nextvlad import VideoAudio 
from utils.gap_score import get_tag_id_dict, parse_input_json, parse_gt_json, calculate_gap 

# ckpt = './checkpoint/m_eval_from_train/0_0.5879.pth'
use_ckpt = True
ckpt = './checkpoint/nextvlad.pth'
name = 'nextvlad'
device_id = 'cuda:0'

use_scheduler = True
test_batch_size = 100

# train batch size
batch_size = 100
lr = 1e-4
max_epoch = 200

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

def padding(features, max_frame):
    n_frame, dim = features.shape
    if n_frame >= max_frame:
        mask = np.ones(max_frame)
        return features[:max_frame, :], mask
    pad = np.zeros((max_frame, dim))
    pad[:n_frame, :] = features
    
    mask = np.zeros(max_frame)
    mask[:n_frame] = 1

    return pad, mask


class Dataset(torch.utils.data.Dataset):
    def __init__(self, video_path_file, audio_path_file, label_id_dic, is_train):
        self.video_path = video_path_file[0::3]
#         self.labels = video_path_file[1::3]
        
        self.audio_path = audio_path_file[1::5]
        self.is_train = is_train

    def __getitem__(self, idx):
        data = {}
        ## --------audio feature------------
        audio_path = self.audio_path[idx].replace('..', '/home/tione/notebook/algo-2021')
        audio = np.load(audio_path)
        audio_feature, audio_mask = padding(audio, 80)
        ## --------video feature-------------
        video_path = self.video_path[idx]
        video = np.load(video_path)
        video_feature, video_mask = padding(video, 300)

        ## ----------label to id-------------- 
#         ids = torch.tensor(eval(self.labels[idx])).float()

        data['audio'] = torch.tensor(audio_feature).float()
        data['video'] = torch.tensor(video_feature).float()
        data['video_mask'] = torch.tensor(video_mask).float()
        data['audio_mask'] = torch.tensor(audio_mask).float()
#         data['labels'] = ids
        
        if self.is_train == False:
            data['video_path'] = self.video_path[idx].split("/")[-1].split(".")[0] + '.mp4' # 返回.mp4文件名
            
        return data

    def __len__(self):
        return len(self.video_path)

# video path file
datafile_dir = './utils/datafile/baseline/'
test_datafile = datafile_dir + 'test.txt'
with open(test_datafile, 'r') as f:
    audio_test = f.read().splitlines() # path_file是一个list

    
datafile_dir = './utils/datafile/features/'
test_datafile = datafile_dir + 'test.txt'
with open(test_datafile, 'r') as f:
    video_test = f.read().splitlines() # path_file是一个list
    

# dataset
test_dataset = Dataset(video_path_file=video_test, audio_path_file=audio_test, label_id_dic=label_id_dic, is_train=False)
# dataloader
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

# Training on cuda
device = torch.device(device_id) if torch.cuda.is_available() else torch.device('cpu')
print(device)

# model
model = VideoAudio(video_dim=768, audio_dim=128, video_max_frames=300, audio_max_frames=80, video_cluster=128, audio_cluster=32, video_lamb=8, audio_lamb=4, groups=8, classify=False)
if use_ckpt:
    model.load_state_dict(torch.load(ckpt), strict=False)

# 如果是多GPU，指定gpu
# if torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     model = nn.DataParallel(model, device_ids=[0,1])
model.to(device)
model.eval()



features = []

dataloader = test_loader

with torch.no_grad():
    for batch in tqdm(dataloader, ncols=40):
        videos = batch['video'].to(device)
        video_mask = batch['video_mask'].to(device)
        audios = batch['audio'].to(device)
        audio_mask = batch['audio_mask'].to(device)

        # optimazation
        feature = model(videos, audios, video_mask, audio_mask)
        features.append(feature.cpu())

    
features = torch.cat(features)

print(features.shape)
save_dir = '/home/tione/notebook/dataset/nextvlad/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
np.save('/home/tione/notebook/dataset/nextvlad/test_features.npy', features)
print('done')