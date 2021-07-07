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

use_ckpt = False
ckpt = './checkpoint/dual_model/1_0.7424.pth'
name = 'nextvlad'
device_id = 'cuda:0'

use_scheduler = True
val_batch_size = 100

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
        self.labels = video_path_file[1::3]
        
        self.audio_path = audio_path_file[1::6]
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
        ids = torch.tensor(eval(self.labels[idx])).float()

        data['audio'] = torch.tensor(audio_feature).float()
        data['video'] = torch.tensor(video_feature).float()
        data['video_mask'] = torch.tensor(video_mask).float()
        data['audio_mask'] = torch.tensor(audio_mask).float()
        data['labels'] = ids
        
        if self.is_train == False:
            data['video_path'] = self.video_path[idx].split("/")[-1].split(".")[0] + '.mp4' # 返回.mp4文件名
            
        return data

    def __len__(self):
        return len(self.video_path)

# video path file
datafile_dir = './utils/datafile/baseline/'
train_datafile = datafile_dir + 'train.txt'
with open(train_datafile, 'r') as f:
    audio_train = f.read().splitlines() # path_file是一个list
del audio_train[2622 * 6 : 2623 * 6]
    
val_datafile = datafile_dir + 'val.txt'
with open(val_datafile, 'r') as f:
    audio_val = f.read().splitlines() # path_file是一个list

    
datafile_dir = './utils/datafile/features/'
train_datafile = datafile_dir + 'train.txt'
with open(train_datafile, 'r') as f:
    video_train = f.read().splitlines() # path_file是一个list
del video_train[2622 * 3 : 2623 * 3]
val_datafile = datafile_dir + 'val.txt'
with open(val_datafile, 'r') as f:
    video_val = f.read().splitlines() # path_file是一个list
    

# dataset
train_dataset = Dataset(video_path_file=video_train, audio_path_file=audio_train, label_id_dic=label_id_dic, is_train=True)
val_dataset = Dataset(video_path_file=video_val, audio_path_file=audio_val, label_id_dic=label_id_dic, is_train=False)
# dataloader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)



@torch.no_grad()
def evaluate(val_loader, model, epoch, device):
    # 一些文件的位置
    tag_id_file = './utils/label_id.txt'
    gt_json = './utils/train5k.txt'    
    
    model.eval()
    # 输出测试的.json文件
    output = {}
    for batch in tqdm(val_loader, ncols=20):
        videos = batch['video'].to(device)
        video_mask = batch['video_mask'].to(device)
        audios = batch['audio'].to(device)
        audio_mask = batch['audio_mask'].to(device)

        # optimazation
        preds = model(videos, audios, video_mask, audio_mask)
        for i in range(preds.shape[0]):
            # 计算分数及标签 + 排序
            scores = torch.sigmoid(preds[i]) # 选取第i个，将数字转换为0-1
            scores_sort = scores.sort(descending=True) # 对score排序
            labels = [ id_label_dic[j.item()] for j in scores_sort.indices ] # 生成排序好的labels
            scores = scores_sort.values # 排序好的scores

            # 保存输出项目到output
            one_output = {}
            mp4_path = batch['video_path'][i]
            output[mp4_path] = one_output
            one_output["result"] = [{"labels": labels[:20], "scores": ["%.2f" % scores[i] for i in range(20)]}]        

    # 输出.json测试文件
    pred_json = name + '.json'
    with open(pred_json, 'w', encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent = 4)

    # 计算GAP
    tag_dict = get_tag_id_dict(tag_id_file)
    pred_dict = parse_input_json(pred_json, tag_dict)
    gt_dict =  parse_gt_json(gt_json, tag_dict)

    preds, labels = [], []
    for k in pred_dict:
        preds.append(pred_dict[k])
        labels.append(gt_dict[k])
    preds = np.stack(preds)
    labels = np.stack(labels)
    gap = calculate_gap(preds, labels, top_k=20)
    print("The " + str(epoch + 1) + " epoch GAP result is {:.4f}".format(gap))
    
    os.remove(pred_json)
    model.train()
    
    return gap


# Training on cuda
device = torch.device(device_id) if torch.cuda.is_available() else torch.device('cpu')
print(device)

# model
model = VideoAudio(video_dim=768, audio_dim=128, video_max_frames=300, audio_max_frames=80, video_cluster=128, audio_cluster=32, video_lamb=8, audio_lamb=4, groups=8, classify=True)
if use_ckpt:
    model.load_state_dict(torch.load(ckpt))

# 如果是多GPU，指定gpu
# if torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     model = nn.DataParallel(model, device_ids=[0,1])
model.to(device)
model.train()


loss_fn = nn.BCEWithLogitsLoss()
optimizer = AdamW(model.parameters(), lr=lr)
if use_scheduler:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

count = 0
max_gap = 0
save_path = None
for epoch in range(max_epoch):
    for idx_b, batch in enumerate(train_loader):
        videos = batch['video'].to(device)
        video_mask = batch['video_mask'].to(device)
        audios = batch['audio'].to(device)
        audio_mask = batch['audio_mask'].to(device)
        
        labels = batch['labels'].to(device)

        # optimazation
        preds = model(videos, audios, video_mask, audio_mask)
        optimizer.zero_grad()
        loss = loss_fn(preds, labels)
        loss.backward()
        optimizer.step()
        
        # log
        print("Epoch:[{}|{}]\t Current:[{}|4500]\t Loss:[{}]".format(epoch, max_epoch,
                                                                     (idx_b+1)*batch_size, loss.item()))
        # scheduler        
        if use_scheduler:
            if idx_b % 3 == 0:
                scheduler.step()

        # evaluate and save ckpt
        if idx_b % 20 == 0:
            gap = evaluate(val_loader, model, epoch, device)
            if gap > max_gap:
                max_gap = gap
                if save_path:
                    os.remove(save_path)
                save_path = './checkpoint/' + "nextvlad.pth"
                torch.save(model.state_dict(), save_path)
                count = 0
                
    if count > 3:
        break
    count += 1