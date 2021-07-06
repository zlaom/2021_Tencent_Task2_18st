import os
import json 
import random
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader 
import numpy as np 
import torchvision.transforms as T
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image
import random
import time

from transformers import AdamW, AutoModel 
from tqdm import tqdm 

# from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from config.cfg import loading_config
from model.video_model_builder import X3D
from utils.gap_score import get_tag_id_dict, parse_input_json, parse_gt_json, calculate_gap 

checkpoint = './checkpoint/m/7_0.7675.pth'
name = 'multiple_val'
batch_size = 26
lr = 1e-5
max_epoch = 40
n_val = 1

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


# read path file: 'val.txt'
datafile_dir = '/home/tione/notebook/datafile/val'



# Dataset, is_train用来在测试时返回.mp4文件名 
class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, datafile_dir, n_val):
        self.n_val = n_val
        self.val = []
        for num in range(self.n_val):
            datafile = os.path.join(datafile_dir, 'val' + str(num) + '.txt')
            with open(datafile, 'r') as f:
                path_file = f.read().splitlines() # path_file是一个list
                self.val.append(path_file[0::3])
        
    def __getitem__(self, idx):
        data = {}
        ## ---------video-------------------
        for num in range(self.n_val):
            data['video' + str(num)] = torch.load(self.val[num][idx])
            
        ## ----------video .mp4 name------------
        data['video_path'] = os.path.basename(self.val[0][idx]).split('.')[0] + '.mp4'
        return data

    def __len__(self):
        return len(self.val[0])



@torch.no_grad()
def evaluate(val_loader, model, epoch, name, n_val=5):
    # 一些文件的位置
    tag_id_file = './utils/label_id.txt'
    gt_json = './utils/train5k.txt'    
    
    model.eval()
    # 输出测试的.json文件
    output = {}
    for batch in tqdm(val_loader, ncols=20):
        preds = None
        for num in range(n_val):
            videos = batch['video' + str(num)].to(device)
            if preds == None:
                preds = model(videos)
            else:
                preds += model(videos)
        preds = preds / n_val
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
    print("The " + str(epoch + 1) + " epoch GAP result is {:.3f}".format(gap))
    
    os.remove(pred_json)
    model.train()
    
    return gap


# Training on cuda
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

# dataset
val_dataset = VideoDataset(datafile_dir, n_val=n_val)
# dataloader
val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)


cfg = loading_config('./config/X3D_M.yaml')

model = X3D(cfg, is_classify=True)
    
model.load_state_dict(torch.load(checkpoint))

# 如果是多GPU，指定gpu
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model, device_ids=[0,1])
model.cuda()

epoch = 0
gap = evaluate(val_loader, model, epoch, name, n_val=n_val)
