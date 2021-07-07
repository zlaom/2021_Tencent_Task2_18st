import os
import json 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader 
import numpy as np 

from transformers import BertTokenizer, AdamW, AutoModel 
from tqdm import tqdm 

from utils.gap_score import get_tag_id_dict, parse_input_json, parse_gt_json, calculate_gap 
from src.models.fusion.models import TVSENet
from src.data.fusion_dataset import TVDataset

device_id = 'cuda:0'
name = 'TVSENet'

batch_size = 200
lr = 1e-4
max_epoch = 500

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


train_datafile = './utils/datafile/baseline/train.txt'
val_datafile = './utils/datafile/baseline/val.txt'

train_tf = '/home/tione/notebook/dataset/text/train_features.npy'
val_tf = '/home/tione/notebook/dataset/text/val_features.npy'

train_vf = '/home/tione/notebook/dataset/x3d/train_features.npy'
val_vf = '/home/tione/notebook/dataset/x3d/val_features.npy'



train_dataset = TVDataset(train_datafile, label_id_dic, train_tf, train_vf, is_train=True)
val_dataset = TVDataset(val_datafile, label_id_dic, val_tf, val_vf, is_train=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False) # here batch_size must be 1


def evaluate(val_loader, tokenizer, model, epoch, name):
    model.eval()
    # 输出测试的.json文件
    output = {}
    with torch.no_grad():
        for batch in tqdm(val_loader, ncols=40):
            text_feature = batch['text'].float().to(device)
            video_feature = batch['video'].float().to(device)
            preds = model(video_feature, text_feature)
            
            for i in range(preds.shape[0]):
                scores = torch.sigmoid(preds[i]) # 数字转换为0-1,且去掉batch维度
                scores_sort = scores.sort(descending=True) # 对score排序
                labels = [ id_label_dic[j.item()] for j in scores_sort.indices ] # 生成排序好的labels
                scores = scores_sort.values # 排序好的scores

                # 保存输出项目到output
                one_output = {}
                mp4_path = batch['video_path'][i]
                output[mp4_path] = one_output
                one_output["result"] = [{"labels": labels[:82], "scores": ["%.4f" % scores[i] for i in range(82)]}]        

    # 输出.json测试文件
    save_dir = '/home/tione/notebook/dataset/json/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir) 
    pred_json = '/home/tione/notebook/dataset/json/val_text_x3d.json'
    with open(pred_json, 'w', encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent = 4)
    

# Training on cuda
device = torch.device(device_id) if torch.cuda.is_available() else torch.device('cpu')
print(device)


model = TVSENet()
ckpt = './checkpoint/text_x3d.pth'
model.load_state_dict(torch.load(ckpt))

model.to(device)
model.train()

tokenizer = BertTokenizer.from_pretrained("hfl/chinese-macbert-base")

epoch = 0
gap = evaluate(val_loader, tokenizer, model, epoch, name)
