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


class TVDataset(torch.utils.data.Dataset):
    def __init__(self, datafile, label_id_dic, text_path, video_path, is_train):
        # baseline datafile for get labels and video names
        with open(datafile, 'r') as f:
            path_file = f.read().splitlines()

        self.text_path = path_file[3::5]    
            
        self.text_features = np.load(text_path)
        self.video_features = np.load(video_path)
            
        self.label_id_dic = label_id_dic
        self.is_train = is_train

    def __getitem__(self, idx):
        data = {}
        ## ---------text to token------------ 
        data['text'] = torch.tensor(self.text_features[idx])
        ##-----------video feature-----------
        data['video'] = torch.tensor(self.video_features[idx])
        ## ----------video name---------------
        if self.is_train == False:
            data['video_path'] = self.text_path[idx].split("/")[-1].split(".")[0] + '.mp4' # 返回.mp4文件名
            
        return data

    def __len__(self):
        return len(self.text_path)

    
device_id = 'cuda:1'
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


test_datafile = './utils/datafile/baseline/test.txt'
test_tf = '/home/tione/notebook/dataset/text/test_features.npy'
test_vf = '/home/tione/notebook/dataset/x3d/test_features.npy'

test_dataset = TVDataset(test_datafile, label_id_dic, test_tf, test_vf, is_train=False)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

def evaluate(val_loader, model, epoch, name):
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
    pred_json = '/home/tione/notebook/dataset/json/test_text_x3d.json'
    with open(pred_json, 'w', encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent = 4)
    
    
    
# Training on cuda
device = torch.device(device_id) if torch.cuda.is_available() else torch.device('cpu')
print(device)


model = TVSENet()
ckpt = './checkpoint/text_x3d.pth'
model.load_state_dict(torch.load(ckpt))

model.to(device)


epoch = 0
gap = evaluate(test_loader, model, epoch, name)

