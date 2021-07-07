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
# from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from config.cfg import loading_config
from src.models.x3d.video_model_builder import X3D
from utils.gap_score import get_tag_id_dict, parse_input_json, parse_gt_json, calculate_gap 

checkpoint = './pretrain_models/x3d/x3d_m.pth'

batch_size = 24
lr = 1e-4
max_epoch = 40
save_dir = './checkpoint'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)  
    
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
        # (3x16x312x312) random crop(3x16x224x224)
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
        data['labels'] = torch.tensor(eval(self.labels[idx])).float()
        
        ## ----------video .mp4 name------------
        if self.is_train == False:
            data['video_path'] = os.path.basename(self.video_path[idx]).split('.')[0] + '.mp4'
            
        return data

    def __len__(self):
        return len(self.video_path)

# 自己定义的dataloader for train
import random 
def dataloader(dataset, batch_size):
    length = len(dataset)
    sequence = [x for x in range(length)]
    random.shuffle(sequence)

    video = []
    labels = []
    for i in range(length): # 一轮数据
        data = dataset[sequence[i]]
        video.append(data['video'])
        labels.append(data['labels'])


        if (i+1) % batch_size == 0 or (i+1) == length:
            video = torch.stack(video)
            labels = torch.stack(labels)   

            yield video, labels

            video = []
            labels = []
                
# dataset
train_dataset = VideoDataset(path_file=train_path_file, label_id_dic=label_id_dic, is_train=True)
val_dataset = VideoDataset(path_file=val_path_file, label_id_dic=label_id_dic, is_train=False)
# dataloader
# train_loader = dataloader(train_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)


@torch.no_grad()
def evaluate(val_loader, model, epoch):
    # 一些文件的位置
    tag_id_file = './utils/label_id.txt'
    gt_json = './utils/train5k.txt'    
    
    model.train()
    # 输出测试的.json文件
    output = {}
    for batch in tqdm(val_loader, ncols=20):
        videos = batch['video'].cuda()
        preds = model(videos)
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
    with open('val_tagging.json', 'w', encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent = 4)

    # 计算GAP
    pred_json = './val_tagging.json' # 临时文件，不用设置
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
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

cfg = loading_config('./config/X3D_M.yaml')

model = X3D(cfg, is_classify=True)
    
model.load_state_dict(torch.load(checkpoint))

# 如果是多GPU，指定gpu
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model, device_ids=[0,1])
model.to(device)
model.train()

loss_fn = nn.BCEWithLogitsLoss()
optimizer = AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

max_gap = 0
save_path = None
count = 0
for epoch in range(max_epoch):
    train_loader = dataloader(train_dataset, batch_size=batch_size)
    for idx_b, (videos, labels) in enumerate(train_loader):
        videos = videos.to(device)
        labels = labels.to(device)
        
        # optimazation
        preds = model(videos)
        optimizer.zero_grad()
        loss = loss_fn(preds, labels)
        loss.backward()
        optimizer.step()
        
        if idx_b % 3 == 0:
            scheduler.step()
        print("Epoch:[{}|{}]\t Current:[{}|4500]\t Loss:[{}]".format(epoch, max_epoch,
                                                                     (idx_b+1)*batch_size, loss.item()))
        
        if idx_b % 20 == 0:
            gap = evaluate(val_loader, model, epoch)
            if gap > max_gap:
                max_gap = gap
                if save_path:
                    os.remove(save_path)
                save_path = os.path.join(save_dir, 'x3d.pth')
                torch.save(model.module.state_dict(), save_path)
                count = 0

    if count > 2:
        break
    count += 1