import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
import torchvision.transforms as T
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image
import time

def read_lines(datafile):
    with open(datafile, 'r') as f:
        pathfile = f.read().splitlines()
    return pathfile

def get_basename(path):
    # return the file basename
    return os.path.basename(path)

def save_to_file(filename, datafile):
    # save datafile to local directory
    with open(filename, 'w') as f:
        for line in datafile:
            f.write(line+'\n')
            

# 创建labels--ids转换字典
def get_label2id_dic(path):
    label2id_dic = {}
    with open(path, 'r') as f:
        while True: 
            line = f.readline()
            if line == '':
                break
            pair = line.strip().split('\t')
            label2id_dic[pair[0]] = int(pair[1])
    return label2id_dic

label2id_dic = get_label2id_dic('./utils/label_id.txt')

def label2id(label2id_dic, labels):
    ids = [0 for i in range(82)]
    labels = labels.split(',')
    for label in labels:
        ids[label2id_dic[label]] = 1
    return ids

transform = T.Compose([
                        T.Resize(256, interpolation=3),
                        T.CenterCrop(256),
                        T.ToTensor(),
                        T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
                        ])   


train_datafile = './utils/datafile/frames/train.txt'
save_dir = '/home/tione/notebook/dataset/transform/train/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir) 

pathfile = read_lines(train_datafile)
train_path = pathfile[0::3]
train_label = pathfile[1::3]



datafile = []
for path, label in tqdm(zip(train_path, train_label)):
    imgs = []
    for i in range(16):
        img_path = os.path.join(path, 'image'+str(i)+'.jpg')
        img = Image.open(img_path)
        img = transform(img) 
        imgs.append(img)
    # TxCxHxW  -->  CxTxHxW
    imgs = torch.stack(imgs, dim=0).permute(1,0,2,3)
    
    
    basename = get_basename(path)
    save_path = os.path.join(save_dir, basename+'.pth')
    torch.save(imgs, save_path)
    
    datafile.append(save_path)
    datafile.append(label)
    datafile.append('')

    
datafile_save_dir = './utils/datafile/transform'  
if not os.path.exists(datafile_save_dir):
    os.makedirs(datafile_save_dir) 
# save to datafile
save_to_file(os.path.join(datafile_save_dir, 'train.txt'), datafile)