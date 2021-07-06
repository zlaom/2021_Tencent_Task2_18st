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
            
transform = T.Compose([
                        T.Resize(256, interpolation=3),
                        T.CenterCrop(224),
                        T.ToTensor(),
                        T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
                        ])   


# these are baseline's datafiles to decide which video is train/val
datafile = '/home/tione/notebook/datafile/frames/val.txt'

pathfile = read_lines(datafile)
paths = pathfile[0::3]
labels = pathfile[1::3]

save_dir = '/home/tione/notebook/data/video/multiple_val/'

for num in range(1):
    num = 10
    # 全部随机挑选的val
    _idx = torch.randint(0,64//16,(16,))
    # _idx[0] = 1
    # _idx[-1] = 3

    datafile = []
    for path, label in tqdm(zip(paths, labels)):
        imgs = []
        for i in range(16):
            select = 4 * i + _idx[i].item()
            img_path = os.path.join(path, 'image'+str(select)+'.jpg')
            img = Image.open(img_path)
            img = transform(img) 
            imgs.append(img)

        # TxCxHxW  -->  CxTxHxW
        imgs = torch.stack(imgs, dim=0).permute(1,0,2,3)

        basename = get_basename(path)
        save_path = os.path.join(save_dir, 'val' + str(num), basename+'.pth')
        torch.save(imgs, save_path)

        datafile.append(save_path)
        datafile.append(label)
        datafile.append('')



    # save to datafile
    save_to_file('val' + str(num) + '.txt', datafile)