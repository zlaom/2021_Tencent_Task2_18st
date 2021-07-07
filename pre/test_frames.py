import os
import torch
import cv2
import numpy as np
from tqdm import tqdm

def read_lines(datafile):
    with open(datafile, 'r') as f:
        pathfile = f.read().splitlines()
    return pathfile

def get_basename(path):
    # return the file basename
    return os.path.basename(path)[:-4]

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


# these are baseline's datafiles to decide which video is train/val
val_datafile = './utils/datafile/baseline/test.txt'

pathfile = read_lines(val_datafile)
val_path = pathfile[0::5]


# video path
mode = 'test'
# 视频的位置
absolute_dir = '/home/tione/notebook/algo-2021/dataset/videos/test_5k_2nd'
# 生成图片存储的位置
datafile_dir = '/home/tione/notebook/dataset/frames/test'
if not os.path.exists(datafile_dir):
    os.makedirs(datafile_dir)

datafile = []
for path in tqdm(val_path):
    
    basename = get_basename(path)
    video = os.path.join(absolute_dir, basename+'.mp4')
    
    datafile_path = os.path.join(datafile_dir, basename)
    if not os.path.exists(datafile_path):
        os.makedirs(datafile_path)    
    
    
    cap = cv2.VideoCapture()
    if not cap.open(video):
        print('cannot open video:{}'.format(video))
    # if video exists, create save dir for this video
    datafile_path = os.path.join(datafile_dir, basename)
    if not os.path.exists(datafile_path):
        os.makedirs(datafile_path)        
    # sample 32 images per video
    n_frames = cap.get(7)
    step = (n_frames - 1) / 15
    index = []
    for i in range(16):
        index.append(round(step * i))
    # begin
    i_image = 0
    i_frame = 0
    while True:
        has_frame, frame = cap.read()
        if has_frame == False:
            break
        if i_frame in index:
            save_path = os.path.join(datafile_path, 'image' + str(i_image) + '.jpg')
            i_image += 1
            cv2.imwrite(save_path, frame)
        i_frame += 1
    cap.release()
    
    # 保存datafile的内容
    datafile.append(datafile_path)
    datafile.append('')
    datafile.append('')

    
datafile_save_dir = './utils/datafile/frames'
if not os.path.exists(datafile_save_dir):
    os.makedirs(datafile_save_dir)    
# save to datafile
save_to_file(os.path.join(datafile_save_dir, 'test.txt'), datafile)