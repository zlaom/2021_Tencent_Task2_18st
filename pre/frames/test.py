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

label2id_dic = get_label2id_dic('label_id.txt')

def label2id(label2id_dic, labels):
    ids = [0 for i in range(82)]
    labels = labels.split(',')
    for label in labels:
        ids[label2id_dic[label]] = 1
    return ids

mode_list = ['test']
for ii in range(1):
    mode = mode_list[ii]
    
    datafile = '/home/tione/notebook/datafile/baseline/' + mode + '.txt'

    pathfile = read_lines(datafile)
    paths = pathfile[0::5]

    # 视频的位置
    if mode == 'train' or mode == 'val':
        absolute_dir = '/home/tione/notebook/data/raw/video/train_5k'
    else:
        absolute_dir = '/home/tione/notebook/data/raw/video/test_5k_2nd'

    # 生成图片存储的位置
    datafile_dir = '/home/tione/notebook/data/video/frames/' + mode


    datafile = []
    for path in tqdm(paths):
        # 生成对应的datafile文件

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
        # sample many images per video
        n_frames = cap.get(7)
        step = (n_frames - 1) / 63
        index = []
        for i in range(64):
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


    # save to datafile
    save_to_file(mode + '.txt', datafile)
    print(mode + '.txt', 'saved')